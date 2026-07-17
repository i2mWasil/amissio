"""
Chat API: conversational RAG endpoint over stored video chunks.

Implements a ``POST /chat`` endpoint that:
  1. Embeds the user's message via Gemini text-embedding.
  2. Queries Qdrant for the most relevant semantic chunks, filtered by
     ``video_id``.
  3. Constructs a context-rich prompt from the retrieved chunks, chat
     history, and user question.
  4. Sends the prompt to Gemini Flash for a grounded, Markdown-formatted
     answer.

Designed for a multi-turn conversational UI where users ask follow-up
questions about a video they've already generated notes for.
"""

from __future__ import annotations

import logging
import time
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from core.config import Settings, get_settings
from models.schemas import ErrorResponse
from services.llm import LLMService, LLMServiceError
from services.vector_db import VectorDBService, SearchError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Request / Response schemas
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ChatMessage(BaseModel):
    """A single message in the conversation history."""

    role: str = Field(
        ...,
        description="Message role: 'user' or 'assistant'",
        examples=["user", "assistant"],
    )
    content: str = Field(
        ...,
        min_length=1,
        description="Message text content",
    )


class ChatRequest(BaseModel):
    """POST /chat request body."""

    video_id: str = Field(
        ...,
        min_length=1,
        description="YouTube video ID to scope the retrieval",
        examples=["dQw4w9WgXcQ"],
    )
    message: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="The user's question or message",
    )
    chat_history: list[ChatMessage] = Field(
        default_factory=list,
        description="Previous conversation turns for multi-turn context",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of chunks to retrieve for context",
    )


class RetrievedChunk(BaseModel):
    """A chunk returned as part of the chat response context."""

    chunk_index: int = Field(default=0, description="Original chunk index")
    text: str = Field(default="", description="Chunk text content")
    start_time: float = Field(default=0.0, description="Chunk start time in seconds")
    end_time: float = Field(default=0.0, description="Chunk end time in seconds")
    score: float = Field(default=0.0, description="Similarity score")
    chapter_title: str | None = Field(default=None)


class ChatResponse(BaseModel):
    """POST /chat response body."""

    answer: str = Field(
        ...,
        description="Gemini-generated answer in Markdown format",
    )
    video_id: str = Field(
        ...,
        description="The video this answer is grounded in",
    )
    sources: list[RetrievedChunk] = Field(
        default_factory=list,
        description="Retrieved chunks used as context for the answer",
    )
    tokens_used: int = Field(
        default=0,
        ge=0,
        description="Approximate tokens consumed by this request",
    )
    processing_time_ms: int = Field(
        default=0,
        ge=0,
        description="Wall-clock processing time in milliseconds",
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Dependency injection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def get_llm_service(
    settings: Annotated[Settings, Depends(get_settings)],
) -> LLMService:
    return LLMService(settings)


def get_vector_db_service(
    llm: Annotated[LLMService, Depends(get_llm_service)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> VectorDBService:
    return VectorDBService(llm, settings)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# System prompt
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


_CHAT_SYSTEM_INSTRUCTION = """You are an expert study assistant that answers questions about educational video content.
You have been given relevant excerpts from the video's transcript, OCR text, and visual descriptions as context.

CRITICAL RULES:
1. Answer ONLY based on the provided context. If the context does not contain enough information to answer the question, say so explicitly — never fabricate information.
2. Format your response in clear, well-structured Markdown. Use headings, bold text, bullet points, and code blocks where appropriate.
3. When referencing specific moments from the video, include timestamps in [MM:SS] or [HH:MM:SS] format.
4. If the question asks about code, include relevant code snippets in fenced code blocks with language identifiers.
5. If the question involves math, use LaTeX notation inside $...$ for inline and $$...$$ for display equations.
6. Be concise but thorough. Prefer precision over verbosity.
7. If the user's question is a follow-up, use the conversation history for context but always ground your answer in the retrieved video content."""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _fmt_ts(seconds: float) -> str:
    """Format seconds into HH:MM:SS or MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _build_context_block(results: list[dict[str, Any]]) -> tuple[str, list[RetrievedChunk]]:
    """
    Build a context block from Qdrant search results and extract
    source metadata for the response.

    Returns:
        Tuple of (formatted context string, list of RetrievedChunk).
    """
    if not results:
        return "[No relevant context found for this video.]", []

    context_parts: list[str] = []
    sources: list[RetrievedChunk] = []

    for i, result in enumerate(results, 1):
        payload = result.get("payload", {})
        score = result.get("score", 0.0)

        text = payload.get("text", "")
        ocr_text = payload.get("ocr_text", "")
        visual_context = payload.get("visual_context", "")
        start_time = payload.get("start_time", 0.0)
        end_time = payload.get("end_time", 0.0)
        chunk_index = payload.get("chunk_index", 0)
        chapter_title = payload.get("chapter_title")
        code_snippets = payload.get("code_snippets", [])
        equations = payload.get("equations", [])

        # Build the context entry
        ts_range = f"[{_fmt_ts(start_time)} – {_fmt_ts(end_time)}]"
        header = f"--- Context {i} {ts_range}"
        if chapter_title:
            header += f" (Chapter: {chapter_title})"
        header += f" [relevance: {score:.2f}] ---"

        parts = [header]

        if text:
            parts.append(f"Transcript: {text}")
        if ocr_text:
            parts.append(f"Visible Text/Slides: {ocr_text}")
        if visual_context:
            parts.append(f"Visual Description: {visual_context}")
        if code_snippets:
            for cs in code_snippets[:3]:
                parts.append(f"Code:\n```\n{cs}\n```")
        if equations:
            parts.append(f"Equations: {', '.join(equations[:5])}")

        context_parts.append("\n".join(parts))

        sources.append(
            RetrievedChunk(
                chunk_index=chunk_index,
                text=text[:500] if text else "",
                start_time=start_time,
                end_time=end_time,
                score=round(score, 4),
                chapter_title=chapter_title,
            )
        )

    return "\n\n".join(context_parts), sources


def _build_chat_prompt(
    message: str,
    context: str,
    chat_history: list[ChatMessage],
) -> str:
    """
    Assemble the full prompt from context, history, and question.
    """
    parts: list[str] = []

    # Context block
    parts.append("## Retrieved Video Context\n")
    parts.append(context)
    parts.append("")

    # Conversation history (last 10 turns max to stay within token limits)
    recent_history = chat_history[-10:]
    if recent_history:
        parts.append("## Conversation History\n")
        for msg in recent_history:
            role_label = "User" if msg.role == "user" else "Assistant"
            parts.append(f"**{role_label}:** {msg.content}")
        parts.append("")

    # Current question
    parts.append("## Current Question\n")
    parts.append(message)
    parts.append("")
    parts.append("---")
    parts.append(
        "Answer the question above using ONLY the retrieved context. "
        "Format your response in Markdown. Include timestamp references "
        "when citing specific video sections."
    )

    return "\n".join(parts)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# POST /chat
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        404: {"model": ErrorResponse, "description": "No chunks found for video"},
        500: {"model": ErrorResponse, "description": "Internal error"},
    },
    summary="Chat with a video's content using RAG",
    description=(
        "Send a question about a previously processed video. "
        "The system retrieves the most relevant chunks from Qdrant, "
        "builds a context-rich prompt, and generates a grounded answer "
        "using Gemini Flash. Supports multi-turn conversation via chat_history."
    ),
)
async def chat(
    request: ChatRequest,
    llm_svc: Annotated[LLMService, Depends(get_llm_service)],
    vector_db_svc: Annotated[VectorDBService, Depends(get_vector_db_service)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> ChatResponse:
    """
    RAG-powered chat endpoint.

    Flow:
      1. Embed the user's message
      2. Retrieve top-k chunks from Qdrant (filtered by video_id)
      3. Build context + history prompt
      4. Generate answer via Gemini
    """
    start_time = time.monotonic()
    tokens_before = llm_svc.total_tokens_used

    video_id = request.video_id
    message = request.message
    top_k = request.top_k

    logger.info(
        "Chat request: video_id=%s, message_len=%d, history_len=%d, top_k=%d",
        video_id,
        len(message),
        len(request.chat_history),
        top_k,
    )

    # ── Step 1: Retrieve relevant chunks ──
    try:
        search_results = await vector_db_svc.search(
            query=message,
            video_id=video_id,
            limit=top_k,
            score_threshold=0.3,
        )
    except SearchError as exc:
        logger.error("Qdrant search failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Vector search failed: {exc}",
        )

    if not search_results:
        logger.warning("No chunks found for video_id=%s", video_id)
        raise HTTPException(
            status_code=404,
            detail=(
                f"No content found for video '{video_id}'. "
                "Make sure the video has been processed via /generate-notes first."
            ),
        )

    # ── Step 2: Build context and source list ──
    context_block, sources = _build_context_block(search_results)

    # ── Step 3: Assemble prompt ──
    prompt = _build_chat_prompt(
        message=message,
        context=context_block,
        chat_history=request.chat_history,
    )

    # ── Step 4: Generate answer ──
    try:
        answer = await llm_svc.generate_text(
            prompt=prompt,
            system_instruction=_CHAT_SYSTEM_INSTRUCTION,
            temperature=0.4,
            max_output_tokens=4096,
        )
    except LLMServiceError as exc:
        logger.error("Chat generation failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Answer generation failed: {exc}",
        )

    elapsed_ms = int((time.monotonic() - start_time) * 1000)
    tokens_used = llm_svc.total_tokens_used - tokens_before

    logger.info(
        "Chat response: video_id=%s, sources=%d, tokens=%d, elapsed=%dms",
        video_id,
        len(sources),
        tokens_used,
        elapsed_ms,
    )

    return ChatResponse(
        answer=answer,
        video_id=video_id,
        sources=sources,
        tokens_used=tokens_used,
        processing_time_ms=elapsed_ms,
    )
