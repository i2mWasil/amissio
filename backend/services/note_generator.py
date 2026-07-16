"""
Note generator service: Map-Reduce hierarchical note synthesis.

Implements the core intelligence of the pipeline — transforming raw
semantic chunks into textbook-quality, structured Markdown notes.

Architecture
────────────
For **short videos** (≤ ``merge_batch_size`` chunks):
  → Direct single-pass generation over all chunks.

For **long videos** (> ``merge_batch_size`` chunks):
  → **Map phase**: Generate detailed notes per chunk, concurrently
    with bounded parallelism.
  → **Reduce phase**: Recursively merge chunk-level notes in batches
    of ``merge_batch_size`` until a single cohesive document remains.
  → **Finalise**: Generate the overview, learning objectives, glossary,
    common mistakes, and quiz from the merged notes.

All LLM calls go through ``LLMService`` which handles retry, rate
limiting, and token tracking.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any
from uuid import UUID

from pydantic import BaseModel

from core.config import Settings, get_settings
from models.schemas import (
    ChunkNotes,
    CodeBlock,
    DifficultyLevel,
    Formula,
    GlossaryEntry,
    MergedNotes,
    QuizQuestion,
    QuizQuestionType,
    Reference,
    SemanticChunk,
    SegmentNotes,
    TimestampEntry,
)
from services.llm import LLMService

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Custom exceptions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class NoteGenerationError(Exception):
    """Raised when note generation fails."""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Timestamp helper
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _fmt_ts(seconds: float) -> str:
    """Format seconds into HH:MM:SS or MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Structured JSON schemas for LLM responses
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class _CodeBlockSchema(BaseModel):
    """Structured code block for LLM response."""

    language: str = ""
    code: str = ""
    explanation: str = ""


class _FormulaSchema(BaseModel):
    """Structured formula for LLM response."""

    expression: str = ""
    name: str = ""
    explanation: str = ""
    variable_names: list[str] = []
    variable_meanings: list[str] = []


class _ChunkNotesResponse(BaseModel):
    """Schema enforced on the LLM when generating per-chunk notes."""

    heading: str = ""
    summary: str = ""
    detailed_explanation: str = ""
    key_concepts: list[str] = []
    code_blocks: list[_CodeBlockSchema] = []
    formulas: list[_FormulaSchema] = []
    diagrams: list[str] = []


class _MergeResponse(BaseModel):
    """Schema enforced on the LLM when merging segment notes."""

    merged_notes: str = ""
    key_concepts: list[str] = []


class _GlossarySchema(BaseModel):
    """Structured glossary entry for LLM response."""

    term: str = ""
    definition: str = ""
    first_mentioned: str = ""
    related_terms: list[str] = []


class _ReferenceSchema(BaseModel):
    """Structured reference for LLM response."""

    title: str = ""
    url: str = ""
    context: str = ""
    timestamp: str = ""


class _TimestampSchema(BaseModel):
    """Structured timestamp for LLM response."""

    time: str = ""
    time_seconds: float = 0.0
    label: str = ""
    chapter: str = ""


class _FinaliseResponse(BaseModel):
    """Schema enforced on the LLM for the final assembly pass."""

    overview: str = ""
    learning_objectives: list[str] = []
    glossary: list[_GlossarySchema] = []
    algorithms: list[str] = []
    common_mistakes: list[str] = []
    references: list[_ReferenceSchema] = []
    timestamps: list[_TimestampSchema] = []


class _QuizItemSchema(BaseModel):
    """Structured quiz question for LLM response."""

    question_id: int = 1
    question_type: str = "short_answer"
    difficulty: str = "medium"
    question: str = ""
    options: list[str] = []
    correct_answer: str = ""
    explanation: str = ""
    related_timestamp: str = ""


class _QuizResponse(BaseModel):
    """Schema enforced on the LLM for quiz generation."""

    questions: list[_QuizItemSchema] = []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Prompts
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


_SYSTEM_NOTE_GENERATOR = """You are an expert educational content writer who transforms video transcripts into textbook-quality study notes.

CRITICAL RULES:
1. Write in clear, precise academic language. Explain concepts as if writing a university textbook.
2. Preserve ALL technical details — every formula, algorithm, code snippet, and definition must be included.
3. Use proper Markdown formatting: headings (##, ###), bold for key terms, bullet lists, numbered lists, code blocks with language tags, and LaTeX for equations.
4. Do NOT add information that is not present in the source material. Synthesise and restructure, but never fabricate.
5. When code is shown, include it in fenced code blocks with the correct language identifier.
6. When equations are shown, use LaTeX notation inside $...$ or $$...$$ blocks.
7. Attribute timestamps to every section so readers can jump to the video."""


_MAP_CHUNK_PROMPT = """Analyze the following chunk from an educational video and generate detailed, structured notes.

## Chunk Information
- **Time Range**: {start} → {end}
- **Chapter**: {chapter}

## Transcript (spoken content)
{transcript}

## Visual Context (what is shown on screen)
{visual_context}

## OCR Text (visible text/slides)
{ocr_text}

## Code Snippets Visible
{code_snippets}

## Equations Visible
{equations}

---

Generate detailed notes for this chunk. Your response must be a JSON object with:
- "heading": A descriptive section heading for this chunk (not just "Introduction" — be specific).
- "summary": A 2-3 sentence concise summary.
- "detailed_explanation": A thorough Markdown explanation covering all concepts, with examples. This should be textbook-quality — imagine a student reading ONLY this section.
- "key_concepts": An array of the key concepts/takeaways (3-8 items).
- "code_blocks": An array of objects with "language", "code", "explanation" for each code block discussed.
- "formulas": An array of objects with "expression" (LaTeX), "name", "explanation", "variables" (dict) for each formula.
- "diagrams": An array of strings describing any diagrams, charts, or visual aids discussed."""


_REDUCE_MERGE_PROMPT = """You are merging multiple sections of educational notes into a single cohesive document.

Below are {count} sections of detailed notes from consecutive parts of an educational video.
Your task is to merge them into ONE continuous, well-structured Markdown document that flows naturally.

RULES:
1. Eliminate redundancy — if the same concept appears in multiple sections, consolidate into one clear explanation.
2. Maintain chronological flow — preserve the order of topics as they appeared in the video.
3. Use a proper heading hierarchy: ## for major topics, ### for subtopics, #### for details.
4. Preserve ALL code blocks, equations, and technical details — do not summarise them away.
5. Add transition sentences between sections so the document reads smoothly.
6. Include timestamp references in the format [MM:SS] or [HH:MM:SS] at the start of each major section.

## Sections to Merge

{sections}

---

Return a JSON object with:
- "merged_notes": The complete merged Markdown notes.
- "key_concepts": A deduplicated array of all key concepts across all sections."""


_FINALISE_PROMPT = """You are finalising a set of educational notes generated from a video titled "{title}".

Below are the complete detailed notes:

{notes}

---

Based on these notes, generate the supplementary sections. Return a JSON object with:

- "overview": A 3-5 sentence executive summary of the entire video's content and purpose.
- "learning_objectives": An array of 5-10 specific, measurable learning objectives (use action verbs: "Explain...", "Implement...", "Compare...", "Derive...").
- "glossary": An array of objects with "term", "definition", "first_mentioned" (timestamp like "MM:SS"), and "related_terms" (array of strings). Include 10-30 key terms.
- "algorithms": An array of key algorithms discussed (empty array if none).
- "common_mistakes": An array of common mistakes or pitfalls mentioned (empty if none).
- "references": An array of objects with "title", "url" (null if not given), "context", "timestamp". Include any papers, books, tools, libraries, or links mentioned.
- "timestamps": An array of objects with "time" (formatted "MM:SS" or "HH:MM:SS"), "time_seconds" (float), "label" (description), "chapter" (string or null). Include 10-30 key moments."""


_QUIZ_PROMPT = """You are generating a comprehensive quiz based on educational video notes.

## Video Title
{title}

## Notes Content
{notes}

---

Generate {count} quiz questions that thoroughly test understanding of the material.

RULES:
1. Mix question types: multiple_choice, true_false, short_answer, fill_in_the_blank.
2. Mix difficulty levels: approximately 30% easy, 50% medium, 20% hard.
3. Questions should test conceptual understanding, not just memorisation.
4. For multiple_choice questions, provide exactly 4 options.
5. Include an explanation for each correct answer.
6. Reference the relevant video timestamp for each question.

Return a JSON object with a "questions" array, where each element has:
- "question_id": Sequential number starting from 1.
- "question_type": One of "multiple_choice", "true_false", "short_answer", "fill_in_the_blank".
- "difficulty": One of "easy", "medium", "hard".
- "question": The question text.
- "options": Array of 4 strings (for multiple_choice only, null otherwise).
- "correct_answer": The correct answer string.
- "explanation": Why the answer is correct (1-2 sentences).
- "related_timestamp": The relevant video timestamp ("MM:SS")."""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Note Generator Service
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class NoteGeneratorService:
    """
    Map-Reduce note generation engine.

    Transforms semantic chunks into textbook-quality structured notes
    with overview, learning objectives, detailed sections, glossary,
    and auto-generated quiz.
    """

    def __init__(
        self,
        llm_service: LLMService,
        settings: Settings | None = None,
    ) -> None:
        self._llm = llm_service
        self._settings = settings or get_settings()
        self._merge_batch_size = self._settings.merge_batch_size
        self._max_parallel = self._settings.max_parallel_segments

    # ─────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────

    async def generate_notes(
        self,
        chunks: list[SemanticChunk],
        video_id: str,
        video_title: str,
        video_duration: int,
    ) -> MergedNotes:
        """
        Full Map-Reduce note generation pipeline.

        Args:
            chunks: Ordered semantic chunks from the chunking service.
            video_id: YouTube video ID.
            video_title: Video title for context.
            video_duration: Video duration in seconds.

        Returns:
            A ``MergedNotes`` object containing all generated content.

        Raises:
            NoteGenerationError: If the pipeline fails critically.
        """
        if not chunks:
            raise NoteGenerationError("No chunks provided for note generation")

        start_time = time.monotonic()
        total_tokens = 0

        logger.info(
            "Starting note generation: video='%s', chunks=%d, duration=%ds",
            video_title,
            len(chunks),
            video_duration,
        )

        # ── Step 1: MAP — generate notes per chunk ──
        logger.info("MAP phase: generating notes for %d chunks", len(chunks))
        chunk_notes_list, map_tokens = await self._map_phase(chunks)
        total_tokens += map_tokens

        logger.info(
            "MAP complete: %d chunk notes generated, %d tokens used",
            len(chunk_notes_list),
            map_tokens,
        )

        # ── Step 2: REDUCE — merge chunk notes hierarchically ──
        logger.info("REDUCE phase: merging %d chunk notes", len(chunk_notes_list))
        merged_markdown, merge_concepts, reduce_tokens = await self._reduce_phase(
            chunk_notes_list, chunks
        )
        total_tokens += reduce_tokens

        logger.info(
            "REDUCE complete: %d chars of merged notes, %d tokens used",
            len(merged_markdown),
            reduce_tokens,
        )

        # ── Step 3: FINALISE — generate overview, glossary, etc. ──
        logger.info("FINALISE phase: generating supplementary sections")
        final_data, final_tokens = await self._finalise_phase(
            merged_markdown, video_title
        )
        total_tokens += final_tokens

        # ── Step 4: QUIZ — generate quiz questions ──
        logger.info("QUIZ phase: generating questions")
        quiz_count = min(20, max(10, len(chunks) * 2))
        quiz_questions, quiz_tokens = await self._quiz_phase(
            merged_markdown, video_title, quiz_count
        )
        total_tokens += quiz_tokens

        # ── Step 5: ASSEMBLE — build the final Markdown document ──
        full_markdown = self._assemble_final_markdown(
            video_title=video_title,
            video_duration=video_duration,
            overview=final_data.get("overview", ""),
            learning_objectives=final_data.get("learning_objectives", []),
            detailed_notes=merged_markdown,
            glossary_entries=final_data.get("glossary", []),
            algorithms=final_data.get("algorithms", []),
            common_mistakes=final_data.get("common_mistakes", []),
            quiz_questions=quiz_questions,
            timestamps=final_data.get("timestamps", []),
        )

        # Build timestamps
        timestamps = self._parse_timestamps(final_data.get("timestamps", []))

        # Build glossary
        glossary = self._parse_glossary(final_data.get("glossary", []))

        # Build references
        references = self._parse_references(final_data.get("references", []))

        elapsed = time.monotonic() - start_time
        logger.info(
            "Note generation complete: %.1fs elapsed, %d tokens total, "
            "%d chars final document",
            elapsed,
            total_tokens,
            len(full_markdown),
        )

        return MergedNotes(
            video_id=video_id,
            title=video_title,
            overview=final_data.get("overview", ""),
            learning_objectives=final_data.get("learning_objectives", []),
            detailed_notes_markdown=full_markdown,
            glossary=glossary,
            algorithms=final_data.get("algorithms", []),
            common_mistakes=final_data.get("common_mistakes", []),
            quiz_questions=quiz_questions,
            timestamps=timestamps,
            references=references,
            total_tokens_used=total_tokens,
        )

    # ─────────────────────────────────────────────
    # MAP Phase
    # ─────────────────────────────────────────────

    async def _map_phase(
        self,
        chunks: list[SemanticChunk],
    ) -> tuple[list[ChunkNotes], int]:
        """
        Generate detailed notes for each semantic chunk concurrently.

        Returns:
            Tuple of (ordered list of ChunkNotes, total tokens used).
        """
        semaphore = asyncio.Semaphore(self._max_parallel)
        results: list[ChunkNotes | None] = [None] * len(chunks)
        total_tokens = 0

        async def process_chunk(idx: int, chunk: SemanticChunk) -> None:
            nonlocal total_tokens
            async with semaphore:
                notes = await self._generate_chunk_notes(chunk)
                results[idx] = notes

        tasks = [process_chunk(i, c) for i, c in enumerate(chunks)]
        await asyncio.gather(*tasks, return_exceptions=False)

        # Filter out any None results (shouldn't happen, but safety)
        chunk_notes: list[ChunkNotes] = [r for r in results if r is not None]

        return chunk_notes, self._llm.total_tokens_used

    async def _generate_chunk_notes(self, chunk: SemanticChunk) -> ChunkNotes:
        """Generate detailed notes for a single semantic chunk."""
        prompt = _MAP_CHUNK_PROMPT.format(
            start=_fmt_ts(chunk.start_time),
            end=_fmt_ts(chunk.end_time),
            chapter=chunk.chapter_title or "N/A",
            transcript=chunk.text or "[No transcript available]",
            visual_context=chunk.visual_context or "[No visual context]",
            ocr_text=chunk.ocr_text or "[No OCR text]",
            code_snippets="\n".join(f"```\n{cs}\n```" for cs in chunk.code_snippets) or "[None]",
            equations=", ".join(chunk.equations) or "[None]",
        )

        try:
            response_text = await self._llm.generate_text(
                prompt=prompt,
                system_instruction=_SYSTEM_NOTE_GENERATOR,
                response_schema=_ChunkNotesResponse,
                temperature=0.3,
                max_output_tokens=4096,
            )

            data = json.loads(response_text)

            # Parse code blocks
            code_blocks: list[CodeBlock] = []
            for cb in data.get("code_blocks", []):
                if isinstance(cb, dict) and cb.get("code"):
                    code_blocks.append(
                        CodeBlock(
                            language=cb.get("language", ""),
                            code=cb["code"],
                            explanation=cb.get("explanation", ""),
                            line_annotations=cb.get("line_annotations", {}),
                        )
                    )

            # Parse formulas
            formulas: list[Formula] = []
            for f in data.get("formulas", []):
                if isinstance(f, dict) and f.get("expression"):
                    # Build variables dict from either format
                    variables: dict[str, str] = f.get("variables", {})
                    if not variables:
                        # New schema: parallel lists
                        names = f.get("variable_names", [])
                        meanings = f.get("variable_meanings", [])
                        variables = dict(zip(names, meanings))
                    formulas.append(
                        Formula(
                            expression=f["expression"],
                            name=f.get("name", ""),
                            explanation=f.get("explanation", ""),
                            variables=variables,
                        )
                    )

            ts_label = f"{_fmt_ts(chunk.start_time)} – {_fmt_ts(chunk.end_time)}"

            return ChunkNotes(
                chunk_id=chunk.chunk_id,
                heading=data.get("heading", f"Section at {ts_label}"),
                summary=data.get("summary", ""),
                detailed_explanation=data.get("detailed_explanation", ""),
                key_concepts=data.get("key_concepts", []),
                code_blocks=code_blocks,
                formulas=formulas,
                diagrams=data.get("diagrams", []),
                start_time=chunk.start_time,
                end_time=chunk.end_time,
                timestamp_label=ts_label,
            )

        except json.JSONDecodeError:
            # If JSON parsing fails, treat the response as plain markdown
            logger.warning(
                "Chunk %d: LLM returned non-JSON response, using as raw notes",
                chunk.chunk_index,
            )
            ts_label = f"{_fmt_ts(chunk.start_time)} – {_fmt_ts(chunk.end_time)}"
            return ChunkNotes(
                chunk_id=chunk.chunk_id,
                heading=f"Section at {ts_label}",
                summary="",
                detailed_explanation=response_text,
                key_concepts=[],
                code_blocks=[],
                formulas=[],
                diagrams=[],
                start_time=chunk.start_time,
                end_time=chunk.end_time,
                timestamp_label=ts_label,
            )
        except Exception as exc:
            logger.error("Chunk notes generation failed: %s", exc)
            ts_label = f"{_fmt_ts(chunk.start_time)} – {_fmt_ts(chunk.end_time)}"
            return ChunkNotes(
                chunk_id=chunk.chunk_id,
                heading=f"Section at {ts_label}",
                summary="[Note generation failed for this section]",
                detailed_explanation=chunk.text,
                key_concepts=[],
                code_blocks=[],
                formulas=[],
                diagrams=[],
                start_time=chunk.start_time,
                end_time=chunk.end_time,
                timestamp_label=ts_label,
            )

    # ─────────────────────────────────────────────
    # REDUCE Phase
    # ─────────────────────────────────────────────

    async def _reduce_phase(
        self,
        chunk_notes: list[ChunkNotes],
        chunks: list[SemanticChunk],
    ) -> tuple[str, list[str], int]:
        """
        Recursively merge chunk notes into a single cohesive document.

        For small sets (≤ merge_batch_size), merges in a single pass.
        For large sets, recursively batches and merges until one document
        remains.

        Returns:
            Tuple of (merged markdown, deduplicated key concepts, tokens used).
        """
        tokens_before = self._llm.total_tokens_used

        # Convert chunk notes to markdown sections for merging
        sections = self._chunk_notes_to_sections(chunk_notes)

        if len(sections) <= self._merge_batch_size:
            # Single-pass merge
            merged, concepts = await self._merge_sections(sections)
            tokens_used = self._llm.total_tokens_used - tokens_before
            return merged, concepts, tokens_used

        # Recursive hierarchical merge
        current_level = sections
        all_concepts: list[str] = []

        while len(current_level) > 1:
            # Split into batches
            batches = [
                current_level[i : i + self._merge_batch_size]
                for i in range(0, len(current_level), self._merge_batch_size)
            ]

            logger.info(
                "Reduce level: %d sections → %d batches of ≤%d",
                len(current_level),
                len(batches),
                self._merge_batch_size,
            )

            # Process batches concurrently
            semaphore = asyncio.Semaphore(self._max_parallel)
            next_level: list[str] = [None] * len(batches)  # type: ignore[list-item]
            batch_concepts: list[list[str]] = [[] for _ in batches]

            async def merge_batch(idx: int, batch: list[str]) -> None:
                async with semaphore:
                    merged, concepts = await self._merge_sections(batch)
                    next_level[idx] = merged
                    batch_concepts[idx] = concepts

            tasks = [merge_batch(i, b) for i, b in enumerate(batches)]
            await asyncio.gather(*tasks)

            current_level = [s for s in next_level if s is not None]
            for bc in batch_concepts:
                all_concepts.extend(bc)

        merged_result = current_level[0] if current_level else ""
        deduped_concepts = list(dict.fromkeys(all_concepts))
        tokens_used = self._llm.total_tokens_used - tokens_before

        return merged_result, deduped_concepts, tokens_used

    async def _merge_sections(
        self,
        sections: list[str],
    ) -> tuple[str, list[str]]:
        """
        Merge a batch of Markdown sections into one cohesive document.

        Returns:
            Tuple of (merged markdown, key concepts).
        """
        numbered_sections = "\n\n---\n\n".join(
            f"### Section {i + 1}\n\n{section}" for i, section in enumerate(sections)
        )

        prompt = _REDUCE_MERGE_PROMPT.format(
            count=len(sections),
            sections=numbered_sections,
        )

        try:
            response_text = await self._llm.generate_text(
                prompt=prompt,
                system_instruction=_SYSTEM_NOTE_GENERATOR,
                response_schema=_MergeResponse,
                temperature=0.2,
                max_output_tokens=8192,
            )

            data = json.loads(response_text)
            merged = data.get("merged_notes", "")
            concepts = data.get("key_concepts", [])

            if not merged:
                # Fallback: concatenate sections if LLM returns empty
                merged = "\n\n---\n\n".join(sections)

            return merged, concepts

        except (json.JSONDecodeError, Exception) as exc:
            logger.warning("Merge failed (%s), falling back to concatenation", exc)
            return "\n\n---\n\n".join(sections), []

    # ─────────────────────────────────────────────
    # FINALISE Phase
    # ─────────────────────────────────────────────

    async def _finalise_phase(
        self,
        merged_notes: str,
        video_title: str,
    ) -> tuple[dict[str, Any], int]:
        """
        Generate supplementary sections: overview, learning objectives,
        glossary, algorithms, common mistakes, references, timestamps.

        Returns:
            Tuple of (parsed data dict, tokens used).
        """
        tokens_before = self._llm.total_tokens_used

        # Truncate notes if too long for the context window
        notes_for_prompt = merged_notes
        if len(notes_for_prompt) > 60000:
            notes_for_prompt = notes_for_prompt[:60000] + "\n\n[... truncated ...]"

        prompt = _FINALISE_PROMPT.format(
            title=video_title,
            notes=notes_for_prompt,
        )

        try:
            response_text = await self._llm.generate_text(
                prompt=prompt,
                system_instruction=_SYSTEM_NOTE_GENERATOR,
                response_schema=_FinaliseResponse,
                temperature=0.2,
                max_output_tokens=8192,
            )

            data = json.loads(response_text)
            tokens_used = self._llm.total_tokens_used - tokens_before
            return data, tokens_used

        except (json.JSONDecodeError, Exception) as exc:
            logger.error("Finalise phase failed: %s", exc)
            tokens_used = self._llm.total_tokens_used - tokens_before
            return {
                "overview": f"Notes generated from: {video_title}",
                "learning_objectives": [],
                "glossary": [],
                "algorithms": [],
                "common_mistakes": [],
                "references": [],
                "timestamps": [],
            }, tokens_used

    # ─────────────────────────────────────────────
    # QUIZ Phase
    # ─────────────────────────────────────────────

    async def _quiz_phase(
        self,
        merged_notes: str,
        video_title: str,
        question_count: int,
    ) -> tuple[list[QuizQuestion], int]:
        """
        Generate quiz questions from the merged notes.

        Returns:
            Tuple of (list of QuizQuestion, tokens used).
        """
        tokens_before = self._llm.total_tokens_used

        notes_for_prompt = merged_notes
        if len(notes_for_prompt) > 50000:
            notes_for_prompt = notes_for_prompt[:50000] + "\n\n[... truncated ...]"

        prompt = _QUIZ_PROMPT.format(
            title=video_title,
            notes=notes_for_prompt,
            count=question_count,
        )

        try:
            response_text = await self._llm.generate_text(
                prompt=prompt,
                system_instruction=_SYSTEM_NOTE_GENERATOR,
                response_schema=_QuizResponse,
                temperature=0.4,
                max_output_tokens=8192,
            )

            data = json.loads(response_text)
            questions = self._parse_quiz_questions(data.get("questions", []))
            tokens_used = self._llm.total_tokens_used - tokens_before
            return questions, tokens_used

        except (json.JSONDecodeError, Exception) as exc:
            logger.error("Quiz generation failed: %s", exc)
            tokens_used = self._llm.total_tokens_used - tokens_before
            return [], tokens_used

    # ─────────────────────────────────────────────
    # Markdown assembly
    # ─────────────────────────────────────────────

    def _assemble_final_markdown(
        self,
        video_title: str,
        video_duration: int,
        overview: str,
        learning_objectives: list[str],
        detailed_notes: str,
        glossary_entries: list[dict[str, Any]],
        algorithms: list[str],
        common_mistakes: list[str],
        quiz_questions: list[QuizQuestion],
        timestamps: list[dict[str, Any]],
    ) -> str:
        """Assemble all sections into the final Markdown document."""
        parts: list[str] = []

        # ── Title ──
        parts.append(f"# 📚 {video_title}")
        parts.append("")
        parts.append(f"*Duration: {_fmt_ts(video_duration)}*")
        parts.append("")

        # ── Video Overview ──
        parts.append("## 📋 Video Overview")
        parts.append("")
        parts.append(overview or "*Overview not available.*")
        parts.append("")

        # ── Key Timestamps ──
        if timestamps:
            parts.append("## ⏱️ Key Timestamps")
            parts.append("")
            parts.append("| Time | Topic |")
            parts.append("|------|-------|")
            for ts in timestamps:
                time_str = ts.get("time", "00:00")
                label = ts.get("label", "")
                parts.append(f"| {time_str} | {label} |")
            parts.append("")

        # ── Learning Objectives ──
        if learning_objectives:
            parts.append("## 🎯 Learning Objectives")
            parts.append("")
            parts.append("After studying these notes, you should be able to:")
            parts.append("")
            for i, obj in enumerate(learning_objectives, 1):
                parts.append(f"{i}. {obj}")
            parts.append("")

        # ── Detailed Notes ──
        parts.append("## 📖 Detailed Notes")
        parts.append("")
        parts.append(detailed_notes)
        parts.append("")

        # ── Algorithms ──
        if algorithms:
            parts.append("## 🔬 Key Algorithms")
            parts.append("")
            for alg in algorithms:
                parts.append(f"- {alg}")
            parts.append("")

        # ── Common Mistakes ──
        if common_mistakes:
            parts.append("## ⚠️ Common Mistakes & Pitfalls")
            parts.append("")
            for mistake in common_mistakes:
                parts.append(f"- {mistake}")
            parts.append("")

        # ── Glossary ──
        if glossary_entries:
            parts.append("## 📖 Glossary")
            parts.append("")
            parts.append("| Term | Definition |")
            parts.append("|------|-----------|")
            for entry in glossary_entries:
                term = entry.get("term", "")
                defn = entry.get("definition", "")
                parts.append(f"| **{term}** | {defn} |")
            parts.append("")

        # ── Quiz ──
        if quiz_questions:
            parts.append("## 🧠 Self-Assessment Quiz")
            parts.append("")
            for q in quiz_questions:
                difficulty_badge = {
                    DifficultyLevel.EASY: "🟢 Easy",
                    DifficultyLevel.MEDIUM: "🟡 Medium",
                    DifficultyLevel.HARD: "🔴 Hard",
                }.get(q.difficulty, "🟡 Medium")

                parts.append(f"### Q{q.question_id}. {q.question}")
                parts.append(f"*{difficulty_badge}* | *{q.question_type.value.replace('_', ' ').title()}*")
                parts.append("")

                if q.options:
                    for i, opt in enumerate(q.options):
                        letter = chr(65 + i)  # A, B, C, D
                        parts.append(f"  {letter}. {opt}")
                    parts.append("")

                parts.append(f"<details><summary>Show Answer</summary>")
                parts.append("")
                parts.append(f"**Answer:** {q.correct_answer}")
                parts.append("")
                if q.explanation:
                    parts.append(f"**Explanation:** {q.explanation}")
                if q.related_timestamp:
                    parts.append(f"")
                    parts.append(f"*📍 See video at {q.related_timestamp}*")
                parts.append("")
                parts.append("</details>")
                parts.append("")

        # ── Footer ──
        parts.append("---")
        parts.append("")
        parts.append("*Notes generated by Amissio — Multimodal RAG Pipeline*")

        return "\n".join(parts)

    # ─────────────────────────────────────────────
    # Conversion helpers
    # ─────────────────────────────────────────────

    @staticmethod
    def _chunk_notes_to_sections(chunk_notes: list[ChunkNotes]) -> list[str]:
        """Convert a list of ChunkNotes into Markdown section strings."""
        sections: list[str] = []
        for cn in chunk_notes:
            parts: list[str] = []

            parts.append(f"## {cn.heading}")
            parts.append(f"*[{cn.timestamp_label}]*")
            parts.append("")

            if cn.summary:
                parts.append(f"> **Summary:** {cn.summary}")
                parts.append("")

            if cn.detailed_explanation:
                parts.append(cn.detailed_explanation)
                parts.append("")

            if cn.key_concepts:
                parts.append("**Key Concepts:**")
                for kc in cn.key_concepts:
                    parts.append(f"- {kc}")
                parts.append("")

            for cb in cn.code_blocks:
                lang = cb.language or ""
                parts.append(f"```{lang}")
                parts.append(cb.code)
                parts.append("```")
                if cb.explanation:
                    parts.append(f"*{cb.explanation}*")
                parts.append("")

            for formula in cn.formulas:
                if formula.name:
                    parts.append(f"**{formula.name}:**")
                parts.append(f"$${formula.expression}$$")
                if formula.explanation:
                    parts.append(f"*{formula.explanation}*")
                if formula.variables:
                    parts.append("Where:")
                    for var, meaning in formula.variables.items():
                        parts.append(f"- ${var}$ = {meaning}")
                parts.append("")

            if cn.diagrams:
                parts.append("**Visual Aids:**")
                for d in cn.diagrams:
                    parts.append(f"- 📊 {d}")
                parts.append("")

            sections.append("\n".join(parts))

        return sections

    @staticmethod
    def _parse_timestamps(raw: list[dict[str, Any]]) -> list[TimestampEntry]:
        """Parse raw timestamp dicts into TimestampEntry models."""
        timestamps: list[TimestampEntry] = []
        for item in raw:
            try:
                timestamps.append(
                    TimestampEntry(
                        time=item.get("time", "00:00"),
                        time_seconds=float(item.get("time_seconds", 0)),
                        label=item.get("label", ""),
                        chapter=item.get("chapter"),
                    )
                )
            except Exception:
                continue
        return timestamps

    @staticmethod
    def _parse_glossary(raw: list[dict[str, Any]]) -> list[GlossaryEntry]:
        """Parse raw glossary dicts into GlossaryEntry models."""
        glossary: list[GlossaryEntry] = []
        for item in raw:
            try:
                glossary.append(
                    GlossaryEntry(
                        term=item.get("term", ""),
                        definition=item.get("definition", ""),
                        first_mentioned=item.get("first_mentioned", ""),
                        related_terms=item.get("related_terms", []),
                    )
                )
            except Exception:
                continue
        return glossary

    @staticmethod
    def _parse_references(raw: list[dict[str, Any]]) -> list[Reference]:
        """Parse raw reference dicts into Reference models."""
        references: list[Reference] = []
        for item in raw:
            try:
                references.append(
                    Reference(
                        title=item.get("title", ""),
                        url=item.get("url"),
                        context=item.get("context", ""),
                        timestamp=item.get("timestamp", ""),
                    )
                )
            except Exception:
                continue
        return references

    @staticmethod
    def _parse_quiz_questions(raw: list[dict[str, Any]]) -> list[QuizQuestion]:
        """Parse raw quiz question dicts into QuizQuestion models."""
        questions: list[QuizQuestion] = []

        # Valid enum maps
        type_map = {
            "multiple_choice": QuizQuestionType.MULTIPLE_CHOICE,
            "true_false": QuizQuestionType.TRUE_FALSE,
            "short_answer": QuizQuestionType.SHORT_ANSWER,
            "fill_in_the_blank": QuizQuestionType.FILL_IN_THE_BLANK,
        }
        diff_map = {
            "easy": DifficultyLevel.EASY,
            "medium": DifficultyLevel.MEDIUM,
            "hard": DifficultyLevel.HARD,
        }

        for i, item in enumerate(raw):
            try:
                q_type_str = item.get("question_type", "short_answer").lower()
                q_type = type_map.get(q_type_str, QuizQuestionType.SHORT_ANSWER)

                diff_str = item.get("difficulty", "medium").lower()
                difficulty = diff_map.get(diff_str, DifficultyLevel.MEDIUM)

                options = item.get("options")
                if q_type != QuizQuestionType.MULTIPLE_CHOICE:
                    options = None

                questions.append(
                    QuizQuestion(
                        question_id=item.get("question_id", i + 1),
                        question_type=q_type,
                        difficulty=difficulty,
                        question=item.get("question", ""),
                        options=options,
                        correct_answer=item.get("correct_answer", ""),
                        explanation=item.get("explanation", ""),
                        related_timestamp=item.get("related_timestamp", ""),
                    )
                )
            except Exception as exc:
                logger.warning("Failed to parse quiz question %d: %s", i, exc)
                continue

        return questions
