"""
API routes: POST /generate-notes endpoint and pipeline orchestration.

This module contains the FastAPI router that wires together every
service in the correct order:

  URL → Metadata → Download (audio + video + captions) → Transcribe
    → Extract frames → Deduplicate → OCR / Vision → Synchronise
    → Chunk → Generate notes (Map-Reduce) → Embed → Store → Respond

Each pipeline stage updates the metadata and records non-fatal errors
so the caller always gets a response — even if individual stages
partially fail.
"""

from __future__ import annotations

import logging
import time
import traceback
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from core.config import Settings, get_settings
from models.schemas import (
    ErrorResponse,
    GenerateNotesRequest,
    GenerateNotesResponse,
    NoteGenerationMetadata,
    PipelineError,
    PipelineStatus,
    TranscriptSource,
)
from services.chunking import ChunkingService
from services.llm import LLMService
from services.note_generator import NoteGenerationError, NoteGeneratorService
from services.ocr import OCRService
from services.transcript import TranscriptService
from services.vector_db import VectorDBService
from services.video import (
    InvalidURLError,
    VideoAgeRestrictedError,
    VideoDurationExceededError,
    VideoDownloadError,
    VideoNotFoundError,
    VideoPrivateError,
    VideoService,
)
from services.vision import VisionService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["notes"])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Dependency injection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def get_video_service(
    settings: Annotated[Settings, Depends(get_settings)],
) -> VideoService:
    return VideoService(settings)


def get_transcript_service(
    settings: Annotated[Settings, Depends(get_settings)],
) -> TranscriptService:
    return TranscriptService(settings)


def get_vision_service(
    settings: Annotated[Settings, Depends(get_settings)],
) -> VisionService:
    return VisionService(settings)


def get_llm_service(
    settings: Annotated[Settings, Depends(get_settings)],
) -> LLMService:
    return LLMService(settings)


def get_chunking_service(
    settings: Annotated[Settings, Depends(get_settings)],
) -> ChunkingService:
    return ChunkingService(settings)


def get_ocr_service(
    llm: Annotated[LLMService, Depends(get_llm_service)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> OCRService:
    return OCRService(llm, settings)


def get_vector_db_service(
    llm: Annotated[LLMService, Depends(get_llm_service)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> VectorDBService:
    return VectorDBService(llm, settings)


def get_note_generator_service(
    llm: Annotated[LLMService, Depends(get_llm_service)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> NoteGeneratorService:
    return NoteGeneratorService(llm, settings)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# POST /generate-notes
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@router.post(
    "/generate-notes",
    response_model=GenerateNotesResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid URL or request"},
        403: {"model": ErrorResponse, "description": "Video is private or age-restricted"},
        404: {"model": ErrorResponse, "description": "Video not found"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal pipeline error"},
    },
    summary="Generate textbook-quality notes from a YouTube video",
    description=(
        "Accepts a YouTube URL and runs the full multimodal RAG pipeline: "
        "metadata extraction, audio/video download, transcription (captions or Whisper), "
        "frame extraction, OCR/vision analysis, semantic chunking, "
        "Map-Reduce note generation, embedding, and vector storage."
    ),
)
async def generate_notes(
    request: GenerateNotesRequest,
    video_svc: Annotated[VideoService, Depends(get_video_service)],
    transcript_svc: Annotated[TranscriptService, Depends(get_transcript_service)],
    vision_svc: Annotated[VisionService, Depends(get_vision_service)],
    ocr_svc: Annotated[OCRService, Depends(get_ocr_service)],
    chunking_svc: Annotated[ChunkingService, Depends(get_chunking_service)],
    note_gen_svc: Annotated[NoteGeneratorService, Depends(get_note_generator_service)],
    vector_db_svc: Annotated[VectorDBService, Depends(get_vector_db_service)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> GenerateNotesResponse:
    """
    Full pipeline orchestration endpoint.

    Runs every stage sequentially (with internal parallelism within
    stages) and returns the final structured response.
    """
    url = str(request.url)
    pipeline_start = time.monotonic()
    pipeline_errors: list[PipelineError] = []
    transcript_source = TranscriptSource.WHISPER  # default, updated later

    # ─────────────────────────────────────────────
    # Stage 1: METADATA EXTRACTION
    # ─────────────────────────────────────────────
    try:
        metadata = await video_svc.extract_metadata(url)
    except InvalidURLError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except VideoNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except VideoPrivateError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except VideoAgeRestrictedError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except VideoDurationExceededError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Metadata extraction failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Metadata extraction failed: {exc}")

    video_id = metadata.video_id
    logger.info(
        "Pipeline started: video_id=%s, title='%s', duration=%ds",
        video_id,
        metadata.title,
        metadata.duration,
    )

    # ─────────────────────────────────────────────
    # Stage 2: DOWNLOAD (audio + video + captions)
    # ─────────────────────────────────────────────
    audio_path = None
    video_path = None
    caption_path = None

    try:
        # Download audio and captions in parallel; video separately
        import asyncio

        audio_task = video_svc.download_audio(url, video_id)
        caption_task = video_svc.download_captions(url, video_id)
        video_task = video_svc.download_video(url, video_id)

        audio_path, caption_path, video_path = await asyncio.gather(
            audio_task, caption_task, video_task
        )

    except VideoDownloadError as exc:
        logger.error("Download failed: %s", exc)
        pipeline_errors.append(
            PipelineError(
                stage=PipelineStatus.DOWNLOADING,
                message=str(exc),
                recoverable=False,
            )
        )
        raise HTTPException(status_code=500, detail=f"Video download failed: {exc}")
    except Exception as exc:
        logger.error("Download failed unexpectedly: %s", exc)
        pipeline_errors.append(
            PipelineError(
                stage=PipelineStatus.DOWNLOADING,
                message=str(exc),
                recoverable=False,
            )
        )
        raise HTTPException(status_code=500, detail=f"Download failed: {exc}")

    # ─────────────────────────────────────────────
    # Stage 3: TRANSCRIPTION
    # ─────────────────────────────────────────────
    try:
        transcript = await transcript_svc.get_transcript(
            caption_path=caption_path,
            audio_path=audio_path,
        )
        transcript_source = transcript.source
        logger.info(
            "Transcript ready: source=%s, segments=%d, words=%d",
            transcript.source.value,
            len(transcript.segments),
            transcript.word_count,
        )
    except Exception as exc:
        logger.error("Transcription failed: %s", exc)
        pipeline_errors.append(
            PipelineError(
                stage=PipelineStatus.TRANSCRIBING,
                message=str(exc),
                recoverable=False,
            )
        )
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}")

    # ─────────────────────────────────────────────
    # Stage 4: FRAME EXTRACTION + DEDUPLICATION
    # ─────────────────────────────────────────────
    frame_result = None
    total_frames = 0
    unique_frames = 0

    if video_path:
        try:
            frame_result = await vision_svc.extract_frames(video_path, video_id)
            total_frames = frame_result.total_extracted
            unique_frames = frame_result.total_unique
            logger.info(
                "Frames: %d extracted, %d unique (%.0f%% dedup)",
                total_frames,
                unique_frames,
                frame_result.dedup_ratio * 100,
            )
        except Exception as exc:
            logger.warning("Frame extraction failed (non-fatal): %s", exc)
            pipeline_errors.append(
                PipelineError(
                    stage=PipelineStatus.EXTRACTING_FRAMES,
                    message=str(exc),
                    recoverable=True,
                )
            )

    # ─────────────────────────────────────────────
    # Stage 5: OCR / VISION ANALYSIS
    # ─────────────────────────────────────────────
    vision_analyses = []

    if frame_result and frame_result.unique_frames:
        try:
            vision_analyses = await ocr_svc.analyze_frames(
                frames=frame_result.unique_frames,
                video_id=video_id,
            )
            logger.info("Vision analysis complete: %d frames analyzed", len(vision_analyses))
        except Exception as exc:
            logger.warning("Vision analysis failed (non-fatal): %s", exc)
            pipeline_errors.append(
                PipelineError(
                    stage=PipelineStatus.ANALYZING_VISUALS,
                    message=str(exc),
                    recoverable=True,
                )
            )

    # ─────────────────────────────────────────────
    # Stage 6: SYNCHRONISE + CHUNK
    # ─────────────────────────────────────────────
    try:
        timeline = chunking_svc.build_timeline(
            transcript=transcript,
            vision_analyses=vision_analyses,
            chapters=metadata.chapters,
        )

        chunks = chunking_svc.chunk_timeline(
            timeline=timeline,
            video_id=video_id,
            chapters=metadata.chapters,
        )

        logger.info("Chunking complete: %d chunks from %d timeline entries", len(chunks), len(timeline))

        if not chunks:
            raise HTTPException(
                status_code=500,
                detail="Chunking produced zero chunks — cannot generate notes",
            )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Chunking failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Chunking failed: {exc}")

    # ─────────────────────────────────────────────
    # Stage 7: NOTE GENERATION (Map-Reduce)
    # ─────────────────────────────────────────────
    try:
        merged_notes = await note_gen_svc.generate_notes(
            chunks=chunks,
            video_id=video_id,
            video_title=metadata.title,
            video_duration=metadata.duration,
        )
        logger.info(
            "Note generation complete: %d chars, %d tokens used",
            len(merged_notes.detailed_notes_markdown),
            merged_notes.total_tokens_used,
        )
    except NoteGenerationError as exc:
        logger.error("Note generation failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Note generation failed: {exc}")
    except Exception as exc:
        logger.error("Note generation failed unexpectedly: %s", exc)
        raise HTTPException(status_code=500, detail=f"Note generation failed: {exc}")

    # ─────────────────────────────────────────────
    # Stage 8: EMBED + STORE IN QDRANT
    # ─────────────────────────────────────────────
    cache_hits = 0

    try:
        upserted = await vector_db_svc.embed_and_upsert(
            chunks=chunks,
            video_title=metadata.title,
        )
        logger.info("Vector DB: %d points upserted", upserted)
    except Exception as exc:
        logger.warning("Vector DB upsert failed (non-fatal): %s", exc)
        pipeline_errors.append(
            PipelineError(
                stage=PipelineStatus.STORING,
                message=str(exc),
                recoverable=True,
            )
        )

    # ─────────────────────────────────────────────
    # ASSEMBLE RESPONSE
    # ─────────────────────────────────────────────
    elapsed = time.monotonic() - pipeline_start

    pipeline_metadata = NoteGenerationMetadata(
        pipeline_status=PipelineStatus.COMPLETED,
        total_processing_time_seconds=round(elapsed, 2),
        transcript_source=transcript_source,
        total_chunks=len(chunks),
        total_frames_extracted=total_frames,
        unique_frames_analyzed=unique_frames,
        total_tokens_used=merged_notes.total_tokens_used,
        gemini_model_used=settings.gemini_model,
        gemini_vision_model_used=settings.gemini_vision_model,
        embedding_model_used=settings.gemini_embedding_model,
        segments_processed=len(chunks),
        cache_hits=cache_hits,
        errors=pipeline_errors,
        created_at=datetime.utcnow(),
        completed_at=datetime.utcnow(),
    )

    response = GenerateNotesResponse(
        video_id=video_id,
        title=metadata.title,
        duration=metadata.duration,
        channel=metadata.channel,
        thumbnail_url=metadata.thumbnail_url,
        summary=merged_notes.overview,
        notes=merged_notes.detailed_notes_markdown,
        timestamps=merged_notes.timestamps,
        references=merged_notes.references,
        glossary=merged_notes.glossary,
        quiz=merged_notes.quiz_questions,
        metadata=pipeline_metadata,
    )

    logger.info(
        "Pipeline complete: video_id=%s, duration=%.1fs, chunks=%d, tokens=%d",
        video_id,
        elapsed,
        len(chunks),
        merged_notes.total_tokens_used,
    )

    return response
