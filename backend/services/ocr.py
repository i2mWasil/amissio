"""
OCR service: batch frame analysis via Gemini Flash Vision.

Orchestrates the visual analysis of deduplicated keyframes by:
  1. Batching frames into groups of ``vision_batch_size``
  2. Sending each batch to Gemini Vision with a structured prompt
  3. Parsing the JSON response into ``VisionAnalysis`` models
  4. Caching results on disk to avoid redundant API calls
  5. Running batches concurrently up to ``vision_max_concurrent``

This module owns the prompts and parsing logic — the actual API call
is delegated to ``LLMService``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from core.config import Settings, get_settings
from models.schemas import (
    ContentType,
    ExtractedFrame,
    VisionAnalysis,
    VisionBatchResult,
)
from services.llm import LLMService, LLMContentBlockedError, LLMServiceError

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Custom Exceptions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class OCRServiceError(Exception):
    """Base exception for OCR service operations."""


class FrameAnalysisError(OCRServiceError):
    """Raised when Gemini Vision fails to analyze a frame batch."""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Response Schema for Structured JSON Output
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class FrameAnalysisItem(BaseModel):
    """Schema for a single frame analysis returned by Gemini."""

    frame_label: str = ""
    description: str = ""
    slide_title: str | None = None
    text_content: str = ""
    code_snippets: list[str] = []
    equations: list[str] = []
    diagram_description: str | None = None
    content_types: list[str] = []
    confidence: float = 0.8


class BatchAnalysisResponse(BaseModel):
    """Top-level schema for Gemini's batch analysis JSON response."""

    frames: list[FrameAnalysisItem] = []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Prompts
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


VISION_SYSTEM_INSTRUCTION = """You are a precise visual analysis assistant for educational video frames.
Your task is to extract ALL visible information from video frames with perfect accuracy.

CRITICAL RULES:
- Describe ONLY what is actually visible in each frame. Never hallucinate or infer content that is not shown.
- Extract ALL visible text exactly as written, preserving formatting where possible.
- If you see code, extract it verbatim with correct indentation and syntax.
- If you see mathematical equations, express them in LaTeX notation.
- If you see diagrams, charts, or tables, describe their structure and content precisely.
- For slides or presentations, always identify the slide title if one is visible.
- Rate your confidence in the analysis accuracy (0.0 to 1.0)."""

SINGLE_FRAME_PROMPT = """Analyze this video frame and extract all visible information.

Respond with a JSON object containing:
- "description": A concise natural-language description of what the frame shows (1-3 sentences).
- "slide_title": The slide/section title if this is a presentation frame, otherwise null.
- "text_content": ALL visible text extracted verbatim.
- "code_snippets": An array of any code blocks visible (preserve indentation and syntax).
- "equations": An array of mathematical equations in LaTeX notation.
- "diagram_description": Description of any diagram, chart, flowchart, or table if present, otherwise null.
- "content_types": Array of content types present. Valid values: "text", "code", "equation", "diagram", "chart", "table", "image".
- "confidence": Your confidence in the analysis accuracy (0.0-1.0)."""

BATCH_FRAME_PROMPT_TEMPLATE = """Analyze these {count} video frames from an educational video.
The frames are labeled [Frame 1] through [Frame {count}] and are ordered chronologically.

For EACH frame, extract all visible information with perfect accuracy.

Respond with a JSON object containing a "frames" array, where each element corresponds to one frame (in order) with these fields:
- "frame_label": The frame label (e.g., "Frame 1").
- "description": A concise natural-language description (1-3 sentences).
- "slide_title": The slide/section title if visible, otherwise null.
- "text_content": ALL visible text extracted verbatim.
- "code_snippets": Array of code blocks visible (preserve indentation).
- "equations": Array of mathematical equations in LaTeX notation.
- "diagram_description": Description of any diagram/chart/table, otherwise null.
- "content_types": Array of content types present. Valid values: "text", "code", "equation", "diagram", "chart", "table", "image".
- "confidence": Your confidence in the analysis accuracy (0.0-1.0).

Return exactly {count} elements in the "frames" array, one per input frame."""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OCR Service
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class OCRService:
    """
    Orchestrates Gemini Vision analysis of extracted video keyframes.

    Groups frames into batches, sends them to Gemini Vision with
    structured output prompts, parses responses into ``VisionAnalysis``
    models, and caches results to avoid redundant API calls on restarts.
    """

    def __init__(
        self,
        llm_service: LLMService,
        settings: Settings | None = None,
    ) -> None:
        self._llm = llm_service
        self._settings = settings or get_settings()
        self._batch_size = self._settings.vision_batch_size
        self._max_concurrent = self._settings.vision_max_concurrent
        self._cache_dir = self._settings.cache_dir / "ocr"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────
    # Public async API
    # ─────────────────────────────────────────────

    async def analyze_frames(
        self,
        frames: list[ExtractedFrame],
        video_id: str,
    ) -> list[VisionAnalysis]:
        """
        Analyze a list of unique (non-duplicate) keyframes.

        Frames are batched, sent to Gemini Vision concurrently (up to
        ``vision_max_concurrent``), and results are aggregated into a
        flat list of ``VisionAnalysis`` objects ordered by timestamp.

        Args:
            frames: Unique keyframes to analyze (must have valid file_path).
            video_id: YouTube video ID (used for cache keying).

        Returns:
            Ordered list of ``VisionAnalysis`` — one per input frame.

        Raises:
            OCRServiceError: If analysis fails for all batches.
        """
        if not frames:
            logger.warning("No frames provided for analysis")
            return []

        # Filter to only non-duplicate frames with valid paths
        valid_frames = [
            f for f in frames
            if not f.is_duplicate and f.file_path and Path(f.file_path).exists()
        ]

        if not valid_frames:
            logger.warning("No valid frame files found for analysis")
            return []

        logger.info(
            "Starting OCR analysis: %d frames, batch_size=%d, max_concurrent=%d",
            len(valid_frames),
            self._batch_size,
            self._max_concurrent,
        )

        # Check cache for the entire video
        cached = self._load_cache(video_id, valid_frames)
        if cached is not None:
            logger.info(
                "Using cached OCR results for %s: %d analyses",
                video_id,
                len(cached),
            )
            return cached

        # Split frames into batches
        batches = self._create_batches(valid_frames)
        logger.info("Created %d batches from %d frames", len(batches), len(valid_frames))

        # Process batches with bounded concurrency
        semaphore = asyncio.Semaphore(self._max_concurrent)
        start_time = time.monotonic()

        async def process_batch(batch_index: int, batch: list[ExtractedFrame]) -> list[VisionAnalysis]:
            async with semaphore:
                return await self._analyze_batch(batch_index, batch)

        tasks = [
            process_batch(i, batch)
            for i, batch in enumerate(batches)
        ]

        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate results, handling per-batch failures gracefully
        all_analyses: list[VisionAnalysis] = []
        failed_batches = 0

        for i, result in enumerate(batch_results):
            if isinstance(result, Exception):
                logger.error("Batch %d failed: %s", i, result)
                failed_batches += 1
                # Create fallback entries for failed batch frames
                for frame in batches[i]:
                    all_analyses.append(
                        VisionAnalysis(
                            frame_index=frame.frame_index,
                            timestamp=frame.timestamp,
                            description="[Analysis failed]",
                            confidence=0.0,
                        )
                    )
            else:
                all_analyses.extend(result)

        # Sort by timestamp
        all_analyses.sort(key=lambda a: a.timestamp)

        elapsed = time.monotonic() - start_time
        logger.info(
            "OCR analysis complete: %d analyses, %d failed batches, %.1fs elapsed",
            len(all_analyses),
            failed_batches,
            elapsed,
        )

        # Cache the results
        self._save_cache(video_id, valid_frames, all_analyses)

        return all_analyses

    async def analyze_single_frame(
        self,
        frame: ExtractedFrame,
    ) -> VisionAnalysis:
        """
        Analyze a single frame (no batching).

        Useful for on-demand analysis outside the batch pipeline.
        """
        image_path = Path(frame.file_path)
        if not image_path.exists():
            raise FrameAnalysisError(f"Frame file not found: {frame.file_path}")

        try:
            response_text = await self._llm.analyze_image(
                image_path=image_path,
                prompt=SINGLE_FRAME_PROMPT,
                system_instruction=VISION_SYSTEM_INSTRUCTION,
                response_schema=FrameAnalysisItem,
            )

            item = self._parse_single_response(response_text)

            return self._item_to_analysis(item, frame)

        except LLMContentBlockedError:
            logger.warning(
                "Frame %d blocked by safety filter — returning empty analysis",
                frame.frame_index,
            )
            return VisionAnalysis(
                frame_index=frame.frame_index,
                timestamp=frame.timestamp,
                description="[Content blocked by safety filter]",
                confidence=0.0,
            )
        except LLMServiceError as exc:
            raise FrameAnalysisError(
                f"Failed to analyze frame {frame.frame_index}: {exc}"
            ) from exc

    # ─────────────────────────────────────────────
    # Batch processing internals
    # ─────────────────────────────────────────────

    def _create_batches(
        self,
        frames: list[ExtractedFrame],
    ) -> list[list[ExtractedFrame]]:
        """Split frames into batches of ``vision_batch_size``."""
        return [
            frames[i : i + self._batch_size]
            for i in range(0, len(frames), self._batch_size)
        ]

    async def _analyze_batch(
        self,
        batch_index: int,
        batch: list[ExtractedFrame],
    ) -> list[VisionAnalysis]:
        """
        Analyze a single batch of frames via Gemini Vision.

        For single-frame batches, uses the simpler single-image prompt.
        For multi-frame batches, uses the batch prompt with structured
        JSON output.
        """
        logger.debug(
            "Processing batch %d: %d frames (timestamps %.1f–%.1f)",
            batch_index,
            len(batch),
            batch[0].timestamp,
            batch[-1].timestamp,
        )

        if len(batch) == 1:
            analysis = await self.analyze_single_frame(batch[0])
            return [analysis]

        # Multi-frame batch
        image_paths = [Path(f.file_path) for f in batch]
        prompt = BATCH_FRAME_PROMPT_TEMPLATE.format(count=len(batch))

        try:
            response_text = await self._llm.analyze_images_batch(
                image_paths=image_paths,
                prompt=prompt,
                system_instruction=VISION_SYSTEM_INSTRUCTION,
                response_schema=BatchAnalysisResponse,
            )

            items = self._parse_batch_response(response_text, len(batch))
            analyses: list[VisionAnalysis] = []

            for i, frame in enumerate(batch):
                if i < len(items):
                    analyses.append(self._item_to_analysis(items[i], frame))
                else:
                    # Gemini returned fewer items than frames
                    logger.warning(
                        "Batch %d: Gemini returned %d items for %d frames",
                        batch_index,
                        len(items),
                        len(batch),
                    )
                    analyses.append(
                        VisionAnalysis(
                            frame_index=frame.frame_index,
                            timestamp=frame.timestamp,
                            description="[Missing from batch response]",
                            confidence=0.0,
                        )
                    )

            return analyses

        except LLMContentBlockedError:
            logger.warning(
                "Batch %d blocked by safety filter — falling back to individual frames",
                batch_index,
            )
            # Retry each frame individually
            return await self._fallback_individual(batch)

        except LLMServiceError as exc:
            logger.error("Batch %d LLM error: %s — falling back to individual", batch_index, exc)
            return await self._fallback_individual(batch)

    async def _fallback_individual(
        self,
        batch: list[ExtractedFrame],
    ) -> list[VisionAnalysis]:
        """
        When a batch fails, retry each frame individually.

        Individual failures produce placeholder analyses rather than
        propagating the error.
        """
        analyses: list[VisionAnalysis] = []
        for frame in batch:
            try:
                analysis = await self.analyze_single_frame(frame)
                analyses.append(analysis)
            except (FrameAnalysisError, LLMServiceError) as exc:
                logger.warning(
                    "Individual frame %d analysis failed: %s",
                    frame.frame_index,
                    exc,
                )
                analyses.append(
                    VisionAnalysis(
                        frame_index=frame.frame_index,
                        timestamp=frame.timestamp,
                        description="[Analysis failed]",
                        confidence=0.0,
                    )
                )
        return analyses

    # ─────────────────────────────────────────────
    # Response parsing
    # ─────────────────────────────────────────────

    def _parse_single_response(self, response_text: str) -> FrameAnalysisItem:
        """Parse Gemini's JSON response for a single frame."""
        try:
            data = json.loads(response_text)
            return FrameAnalysisItem.model_validate(data)
        except (json.JSONDecodeError, Exception) as exc:
            logger.warning("Failed to parse single frame response as JSON: %s", exc)
            # Treat the entire response as a description
            return FrameAnalysisItem(
                description=response_text[:2000],
                confidence=0.3,
                content_types=["text"],
            )

    def _parse_batch_response(
        self,
        response_text: str,
        expected_count: int,
    ) -> list[FrameAnalysisItem]:
        """Parse Gemini's JSON response for a batch of frames."""
        try:
            data = json.loads(response_text)

            # Handle both {"frames": [...]} and direct array [...]
            if isinstance(data, dict):
                frames_data = data.get("frames", [])
            elif isinstance(data, list):
                frames_data = data
            else:
                logger.warning("Unexpected batch response type: %s", type(data))
                return [FrameAnalysisItem(description=response_text[:2000], confidence=0.3)]

            items: list[FrameAnalysisItem] = []
            for item_data in frames_data:
                try:
                    items.append(FrameAnalysisItem.model_validate(item_data))
                except Exception as exc:
                    logger.warning("Failed to parse frame item: %s", exc)
                    items.append(
                        FrameAnalysisItem(
                            description=str(item_data)[:1000],
                            confidence=0.2,
                        )
                    )

            if len(items) < expected_count:
                logger.warning(
                    "Batch parse: got %d items, expected %d",
                    len(items),
                    expected_count,
                )

            return items

        except json.JSONDecodeError as exc:
            logger.warning("Batch response is not valid JSON: %s", exc)
            # Return a single item treating the whole response as text
            return [
                FrameAnalysisItem(
                    description=response_text[:2000],
                    confidence=0.2,
                    content_types=["text"],
                )
            ]

    @staticmethod
    def _item_to_analysis(
        item: FrameAnalysisItem,
        frame: ExtractedFrame,
    ) -> VisionAnalysis:
        """Convert a parsed ``FrameAnalysisItem`` into a ``VisionAnalysis``."""
        # Map string content types to the ContentType enum
        content_types: list[ContentType] = []
        valid_types = {ct.value for ct in ContentType}
        for ct_str in item.content_types:
            ct_lower = ct_str.lower().strip()
            if ct_lower in valid_types:
                content_types.append(ContentType(ct_lower))

        return VisionAnalysis(
            frame_index=frame.frame_index,
            timestamp=frame.timestamp,
            description=item.description,
            slide_title=item.slide_title,
            text_content=item.text_content,
            code_snippets=item.code_snippets,
            equations=item.equations,
            diagram_description=item.diagram_description,
            content_types=content_types,
            confidence=max(0.0, min(1.0, item.confidence)),
        )

    # ─────────────────────────────────────────────
    # Disk caching
    # ─────────────────────────────────────────────

    def _cache_key(
        self,
        video_id: str,
        frames: list[ExtractedFrame],
    ) -> str:
        """
        Generate a deterministic cache key from the video ID and
        frame file paths.

        The key changes if frames are re-extracted (different files
        or count), ensuring stale caches are not used.
        """
        hasher = hashlib.sha256()
        hasher.update(video_id.encode())
        for f in frames:
            hasher.update(f.file_path.encode())
            hasher.update(str(f.timestamp).encode())
        return hasher.hexdigest()[:24]

    def _cache_path(self, cache_key: str) -> Path:
        return self._cache_dir / f"{cache_key}.json"

    def _load_cache(
        self,
        video_id: str,
        frames: list[ExtractedFrame],
    ) -> list[VisionAnalysis] | None:
        """Load cached OCR results if they exist and match the current frames."""
        key = self._cache_key(video_id, frames)
        path = self._cache_path(key)

        if not path.exists():
            return None

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            analyses = [VisionAnalysis.model_validate(item) for item in raw]
            logger.debug("Cache hit for OCR key %s: %d analyses", key, len(analyses))
            return analyses
        except (json.JSONDecodeError, Exception) as exc:
            logger.warning("Failed to load OCR cache %s: %s", key, exc)
            return None

    def _save_cache(
        self,
        video_id: str,
        frames: list[ExtractedFrame],
        analyses: list[VisionAnalysis],
    ) -> None:
        """Persist OCR results to disk cache."""
        key = self._cache_key(video_id, frames)
        path = self._cache_path(key)

        try:
            data = [a.model_dump(mode="json") for a in analyses]
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            logger.debug("Saved OCR cache: %s (%d analyses)", key, len(analyses))
        except Exception as exc:
            logger.warning("Failed to save OCR cache %s: %s", key, exc)
