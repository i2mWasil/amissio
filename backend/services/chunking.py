"""
Chunking service: multimodal timeline synchronization and semantic chunking.

This module is the bridge between raw extraction outputs (transcript +
vision) and the structured semantic chunks consumed by note generation
and vector storage.

Pipeline:
  1. **Synchronize** — Interleave transcript segments with vision analyses
     into a unified, time-ordered ``TimelineEntry`` stream.
  2. **Detect boundaries** — Identify natural split points from chapters,
     slide changes, and topic shifts.
  3. **Chunk** — Group timeline entries into ``SemanticChunk`` objects of
     500–1200 tokens, respecting detected boundaries.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from core.config import Settings, get_settings
from models.schemas import (
    ChunkType,
    ContentType,
    SemanticChunk,
    TimelineEntry,
    TranscriptData,
    TranscriptSegment,
    VideoChapter,
    VisionAnalysis,
)

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Token estimation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def estimate_tokens(text: str) -> int:
    """
    Fast approximate token count.

    Uses the heuristic that 1 token ≈ 4 characters for English text.
    This avoids importing a full tokenizer while staying within ±10%
    accuracy for Gemini models.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def _format_timestamp(seconds: float) -> str:
    """Format seconds into MM:SS or HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Custom exceptions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ChunkingError(Exception):
    """Raised when the chunking pipeline fails."""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Boundary detection helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class _BoundaryMarker:
    """A detected natural boundary in the timeline."""

    __slots__ = ("time", "chunk_type", "label")

    def __init__(self, time: float, chunk_type: ChunkType, label: str) -> None:
        self.time = time
        self.chunk_type = chunk_type
        self.label = label

    def __repr__(self) -> str:
        return f"Boundary({self.chunk_type.value}@{self.time:.1f}s '{self.label}')"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chunking Service
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ChunkingService:
    """
    Synchronises transcript and vision data into a unified timeline,
    then segments it into semantically coherent chunks.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._min_tokens = self._settings.chunk_min_tokens
        self._max_tokens = self._settings.chunk_max_tokens
        self._overlap_tokens = self._settings.chunk_overlap_tokens

    # ─────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────

    def build_timeline(
        self,
        transcript: TranscriptData,
        vision_analyses: list[VisionAnalysis],
        chapters: list[VideoChapter] | None = None,
    ) -> list[TimelineEntry]:
        """
        Synchronise transcript segments with vision analyses into a
        unified, chronologically ordered timeline.

        Each timeline entry covers a transcript segment's time window
        and is enriched with any vision data whose timestamp falls
        within (or near) that window.

        Args:
            transcript: Complete transcript with timed segments.
            vision_analyses: Ordered list of frame-level analyses.
            chapters: Optional chapter list for chapter tagging.

        Returns:
            A chronologically sorted list of ``TimelineEntry`` objects.
        """
        if not transcript.segments:
            logger.warning("Empty transcript — timeline will be vision-only")
            return self._vision_only_timeline(vision_analyses, chapters)

        chapters = chapters or []
        # Index vision analyses by timestamp for efficient lookup
        sorted_vision = sorted(vision_analyses, key=lambda v: v.timestamp)

        timeline: list[TimelineEntry] = []
        vision_idx = 0

        for seg in transcript.segments:
            # Collect vision analyses whose timestamp falls in or near this segment
            seg_visual_parts: list[str] = []
            seg_ocr_parts: list[str] = []
            seg_code: list[str] = []
            seg_equations: list[str] = []
            seg_content_types: set[ContentType] = set()
            active_slide_title: str | None = None

            # Window: allow frames within ±1.5 seconds of the segment edges
            window_start = seg.start - 1.5
            window_end = seg.end + 1.5

            while vision_idx < len(sorted_vision):
                va = sorted_vision[vision_idx]

                if va.timestamp < window_start:
                    vision_idx += 1
                    continue
                if va.timestamp > window_end:
                    break

                # This vision analysis belongs to the current segment
                if va.description:
                    seg_visual_parts.append(va.description)
                if va.text_content:
                    seg_ocr_parts.append(va.text_content)
                if va.slide_title:
                    active_slide_title = va.slide_title
                seg_code.extend(va.code_snippets)
                seg_equations.extend(va.equations)
                seg_content_types.update(va.content_types)

                vision_idx += 1

            # Find the active chapter
            chapter = self._find_chapter(seg.start, chapters)

            # Always add TEXT content type for spoken text
            seg_content_types.add(ContentType.TEXT)

            entry = TimelineEntry(
                start_time=seg.start,
                end_time=seg.end,
                transcript_text=seg.text,
                visual_context=" | ".join(seg_visual_parts) if seg_visual_parts else "",
                slide_title=active_slide_title,
                ocr_text="\n".join(seg_ocr_parts) if seg_ocr_parts else "",
                code_snippets=seg_code,
                equations=seg_equations,
                content_types=list(seg_content_types),
                chapter=chapter,
            )
            timeline.append(entry)

        # Handle any remaining vision analyses that fell after the last transcript segment
        remaining_vision = sorted_vision[vision_idx:]
        if remaining_vision:
            self._append_orphan_vision(timeline, remaining_vision, chapters)

        logger.info(
            "Timeline built: %d entries from %d transcript segments + %d vision analyses",
            len(timeline),
            len(transcript.segments),
            len(vision_analyses),
        )

        return timeline

    def chunk_timeline(
        self,
        timeline: list[TimelineEntry],
        video_id: str,
        chapters: list[VideoChapter] | None = None,
    ) -> list[SemanticChunk]:
        """
        Segment a timeline into semantically coherent chunks.

        Boundary placement priority:
          1. **Chapter boundaries** — always split at chapter changes.
          2. **Slide changes** — split when the slide title changes.
          3. **Token overflow** — split when accumulated tokens exceed
             ``chunk_max_tokens``.
          4. **Topic shift** — heuristic detection of topic changes
             (transitional phrases, long pauses).

        Chunks are kept within ``[chunk_min_tokens, chunk_max_tokens]``
        whenever possible. Under-sized tail chunks are merged with
        their predecessor.

        Args:
            timeline: Ordered timeline entries.
            video_id: YouTube video ID for the chunk metadata.
            chapters: Optional chapters for chapter-based splitting.

        Returns:
            Ordered list of ``SemanticChunk`` objects.
        """
        if not timeline:
            return []

        chapters = chapters or []

        # Step 1: Detect all natural boundaries
        boundaries = self._detect_boundaries(timeline, chapters)

        # Step 2: Group timeline entries by boundaries
        groups = self._split_at_boundaries(timeline, boundaries)

        # Step 3: Split or merge groups to fit token bounds
        final_groups = self._enforce_token_bounds(groups)

        # Step 4: Convert groups into SemanticChunk objects
        chunks = self._groups_to_chunks(final_groups, video_id, chapters)

        logger.info(
            "Chunking complete: %d chunks from %d timeline entries "
            "(tokens: min=%d, max=%d, avg=%d)",
            len(chunks),
            len(timeline),
            min(c.token_count for c in chunks) if chunks else 0,
            max(c.token_count for c in chunks) if chunks else 0,
            sum(c.token_count for c in chunks) // max(len(chunks), 1),
        )

        return chunks

    # ─────────────────────────────────────────────
    # Boundary detection
    # ─────────────────────────────────────────────

    def _detect_boundaries(
        self,
        timeline: list[TimelineEntry],
        chapters: list[VideoChapter],
    ) -> list[_BoundaryMarker]:
        """
        Detect natural boundary points in the timeline.

        Returns a sorted list of ``_BoundaryMarker`` objects.
        """
        boundaries: list[_BoundaryMarker] = []

        # 1) Chapter boundaries
        for ch in chapters:
            boundaries.append(
                _BoundaryMarker(
                    time=ch.start_time,
                    chunk_type=ChunkType.CHAPTER,
                    label=ch.title,
                )
            )

        # 2) Slide changes — when slide_title changes between entries
        prev_slide: str | None = None
        for entry in timeline:
            if entry.slide_title and entry.slide_title != prev_slide:
                boundaries.append(
                    _BoundaryMarker(
                        time=entry.start_time,
                        chunk_type=ChunkType.SLIDE_CHANGE,
                        label=entry.slide_title,
                    )
                )
            prev_slide = entry.slide_title

        # 3) Topic shift heuristics — transitional phrases in the text
        topic_markers = (
            "now let's", "moving on", "next topic", "next we",
            "let's talk about", "another important", "on the other hand",
            "in summary", "to summarize", "in conclusion",
            "the next section", "switching to", "turning to",
        )
        for entry in timeline:
            text_lower = entry.transcript_text.lower().strip()
            for marker in topic_markers:
                if text_lower.startswith(marker) or f". {marker}" in text_lower:
                    boundaries.append(
                        _BoundaryMarker(
                            time=entry.start_time,
                            chunk_type=ChunkType.TOPIC_CHANGE,
                            label=marker,
                        )
                    )
                    break  # one boundary per entry

        # Deduplicate boundaries that are too close (within 5 seconds)
        boundaries.sort(key=lambda b: b.time)
        deduped: list[_BoundaryMarker] = []
        for b in boundaries:
            if deduped and abs(b.time - deduped[-1].time) < 5.0:
                # Keep the higher-priority boundary (chapter > slide > topic)
                priority = {ChunkType.CHAPTER: 3, ChunkType.SLIDE_CHANGE: 2, ChunkType.TOPIC_CHANGE: 1}
                if priority.get(b.chunk_type, 0) > priority.get(deduped[-1].chunk_type, 0):
                    deduped[-1] = b
            else:
                deduped.append(b)

        logger.debug("Detected %d boundaries: %s", len(deduped), deduped)
        return deduped

    # ─────────────────────────────────────────────
    # Splitting and merging
    # ─────────────────────────────────────────────

    def _split_at_boundaries(
        self,
        timeline: list[TimelineEntry],
        boundaries: list[_BoundaryMarker],
    ) -> list[_EntryGroup]:
        """
        Split the timeline into groups at detected boundaries.

        Each group has a ``chunk_type`` from the boundary that starts it.
        """
        if not boundaries:
            return [_EntryGroup(entries=timeline, chunk_type=ChunkType.SEMANTIC_SPLIT)]

        groups: list[_EntryGroup] = []
        boundary_times = [b.time for b in boundaries]
        boundary_types = [b.chunk_type for b in boundaries]
        b_idx = 0
        current_entries: list[TimelineEntry] = []
        current_type = ChunkType.SEMANTIC_SPLIT

        for entry in timeline:
            # Check if this entry crosses a boundary
            while b_idx < len(boundary_times) and entry.start_time >= boundary_times[b_idx]:
                if current_entries:
                    groups.append(_EntryGroup(entries=current_entries, chunk_type=current_type))
                    current_entries = []
                current_type = boundary_types[b_idx]
                b_idx += 1

            current_entries.append(entry)

        # Don't forget the last group
        if current_entries:
            groups.append(_EntryGroup(entries=current_entries, chunk_type=current_type))

        return groups

    def _enforce_token_bounds(
        self,
        groups: list[_EntryGroup],
    ) -> list[_EntryGroup]:
        """
        Split oversized groups and merge undersized ones.

        - Groups exceeding ``chunk_max_tokens`` are split into sub-groups.
        - Groups under ``chunk_min_tokens`` are merged with the previous
          group unless they are chapter boundaries.
        """
        result: list[_EntryGroup] = []

        for group in groups:
            tokens = group.token_count
            if tokens <= self._max_tokens:
                result.append(group)
            else:
                # Split oversized group into sub-groups
                sub_groups = self._split_oversized_group(group)
                result.extend(sub_groups)

        # Merge undersized groups
        merged: list[_EntryGroup] = []
        for group in result:
            if (
                merged
                and group.token_count < self._min_tokens
                and group.chunk_type != ChunkType.CHAPTER
                and (merged[-1].token_count + group.token_count) <= self._max_tokens
            ):
                # Merge with predecessor
                merged[-1].entries.extend(group.entries)
            else:
                merged.append(group)

        # Handle the case where the very last group is still undersized
        if (
            len(merged) >= 2
            and merged[-1].token_count < self._min_tokens
            and merged[-1].chunk_type != ChunkType.CHAPTER
        ):
            last = merged.pop()
            if (merged[-1].token_count + last.token_count) <= self._max_tokens * 1.2:
                merged[-1].entries.extend(last.entries)
            else:
                merged.append(last)  # put it back; it's small but can't merge

        return merged

    def _split_oversized_group(self, group: _EntryGroup) -> list[_EntryGroup]:
        """
        Split a group that exceeds ``chunk_max_tokens`` into smaller
        sub-groups at sentence boundaries.
        """
        sub_groups: list[_EntryGroup] = []
        current_entries: list[TimelineEntry] = []
        current_tokens = 0

        for entry in group.entries:
            entry_tokens = estimate_tokens(entry.transcript_text) + estimate_tokens(entry.ocr_text)

            if current_tokens + entry_tokens > self._max_tokens and current_entries:
                sub_groups.append(
                    _EntryGroup(entries=current_entries, chunk_type=group.chunk_type)
                )
                # Overlap: carry the last few entries forward
                overlap_entries = self._get_overlap_entries(current_entries)
                current_entries = overlap_entries
                current_tokens = sum(
                    estimate_tokens(e.transcript_text) + estimate_tokens(e.ocr_text)
                    for e in current_entries
                )

            current_entries.append(entry)
            current_tokens += entry_tokens

        if current_entries:
            sub_groups.append(
                _EntryGroup(
                    entries=current_entries,
                    chunk_type=ChunkType.SEMANTIC_SPLIT if len(sub_groups) > 0 else group.chunk_type,
                )
            )

        return sub_groups

    def _get_overlap_entries(self, entries: list[TimelineEntry]) -> list[TimelineEntry]:
        """
        Select entries from the tail of a group to carry forward as
        overlap for the next chunk, up to ``chunk_overlap_tokens``.
        """
        if self._overlap_tokens <= 0 or not entries:
            return []

        overlap: list[TimelineEntry] = []
        tokens = 0
        for entry in reversed(entries):
            entry_tokens = estimate_tokens(entry.transcript_text)
            if tokens + entry_tokens > self._overlap_tokens:
                break
            overlap.insert(0, entry)
            tokens += entry_tokens

        return overlap

    # ─────────────────────────────────────────────
    # Chunk construction
    # ─────────────────────────────────────────────

    def _groups_to_chunks(
        self,
        groups: list[_EntryGroup],
        video_id: str,
        chapters: list[VideoChapter],
    ) -> list[SemanticChunk]:
        """Convert ``_EntryGroup`` objects into ``SemanticChunk`` models."""
        chunks: list[SemanticChunk] = []

        for idx, group in enumerate(groups):
            if not group.entries:
                continue

            # Aggregate content from all entries in the group
            text_parts: list[str] = []
            visual_parts: list[str] = []
            ocr_parts: list[str] = []
            all_code: list[str] = []
            all_equations: list[str] = []
            all_content_types: set[ContentType] = set()
            chapter_title: str | None = None

            for entry in group.entries:
                if entry.transcript_text:
                    text_parts.append(entry.transcript_text)
                if entry.visual_context:
                    visual_parts.append(entry.visual_context)
                if entry.ocr_text:
                    ocr_parts.append(entry.ocr_text)
                all_code.extend(entry.code_snippets)
                all_equations.extend(entry.equations)
                all_content_types.update(entry.content_types)
                if entry.chapter and not chapter_title:
                    chapter_title = entry.chapter.title

            text = " ".join(text_parts)
            visual_context = " | ".join(dict.fromkeys(visual_parts))  # deduplicate preserving order
            ocr_text = "\n".join(dict.fromkeys(ocr_parts))
            code_snippets = list(dict.fromkeys(all_code))  # deduplicate
            equations = list(dict.fromkeys(all_equations))

            start_time = group.entries[0].start_time
            end_time = group.entries[-1].end_time
            token_count = estimate_tokens(text) + estimate_tokens(ocr_text)

            chunk = SemanticChunk(
                video_id=video_id,
                chunk_index=idx,
                chunk_type=group.chunk_type,
                text=text,
                visual_context=visual_context,
                ocr_text=ocr_text,
                code_snippets=code_snippets,
                equations=equations,
                content_types=list(all_content_types),
                start_time=start_time,
                end_time=end_time,
                chapter_title=chapter_title,
                token_count=token_count,
            )
            chunks.append(chunk)

        return chunks

    # ─────────────────────────────────────────────
    # Vision-only timeline (no transcript case)
    # ─────────────────────────────────────────────

    def _vision_only_timeline(
        self,
        vision_analyses: list[VisionAnalysis],
        chapters: list[VideoChapter] | None,
    ) -> list[TimelineEntry]:
        """
        Build a timeline from vision analyses alone when no transcript
        is available (silent video or failed transcription).
        """
        chapters = chapters or []
        timeline: list[TimelineEntry] = []

        for va in sorted(vision_analyses, key=lambda v: v.timestamp):
            chapter = self._find_chapter(va.timestamp, chapters)
            entry = TimelineEntry(
                start_time=va.timestamp,
                end_time=va.timestamp + 1.0,  # approx 1s per frame
                transcript_text="",
                visual_context=va.description,
                slide_title=va.slide_title,
                ocr_text=va.text_content,
                code_snippets=va.code_snippets,
                equations=va.equations,
                content_types=va.content_types,
                chapter=chapter,
            )
            timeline.append(entry)

        return timeline

    def _append_orphan_vision(
        self,
        timeline: list[TimelineEntry],
        orphans: list[VisionAnalysis],
        chapters: list[VideoChapter],
    ) -> None:
        """
        Append vision analyses that fell outside all transcript segments
        as standalone timeline entries.
        """
        for va in orphans:
            chapter = self._find_chapter(va.timestamp, chapters)
            entry = TimelineEntry(
                start_time=va.timestamp,
                end_time=va.timestamp + 1.0,
                transcript_text="",
                visual_context=va.description,
                slide_title=va.slide_title,
                ocr_text=va.text_content,
                code_snippets=va.code_snippets,
                equations=va.equations,
                content_types=va.content_types,
                chapter=chapter,
            )
            timeline.append(entry)

    @staticmethod
    def _find_chapter(
        time_seconds: float,
        chapters: list[VideoChapter],
    ) -> VideoChapter | None:
        """Find the chapter that a timestamp falls within."""
        for ch in chapters:
            end = ch.end_time if ch.end_time is not None else float("inf")
            if ch.start_time <= time_seconds < end:
                return ch
        return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Internal group container
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class _EntryGroup:
    """
    Mutable container for a group of timeline entries that will become
    a single chunk.
    """

    __slots__ = ("entries", "chunk_type")

    def __init__(
        self,
        entries: list[TimelineEntry],
        chunk_type: ChunkType,
    ) -> None:
        self.entries = list(entries)
        self.chunk_type = chunk_type

    @property
    def token_count(self) -> int:
        return sum(
            estimate_tokens(e.transcript_text) + estimate_tokens(e.ocr_text)
            for e in self.entries
        )

    @property
    def start_time(self) -> float:
        return self.entries[0].start_time if self.entries else 0.0

    @property
    def end_time(self) -> float:
        return self.entries[-1].end_time if self.entries else 0.0
