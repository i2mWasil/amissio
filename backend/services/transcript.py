"""
Transcript service: caption parsing and Whisper-based transcription.

Priority order:
  1. YouTube captions (manually uploaded → auto-generated)
  2. faster-whisper local transcription with word-level timestamps

All heavy computation (Whisper inference, file parsing) is offloaded to
``asyncio.to_thread`` so the event loop stays responsive.
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Any

from core.config import Settings, get_settings
from models.schemas import (
    TranscriptData,
    TranscriptSegment,
    TranscriptSource,
    WordTimestamp,
)

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Custom Exceptions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TranscriptServiceError(Exception):
    """Base exception for transcript service operations."""


class CaptionParseError(TranscriptServiceError):
    """Raised when a caption/subtitle file cannot be parsed."""


class WhisperModelLoadError(TranscriptServiceError):
    """Raised when the faster-whisper model fails to load."""


class WhisperTranscriptionError(TranscriptServiceError):
    """Raised when Whisper transcription fails."""


class NoTranscriptAvailableError(TranscriptServiceError):
    """Raised when neither captions nor audio are available for transcription."""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VTT / SRT Parser
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class CaptionParser:
    """
    Parse VTT and SRT subtitle files into ``TranscriptSegment`` objects.

    Handles the quirks of YouTube-generated VTT files including:
      - Duplicate cue text across overlapping time windows
      - HTML tags and positioning metadata inside cues
      - UTF-8 BOM markers
    """

    # Regex for VTT/SRT timestamp lines
    _VTT_TIMESTAMP_RE = re.compile(
        r"(\d{1,2}:)?(\d{2}):(\d{2})[.,](\d{3})"
        r"\s*-->\s*"
        r"(\d{1,2}:)?(\d{2}):(\d{2})[.,](\d{3})"
    )

    # HTML / VTT formatting tags to strip
    _TAG_RE = re.compile(r"<[^>]+>")
    # VTT positioning metadata (e.g. "position:10% align:start")
    _VTT_POSITION_RE = re.compile(r"(position|align|line|size|vertical):[^\s]+", re.IGNORECASE)
    # SRT sequence number lines
    _SRT_SEQ_RE = re.compile(r"^\d+\s*$")

    @classmethod
    def parse_file(cls, caption_path: Path) -> list[TranscriptSegment]:
        """
        Parse a VTT or SRT file and return ordered transcript segments.

        Deduplicates overlapping cues that YouTube VTT files commonly contain.
        """
        suffix = caption_path.suffix.lower()
        if suffix not in (".vtt", ".srt"):
            raise CaptionParseError(f"Unsupported caption format: {suffix}")

        text = caption_path.read_text(encoding="utf-8-sig")  # handles BOM

        if suffix == ".vtt":
            return cls._parse_vtt(text)
        return cls._parse_srt(text)

    @classmethod
    def _parse_vtt(cls, raw_text: str) -> list[TranscriptSegment]:
        """Parse a WebVTT string into segments."""
        lines = raw_text.strip().splitlines()
        segments: list[TranscriptSegment] = []
        seen_texts: set[str] = set()

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Look for timestamp lines
            match = cls._VTT_TIMESTAMP_RE.match(line)
            if match is None:
                i += 1
                continue

            start_seconds = cls._parse_timestamp_match(match, is_start=True)
            end_seconds = cls._parse_timestamp_match(match, is_start=False)
            i += 1

            # Collect all cue text lines until the next blank line or timestamp
            cue_lines: list[str] = []
            while i < len(lines):
                next_line = lines[i].strip()
                if not next_line:
                    i += 1
                    break
                if cls._VTT_TIMESTAMP_RE.match(next_line):
                    break
                cue_lines.append(next_line)
                i += 1

            # Clean and assemble the cue text
            cue_text = cls._clean_cue_text(" ".join(cue_lines))
            if not cue_text:
                continue

            # Deduplicate: YouTube VTT files often repeat cues with overlapping times
            dedup_key = cue_text.lower().strip()
            if dedup_key in seen_texts:
                continue
            seen_texts.add(dedup_key)

            segments.append(
                TranscriptSegment(
                    text=cue_text,
                    start=start_seconds,
                    end=end_seconds,
                )
            )

        return segments

    @classmethod
    def _parse_srt(cls, raw_text: str) -> list[TranscriptSegment]:
        """Parse an SRT string into segments."""
        lines = raw_text.strip().splitlines()
        segments: list[TranscriptSegment] = []

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Skip sequence number lines
            if cls._SRT_SEQ_RE.match(line):
                i += 1
                continue

            # Look for timestamp line
            match = cls._VTT_TIMESTAMP_RE.match(line)
            if match is None:
                i += 1
                continue

            start_seconds = cls._parse_timestamp_match(match, is_start=True)
            end_seconds = cls._parse_timestamp_match(match, is_start=False)
            i += 1

            # Collect subtitle text until blank line
            sub_lines: list[str] = []
            while i < len(lines) and lines[i].strip():
                sub_lines.append(lines[i].strip())
                i += 1

            cue_text = cls._clean_cue_text(" ".join(sub_lines))
            if cue_text:
                segments.append(
                    TranscriptSegment(
                        text=cue_text,
                        start=start_seconds,
                        end=end_seconds,
                    )
                )

            i += 1  # skip blank line

        return segments

    @classmethod
    def _parse_timestamp_match(cls, match: re.Match[str], is_start: bool) -> float:
        """Convert a regex match of a VTT/SRT timestamp to seconds."""
        if is_start:
            hours_str = match.group(1)
            minutes = int(match.group(2))
            seconds = int(match.group(3))
            millis = int(match.group(4))
        else:
            hours_str = match.group(5)
            minutes = int(match.group(6))
            seconds = int(match.group(7))
            millis = int(match.group(8))

        hours = int(hours_str.rstrip(":")) if hours_str else 0
        return hours * 3600 + minutes * 60 + seconds + millis / 1000.0

    @classmethod
    def _clean_cue_text(cls, text: str) -> str:
        """Strip HTML tags, VTT positioning metadata, and normalize whitespace."""
        text = cls._TAG_RE.sub("", text)
        text = cls._VTT_POSITION_RE.sub("", text)
        text = text.replace("\n", " ").replace("\r", "")
        # Collapse multiple spaces
        text = re.sub(r"\s+", " ", text).strip()
        return text


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Whisper Transcriber
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class WhisperTranscriber:
    """
    Transcribe audio files using ``faster-whisper`` with word-level timestamps.

    The model is loaded lazily on first use and cached for the lifetime
    of the service instance. Loading is thread-safe via a lock.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._model: Any | None = None
        self._load_lock = asyncio.Lock()

    async def _ensure_model_loaded(self) -> None:
        """Load the Whisper model if not already loaded (thread-safe)."""
        if self._model is not None:
            return

        async with self._load_lock:
            # Double-check after acquiring the lock
            if self._model is not None:
                return

            logger.info(
                "Loading faster-whisper model: size=%s, device=%s, compute=%s",
                self._settings.whisper_model_size,
                self._settings.whisper_device,
                self._settings.whisper_compute_type,
            )

            try:
                self._model = await asyncio.to_thread(self._load_model)
                logger.info("faster-whisper model loaded successfully")
            except Exception as exc:
                raise WhisperModelLoadError(
                    f"Failed to load faster-whisper model "
                    f"'{self._settings.whisper_model_size}': {exc}"
                ) from exc

    def _load_model(self) -> Any:
        """Synchronous model loading (runs in thread pool)."""
        from faster_whisper import WhisperModel

        device = self._settings.whisper_device
        compute_type = self._settings.whisper_compute_type

        # Auto-detect device: prefer CUDA if available
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        # Auto-select compute type based on device
        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "float32"

        return WhisperModel(
            self._settings.whisper_model_size,
            device=device,
            compute_type=compute_type,
        )

    async def transcribe(self, audio_path: Path) -> TranscriptData:
        """
        Transcribe an audio file and return structured transcript data
        with word-level timestamps.

        Args:
            audio_path: Path to the audio file (wav, mp3, or m4a).

        Returns:
            A populated ``TranscriptData`` with segments and word timestamps.

        Raises:
            WhisperTranscriptionError: If transcription fails.
            WhisperModelLoadError: If the model cannot be loaded.
        """
        if not audio_path.exists():
            raise WhisperTranscriptionError(f"Audio file not found: {audio_path}")

        await self._ensure_model_loaded()

        logger.info("Starting Whisper transcription: %s", audio_path)

        try:
            segments_data, detected_lang = await asyncio.to_thread(
                self._run_transcription, audio_path
            )
        except WhisperTranscriptionError:
            raise
        except Exception as exc:
            raise WhisperTranscriptionError(
                f"Whisper transcription failed: {exc}"
            ) from exc

        # Build the transcript
        segments: list[TranscriptSegment] = []
        all_text_parts: list[str] = []
        max_end_time: float = 0.0

        for seg_data in segments_data:
            segment = TranscriptSegment(
                text=seg_data["text"],
                start=seg_data["start"],
                end=seg_data["end"],
                words=seg_data["words"],
            )
            segments.append(segment)
            all_text_parts.append(seg_data["text"])
            max_end_time = max(max_end_time, seg_data["end"])

        full_text = " ".join(all_text_parts)

        transcript = TranscriptData(
            source=TranscriptSource.WHISPER,
            language=detected_lang,
            segments=segments,
            full_text=full_text,
            duration=max_end_time,
        )

        logger.info(
            "Whisper transcription complete: language=%s, segments=%d, "
            "words=%d, duration=%.1fs",
            detected_lang,
            len(segments),
            transcript.word_count,
            max_end_time,
        )
        return transcript

    def _run_transcription(self, audio_path: Path) -> tuple[list[dict[str, Any]], str]:
        """
        Run faster-whisper inference synchronously (called via to_thread).

        Returns a list of segment dicts and the detected language code.
        """
        segments_iter, info = self._model.transcribe(
            str(audio_path),
            beam_size=self._settings.whisper_beam_size,
            language=self._settings.whisper_language,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=200,
            ),
        )

        detected_lang = info.language if hasattr(info, "language") else "en"
        segments_data: list[dict[str, Any]] = []

        for segment in segments_iter:
            word_timestamps: list[WordTimestamp] = []

            if segment.words:
                for word_info in segment.words:
                    word_timestamps.append(
                        WordTimestamp(
                            word=word_info.word.strip(),
                            start=round(word_info.start, 3),
                            end=round(word_info.end, 3),
                            confidence=round(word_info.probability, 4),
                        )
                    )

            # Clean up segment text
            text = segment.text.strip()
            if not text:
                continue

            segments_data.append(
                {
                    "text": text,
                    "start": round(segment.start, 3),
                    "end": round(segment.end, 3),
                    "words": word_timestamps,
                }
            )

        return segments_data, detected_lang


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Transcript Service (Orchestrator)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TranscriptService:
    """
    High-level transcript orchestrator.

    Decides whether to use YouTube captions or Whisper transcription,
    and provides a unified interface for the pipeline.

    Priority:
      1. YouTube manual captions (highest quality)
      2. YouTube auto-generated captions (decent for most content)
      3. Whisper local transcription (fallback, most expensive)
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._whisper = WhisperTranscriber(self._settings)
        self._caption_parser = CaptionParser()

    async def get_transcript(
        self,
        caption_path: Path | None,
        audio_path: Path | None,
        prefer_whisper: bool = False,
    ) -> TranscriptData:
        """
        Obtain the transcript using the best available source.

        Args:
            caption_path: Path to a downloaded VTT/SRT file (or None).
            audio_path: Path to the audio file for Whisper (or None).
            prefer_whisper: Force Whisper even if captions exist (e.g.,
                when word-level timestamps are required and captions
                don't provide them).

        Returns:
            A populated ``TranscriptData`` object.

        Raises:
            NoTranscriptAvailableError: Neither captions nor audio available.
            CaptionParseError: Caption file exists but cannot be parsed.
            WhisperTranscriptionError: Whisper transcription failed.
        """
        # Case 1: Captions available and not forcing Whisper
        if caption_path is not None and caption_path.exists() and not prefer_whisper:
            logger.info("Using YouTube captions: %s", caption_path)
            try:
                transcript = await self._transcript_from_captions(caption_path)
                logger.info(
                    "Caption transcript ready: %d segments, %d words",
                    len(transcript.segments),
                    transcript.word_count,
                )
                return transcript
            except CaptionParseError as exc:
                logger.warning(
                    "Caption parsing failed, falling back to Whisper: %s", exc
                )
                # Fall through to Whisper

        # Case 2: Whisper transcription
        if audio_path is not None and audio_path.exists():
            logger.info("Using Whisper transcription: %s", audio_path)
            return await self._whisper.transcribe(audio_path)

        # Case 3: Nothing available
        raise NoTranscriptAvailableError(
            "No transcript source available. Provide either a caption file "
            "or an audio file for Whisper transcription."
        )

    async def _transcript_from_captions(self, caption_path: Path) -> TranscriptData:
        """
        Parse a caption file into a ``TranscriptData`` object.

        Parsing is offloaded to a thread since it involves file I/O.
        """
        segments = await asyncio.to_thread(
            self._caption_parser.parse_file, caption_path
        )

        if not segments:
            raise CaptionParseError(
                f"Caption file parsed but contained no usable segments: {caption_path}"
            )

        # Sort by start time to guarantee order
        segments.sort(key=lambda s: s.start)

        full_text = " ".join(seg.text for seg in segments)
        max_end = max(seg.end for seg in segments) if segments else 0.0

        return TranscriptData(
            source=TranscriptSource.YOUTUBE_CAPTIONS,
            language="en",  # YouTube captions we download are English
            segments=segments,
            full_text=full_text,
            duration=max_end,
        )

    async def transcribe_with_whisper(self, audio_path: Path) -> TranscriptData:
        """
        Force Whisper transcription regardless of caption availability.

        Useful when the caller specifically needs word-level timestamps
        that YouTube captions don't provide.
        """
        return await self._whisper.transcribe(audio_path)
