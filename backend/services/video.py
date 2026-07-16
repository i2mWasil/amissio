"""
Video service: YouTube metadata extraction, video download, and audio extraction.

Uses ``yt-dlp`` as a library (not CLI subprocess) for reliability and structured
output. All I/O-bound work is offloaded to a thread-pool via ``asyncio.to_thread``
so the FastAPI event loop is never blocked.
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import yt_dlp

from core.config import Settings, get_settings
from models.schemas import VideoChapter, VideoMetadata

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Custom Exceptions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class VideoServiceError(Exception):
    """Base exception for video service operations."""


class VideoNotFoundError(VideoServiceError):
    """Raised when the video does not exist or has been removed."""


class VideoPrivateError(VideoServiceError):
    """Raised when the video is private and cannot be accessed."""


class VideoAgeRestrictedError(VideoServiceError):
    """Raised when the video is age-restricted and cannot be downloaded without auth."""


class VideoDurationExceededError(VideoServiceError):
    """Raised when the video exceeds the configured maximum duration."""


class VideoDownloadError(VideoServiceError):
    """Raised when yt-dlp fails to download the video or audio."""


class InvalidURLError(VideoServiceError):
    """Raised when the URL is not a valid YouTube link."""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# URL Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def extract_video_id(url: str) -> str:
    """
    Extract the 11-character YouTube video ID from various URL formats.

    Supported formats:
      - https://www.youtube.com/watch?v=VIDEO_ID
      - https://youtu.be/VIDEO_ID
      - https://www.youtube.com/embed/VIDEO_ID
      - https://www.youtube.com/shorts/VIDEO_ID
      - https://m.youtube.com/watch?v=VIDEO_ID

    Raises:
        InvalidURLError: If the video ID cannot be extracted.
    """
    parsed = urlparse(url)
    hostname = parsed.hostname or ""

    video_id: str | None = None

    # Standard watch URL: youtube.com/watch?v=...
    if hostname in ("youtube.com", "www.youtube.com", "m.youtube.com"):
        if parsed.path == "/watch":
            qs = parse_qs(parsed.query)
            video_id = qs.get("v", [None])[0]  # type: ignore[assignment]
        elif parsed.path.startswith(("/embed/", "/shorts/", "/v/")):
            # /embed/VIDEO_ID, /shorts/VIDEO_ID, /v/VIDEO_ID
            parts = parsed.path.split("/")
            if len(parts) >= 3:
                video_id = parts[2]

    # Short URL: youtu.be/VIDEO_ID
    elif hostname in ("youtu.be", "www.youtu.be"):
        video_id = parsed.path.lstrip("/").split("/")[0] or None

    # Nocookie embed: youtube-nocookie.com/embed/VIDEO_ID
    elif hostname in ("youtube-nocookie.com", "www.youtube-nocookie.com"):
        if parsed.path.startswith("/embed/"):
            parts = parsed.path.split("/")
            if len(parts) >= 3:
                video_id = parts[2]

    if not video_id:
        raise InvalidURLError(f"Could not extract video ID from URL: {url}")

    # Validate the ID format (11 alphanumeric chars, hyphens, underscores)
    if not re.match(r"^[a-zA-Z0-9_-]{11}$", video_id):
        raise InvalidURLError(
            f"Extracted video ID '{video_id}' does not match expected format"
        )

    return video_id


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Video Service
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class VideoService:
    """
    Handles all interactions with YouTube via yt-dlp:
      - Metadata extraction (no download)
      - Audio-only download for transcription
      - Full video download for frame extraction
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._download_dir = self._settings.download_dir
        self._download_dir.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────
    # Public async API
    # ─────────────────────────────────────────────

    async def extract_metadata(self, url: str) -> VideoMetadata:
        """
        Extract video metadata without downloading.

        Returns a populated ``VideoMetadata`` model with title, duration,
        chapters, caption availability, and more.

        Raises:
            VideoNotFoundError: Video does not exist.
            VideoPrivateError: Video is private.
            VideoAgeRestrictedError: Video requires age verification.
            VideoDurationExceededError: Video exceeds max allowed duration.
            InvalidURLError: URL is not a valid YouTube link.
        """
        video_id = extract_video_id(url)
        logger.info("Extracting metadata for video: %s", video_id)

        info = await asyncio.to_thread(self._fetch_info, url)

        duration = int(info.get("duration", 0) or 0)
        if duration > self._settings.max_video_duration_seconds:
            raise VideoDurationExceededError(
                f"Video duration ({duration}s) exceeds maximum "
                f"({self._settings.max_video_duration_seconds}s)"
            )

        chapters = self._parse_chapters(info, duration)
        caption_info = self._parse_captions(info)

        # Pick the best thumbnail
        thumbnails = info.get("thumbnails", [])
        thumbnail_url = ""
        if thumbnails:
            # yt-dlp orders thumbnails by preference; last is usually best
            thumbnail_url = thumbnails[-1].get("url", "")

        metadata = VideoMetadata(
            video_id=video_id,
            title=info.get("title", "Untitled"),
            description=info.get("description", "") or "",
            channel=info.get("uploader", "") or info.get("channel", "") or "",
            channel_id=info.get("channel_id", "") or "",
            duration=duration,
            thumbnail_url=thumbnail_url,
            upload_date=info.get("upload_date", "") or "",
            view_count=int(info.get("view_count", 0) or 0),
            chapters=chapters,
            tags=info.get("tags", []) or [],
            language=info.get("language") or None,
            has_captions=caption_info["has_captions"],
            caption_languages=caption_info["languages"],
        )

        logger.info(
            "Metadata extracted: title='%s', duration=%ds, chapters=%d, captions=%s",
            metadata.title,
            metadata.duration,
            len(metadata.chapters),
            metadata.has_captions,
        )
        return metadata

    async def download_audio(self, url: str, video_id: str) -> Path:
        """
        Download audio-only from the YouTube video.

        Returns the path to the downloaded audio file. Uses the configured
        audio format (default: wav) for Whisper compatibility.

        Raises:
            VideoDownloadError: If the download fails for any reason.
        """
        output_path = self._download_dir / video_id / f"audio.{self._settings.audio_format}"
        if output_path.exists():
            logger.info("Audio already downloaded: %s", output_path)
            return output_path

        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading audio for %s -> %s", video_id, output_path)

        await asyncio.to_thread(
            self._download,
            url=url,
            output_template=str(output_path.parent / "audio.%(ext)s"),
            extract_audio=True,
            download_video=False,
        )

        # yt-dlp may produce a slightly different extension; find the actual file
        actual_path = self._find_downloaded_file(output_path.parent, "audio")
        if actual_path is None:
            raise VideoDownloadError(
                f"Audio download completed but file not found in {output_path.parent}"
            )

        logger.info("Audio downloaded: %s (%.1f MB)", actual_path, actual_path.stat().st_size / 1e6)
        return actual_path

    async def download_video(self, url: str, video_id: str) -> Path:
        """
        Download the video file for frame extraction.

        Downloads at a reasonable quality (720p max) to balance frame
        quality and disk/bandwidth usage.

        Raises:
            VideoDownloadError: If the download fails.
        """
        output_dir = self._download_dir / video_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Check if video already exists
        existing = self._find_downloaded_file(output_dir, "video")
        if existing is not None:
            logger.info("Video already downloaded: %s", existing)
            return existing

        logger.info("Downloading video for %s -> %s", video_id, output_dir)

        await asyncio.to_thread(
            self._download,
            url=url,
            output_template=str(output_dir / "video.%(ext)s"),
            extract_audio=False,
            download_video=True,
        )

        actual_path = self._find_downloaded_file(output_dir, "video")
        if actual_path is None:
            raise VideoDownloadError(
                f"Video download completed but file not found in {output_dir}"
            )

        logger.info("Video downloaded: %s (%.1f MB)", actual_path, actual_path.stat().st_size / 1e6)
        return actual_path

    async def download_captions(self, url: str, video_id: str) -> Path | None:
        """
        Download YouTube subtitles as a VTT file.

        Returns the path to the subtitle file, or None if no captions are
        available. Prefers manually created captions over auto-generated ones.
        """
        output_dir = self._download_dir / video_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Check for existing subtitle files
        for ext in ("vtt", "srt", "ass"):
            for f in output_dir.glob(f"*.{ext}"):
                logger.info("Captions already downloaded: %s", f)
                return f

        logger.info("Attempting to download captions for %s", video_id)

        try:
            await asyncio.to_thread(self._download_subs, url, output_dir)
        except Exception as exc:
            logger.warning("Caption download failed (non-fatal): %s", exc)
            return None

        # Find the downloaded subtitle file
        for ext in ("vtt", "srt", "ass"):
            for f in output_dir.glob(f"*.{ext}"):
                logger.info("Captions downloaded: %s", f)
                return f

        logger.info("No caption file found after download attempt for %s", video_id)
        return None

    # ─────────────────────────────────────────────
    # Private sync helpers (run in thread pool)
    # ─────────────────────────────────────────────

    def _fetch_info(self, url: str) -> dict[str, Any]:
        """
        Fetch video info dict using yt-dlp without downloading.

        Maps yt-dlp errors to typed exceptions.
        """
        ydl_opts: dict[str, Any] = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "no_color": True,
            # Request subtitle info so we can check availability
            "writesubtitles": False,
            "writeautomaticsub": False,
            "extract_flat": False,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info: dict[str, Any] = ydl.extract_info(url, download=False) or {}
                return info

        except yt_dlp.utils.DownloadError as exc:
            error_msg = str(exc).lower()
            self._raise_typed_error(error_msg, url)
            # _raise_typed_error always raises, but just in case:
            raise VideoDownloadError(f"Failed to extract info: {exc}") from exc

    def _download(
        self,
        url: str,
        output_template: str,
        extract_audio: bool,
        download_video: bool,
    ) -> None:
        """Execute the actual yt-dlp download."""
        ydl_opts: dict[str, Any] = {
            "outtmpl": output_template,
            "quiet": True,
            "no_warnings": True,
            "no_color": True,
            "retries": self._settings.retry_max_attempts,
            "fragment_retries": self._settings.retry_max_attempts,
            "overwrites": False,
        }

        if extract_audio and not download_video:
            # Audio-only extraction
            ydl_opts.update(
                {
                    "format": "bestaudio/best",
                    "postprocessors": [
                        {
                            "key": "FFmpegExtractAudio",
                            "preferredcodec": self._settings.audio_format,
                            "preferredquality": "192",
                        }
                    ],
                }
            )
        elif download_video:
            # Video download — cap at 720p for reasonable file size
            ydl_opts.update(
                {
                    "format": "bestvideo[height<=720]+bestaudio/best[height<=720]/best",
                    "merge_output_format": "mp4",
                }
            )

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

        except yt_dlp.utils.DownloadError as exc:
            error_msg = str(exc).lower()
            self._raise_typed_error(error_msg, url)
            raise VideoDownloadError(f"Download failed: {exc}") from exc

    def _download_subs(self, url: str, output_dir: Path) -> None:
        """Download subtitles (manual preferred, then auto-generated)."""
        ydl_opts: dict[str, Any] = {
            "outtmpl": str(output_dir / "%(title)s.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
            "no_color": True,
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitlesformat": "vtt",
            "subtitleslangs": ["en", "en-US", "en-GB"],
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

    def _raise_typed_error(self, error_msg: str, url: str) -> None:
        """
        Parse yt-dlp error messages and raise the appropriate typed exception.

        This handles the most common failure modes so the caller gets
        structured errors instead of generic DownloadError.
        """
        if any(
            phrase in error_msg
            for phrase in ("video unavailable", "not available", "been removed", "does not exist")
        ):
            raise VideoNotFoundError(f"Video not found or unavailable: {url}")

        if any(phrase in error_msg for phrase in ("private video", "is private")):
            raise VideoPrivateError(f"Video is private: {url}")

        if any(
            phrase in error_msg
            for phrase in (
                "age-restricted",
                "age restricted",
                "sign in to confirm your age",
                "age gate",
                "age verification",
            )
        ):
            raise VideoAgeRestrictedError(
                f"Video is age-restricted and requires authentication: {url}"
            )

        if "members-only" in error_msg or "member" in error_msg:
            raise VideoPrivateError(f"Video is members-only content: {url}")

        if "copyright" in error_msg:
            raise VideoNotFoundError(f"Video removed due to copyright: {url}")

        if any(phrase in error_msg for phrase in ("geo restricted", "geo-restricted", "not available in your country")):
            raise VideoNotFoundError(f"Video is geo-restricted: {url}")

    # ─────────────────────────────────────────────
    # Parsing helpers
    # ─────────────────────────────────────────────

    def _parse_chapters(
        self,
        info: dict[str, Any],
        total_duration: int,
    ) -> list[VideoChapter]:
        """
        Extract chapter information from the yt-dlp info dict.

        Chapters come from either the video description timestamps or
        YouTube's native chapter API. yt-dlp normalizes both into the
        ``chapters`` key.
        """
        raw_chapters: list[dict[str, Any]] = info.get("chapters") or []
        if not raw_chapters:
            return []

        chapters: list[VideoChapter] = []
        for i, ch in enumerate(raw_chapters):
            start = float(ch.get("start_time", 0))
            # Use the next chapter's start as this chapter's end
            if i + 1 < len(raw_chapters):
                end = float(raw_chapters[i + 1].get("start_time", total_duration))
            else:
                end = float(total_duration)

            chapters.append(
                VideoChapter(
                    title=str(ch.get("title", f"Chapter {i + 1}")),
                    start_time=start,
                    end_time=end,
                )
            )

        logger.debug("Parsed %d chapters from video info", len(chapters))
        return chapters

    def _parse_captions(self, info: dict[str, Any]) -> dict[str, Any]:
        """
        Determine caption availability from the info dict.

        Returns a dict with ``has_captions`` and ``languages`` keys.
        """
        subtitles: dict[str, Any] = info.get("subtitles") or {}
        automatic_captions: dict[str, Any] = info.get("automatic_captions") or {}

        # Manual subtitles are preferred
        manual_langs = list(subtitles.keys())
        auto_langs = list(automatic_captions.keys())
        all_langs = list(set(manual_langs + auto_langs))

        has_captions = bool(all_langs)

        return {
            "has_captions": has_captions,
            "languages": sorted(all_langs),
        }

    @staticmethod
    def _find_downloaded_file(directory: Path, prefix: str) -> Path | None:
        """
        Find the actual downloaded file in a directory by prefix.

        yt-dlp may add extensions or modify the filename slightly,
        so we search for any file matching the prefix.
        """
        if not directory.exists():
            return None

        # Direct match first
        for f in directory.iterdir():
            if f.is_file() and f.stem == prefix:
                return f

        # Prefix match (e.g. "audio.wav" or "audio.mp3")
        for f in directory.iterdir():
            if f.is_file() and f.name.startswith(prefix):
                return f

        return None
