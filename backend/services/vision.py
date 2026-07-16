"""
Vision service: frame extraction, perceptual deduplication, and resize.

Extracts frames from a video at a configurable FPS rate using OpenCV,
deduplicates near-identical frames via ``imagehash`` perceptual hashing,
resizes kept frames to a max dimension for bandwidth-efficient Gemini
Vision calls, and returns structured ``ExtractedFrame`` metadata.

All CPU-heavy OpenCV work is offloaded to ``asyncio.to_thread`` so the
FastAPI event loop is never blocked.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

import cv2
import imagehash
import numpy as np
from PIL import Image

from core.config import Settings, get_settings
from models.schemas import ExtractedFrame

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Custom Exceptions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class VisionServiceError(Exception):
    """Base exception for vision service operations."""


class VideoOpenError(VisionServiceError):
    """Raised when OpenCV cannot open the video file."""


class FrameExtractionError(VisionServiceError):
    """Raised when frame extraction fails mid-process."""


class NoFramesExtractedError(VisionServiceError):
    """Raised when zero frames are extracted from the video."""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Frame Extraction Result
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class FrameExtractionResult:
    """
    Container for the result of a frame extraction run.

    Provides convenient access to both the full frame list and the
    deduplicated subset of unique (scene-change) keyframes.
    """

    def __init__(
        self,
        all_frames: list[ExtractedFrame],
        unique_frames: list[ExtractedFrame],
        video_duration: float,
        processing_time_seconds: float,
    ) -> None:
        self.all_frames = all_frames
        self.unique_frames = unique_frames
        self.video_duration = video_duration
        self.processing_time_seconds = processing_time_seconds

    @property
    def total_extracted(self) -> int:
        """Total number of frames sampled from the video."""
        return len(self.all_frames)

    @property
    def total_unique(self) -> int:
        """Number of unique frames after deduplication."""
        return len(self.unique_frames)

    @property
    def dedup_ratio(self) -> float:
        """Fraction of frames removed by deduplication (0.0–1.0)."""
        if not self.all_frames:
            return 0.0
        return 1.0 - (len(self.unique_frames) / len(self.all_frames))

    @property
    def keyframe_paths(self) -> dict[str, float]:
        """
        Map of unique keyframe file paths to their timestamps in seconds.

        This is the primary output consumed by downstream Gemini Vision calls.
        """
        return {frame.file_path: frame.timestamp for frame in self.unique_frames}

    @property
    def keyframe_list(self) -> list[dict[str, Any]]:
        """
        List of dicts with ``path``, ``timestamp``, and ``frame_index``
        for each unique keyframe — useful for ordered iteration.
        """
        return [
            {
                "path": frame.file_path,
                "timestamp": frame.timestamp,
                "frame_index": frame.frame_index,
                "width": frame.width,
                "height": frame.height,
            }
            for frame in self.unique_frames
        ]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Vision Service
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class VisionService:
    """
    Handles video frame extraction and perceptual deduplication.

    Pipeline:
      1. Open video with OpenCV
      2. Sample frames at ``frame_extraction_fps`` (default 1 fps)
      3. Compute a perceptual hash (pHash) for each frame
      4. Compare to the running set of seen hashes; if the hamming
         distance to every existing hash exceeds the threshold the
         frame is a **new scene** and is kept
      5. Resize kept frames so the largest dimension ≤ ``frame_max_dimension``
      6. Save as JPEG and return structured metadata
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._frames_dir = self._settings.frames_dir
        self._fps = self._settings.frame_extraction_fps
        self._hash_threshold = self._settings.frame_hash_threshold
        self._max_dim = self._settings.frame_max_dimension

    # ─────────────────────────────────────────────
    # Public async API
    # ─────────────────────────────────────────────

    async def extract_frames(
        self,
        video_path: Path,
        video_id: str,
    ) -> FrameExtractionResult:
        """
        Extract and deduplicate frames from a video file.

        Args:
            video_path: Absolute path to the downloaded video file.
            video_id: YouTube video ID (used for output directory naming).

        Returns:
            A ``FrameExtractionResult`` with all frames and unique keyframes.

        Raises:
            VideoOpenError: OpenCV cannot open the video.
            FrameExtractionError: Extraction fails mid-process.
            NoFramesExtractedError: Zero frames could be sampled.
        """
        if not video_path.exists():
            raise VideoOpenError(f"Video file does not exist: {video_path}")

        # Each video gets its own subdirectory under frames_dir
        output_dir = self._frames_dir / video_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Check for existing extraction result (cache hit)
        cached = self._load_cached_result(output_dir)
        if cached is not None:
            logger.info(
                "Using cached frames for %s: %d unique / %d total",
                video_id,
                cached.total_unique,
                cached.total_extracted,
            )
            return cached

        logger.info(
            "Starting frame extraction: video=%s, fps=%.1f, hash_threshold=%d",
            video_id,
            self._fps,
            self._hash_threshold,
        )

        result = await asyncio.to_thread(
            self._extract_and_deduplicate,
            video_path=video_path,
            output_dir=output_dir,
        )

        logger.info(
            "Frame extraction complete for %s: %d unique / %d total "
            "(%.1f%% dedup) in %.1fs",
            video_id,
            result.total_unique,
            result.total_extracted,
            result.dedup_ratio * 100,
            result.processing_time_seconds,
        )

        return result

    # ─────────────────────────────────────────────
    # Core extraction logic (runs in thread pool)
    # ─────────────────────────────────────────────

    def _extract_and_deduplicate(
        self,
        video_path: Path,
        output_dir: Path,
    ) -> FrameExtractionResult:
        """
        Synchronous frame extraction + deduplication pipeline.

        This is the heavy-lifting method, always called via
        ``asyncio.to_thread``.
        """
        start_time = time.monotonic()

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise VideoOpenError(f"OpenCV failed to open video: {video_path}")

        try:
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_duration = total_frames / video_fps if video_fps > 0 else 0.0

            if video_fps <= 0:
                raise FrameExtractionError(
                    f"Invalid video FPS ({video_fps}) — file may be corrupt"
                )

            # How many native frames to skip between samples
            # e.g. 30fps video at 1fps extraction → sample every 30th frame
            frame_interval = int(video_fps / self._fps)
            if frame_interval < 1:
                frame_interval = 1

            logger.debug(
                "Video properties: fps=%.2f, total_frames=%d, duration=%.1fs, "
                "sample_interval=%d",
                video_fps,
                total_frames,
                video_duration,
                frame_interval,
            )

            all_frames: list[ExtractedFrame] = []
            unique_frames: list[ExtractedFrame] = []
            seen_hashes: list[imagehash.ImageHash] = []
            frame_counter = 0
            global_index = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Only process frames at the target interval
                if frame_counter % frame_interval != 0:
                    frame_counter += 1
                    continue

                timestamp = frame_counter / video_fps
                frame_counter += 1

                # Compute perceptual hash
                phash = self._compute_phash(frame)
                phash_hex = str(phash)

                # Check for duplicates against all previously seen hashes
                is_duplicate = self._is_duplicate(phash, seen_hashes)

                h, w = frame.shape[:2]

                if not is_duplicate:
                    # Resize the frame if it exceeds max dimension
                    resized = self._resize_frame(frame)
                    rh, rw = resized.shape[:2]

                    # Save the unique frame as JPEG
                    file_path = output_dir / f"frame_{global_index:05d}_{timestamp:.2f}s.jpg"
                    cv2.imwrite(
                        str(file_path),
                        resized,
                        [cv2.IMWRITE_JPEG_QUALITY, 90],
                    )

                    extracted = ExtractedFrame(
                        frame_index=global_index,
                        timestamp=round(timestamp, 3),
                        file_path=str(file_path.resolve()),
                        perceptual_hash=phash_hex,
                        is_duplicate=False,
                        width=rw,
                        height=rh,
                    )
                    unique_frames.append(extracted)
                    seen_hashes.append(phash)
                else:
                    # Record the duplicate but don't save to disk
                    extracted = ExtractedFrame(
                        frame_index=global_index,
                        timestamp=round(timestamp, 3),
                        file_path="",
                        perceptual_hash=phash_hex,
                        is_duplicate=True,
                        width=w,
                        height=h,
                    )

                all_frames.append(extracted)
                global_index += 1

        finally:
            cap.release()

        if not all_frames:
            raise NoFramesExtractedError(
                f"No frames could be extracted from {video_path}"
            )

        elapsed = time.monotonic() - start_time

        return FrameExtractionResult(
            all_frames=all_frames,
            unique_frames=unique_frames,
            video_duration=video_duration,
            processing_time_seconds=round(elapsed, 2),
        )

    # ─────────────────────────────────────────────
    # Perceptual hashing & deduplication
    # ─────────────────────────────────────────────

    def _compute_phash(self, frame: np.ndarray) -> imagehash.ImageHash:
        """
        Compute the perceptual hash (pHash) of an OpenCV BGR frame.

        Converts from BGR → RGB → PIL → pHash. The 64-bit pHash is
        robust to minor compression artifacts, slight crops, and
        brightness changes while still detecting meaningful scene changes.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        return imagehash.phash(pil_image, hash_size=8)

    def _is_duplicate(
        self,
        current_hash: imagehash.ImageHash,
        seen_hashes: list[imagehash.ImageHash],
    ) -> bool:
        """
        Determine if a frame is a duplicate of any previously seen frame.

        Uses hamming distance between perceptual hashes. A distance of 0
        means identical; the configured threshold determines how much
        visual change is needed to consider a frame "new".

        Comparison strategy: check against only the **most recent** N
        hashes (sliding window) to handle gradual scene transitions
        efficiently. For most videos, comparing against the last 10
        hashes is sufficient since scenes don't revert frequently.
        """
        if not seen_hashes:
            return False

        # Compare against the most recent hashes (sliding window)
        # This catches gradual transitions without O(n²) growth
        window_size = min(len(seen_hashes), 10)
        recent_hashes = seen_hashes[-window_size:]

        for seen in recent_hashes:
            distance = current_hash - seen  # hamming distance
            if distance <= self._hash_threshold:
                return True

        return False

    # ─────────────────────────────────────────────
    # Frame resizing
    # ─────────────────────────────────────────────

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Resize a frame so its largest dimension ≤ ``frame_max_dimension``.

        Preserves aspect ratio. Uses INTER_AREA for downscaling (best
        quality for shrinking) and INTER_LANCZOS4 for the rare upscale case.

        Returns the original frame unchanged if it's already within bounds.
        """
        h, w = frame.shape[:2]
        max_side = max(h, w)

        if max_side <= self._max_dim:
            return frame

        scale = self._max_dim / max_side
        new_w = int(w * scale)
        new_h = int(h * scale)

        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4

        return cv2.resize(frame, (new_w, new_h), interpolation=interpolation)

    # ─────────────────────────────────────────────
    # Cache helpers
    # ─────────────────────────────────────────────

    def _load_cached_result(self, output_dir: Path) -> FrameExtractionResult | None:
        """
        Check if frames were already extracted for this video.

        A simple heuristic: if the output directory contains JPEG files,
        we reconstruct the result from the filenames (which encode the
        index and timestamp). This avoids re-running OpenCV on subsequent
        pipeline restarts.
        """
        if not output_dir.exists():
            return None

        jpg_files = sorted(output_dir.glob("frame_*.jpg"))
        if not jpg_files:
            return None

        unique_frames: list[ExtractedFrame] = []

        for jpg_path in jpg_files:
            # Parse index and timestamp from filename: frame_00042_123.45s.jpg
            stem = jpg_path.stem  # frame_00042_123.45s
            parts = stem.split("_")
            if len(parts) < 3:
                continue

            try:
                frame_index = int(parts[1])
                timestamp = float(parts[2].rstrip("s"))
            except (ValueError, IndexError):
                continue

            # Read dimensions without loading full image into memory
            img = cv2.imread(str(jpg_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            rh, rw = img.shape[:2]

            unique_frames.append(
                ExtractedFrame(
                    frame_index=frame_index,
                    timestamp=timestamp,
                    file_path=str(jpg_path.resolve()),
                    perceptual_hash="",  # Not recomputed from cache
                    is_duplicate=False,
                    width=rw,
                    height=rh,
                )
            )

        if not unique_frames:
            return None

        # Sort by timestamp to maintain order
        unique_frames.sort(key=lambda f: f.timestamp)

        return FrameExtractionResult(
            all_frames=unique_frames,  # We only have unique frames from cache
            unique_frames=unique_frames,
            video_duration=unique_frames[-1].timestamp if unique_frames else 0.0,
            processing_time_seconds=0.0,
        )

    # ─────────────────────────────────────────────
    # Utility methods
    # ─────────────────────────────────────────────

    async def cleanup_frames(self, video_id: str) -> int:
        """
        Delete all extracted frames for a video to free disk space.

        Returns the number of files removed.
        """
        output_dir = self._frames_dir / video_id
        if not output_dir.exists():
            return 0

        return await asyncio.to_thread(self._delete_directory_contents, output_dir)

    @staticmethod
    def _delete_directory_contents(directory: Path) -> int:
        """Remove all files in a directory (sync, for thread pool)."""
        count = 0
        for f in directory.iterdir():
            if f.is_file():
                f.unlink()
                count += 1
        return count

    @staticmethod
    def get_video_info(video_path: Path) -> dict[str, Any]:
        """
        Get basic video properties via OpenCV without extracting frames.

        Returns a dict with fps, total_frames, duration, width, height, and codec.

        Raises:
            VideoOpenError: If the file cannot be opened.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise VideoOpenError(f"Cannot open video file: {video_path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            codec_int = int(cap.get(cv2.CAP_PROP_FOURCC))

            # Decode FourCC integer to string
            codec = "".join(
                chr((codec_int >> 8 * i) & 0xFF) for i in range(4)
            )

            duration = total_frames / fps if fps > 0 else 0.0

            return {
                "fps": round(fps, 2),
                "total_frames": total_frames,
                "duration": round(duration, 2),
                "width": width,
                "height": height,
                "codec": codec.strip(),
            }
        finally:
            cap.release()
