"""
Centralized application configuration using Pydantic BaseSettings.

All environment variables are loaded from a .env file (or process env)
and validated at startup. A singleton `settings` instance is exported
for use across the entire application.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Environment variables can be set directly or via a `.env` file placed
    at the project root. Variable names are case-insensitive.
    """

    # ──────────────────────────────────────────────
    # Application
    # ──────────────────────────────────────────────
    app_name: str = Field(default="Amissio", description="Application display name")
    app_version: str = Field(default="1.0.0", description="Semantic version string")
    debug: bool = Field(default=False, description="Enable debug logging and hot-reload")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Python logging level",
    )
    environment: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Deployment environment",
    )

    # ──────────────────────────────────────────────
    # Server
    # ──────────────────────────────────────────────
    host: str = Field(default="0.0.0.0", description="Uvicorn bind address")
    port: int = Field(default=8000, ge=1, le=65535, description="Uvicorn bind port")
    workers: int = Field(default=1, ge=1, description="Uvicorn worker count")
    cors_origins: list[str] = Field(
        default=["*"],
        description="Allowed CORS origins (list of URL strings)",
    )

    # ──────────────────────────────────────────────
    # Google Gemini
    # ──────────────────────────────────────────────
    gemini_api_key: str = Field(
        ...,
        description="Google AI / Gemini API key (required)",
    )
    gemini_model: str = Field(
        default="gemini-2.5-flash",
        description="Gemini model for text generation / note synthesis",
    )
    gemini_vision_model: str = Field(
        default="gemini-2.5-flash",
        description="Gemini model for vision / OCR tasks",
    )
    gemini_embedding_model: str = Field(
        default="text-embedding-004",
        description="Gemini model for text embeddings",
    )
    gemini_embedding_dimension: int = Field(
        default=768,
        ge=64,
        le=3072,
        description="Dimensionality of embedding vectors",
    )
    gemini_max_output_tokens: int = Field(
        default=8192,
        ge=256,
        description="Maximum output tokens per Gemini generation call",
    )
    gemini_temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for Gemini generation",
    )
    gemini_requests_per_minute: int = Field(
        default=15,
        ge=1,
        description="Rate limit: max Gemini requests per minute",
    )
    gemini_max_concurrent: int = Field(
        default=5,
        ge=1,
        description="Max concurrent Gemini API calls",
    )

    # ──────────────────────────────────────────────
    # Qdrant Vector Database
    # ──────────────────────────────────────────────
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="Qdrant server URL",
    )
    qdrant_api_key: str | None = Field(
        default=None,
        description="Qdrant API key (optional for local dev)",
    )
    qdrant_collection_name: str = Field(
        default="amissio_chunks",
        description="Default Qdrant collection name for video chunks",
    )
    qdrant_timeout: int = Field(
        default=30,
        ge=5,
        description="Qdrant client timeout in seconds",
    )

    # ──────────────────────────────────────────────
    # Whisper (faster-whisper) – offline transcription
    # ──────────────────────────────────────────────
    whisper_model_size: Literal[
        "tiny", "tiny.en", "base", "base.en",
        "small", "small.en", "medium", "medium.en",
        "large-v2", "large-v3", "distil-large-v3",
    ] = Field(
        default="base",
        description="faster-whisper model size",
    )
    whisper_device: Literal["cpu", "cuda", "auto"] = Field(
        default="auto",
        description="Compute device for Whisper inference",
    )
    whisper_compute_type: Literal[
        "float32", "float16", "int8", "int8_float16", "auto",
    ] = Field(
        default="auto",
        description="Whisper model quantization / compute type",
    )
    whisper_beam_size: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Beam search width for Whisper decoding",
    )
    whisper_language: str | None = Field(
        default=None,
        description="Force transcription language (ISO 639-1). None = auto-detect.",
    )

    # ──────────────────────────────────────────────
    # Video / Audio Processing
    # ──────────────────────────────────────────────
    download_dir: Path = Field(
        default=Path("./data/downloads"),
        description="Directory for downloaded video/audio files",
    )
    frames_dir: Path = Field(
        default=Path("./data/frames"),
        description="Directory for extracted video frames",
    )
    cache_dir: Path = Field(
        default=Path("./data/cache"),
        description="Directory for pipeline caches (OCR, embeddings, summaries)",
    )
    max_video_duration_seconds: int = Field(
        default=14400,
        ge=60,
        description="Maximum allowed video duration in seconds (default: 4 hours)",
    )
    frame_extraction_fps: float = Field(
        default=1.0,
        ge=0.1,
        le=5.0,
        description="Frame extraction rate (frames per second)",
    )
    frame_hash_threshold: int = Field(
        default=8,
        ge=0,
        le=64,
        description="Perceptual hash hamming-distance threshold for deduplication. "
        "Lower = stricter deduplication.",
    )
    frame_max_dimension: int = Field(
        default=1280,
        ge=480,
        description="Resize frames so the largest dimension does not exceed this value",
    )
    audio_format: Literal["wav", "mp3", "m4a"] = Field(
        default="wav",
        description="Audio format for yt-dlp extraction",
    )

    # ──────────────────────────────────────────────
    # Semantic Chunking
    # ──────────────────────────────────────────────
    chunk_min_tokens: int = Field(
        default=500,
        ge=100,
        description="Minimum tokens per semantic chunk",
    )
    chunk_max_tokens: int = Field(
        default=1200,
        ge=200,
        description="Maximum tokens per semantic chunk",
    )
    chunk_overlap_tokens: int = Field(
        default=50,
        ge=0,
        description="Token overlap between adjacent chunks for continuity",
    )
    similarity_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Cosine similarity threshold for semantic split decisions",
    )

    # ──────────────────────────────────────────────
    # Retry / Resilience
    # ──────────────────────────────────────────────
    retry_max_attempts: int = Field(
        default=5,
        ge=1,
        le=15,
        description="Max retry attempts for transient failures",
    )
    retry_base_delay: float = Field(
        default=1.0,
        ge=0.1,
        description="Base delay in seconds for exponential backoff",
    )
    retry_max_delay: float = Field(
        default=60.0,
        ge=1.0,
        description="Maximum delay cap in seconds for retries",
    )
    retry_exponential_base: float = Field(
        default=2.0,
        ge=1.5,
        le=4.0,
        description="Exponent base for backoff calculation",
    )

    # ──────────────────────────────────────────────
    # Map-Reduce / Hierarchical Summarization
    # ──────────────────────────────────────────────
    segment_duration_seconds: int = Field(
        default=600,
        ge=60,
        description="Duration of each video segment for map-reduce (default: 10 min)",
    )
    merge_batch_size: int = Field(
        default=5,
        ge=2,
        description="Number of segment summaries to merge per reduce step",
    )
    max_parallel_segments: int = Field(
        default=4,
        ge=1,
        description="Max segments to process concurrently in map phase",
    )

    # ──────────────────────────────────────────────
    # Vision / OCR batching
    # ──────────────────────────────────────────────
    vision_batch_size: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of frames sent per Gemini Vision request",
    )
    vision_max_concurrent: int = Field(
        default=3,
        ge=1,
        description="Max concurrent Gemini Vision API calls",
    )

    # ──────────────────────────────────────────────
    # Validators
    # ──────────────────────────────────────────────
    @field_validator("gemini_api_key")
    @classmethod
    def validate_api_key_not_empty(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("GEMINI_API_KEY must not be empty")
        return stripped

    @field_validator("download_dir", "frames_dir", "cache_dir", mode="before")
    @classmethod
    def resolve_path(cls, v: str | Path) -> Path:
        return Path(v).resolve()

    @model_validator(mode="after")
    def validate_chunk_bounds(self) -> Settings:
        if self.chunk_min_tokens >= self.chunk_max_tokens:
            raise ValueError(
                f"chunk_min_tokens ({self.chunk_min_tokens}) must be less than "
                f"chunk_max_tokens ({self.chunk_max_tokens})"
            )
        if self.chunk_overlap_tokens >= self.chunk_min_tokens:
            raise ValueError(
                f"chunk_overlap_tokens ({self.chunk_overlap_tokens}) must be less than "
                f"chunk_min_tokens ({self.chunk_min_tokens})"
            )
        return self

    @model_validator(mode="after")
    def ensure_directories_exist(self) -> Settings:
        for dir_path in (self.download_dir, self.frames_dir, self.cache_dir):
            dir_path.mkdir(parents=True, exist_ok=True)
        return self

    # ──────────────────────────────────────────────
    # Settings config
    # ──────────────────────────────────────────────
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return a cached singleton Settings instance.

    Uses ``lru_cache`` so the .env file is read exactly once per process
    and the same object is reused across the application.
    """
    return Settings()  # type: ignore[call-arg]
