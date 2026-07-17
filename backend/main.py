"""
Amissio — Multimodal RAG API for YouTube Note Generation.

FastAPI application entry point with:
  - Lifespan management for service startup/shutdown
  - Structured logging configuration
  - Global exception handlers for typed + unhandled errors
  - CORS middleware
  - Health check endpoint
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.chat import router as chat_router
from api.routes import router as notes_router
from core.config import get_settings
from models.schemas import ErrorResponse

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Logging
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _configure_logging() -> None:
    """Configure structured logging to stdout."""
    log_format = (
        "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
    )
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    # Quieten noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("qdrant_client").setLevel(logging.WARNING)
    logging.getLogger("faster_whisper").setLevel(logging.WARNING)
    logging.getLogger("google").setLevel(logging.WARNING)


_configure_logging()
logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Lifespan
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan handler.

    Startup: validate configuration, create data directories.
    Shutdown: log clean exit.
    """
    settings = get_settings()

    # Create required directories
    settings.download_dir.mkdir(parents=True, exist_ok=True)
    settings.frames_dir.mkdir(parents=True, exist_ok=True)
    settings.cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info("═" * 60)
    logger.info("  Amissio API starting")
    logger.info("═" * 60)
    logger.info("  Gemini model    : %s", settings.gemini_model)
    logger.info("  Vision model    : %s", settings.gemini_vision_model)
    logger.info("  Embedding model : %s", settings.gemini_embedding_model)
    logger.info("  Whisper model   : %s", settings.whisper_model_size)
    logger.info("  Qdrant URL      : %s", settings.qdrant_url)
    logger.info("  Max concurrent  : %d", settings.gemini_max_concurrent)
    logger.info("  Chunk tokens    : %d–%d", settings.chunk_min_tokens, settings.chunk_max_tokens)
    logger.info("═" * 60)

    yield

    logger.info("Amissio API shutting down")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Application
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


app = FastAPI(
    title="Amissio — Multimodal RAG Notes API",
    description=(
        "Generate textbook-quality study notes from YouTube videos using "
        "a multimodal Retrieval-Augmented Generation pipeline. Combines "
        "transcript analysis, visual frame extraction, OCR, and Gemini-powered "
        "note synthesis with Map-Reduce hierarchical summarization."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Middleware
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_timing_middleware(request: Request, call_next):
    """Log request timing for observability."""
    start = time.monotonic()
    response = await call_next(request)
    elapsed_ms = int((time.monotonic() - start) * 1000)

    logger.info(
        "%s %s → %d (%dms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    response.headers["X-Processing-Time-Ms"] = str(elapsed_ms)
    return response


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Exception Handlers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Convert Pydantic validation errors into our ErrorResponse schema."""
    errors = exc.errors()
    detail_parts = []
    for err in errors:
        loc = " → ".join(str(l) for l in err.get("loc", []))
        msg = err.get("msg", "")
        detail_parts.append(f"{loc}: {msg}")

    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error="Validation error",
            detail="; ".join(detail_parts),
            status_code=422,
        ).model_dump(mode="json"),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Catch-all for unhandled exceptions — never expose internals."""
    logger.error(
        "Unhandled exception on %s %s: %s",
        request.method,
        request.url.path,
        exc,
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail="An unexpected error occurred. Check server logs for details.",
            status_code=500,
        ).model_dump(mode="json"),
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Routes
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


app.include_router(notes_router, prefix="/api/v1")
app.include_router(chat_router, prefix="/api/v1")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Health Check
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@app.get(
    "/health",
    summary="Health check",
    tags=["system"],
)
async def health_check() -> dict:
    """
    Basic liveness probe.

    Returns the application status and current configuration summary.
    """
    settings = get_settings()
    return {
        "status": "healthy",
        "service": "amissio",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "config": {
            "gemini_model": settings.gemini_model,
            "gemini_vision_model": settings.gemini_vision_model,
            "embedding_model": settings.gemini_embedding_model,
            "whisper_model": settings.whisper_model_size,
            "qdrant_url": settings.qdrant_url,
        },
    }
