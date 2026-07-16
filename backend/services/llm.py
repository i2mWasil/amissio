"""
LLM service: async Gemini client with retry, rate limiting, and structured output.

Wraps the official ``google-genai`` SDK to provide:
  - Singleton async client initialisation
  - Text generation (with optional JSON schema enforcement)
  - Vision / multimodal generation (images + text)
  - Text embedding
  - Retry with exponential backoff via ``tenacity``
  - Semaphore-based concurrency limiting
  - Token usage tracking
"""

from __future__ import annotations

import asyncio
import base64
import logging
import mimetypes
import time
from pathlib import Path
from typing import Any, Type

from google import genai
from google.genai import types
from pydantic import BaseModel
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from core.config import Settings, get_settings

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Custom Exceptions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class LLMServiceError(Exception):
    """Base exception for LLM service operations."""


class LLMRateLimitError(LLMServiceError):
    """Raised when Gemini returns a rate-limit (429) error."""


class LLMContentBlockedError(LLMServiceError):
    """Raised when Gemini blocks the response due to safety filters."""


class LLMResponseError(LLMServiceError):
    """Raised when the Gemini response is malformed or empty."""


class LLMConnectionError(LLMServiceError):
    """Raised on network-level failures talking to Gemini."""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Retry Predicate
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _is_retryable(exc: BaseException) -> bool:
    """
    Determine if an exception is transient and worth retrying.

    Retryable:
      - Rate-limit errors (429)
      - Server errors (500, 503)
      - Network / connection errors
      - google.genai API errors that indicate transient issues

    Not retryable:
      - Content blocked by safety filters
      - Invalid request (400)
      - Authentication failures (401, 403)
    """
    exc_str = str(exc).lower()

    # Our own typed errors
    if isinstance(exc, LLMRateLimitError):
        return True
    if isinstance(exc, LLMContentBlockedError):
        return False
    if isinstance(exc, LLMConnectionError):
        return True

    # google-genai SDK errors
    if isinstance(exc, genai.errors.APIError):
        code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        if code in (429, 500, 502, 503, 504):
            return True
        if code in (400, 401, 403, 404):
            return False
        return True  # Unknown API error → retry conservatively

    # Network-level errors
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return True

    # Check string patterns for errors we may not have typed
    if any(kw in exc_str for kw in ("rate limit", "quota", "resource exhausted", "429")):
        return True
    if any(kw in exc_str for kw in ("500", "503", "service unavailable", "internal")):
        return True

    return False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LLM Service
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class LLMService:
    """
    Async wrapper around the ``google-genai`` SDK.

    Provides retried, rate-limited methods for text generation,
    vision analysis, and embedding. Designed to be instantiated once
    and shared across the application.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

        # Initialise the google-genai client
        self._client = genai.Client(api_key=self._settings.gemini_api_key)
        self._async_client = self._client.aio

        # Concurrency semaphores — prevent overwhelming the API
        self._gen_semaphore = asyncio.Semaphore(self._settings.gemini_max_concurrent)
        self._vision_semaphore = asyncio.Semaphore(self._settings.vision_max_concurrent)
        self._embed_semaphore = asyncio.Semaphore(self._settings.gemini_max_concurrent)

        # Cumulative token counters
        self._total_input_tokens = 0
        self._total_output_tokens = 0

        logger.info(
            "LLMService initialised: text_model=%s, vision_model=%s, "
            "embed_model=%s, max_concurrent=%d",
            self._settings.gemini_model,
            self._settings.gemini_vision_model,
            self._settings.gemini_embedding_model,
            self._settings.gemini_max_concurrent,
        )

    # ─────────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────────

    @property
    def total_tokens_used(self) -> int:
        """Cumulative token usage across all calls."""
        return self._total_input_tokens + self._total_output_tokens

    @property
    def token_stats(self) -> dict[str, int]:
        return {
            "input_tokens": self._total_input_tokens,
            "output_tokens": self._total_output_tokens,
            "total_tokens": self.total_tokens_used,
        }

    # ─────────────────────────────────────────────
    # Text Generation
    # ─────────────────────────────────────────────

    async def generate_text(
        self,
        prompt: str,
        *,
        system_instruction: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        response_schema: Type[BaseModel] | None = None,
    ) -> str:
        """
        Generate text from a prompt using Gemini.

        Args:
            prompt: The user prompt.
            system_instruction: Optional system-level instruction.
            model: Override the default text model.
            temperature: Override the default temperature.
            max_output_tokens: Override the default max tokens.
            response_schema: If provided, forces JSON output matching
                this Pydantic model schema.

        Returns:
            The generated text (or JSON string if schema is provided).

        Raises:
            LLMServiceError: On unrecoverable generation failures.
        """
        model_name = model or self._settings.gemini_model
        temp = temperature if temperature is not None else self._settings.gemini_temperature
        max_tokens = max_output_tokens or self._settings.gemini_max_output_tokens

        config = types.GenerateContentConfig(
            temperature=temp,
            max_output_tokens=max_tokens,
            system_instruction=system_instruction,
        )

        # Enforce structured JSON output if a schema is provided
        if response_schema is not None:
            config.response_mime_type = "application/json"
            config.response_schema = response_schema

        async with self._gen_semaphore:
            return await self._generate_with_retry(
                model=model_name,
                contents=[prompt],
                config=config,
            )

    async def generate_text_from_parts(
        self,
        parts: list[str | types.Part],
        *,
        system_instruction: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        response_schema: Type[BaseModel] | None = None,
    ) -> str:
        """
        Generate text from a mixed list of text strings and Part objects.

        Useful for multimodal prompts with inline images, or multi-turn
        content that needs explicit Part construction.
        """
        model_name = model or self._settings.gemini_model
        temp = temperature if temperature is not None else self._settings.gemini_temperature
        max_tokens = max_output_tokens or self._settings.gemini_max_output_tokens

        config = types.GenerateContentConfig(
            temperature=temp,
            max_output_tokens=max_tokens,
            system_instruction=system_instruction,
        )

        if response_schema is not None:
            config.response_mime_type = "application/json"
            config.response_schema = response_schema

        async with self._gen_semaphore:
            return await self._generate_with_retry(
                model=model_name,
                contents=parts,
                config=config,
            )

    # ─────────────────────────────────────────────
    # Vision / Multimodal Generation
    # ─────────────────────────────────────────────

    async def analyze_image(
        self,
        image_path: Path,
        prompt: str,
        *,
        system_instruction: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        response_schema: Type[BaseModel] | None = None,
    ) -> str:
        """
        Send a single image + prompt to Gemini Vision.

        The image is sent as inline bytes (no upload needed for frames
        under the ~20 MB limit).

        Returns:
            The generated text/JSON response.
        """
        model_name = model or self._settings.gemini_vision_model
        image_part = self._load_image_part(image_path)

        config = types.GenerateContentConfig(
            temperature=temperature if temperature is not None else 0.2,
            max_output_tokens=max_output_tokens or self._settings.gemini_max_output_tokens,
            system_instruction=system_instruction,
        )

        if response_schema is not None:
            config.response_mime_type = "application/json"
            config.response_schema = response_schema

        async with self._vision_semaphore:
            return await self._generate_with_retry(
                model=model_name,
                contents=[image_part, prompt],
                config=config,
            )

    async def analyze_images_batch(
        self,
        image_paths: list[Path],
        prompt: str,
        *,
        system_instruction: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        response_schema: Type[BaseModel] | None = None,
    ) -> str:
        """
        Send multiple images + a shared prompt to Gemini Vision in a
        single request.

        Used for batch OCR / vision analysis where frames are from
        the same time window and benefit from shared context.

        Returns:
            The generated text/JSON response covering all images.
        """
        model_name = model or self._settings.gemini_vision_model

        # Build the content parts: images first, then the prompt
        parts: list[types.Part | str] = []
        for i, path in enumerate(image_paths):
            parts.append(self._load_image_part(path))
            parts.append(f"[Frame {i + 1}]")
        parts.append(prompt)

        config = types.GenerateContentConfig(
            temperature=temperature if temperature is not None else 0.2,
            max_output_tokens=max_output_tokens or self._settings.gemini_max_output_tokens,
            system_instruction=system_instruction,
        )

        if response_schema is not None:
            config.response_mime_type = "application/json"
            config.response_schema = response_schema

        async with self._vision_semaphore:
            return await self._generate_with_retry(
                model=model_name,
                contents=parts,
                config=config,
            )

    # ─────────────────────────────────────────────
    # Embeddings
    # ─────────────────────────────────────────────

    async def embed_text(self, text: str) -> list[float]:
        """
        Generate an embedding vector for a single text string.

        Returns:
            A list of floats (the embedding vector).
        """
        async with self._embed_semaphore:
            return await self._embed_with_retry(text)

    async def embed_texts_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts concurrently.

        Respects the concurrency semaphore to avoid rate-limiting.
        Each text is embedded in a separate API call (the embed
        endpoint does not support true batching of heterogeneous
        texts in the google-genai SDK).

        Returns:
            A list of embedding vectors in the same order as input.
        """
        tasks = [self.embed_text(t) for t in texts]
        return await asyncio.gather(*tasks)

    # ─────────────────────────────────────────────
    # Core retry wrapper
    # ─────────────────────────────────────────────

    async def _generate_with_retry(
        self,
        model: str,
        contents: list[Any],
        config: types.GenerateContentConfig,
    ) -> str:
        """
        Call ``generate_content`` with tenacity retry logic.

        Retries on transient errors with exponential backoff.
        Raises typed exceptions for permanent failures.
        """
        last_exception: BaseException | None = None

        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self._settings.retry_max_attempts),
                wait=wait_exponential(
                    multiplier=self._settings.retry_base_delay,
                    min=self._settings.retry_base_delay,
                    max=self._settings.retry_max_delay,
                    exp_base=self._settings.retry_exponential_base,
                ),
                retry=retry_if_exception(_is_retryable),
                reraise=True,
            ):
                with attempt:
                    start = time.monotonic()

                    response = await self._async_client.models.generate_content(
                        model=model,
                        contents=contents,
                        config=config,
                    )

                    elapsed_ms = int((time.monotonic() - start) * 1000)

                    # Track token usage
                    if response.usage_metadata:
                        self._total_input_tokens += (
                            response.usage_metadata.prompt_token_count or 0
                        )
                        self._total_output_tokens += (
                            response.usage_metadata.candidates_token_count or 0
                        )

                    # Validate response
                    text = self._extract_text(response)

                    logger.debug(
                        "Gemini call: model=%s, elapsed=%dms, tokens=%s",
                        model,
                        elapsed_ms,
                        response.usage_metadata,
                    )

                    return text

        except RetryError as exc:
            last_exception = exc.last_attempt.exception()
            logger.error(
                "Gemini call failed after %d retries: %s",
                self._settings.retry_max_attempts,
                last_exception,
            )
            raise LLMServiceError(
                f"Gemini generation failed after {self._settings.retry_max_attempts} "
                f"retries: {last_exception}"
            ) from last_exception

        # Should not reach here, but satisfy type checker
        raise LLMServiceError("Unexpected retry exit")  # pragma: no cover

    async def _embed_with_retry(self, text: str) -> list[float]:
        """
        Call ``embed_content`` with tenacity retry logic.
        """
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self._settings.retry_max_attempts),
                wait=wait_exponential(
                    multiplier=self._settings.retry_base_delay,
                    min=self._settings.retry_base_delay,
                    max=self._settings.retry_max_delay,
                    exp_base=self._settings.retry_exponential_base,
                ),
                retry=retry_if_exception(_is_retryable),
                reraise=True,
            ):
                with attempt:
                    response = await self._async_client.models.embed_content(
                        model=self._settings.gemini_embedding_model,
                        contents=text,
                    )

                    if not response.embeddings or not response.embeddings[0].values:
                        raise LLMResponseError("Empty embedding response from Gemini")

                    return list(response.embeddings[0].values)

        except RetryError as exc:
            last_exception = exc.last_attempt.exception()
            raise LLMServiceError(
                f"Embedding failed after {self._settings.retry_max_attempts} "
                f"retries: {last_exception}"
            ) from last_exception

        raise LLMServiceError("Unexpected retry exit")  # pragma: no cover

    # ─────────────────────────────────────────────
    # Response parsing helpers
    # ─────────────────────────────────────────────

    @staticmethod
    def _extract_text(response: types.GenerateContentResponse) -> str:
        """
        Extract the text from a Gemini GenerateContentResponse.

        Handles safety-blocked responses, empty candidates, and
        multi-part responses gracefully.

        Raises:
            LLMContentBlockedError: Safety filter blocked the response.
            LLMResponseError: Response is empty or malformed.
        """
        # Check for prompt-level blocking
        if response.prompt_feedback and response.prompt_feedback.block_reason:
            raise LLMContentBlockedError(
                f"Prompt blocked by safety filter: "
                f"{response.prompt_feedback.block_reason}"
            )

        if not response.candidates:
            raise LLMResponseError("Gemini returned no candidates")

        candidate = response.candidates[0]

        # Check candidate-level finish reason
        finish_reason = getattr(candidate, "finish_reason", None)
        if finish_reason and "SAFETY" in str(finish_reason).upper():
            raise LLMContentBlockedError(
                f"Response blocked by safety filter: {finish_reason}"
            )

        # Extract text from all parts
        text_parts: list[str] = []
        if candidate.content and candidate.content.parts:
            for part in candidate.content.parts:
                if hasattr(part, "text") and part.text:
                    text_parts.append(part.text)

        result = "\n".join(text_parts).strip()
        if not result:
            raise LLMResponseError(
                f"Gemini returned empty text. Finish reason: {finish_reason}"
            )

        return result

    # ─────────────────────────────────────────────
    # Image loading helpers
    # ─────────────────────────────────────────────

    @staticmethod
    def _load_image_part(image_path: Path) -> types.Part:
        """
        Load an image file as a ``types.Part`` for inline sending.

        Detects MIME type from the file extension. Supports JPEG, PNG,
        WebP, and GIF.

        Raises:
            FileNotFoundError: If the image file does not exist.
            ValueError: If the MIME type cannot be determined.
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        mime_type, _ = mimetypes.guess_type(str(image_path))
        if mime_type is None:
            # Fallback based on common extensions
            ext_map = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".webp": "image/webp",
                ".gif": "image/gif",
            }
            mime_type = ext_map.get(image_path.suffix.lower())

        if not mime_type:
            raise ValueError(
                f"Cannot determine MIME type for: {image_path}"
            )

        image_bytes = image_path.read_bytes()

        return types.Part.from_bytes(
            data=image_bytes,
            mime_type=mime_type,
        )
