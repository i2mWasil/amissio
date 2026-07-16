"""
Vector database service: Qdrant async client for chunk storage and retrieval.

Handles the full lifecycle:
  1. **Collection management** — create or verify the Qdrant collection
     with the correct dimensionality and distance metric.
  2. **Embedding** — embed chunk text via ``LLMService`` (Gemini text-embedding).
  3. **Upsert** — store vectors with rich ``QdrantPointPayload`` metadata.
  4. **Search** — similarity search for retrieval-augmented generation.
  5. **Cleanup** — delete points by video ID.

Uses the official ``qdrant-client`` async interface exclusively.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from core.config import Settings, get_settings
from models.schemas import (
    EmbeddingResult,
    QdrantPointPayload,
    SemanticChunk,
)
from services.llm import LLMService

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Custom exceptions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class VectorDBError(Exception):
    """Base exception for vector DB operations."""


class CollectionError(VectorDBError):
    """Raised when collection creation or verification fails."""


class UpsertError(VectorDBError):
    """Raised when upserting points into Qdrant fails."""


class SearchError(VectorDBError):
    """Raised when a similarity search fails."""


class EmbeddingError(VectorDBError):
    """Raised when embedding generation fails."""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Vector DB Service
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class VectorDBService:
    """
    Async Qdrant client for embedding, storing, and searching video chunks.

    Designed to be instantiated once at application startup and reused
    across the lifetime of the process.
    """

    def __init__(
        self,
        llm_service: LLMService,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._llm = llm_service
        self._collection_name = self._settings.qdrant_collection_name
        self._dimension = self._settings.gemini_embedding_dimension

        # Initialise the async Qdrant client
        self._client = AsyncQdrantClient(
            url=self._settings.qdrant_url,
            api_key=self._settings.qdrant_api_key,
            timeout=self._settings.qdrant_timeout,
        )

        self._collection_verified = False

        logger.info(
            "VectorDBService initialised: url=%s, collection=%s, dimension=%d",
            self._settings.qdrant_url,
            self._collection_name,
            self._dimension,
        )

    # ─────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────

    async def close(self) -> None:
        """Close the Qdrant client connection."""
        await self._client.close()
        logger.info("Qdrant client closed")

    async def ensure_collection(self) -> None:
        """
        Create the collection if it does not exist, or verify that
        the existing collection has the correct vector configuration.

        Idempotent — safe to call multiple times.

        Raises:
            CollectionError: If the collection exists with incompatible config.
        """
        if self._collection_verified:
            return

        try:
            exists = await self._client.collection_exists(self._collection_name)

            if exists:
                # Verify the existing collection's vector dimensions match
                info = await self._client.get_collection(self._collection_name)
                existing_config = info.config.params.vectors

                if isinstance(existing_config, VectorParams):
                    if existing_config.size != self._dimension:
                        raise CollectionError(
                            f"Collection '{self._collection_name}' exists with "
                            f"dimension {existing_config.size}, but configured "
                            f"dimension is {self._dimension}. Delete the collection "
                            f"or update GEMINI_EMBEDDING_DIMENSION."
                        )
                logger.info(
                    "Collection '%s' verified (points: %d)",
                    self._collection_name,
                    info.points_count or 0,
                )
            else:
                await self._client.create_collection(
                    collection_name=self._collection_name,
                    vectors_config=VectorParams(
                        size=self._dimension,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(
                    "Created collection '%s' (dim=%d, cosine)",
                    self._collection_name,
                    self._dimension,
                )

            self._collection_verified = True

        except UnexpectedResponse as exc:
            raise CollectionError(
                f"Qdrant collection operation failed: {exc}"
            ) from exc

    # ─────────────────────────────────────────────
    # Embedding
    # ─────────────────────────────────────────────

    async def embed_chunk(self, chunk: SemanticChunk) -> EmbeddingResult:
        """
        Generate an embedding for a single chunk.

        The embedding text is constructed by concatenating the spoken
        transcript text with OCR text and visual context to capture
        the full multimodal meaning.

        Returns:
            An ``EmbeddingResult`` with the vector and metadata.

        Raises:
            EmbeddingError: If embedding generation fails.
        """
        embedding_text = self._build_embedding_text(chunk)

        try:
            vector = await self._llm.embed_text(embedding_text)
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to embed chunk {chunk.chunk_index}: {exc}"
            ) from exc

        return EmbeddingResult(
            chunk_id=chunk.chunk_id,
            vector=vector,
            model=self._settings.gemini_embedding_model,
            dimension=len(vector),
        )

    async def embed_chunks_batch(
        self,
        chunks: list[SemanticChunk],
        max_concurrent: int | None = None,
    ) -> list[EmbeddingResult]:
        """
        Embed multiple chunks concurrently with bounded parallelism.

        Args:
            chunks: List of semantic chunks to embed.
            max_concurrent: Override the concurrency limit (defaults to
                ``gemini_max_concurrent``).

        Returns:
            Ordered list of ``EmbeddingResult`` objects.
        """
        concurrency = max_concurrent or self._settings.gemini_max_concurrent
        semaphore = asyncio.Semaphore(concurrency)
        results: list[EmbeddingResult | Exception] = [None] * len(chunks)  # type: ignore[list-item]

        async def _embed(idx: int, chunk: SemanticChunk) -> None:
            async with semaphore:
                try:
                    results[idx] = await self.embed_chunk(chunk)
                except Exception as exc:
                    logger.error("Embedding failed for chunk %d: %s", chunk.chunk_index, exc)
                    results[idx] = exc

        tasks = [_embed(i, c) for i, c in enumerate(chunks)]
        await asyncio.gather(*tasks)

        # Separate successes and failures
        embeddings: list[EmbeddingResult] = []
        failures = 0
        for r in results:
            if isinstance(r, EmbeddingResult):
                embeddings.append(r)
            else:
                failures += 1

        if failures > 0:
            logger.warning(
                "Embedding batch: %d succeeded, %d failed out of %d",
                len(embeddings),
                failures,
                len(chunks),
            )

        return embeddings

    # ─────────────────────────────────────────────
    # Upsert
    # ─────────────────────────────────────────────

    async def upsert_chunks(
        self,
        chunks: list[SemanticChunk],
        embeddings: list[EmbeddingResult],
        video_title: str = "",
        batch_size: int = 100,
    ) -> int:
        """
        Upsert chunks with their embeddings into Qdrant.

        Builds a rich ``QdrantPointPayload`` for each chunk so that
        search results contain all metadata needed for display without
        a secondary lookup.

        Args:
            chunks: Semantic chunks to store.
            embeddings: Corresponding embedding vectors (matched by chunk_id).
            video_title: Video title for the payload metadata.
            batch_size: Number of points per Qdrant upsert call.

        Returns:
            Number of points successfully upserted.

        Raises:
            UpsertError: If the upsert operation fails.
        """
        await self.ensure_collection()

        # Build a lookup from chunk_id to embedding vector
        embedding_map: dict[str, list[float]] = {
            str(e.chunk_id): e.vector for e in embeddings
        }

        # Build PointStruct objects
        points: list[PointStruct] = []
        skipped = 0

        for chunk in chunks:
            chunk_id_str = str(chunk.chunk_id)
            vector = embedding_map.get(chunk_id_str)

            if vector is None:
                logger.warning(
                    "No embedding found for chunk %d (id=%s) — skipping",
                    chunk.chunk_index,
                    chunk_id_str,
                )
                skipped += 1
                continue

            payload = QdrantPointPayload.from_chunk(chunk, video_title=video_title)

            # Use a deterministic UUID derived from chunk_id for the Qdrant point ID
            point_id = str(chunk.chunk_id)

            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload.model_dump(mode="json"),
                )
            )

        if not points:
            logger.warning("No points to upsert (all chunks lacked embeddings)")
            return 0

        # Upsert in batches
        total_upserted = 0

        try:
            for i in range(0, len(points), batch_size):
                batch = points[i : i + batch_size]
                await self._client.upsert(
                    collection_name=self._collection_name,
                    points=batch,
                    wait=True,
                )
                total_upserted += len(batch)
                logger.debug(
                    "Upserted batch %d/%d (%d points)",
                    i // batch_size + 1,
                    (len(points) + batch_size - 1) // batch_size,
                    len(batch),
                )

        except UnexpectedResponse as exc:
            raise UpsertError(
                f"Qdrant upsert failed after {total_upserted} points: {exc}"
            ) from exc

        logger.info(
            "Upserted %d points into '%s' (%d skipped)",
            total_upserted,
            self._collection_name,
            skipped,
        )

        return total_upserted

    async def embed_and_upsert(
        self,
        chunks: list[SemanticChunk],
        video_title: str = "",
    ) -> int:
        """
        Convenience method that embeds chunks and upserts them in one call.

        Returns:
            Number of points upserted.
        """
        if not chunks:
            return 0

        logger.info("Embedding and upserting %d chunks for '%s'", len(chunks), video_title)

        embeddings = await self.embed_chunks_batch(chunks)
        return await self.upsert_chunks(chunks, embeddings, video_title=video_title)

    # ─────────────────────────────────────────────
    # Search
    # ─────────────────────────────────────────────

    async def search(
        self,
        query: str,
        video_id: str | None = None,
        limit: int = 10,
        score_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Perform a similarity search over stored chunks.

        Args:
            query: Natural-language query text.
            video_id: Optional filter to search within a specific video.
            limit: Maximum number of results to return.
            score_threshold: Minimum similarity score (0.0–1.0).

        Returns:
            List of dicts with ``score``, ``payload``, and ``id`` keys,
            ordered by descending similarity.

        Raises:
            SearchError: If the search operation fails.
        """
        await self.ensure_collection()

        try:
            query_vector = await self._llm.embed_text(query)
        except Exception as exc:
            raise SearchError(f"Failed to embed query: {exc}") from exc

        # Build optional filter
        query_filter = None
        if video_id:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="video_id",
                        match=MatchValue(value=video_id),
                    )
                ]
            )

        try:
            results = await self._client.query_points(
                collection_name=self._collection_name,
                query=query_vector,
                query_filter=query_filter,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
            )

            return [
                {
                    "id": str(point.id),
                    "score": point.score,
                    "payload": point.payload,
                }
                for point in results.points
            ]

        except UnexpectedResponse as exc:
            raise SearchError(f"Qdrant search failed: {exc}") from exc

    async def get_chunks_by_video(
        self,
        video_id: str,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """
        Retrieve all stored chunks for a specific video.

        Returns points ordered by ``chunk_index`` (ascending).
        """
        await self.ensure_collection()

        try:
            results, _offset = await self._client.scroll(
                collection_name=self._collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="video_id",
                            match=MatchValue(value=video_id),
                        )
                    ]
                ),
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )

            points = [
                {"id": str(p.id), "payload": p.payload}
                for p in results
            ]

            # Sort by chunk_index
            points.sort(
                key=lambda p: p.get("payload", {}).get("chunk_index", 0)
            )

            return points

        except UnexpectedResponse as exc:
            raise SearchError(
                f"Failed to scroll chunks for video {video_id}: {exc}"
            ) from exc

    # ─────────────────────────────────────────────
    # Deletion
    # ─────────────────────────────────────────────

    async def delete_by_video_id(self, video_id: str) -> int:
        """
        Delete all points associated with a specific video.

        Returns the number of points deleted.
        """
        await self.ensure_collection()

        try:
            # Count existing points before deletion
            existing = await self.get_chunks_by_video(video_id)
            count = len(existing)

            if count == 0:
                return 0

            await self._client.delete(
                collection_name=self._collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="video_id",
                            match=MatchValue(value=video_id),
                        )
                    ]
                ),
                wait=True,
            )

            logger.info(
                "Deleted %d points for video_id='%s' from '%s'",
                count,
                video_id,
                self._collection_name,
            )
            return count

        except UnexpectedResponse as exc:
            raise VectorDBError(
                f"Failed to delete points for video {video_id}: {exc}"
            ) from exc

    # ─────────────────────────────────────────────
    # Statistics
    # ─────────────────────────────────────────────

    async def get_collection_stats(self) -> dict[str, Any]:
        """Return basic statistics about the collection."""
        await self.ensure_collection()

        try:
            info = await self._client.get_collection(self._collection_name)
            return {
                "collection_name": self._collection_name,
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
                "status": str(info.status),
                "dimension": self._dimension,
            }
        except UnexpectedResponse as exc:
            raise VectorDBError(f"Failed to get collection stats: {exc}") from exc

    # ─────────────────────────────────────────────
    # Embedding text construction
    # ─────────────────────────────────────────────

    @staticmethod
    def _build_embedding_text(chunk: SemanticChunk) -> str:
        """
        Construct the text string used for embedding.

        Concatenates transcript text, OCR text, and visual context
        into a single string. The order prioritises spoken content
        (most semantically meaningful) followed by visual context
        (enriches retrieval for slide-heavy content).

        The total length is capped to avoid exceeding embedding model
        input limits (~2048 tokens for text-embedding-004).
        """
        parts: list[str] = []

        # Primary: spoken transcript text
        if chunk.text:
            parts.append(chunk.text)

        # Secondary: OCR text (slide content, code)
        if chunk.ocr_text:
            parts.append(f"[Visible text] {chunk.ocr_text}")

        # Tertiary: visual context descriptions
        if chunk.visual_context:
            parts.append(f"[Visual] {chunk.visual_context}")

        # Quaternary: equations and code snippets (brief)
        if chunk.equations:
            parts.append(f"[Equations] {' | '.join(chunk.equations[:5])}")
        if chunk.code_snippets:
            # Include just the first code snippet to stay within token limits
            parts.append(f"[Code] {chunk.code_snippets[0][:500]}")

        combined = "\n".join(parts)

        # Cap at approximately 8000 characters (~2000 tokens) for embedding
        max_chars = 8000
        if len(combined) > max_chars:
            combined = combined[:max_chars]

        return combined
