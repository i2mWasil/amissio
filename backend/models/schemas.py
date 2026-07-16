"""
Pydantic models and schemas for the Amissio Multimodal RAG API.

Covers every data shape flowing through the pipeline:
  - API request / response envelopes
  - Video metadata extracted from YouTube
  - Transcript segments (captions & Whisper)
  - Extracted frame & vision / OCR data
  - Unified multimodal timeline entries
  - Semantic chunks with embeddings
  - Qdrant vector DB payloads
  - Generated notes & final output
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, HttpUrl, field_validator


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Enumerations
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class PipelineStatus(str, Enum):
    """Status of the overall generation pipeline."""

    PENDING = "pending"
    DOWNLOADING = "downloading"
    EXTRACTING_AUDIO = "extracting_audio"
    TRANSCRIBING = "transcribing"
    EXTRACTING_FRAMES = "extracting_frames"
    ANALYZING_VISUALS = "analyzing_visuals"
    SYNCHRONIZING = "synchronizing"
    CHUNKING = "chunking"
    GENERATING_NOTES = "generating_notes"
    EMBEDDING = "embedding"
    STORING = "storing"
    COMPLETED = "completed"
    FAILED = "failed"


class TranscriptSource(str, Enum):
    """How the transcript was obtained."""

    YOUTUBE_CAPTIONS = "youtube_captions"
    WHISPER = "whisper"


class ChunkType(str, Enum):
    """Semantic reason a chunk boundary was placed."""

    CHAPTER = "chapter"
    TOPIC_CHANGE = "topic_change"
    SLIDE_CHANGE = "slide_change"
    SEMANTIC_SPLIT = "semantic_split"
    TIME_BASED = "time_based"


class ContentType(str, Enum):
    """Type of content detected inside a chunk."""

    TEXT = "text"
    CODE = "code"
    EQUATION = "equation"
    DIAGRAM = "diagram"
    CHART = "chart"
    TABLE = "table"
    IMAGE = "image"


class QuizQuestionType(str, Enum):
    """Supported quiz question formats."""

    MULTIPLE_CHOICE = "multiple_choice"
    TRUE_FALSE = "true_false"
    SHORT_ANSWER = "short_answer"
    FILL_IN_THE_BLANK = "fill_in_the_blank"


class DifficultyLevel(str, Enum):
    """Difficulty tier for quiz questions."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# API Request / Response
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class GenerateNotesRequest(BaseModel):
    """POST /generate-notes request body."""

    url: HttpUrl = Field(
        ...,
        description="YouTube video URL",
        examples=["https://www.youtube.com/watch?v=dQw4w9WgXcQ"],
    )

    @field_validator("url")
    @classmethod
    def validate_youtube_url(cls, v: HttpUrl) -> HttpUrl:
        url_str = str(v)
        valid_hosts = (
            "youtube.com", "www.youtube.com",
            "m.youtube.com", "youtu.be",
            "www.youtube-nocookie.com",
        )
        from urllib.parse import urlparse

        parsed = urlparse(url_str)
        hostname = parsed.hostname or ""
        if not any(hostname == h or hostname.endswith(f".{h}") for h in valid_hosts):
            raise ValueError(
                f"URL must be a valid YouTube link. Got host: {hostname}"
            )
        return v


class GenerateNotesResponse(BaseModel):
    """POST /generate-notes response body."""

    video_id: str = Field(..., description="YouTube video ID")
    title: str = Field(..., description="Video title")
    duration: int = Field(..., ge=0, description="Video duration in seconds")
    channel: str = Field(default="", description="Channel name")
    thumbnail_url: str = Field(default="", description="Thumbnail URL")
    summary: str = Field(..., description="Executive summary of the video")
    notes: str = Field(
        ...,
        description="Full generated notes in Markdown format",
    )
    timestamps: list[TimestampEntry] = Field(
        default_factory=list,
        description="Key moments with timestamps",
    )
    references: list[Reference] = Field(
        default_factory=list,
        description="External references mentioned in the video",
    )
    glossary: list[GlossaryEntry] = Field(
        default_factory=list,
        description="Key terms and definitions",
    )
    quiz: list[QuizQuestion] = Field(
        default_factory=list,
        description="Auto-generated quiz questions (10-20)",
    )
    metadata: NoteGenerationMetadata = Field(
        ...,
        description="Pipeline execution metadata",
    )


class ErrorResponse(BaseModel):
    """Standard error envelope."""

    error: str = Field(..., description="Human-readable error message")
    detail: str | None = Field(default=None, description="Technical detail / traceback hint")
    status_code: int = Field(..., ge=400, le=599, description="HTTP status code")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Video Metadata
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class VideoChapter(BaseModel):
    """A single chapter / section from the video description."""

    title: str = Field(..., description="Chapter title")
    start_time: float = Field(..., ge=0.0, description="Start time in seconds")
    end_time: float | None = Field(default=None, description="End time in seconds")


class VideoMetadata(BaseModel):
    """Metadata extracted from YouTube before downloading."""

    video_id: str = Field(..., description="YouTube video ID")
    title: str = Field(..., description="Video title")
    description: str = Field(default="", description="Video description")
    channel: str = Field(default="", description="Channel name")
    channel_id: str = Field(default="", description="Channel ID")
    duration: int = Field(..., ge=0, description="Duration in seconds")
    thumbnail_url: str = Field(default="", description="Best thumbnail URL")
    upload_date: str = Field(default="", description="Upload date string (YYYYMMDD)")
    view_count: int = Field(default=0, ge=0, description="View count")
    chapters: list[VideoChapter] = Field(
        default_factory=list,
        description="Video chapters if available",
    )
    tags: list[str] = Field(default_factory=list, description="Video tags")
    language: str | None = Field(
        default=None,
        description="Detected or declared language (ISO 639-1)",
    )
    has_captions: bool = Field(
        default=False,
        description="Whether YouTube captions are available",
    )
    caption_languages: list[str] = Field(
        default_factory=list,
        description="Available caption language codes",
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Transcript Segments
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class WordTimestamp(BaseModel):
    """A single word with its precise timing."""

    word: str = Field(..., description="The word text")
    start: float = Field(..., ge=0.0, description="Start time in seconds")
    end: float = Field(..., ge=0.0, description="End time in seconds")
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Recognition confidence (1.0 for captions)",
    )


class TranscriptSegment(BaseModel):
    """A contiguous segment of the transcript (a sentence or subtitle block)."""

    text: str = Field(..., description="Segment text content")
    start: float = Field(..., ge=0.0, description="Start time in seconds")
    end: float = Field(..., ge=0.0, description="End time in seconds")
    words: list[WordTimestamp] = Field(
        default_factory=list,
        description="Word-level timestamps (from Whisper or aligned captions)",
    )
    speaker: str | None = Field(
        default=None,
        description="Speaker label if diarization is available",
    )


class TranscriptData(BaseModel):
    """Complete transcript for the video."""

    source: TranscriptSource = Field(..., description="How the transcript was produced")
    language: str = Field(default="en", description="Detected language (ISO 639-1)")
    segments: list[TranscriptSegment] = Field(
        default_factory=list,
        description="Ordered transcript segments",
    )
    full_text: str = Field(default="", description="Concatenated plain text of the transcript")
    duration: float = Field(default=0.0, ge=0.0, description="Total transcript duration")

    @property
    def word_count(self) -> int:
        return len(self.full_text.split())


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Frame & Vision / OCR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ExtractedFrame(BaseModel):
    """A single frame extracted from the video."""

    frame_index: int = Field(..., ge=0, description="Sequential frame index")
    timestamp: float = Field(..., ge=0.0, description="Frame timestamp in seconds")
    file_path: str = Field(..., description="Absolute path to the saved frame image")
    perceptual_hash: str = Field(
        default="",
        description="Perceptual hash (hex) for deduplication",
    )
    is_duplicate: bool = Field(
        default=False,
        description="True if this frame was flagged as a duplicate",
    )
    width: int = Field(default=0, ge=0, description="Frame width in pixels")
    height: int = Field(default=0, ge=0, description="Frame height in pixels")


class VisionAnalysis(BaseModel):
    """Gemini Vision output for a single frame."""

    frame_index: int = Field(..., ge=0, description="Index of the analyzed frame")
    timestamp: float = Field(..., ge=0.0, description="Frame timestamp in seconds")
    description: str = Field(
        default="",
        description="Natural-language description of the frame",
    )
    slide_title: str | None = Field(
        default=None,
        description="Detected slide title (if presentation)",
    )
    text_content: str = Field(
        default="",
        description="Extracted visible text / OCR content",
    )
    code_snippets: list[str] = Field(
        default_factory=list,
        description="Code blocks visible in the frame",
    )
    equations: list[str] = Field(
        default_factory=list,
        description="Mathematical equations (LaTeX or plain)",
    )
    diagram_description: str | None = Field(
        default=None,
        description="Description of diagrams or charts",
    )
    content_types: list[ContentType] = Field(
        default_factory=list,
        description="Types of content detected in this frame",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Overall analysis confidence",
    )


class VisionBatchResult(BaseModel):
    """Result of a batch vision analysis call."""

    analyses: list[VisionAnalysis] = Field(default_factory=list)
    tokens_used: int = Field(default=0, ge=0, description="Total tokens consumed")
    processing_time_ms: int = Field(default=0, ge=0, description="Wall-clock time in ms")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Multimodal Timeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TimelineEntry(BaseModel):
    """
    A single event in the unified multimodal timeline.

    Merges transcript text with visual context at a given point in time.
    """

    start_time: float = Field(..., ge=0.0, description="Start time in seconds")
    end_time: float = Field(..., ge=0.0, description="End time in seconds")
    transcript_text: str = Field(default="", description="Spoken text in this window")
    visual_context: str = Field(
        default="",
        description="Aggregated visual descriptions for this window",
    )
    slide_title: str | None = Field(
        default=None,
        description="Active slide title (if any)",
    )
    ocr_text: str = Field(
        default="",
        description="Concatenated OCR text from frames in this window",
    )
    code_snippets: list[str] = Field(
        default_factory=list,
        description="Code snippets visible during this window",
    )
    equations: list[str] = Field(
        default_factory=list,
        description="Equations visible during this window",
    )
    content_types: list[ContentType] = Field(
        default_factory=list,
        description="Content types present in this window",
    )
    chapter: VideoChapter | None = Field(
        default=None,
        description="Chapter this entry belongs to (if chapters exist)",
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Semantic Chunks
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class SemanticChunk(BaseModel):
    """
    A semantically coherent unit of content from the video.

    This is the fundamental unit for embedding, storage, and note generation.
    """

    chunk_id: UUID = Field(
        default_factory=uuid4,
        description="Unique chunk identifier",
    )
    video_id: str = Field(..., description="Parent YouTube video ID")
    chunk_index: int = Field(..., ge=0, description="Sequential chunk index")
    chunk_type: ChunkType = Field(
        default=ChunkType.SEMANTIC_SPLIT,
        description="Reason for the chunk boundary",
    )

    # Content
    text: str = Field(..., description="Primary text content of the chunk")
    summary: str = Field(default="", description="Concise summary of the chunk")
    visual_context: str = Field(
        default="",
        description="Aggregated visual context descriptions",
    )
    ocr_text: str = Field(default="", description="OCR text associated with the chunk")
    code_snippets: list[str] = Field(
        default_factory=list,
        description="Code snippets in this chunk",
    )
    equations: list[str] = Field(
        default_factory=list,
        description="Mathematical equations in this chunk",
    )
    content_types: list[ContentType] = Field(
        default_factory=list,
        description="Content types present in the chunk",
    )

    # Timing
    start_time: float = Field(..., ge=0.0, description="Chunk start time in seconds")
    end_time: float = Field(..., ge=0.0, description="Chunk end time in seconds")
    chapter_title: str | None = Field(
        default=None,
        description="Chapter this chunk belongs to",
    )

    # Token counts
    token_count: int = Field(default=0, ge=0, description="Approximate token count")

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def content_hash(self) -> str:
        """Deterministic hash of the chunk text for cache keying."""
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Embeddings & Vector DB Payloads
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class EmbeddingResult(BaseModel):
    """Embedding vector for a single chunk."""

    chunk_id: UUID = Field(..., description="Chunk this embedding belongs to")
    vector: list[float] = Field(..., description="Dense embedding vector")
    model: str = Field(default="", description="Model used for embedding")
    dimension: int = Field(default=0, ge=0, description="Vector dimensionality")


class QdrantPointPayload(BaseModel):
    """
    Payload stored alongside the vector in Qdrant.

    Contains all metadata needed for retrieval and display without
    a secondary lookup.
    """

    chunk_id: str = Field(..., description="UUID string of the chunk")
    video_id: str = Field(..., description="YouTube video ID")
    video_title: str = Field(default="", description="Video title for display")
    chunk_index: int = Field(..., ge=0, description="Sequential chunk index")
    chunk_type: str = Field(default="semantic_split", description="Chunk boundary type")

    # Content
    text: str = Field(..., description="Primary chunk text")
    summary: str = Field(default="", description="Chunk summary")
    visual_context: str = Field(default="", description="Visual descriptions")
    ocr_text: str = Field(default="", description="OCR text")
    code_snippets: list[str] = Field(default_factory=list)
    equations: list[str] = Field(default_factory=list)
    content_types: list[str] = Field(default_factory=list)

    # Timing
    start_time: float = Field(..., ge=0.0)
    end_time: float = Field(..., ge=0.0)
    chapter_title: str | None = Field(default=None)

    # Metadata
    token_count: int = Field(default=0, ge=0)
    created_at: str = Field(
        default="",
        description="ISO 8601 timestamp of when this point was created",
    )

    @classmethod
    def from_chunk(
        cls,
        chunk: SemanticChunk,
        video_title: str = "",
    ) -> QdrantPointPayload:
        """Convert a SemanticChunk into a Qdrant-ready payload."""
        return cls(
            chunk_id=str(chunk.chunk_id),
            video_id=chunk.video_id,
            video_title=video_title,
            chunk_index=chunk.chunk_index,
            chunk_type=chunk.chunk_type.value,
            text=chunk.text,
            summary=chunk.summary,
            visual_context=chunk.visual_context,
            ocr_text=chunk.ocr_text,
            code_snippets=chunk.code_snippets,
            equations=chunk.equations,
            content_types=[ct.value for ct in chunk.content_types],
            start_time=chunk.start_time,
            end_time=chunk.end_time,
            chapter_title=chunk.chapter_title,
            token_count=chunk.token_count,
            created_at=datetime.utcnow().isoformat(),
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Note Generation (per-chunk and final output)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ChunkNotes(BaseModel):
    """Generated notes for a single semantic chunk."""

    chunk_id: UUID = Field(..., description="Source chunk ID")
    heading: str = Field(default="", description="Section heading")
    summary: str = Field(default="", description="Concise summary")
    detailed_explanation: str = Field(
        default="",
        description="Detailed explanation in Markdown",
    )
    key_concepts: list[str] = Field(
        default_factory=list,
        description="Key concepts / takeaways",
    )
    code_blocks: list[CodeBlock] = Field(
        default_factory=list,
        description="Explained code blocks",
    )
    formulas: list[Formula] = Field(
        default_factory=list,
        description="Mathematical formulas with explanations",
    )
    diagrams: list[str] = Field(
        default_factory=list,
        description="Diagram / chart descriptions",
    )
    start_time: float = Field(..., ge=0.0)
    end_time: float = Field(..., ge=0.0)
    timestamp_label: str = Field(
        default="",
        description="Human-readable timestamp (e.g. '05:23 - 12:41')",
    )


class CodeBlock(BaseModel):
    """A code snippet with language annotation and explanation."""

    language: str = Field(default="", description="Programming language")
    code: str = Field(..., description="The code content")
    explanation: str = Field(default="", description="Explanation of the code")
    line_annotations: dict[int, str] = Field(
        default_factory=dict,
        description="Line-number → annotation mapping",
    )


class Formula(BaseModel):
    """A mathematical formula with context."""

    expression: str = Field(..., description="LaTeX or plain-text formula")
    name: str = Field(default="", description="Formula name (e.g., 'Bayes Theorem')")
    explanation: str = Field(default="", description="Plain-English explanation")
    variables: dict[str, str] = Field(
        default_factory=dict,
        description="Variable → meaning mapping",
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Final Output Sub-models
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TimestampEntry(BaseModel):
    """A key moment in the video."""

    time: str = Field(
        ...,
        description="Timestamp string (HH:MM:SS or MM:SS)",
        examples=["05:23", "01:12:05"],
    )
    time_seconds: float = Field(..., ge=0.0, description="Time in seconds")
    label: str = Field(..., description="Description of the moment")
    chapter: str | None = Field(default=None, description="Associated chapter title")


class Reference(BaseModel):
    """An external reference mentioned in the video."""

    title: str = Field(..., description="Reference title")
    url: str | None = Field(default=None, description="URL if mentioned or inferred")
    context: str = Field(
        default="",
        description="Context in which the reference was mentioned",
    )
    timestamp: str = Field(
        default="",
        description="When in the video it was mentioned (MM:SS)",
    )


class GlossaryEntry(BaseModel):
    """A key term and its definition."""

    term: str = Field(..., description="The term")
    definition: str = Field(..., description="Concise definition")
    first_mentioned: str = Field(
        default="",
        description="Timestamp when first mentioned (MM:SS)",
    )
    related_terms: list[str] = Field(
        default_factory=list,
        description="Related glossary terms",
    )


class QuizQuestion(BaseModel):
    """An auto-generated quiz question."""

    question_id: int = Field(..., ge=1, description="Question number")
    question_type: QuizQuestionType = Field(
        ...,
        description="Type of question",
    )
    difficulty: DifficultyLevel = Field(
        default=DifficultyLevel.MEDIUM,
        description="Difficulty level",
    )
    question: str = Field(..., description="The question text")
    options: list[str] | None = Field(
        default=None,
        description="Answer options (for multiple choice)",
    )
    correct_answer: str = Field(..., description="The correct answer")
    explanation: str = Field(
        default="",
        description="Explanation of the correct answer",
    )
    related_timestamp: str = Field(
        default="",
        description="Video timestamp for the relevant section (MM:SS)",
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Pipeline Metadata
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class NoteGenerationMetadata(BaseModel):
    """Metadata about the pipeline run for observability."""

    pipeline_status: PipelineStatus = Field(
        default=PipelineStatus.COMPLETED,
        description="Final pipeline status",
    )
    total_processing_time_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Total wall-clock processing time",
    )
    transcript_source: TranscriptSource = Field(
        ...,
        description="How the transcript was obtained",
    )
    total_chunks: int = Field(default=0, ge=0, description="Number of semantic chunks")
    total_frames_extracted: int = Field(
        default=0, ge=0, description="Raw frames extracted"
    )
    unique_frames_analyzed: int = Field(
        default=0, ge=0, description="Deduplicated frames analyzed by vision"
    )
    total_tokens_used: int = Field(
        default=0, ge=0, description="Estimated total Gemini tokens used"
    )
    gemini_model_used: str = Field(default="", description="Gemini model for text gen")
    gemini_vision_model_used: str = Field(
        default="", description="Gemini model for vision"
    )
    embedding_model_used: str = Field(default="", description="Embedding model name")
    segments_processed: int = Field(
        default=0,
        ge=0,
        description="Number of map-reduce segments",
    )
    cache_hits: int = Field(
        default=0, ge=0, description="Number of cache hits during processing"
    )
    errors: list[PipelineError] = Field(
        default_factory=list,
        description="Non-fatal errors encountered during processing",
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the pipeline run started",
    )
    completed_at: datetime | None = Field(
        default=None,
        description="When the pipeline run finished",
    )


class PipelineError(BaseModel):
    """A non-fatal error that occurred during pipeline execution."""

    stage: PipelineStatus = Field(..., description="Pipeline stage where error occurred")
    message: str = Field(..., description="Error description")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    recoverable: bool = Field(
        default=True,
        description="Whether the pipeline continued after this error",
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Segment-level schemas (Map-Reduce)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class VideoSegment(BaseModel):
    """A time-bounded segment of the video for map-reduce processing."""

    segment_index: int = Field(..., ge=0, description="Sequential segment index")
    start_time: float = Field(..., ge=0.0, description="Segment start in seconds")
    end_time: float = Field(..., ge=0.0, description="Segment end in seconds")
    transcript_segments: list[TranscriptSegment] = Field(
        default_factory=list,
        description="Transcript segments within this time range",
    )
    vision_analyses: list[VisionAnalysis] = Field(
        default_factory=list,
        description="Vision analyses for frames in this time range",
    )
    chapters: list[VideoChapter] = Field(
        default_factory=list,
        description="Chapters overlapping this segment",
    )

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def full_text(self) -> str:
        return " ".join(seg.text for seg in self.transcript_segments)


class SegmentNotes(BaseModel):
    """Notes generated for a single map-reduce segment."""

    segment_index: int = Field(..., ge=0, description="Segment index")
    start_time: float = Field(..., ge=0.0)
    end_time: float = Field(..., ge=0.0)
    summary: str = Field(default="", description="Segment summary")
    detailed_notes: str = Field(default="", description="Detailed notes Markdown")
    key_concepts: list[str] = Field(default_factory=list)
    chunk_notes: list[ChunkNotes] = Field(
        default_factory=list,
        description="Notes per chunk within this segment",
    )
    tokens_used: int = Field(default=0, ge=0)


class MergedNotes(BaseModel):
    """Result of the reduce / merge phase across segments."""

    video_id: str = Field(..., description="YouTube video ID")
    title: str = Field(default="", description="Video title")
    overview: str = Field(default="", description="Video overview section")
    learning_objectives: list[str] = Field(
        default_factory=list,
        description="Extracted learning objectives",
    )
    detailed_notes_markdown: str = Field(
        default="",
        description="Full merged detailed notes in Markdown",
    )
    glossary: list[GlossaryEntry] = Field(default_factory=list)
    algorithms: list[str] = Field(
        default_factory=list,
        description="Key algorithms discussed",
    )
    common_mistakes: list[str] = Field(
        default_factory=list,
        description="Common mistakes or pitfalls mentioned",
    )
    quiz_questions: list[QuizQuestion] = Field(default_factory=list)
    timestamps: list[TimestampEntry] = Field(default_factory=list)
    references: list[Reference] = Field(default_factory=list)
    total_tokens_used: int = Field(default=0, ge=0)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Cache schemas
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class CacheEntry(BaseModel):
    """Generic cache entry wrapper for disk-based caching."""

    key: str = Field(..., description="Cache key (usually a content hash)")
    data: Any = Field(..., description="Cached data (serializable)")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    ttl_seconds: int = Field(
        default=86400,
        ge=0,
        description="Time-to-live in seconds (0 = forever)",
    )
    hit_count: int = Field(default=0, ge=0, description="Number of cache hits")

    @property
    def is_expired(self) -> bool:
        if self.ttl_seconds == 0:
            return False
        elapsed = (datetime.utcnow() - self.created_at).total_seconds()
        return elapsed > self.ttl_seconds


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Rebuild forward references (required for models
# referencing other models defined later in this file)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GenerateNotesResponse.model_rebuild()
ChunkNotes.model_rebuild()
NoteGenerationMetadata.model_rebuild()
