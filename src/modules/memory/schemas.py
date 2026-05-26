from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


def _now() -> datetime:
    """Helper to get current time in UTC."""
    return datetime.now(UTC)


class MemoryDomain(StrEnum):
    """Domains of operational memory."""
    OPERATIONAL = "operational"
    ARCHITECTURE = "architecture"
    RELATIONSHIP = "relationship"
    CODEBASE = "codebase"
    RETRIEVAL = "retrieval"


class MemoryImportance(StrEnum):
    """Qualitative significance classification bands."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MemorySource(StrEnum):
    """Origins of memory entries."""
    USER = "user"
    SYSTEM = "system"
    AI = "ai"
    OBSIDIAN = "obsidian"
    INGESTION = "ingestion"


class MemoryType(StrEnum):
    """Structural classifications of memory entries."""
    SUMMARY = "summary"
    DECISION = "decision"
    FACT = "fact"
    METADATA = "metadata"
    DOCUMENT = "document"


class MemoryCreate(BaseModel):
    """Schema for creating a new memory entry."""
    content: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Concise memory text or summary. Restricted to 10,000 characters to prevent giant raw blobs."
    )
    domain: MemoryDomain = Field(..., description="Bounded domain categorization")
    memory_type: MemoryType = Field(..., description="Structural memory type")
    source: MemorySource = Field(..., description="Origin of the memory")
    importance_level: MemoryImportance = Field(
        default=MemoryImportance.MEDIUM,
        description="Qualitative significance classification"
    )
    importance_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Quantitative score from 0.0 (least important) to 1.0 (most important)"
    )
    tags: list[str] = Field(default_factory=list, description="Categorization tags")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Lightweight arbitrary metadata, e.g. for Obsidian sync, coordinates, or specific entity references"
    )


class MemoryUpdate(BaseModel):
    """Schema for updating an existing memory entry."""
    content: str | None = Field(
        default=None,
        min_length=1,
        max_length=10000,
        description="Updated memory content"
    )
    domain: MemoryDomain | None = Field(default=None, description="Updated domain categorization")
    memory_type: MemoryType | None = Field(default=None, description="Updated memory type")
    source: MemorySource | None = Field(default=None, description="Updated source of the memory")
    importance_level: MemoryImportance | None = Field(
        default=None,
        description="Updated qualitative significance level"
    )
    importance_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Updated quantitative score from 0.0 to 1.0"
    )
    tags: list[str] | None = Field(default=None, description="Updated categorization tags")
    metadata: dict[str, Any] | None = Field(default=None, description="Updated metadata dictionary")


class MemoryEntry(BaseModel):
    """Core memory entry schema mapping SQLite database rows."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID = Field(default_factory=uuid4, description="Unique memory identifier")
    content: str = Field(
        ...,
        max_length=10000,
        description="The content of the memory entry"
    )
    domain: MemoryDomain = Field(..., description="Bounded domain categorization")
    memory_type: MemoryType = Field(..., description="Structural memory type")
    source: MemorySource = Field(..., description="Origin of the memory")
    importance_level: MemoryImportance = Field(..., description="Qualitative significance level")
    importance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Quantitative importance score from 0.0 to 1.0"
    )
    tags: list[str] = Field(default_factory=list, description="Categorization tags")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Lightweight arbitrary metadata"
    )
    created_at: datetime = Field(default_factory=_now, description="Record creation timestamp")
    updated_at: datetime = Field(default_factory=_now, description="Record last update timestamp")


class MemorySummary(BaseModel):
    """Schema for representing compressed operational summaries."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID = Field(default_factory=uuid4, description="Unique summary identifier")
    memory_id: UUID | None = Field(
        default=None,
        description="Optional associated source memory ID"
    )
    summary_text: str = Field(
        ...,
        max_length=2000,
        description="Highly compressed operational summary text. Restricted to 2,000 characters."
    )
    original_length: int = Field(
        ...,
        ge=0,
        description="Length of the original content in characters"
    )
    compressed_length: int = Field(
        ...,
        ge=0,
        description="Length of the summarized content in characters"
    )
    key_takeaways: list[str] = Field(
        default_factory=list,
        description="Key takeaways or actions extracted from the memory"
    )
    created_at: datetime = Field(default_factory=_now, description="Summary creation timestamp")


class MemoryEmbedding(BaseModel):
    """Metadata schema representing the vector embedding of a memory."""
    model_config = ConfigDict(from_attributes=True)

    memory_id: UUID = Field(..., description="Associated memory ID")
    embedding: list[float] = Field(..., description="Dense float vector embedding list")
    model_name: str = Field(..., description="Name of the embedding model used")
    dimension: int = Field(..., ge=1, description="Embedding vector dimensionality")
    created_at: datetime = Field(default_factory=_now, description="Embedding generation timestamp")


class MemoryRetrievalResult(BaseModel):
    """Represents an individual retrieved memory match."""
    model_config = ConfigDict(from_attributes=True)

    entry: MemoryEntry = Field(..., description="The retrieved memory entry")
    similarity_score: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Cosine similarity score between query and entry embeddings"
    )
    rerank_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional significance or recency adjusted reranking score"
    )


class RetrievalContext(BaseModel):
    """Context schema optimized for token-efficient prompt injection."""
    model_config = ConfigDict(from_attributes=True)

    query: str = Field(..., description="The original retrieval search query")
    results: list[MemoryRetrievalResult] = Field(..., description="Matched memory results")
    assembled_context: str = Field(
        ...,
        description="Formatted, token-efficient context text block ready for prompt template"
    )
    total_tokens: int = Field(
        ...,
        ge=0,
        description="Estimated token count of the assembled context block"
    )
    domain_filters: list[MemoryDomain] = Field(
        default_factory=list,
        description="Domains used to filter the retrieval"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context metadata, e.g., processing latency or similarity cutoffs"
    )
