from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import JSON, DateTime, Float, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from src.core.database.models import Base


def _uuid_str() -> str:
    return str(uuid4())


def _now() -> datetime:
    return datetime.now(UTC)


class MemoryEntryModel(Base):
    """
    Persists compressed operational memory entries.
    Content is bounded to 10,000 characters — transcripts and raw blobs are not stored.
    """

    __tablename__ = "memory_entries"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    content: Mapped[str] = mapped_column(Text, nullable=False)           # max 10k enforced in schema
    domain: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    memory_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    source: Mapped[str] = mapped_column(String(50), nullable=False)
    importance_level: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    importance_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    tags: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )

    __table_args__ = (
        Index("ix_memory_entries_importance_score", "importance_score"),
        Index("ix_memory_entries_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<MemoryEntryModel id={self.id[:8]} domain={self.domain} type={self.memory_type}>"


class MemorySummaryModel(Base):
    """
    Persists compressed operational summaries linked to source memory entries.
    Summary text is bounded to 2,000 characters.
    """

    __tablename__ = "memory_summaries"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    memory_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    summary_text: Mapped[str] = mapped_column(String(2000), nullable=False)
    original_length: Mapped[int] = mapped_column(Integer, nullable=False)
    compressed_length: Mapped[int] = mapped_column(Integer, nullable=False)
    key_takeaways: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    def __repr__(self) -> str:
        return f"<MemorySummaryModel id={self.id[:8]} memory_id={self.memory_id}>"


class MemoryEmbeddingModel(Base):
    """
    Persists embedding vector metadata for memory entries.
    Stores dense float list (e.g. 384 dimensions) as JSON alongside model provenance.
    """

    __tablename__ = "memory_embeddings"

    memory_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    embedding: Mapped[list[float]] = mapped_column(JSON, nullable=False)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    def __repr__(self) -> str:
        return f"<MemoryEmbeddingModel memory_id={self.memory_id[:8]} dim={self.dimension}>"
