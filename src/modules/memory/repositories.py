from collections.abc import Sequence
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy.orm import Session

from src.core.database.repositories.base import BaseRepository
from src.modules.memory.models import (
    MemoryEmbeddingModel,
    MemoryEntryModel,
    MemorySummaryModel,
)
from src.modules.memory.schemas import (
    MemoryCreate,
    MemoryDomain,
    MemoryEmbedding,
    MemoryEntry,
    MemoryImportance,
    MemorySource,
    MemorySummary,
    MemoryType,
    MemoryUpdate,
)


def _now() -> datetime:
    return datetime.now(UTC)


# ---------------------------------------------------------------------------
# MemoryRepository
# ---------------------------------------------------------------------------


class MemoryRepository(BaseRepository[MemoryEntryModel]):
    """
    Persists and queries operational memory entries.

    Supported operations:
    - create, update, get by UUID
    - list with optional domain / importance_level / importance_score / tag filters
    - retrieval candidate query (ordered by importance_score desc, created_at desc)
    """

    def __init__(self, session: Session) -> None:
        super().__init__(MemoryEntryModel, session)

    # ------------------------------------------------------------------
    # Mapping helpers
    # ------------------------------------------------------------------

    def _to_schema(self, model: MemoryEntryModel) -> MemoryEntry:
        return MemoryEntry(
            id=UUID(model.id),
            content=model.content,
            domain=MemoryDomain(model.domain),
            memory_type=MemoryType(model.memory_type),
            source=MemorySource(model.source),
            importance_level=MemoryImportance(model.importance_level),
            importance_score=model.importance_score,
            tags=model.tags or [],
            metadata=model.metadata_json or {},
            created_at=model.created_at,
            updated_at=model.updated_at,
        )

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def create_entry(self, create_schema: MemoryCreate) -> MemoryEntry:
        """Persist a new memory entry."""
        model = MemoryEntryModel(
            content=create_schema.content,
            domain=create_schema.domain.value,
            memory_type=create_schema.memory_type.value,
            source=create_schema.source.value,
            importance_level=create_schema.importance_level.value,
            importance_score=create_schema.importance_score,
            tags=create_schema.tags or None,
            metadata_json=create_schema.metadata or None,
        )
        self.session.add(model)
        self.session.flush()
        return self._to_schema(model)

    def update_entry(self, entry_id: UUID, update_schema: MemoryUpdate) -> MemoryEntry | None:
        """Update an existing memory entry by UUID. Returns None if not found."""
        model = self.session.query(MemoryEntryModel).filter_by(id=str(entry_id)).first()
        if not model:
            return None

        update_data = update_schema.model_dump(exclude_unset=True)
        if "domain" in update_data:
            model.domain = update_data["domain"]
        if "memory_type" in update_data:
            model.memory_type = update_data["memory_type"]
        if "source" in update_data:
            model.source = update_data["source"]
        if "importance_level" in update_data:
            model.importance_level = update_data["importance_level"]
        if "importance_score" in update_data:
            model.importance_score = update_data["importance_score"]
        if "content" in update_data:
            model.content = update_data["content"]
        if "tags" in update_data:
            model.tags = update_data["tags"]
        if "metadata" in update_data:
            model.metadata_json = update_data["metadata"]

        model.updated_at = _now()
        self.session.flush()
        return self._to_schema(model)

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_by_uuid(self, entry_id: UUID) -> MemoryEntry | None:
        """Fetch a memory entry by UUID. Returns None if not found."""
        model = self.session.query(MemoryEntryModel).filter_by(id=str(entry_id)).first()
        return self._to_schema(model) if model else None

    def list_entries(
        self,
        *,
        domain: MemoryDomain | None = None,
        importance_level: MemoryImportance | None = None,
        min_importance_score: float | None = None,
        tag: str | None = None,
        limit: int = 100,
    ) -> Sequence[MemoryEntry]:
        """
        Fetch memory entries with optional filters.

        Filters:
        - domain: restrict to a specific MemoryDomain
        - importance_level: restrict to a qualitative band
        - min_importance_score: minimum score threshold (0.0-1.0)
        - tag: single tag substring match (checked via JSON contains)
        - limit: max results returned (default 100)
        """
        query = self.session.query(MemoryEntryModel)
        if domain:
            query = query.filter(MemoryEntryModel.domain == domain.value)
        if importance_level:
            query = query.filter(MemoryEntryModel.importance_level == importance_level.value)
        if min_importance_score is not None:
            query = query.filter(MemoryEntryModel.importance_score >= min_importance_score)
        models = query.limit(limit).all()

        # Tag post-filter: SQLite JSON arrays don't support efficient LIKE queries —
        # iterate in Python for correctness without adding complexity.
        if tag:
            tag_lower = tag.lower()
            models = [m for m in models if m.tags and any(tag_lower in t.lower() for t in m.tags)]

        return [self._to_schema(m) for m in models]

    def get_retrieval_candidates(
        self,
        *,
        domain: MemoryDomain | None = None,
        min_importance_score: float = 0.0,
        limit: int = 50,
    ) -> Sequence[MemoryEntry]:
        """
        Return top-ranked memory candidates for semantic retrieval.
        Ordered by importance_score descending, then created_at descending.
        Used as the candidate pool for embedding similarity ranking.
        """
        query = self.session.query(MemoryEntryModel)
        if domain:
            query = query.filter(MemoryEntryModel.domain == domain.value)
        if min_importance_score > 0.0:
            query = query.filter(MemoryEntryModel.importance_score >= min_importance_score)
        models = (
            query
            .order_by(MemoryEntryModel.importance_score.desc(), MemoryEntryModel.created_at.desc())
            .limit(limit)
            .all()
        )
        return [self._to_schema(m) for m in models]

    def delete_entry(self, entry_id: UUID) -> bool:
        """Remove a memory entry by UUID. Returns True if deleted, False if not found."""
        model = self.session.query(MemoryEntryModel).filter_by(id=str(entry_id)).first()
        if not model:
            return False
        self.session.delete(model)
        self.session.flush()
        return True


# ---------------------------------------------------------------------------
# MemorySummaryRepository
# ---------------------------------------------------------------------------


class MemorySummaryRepository(BaseRepository[MemorySummaryModel]):
    """
    Persists and retrieves compressed operational memory summaries.

    Supported operations:
    - save (create or overwrite by memory_id)
    - get by summary UUID
    - get by source memory UUID
    - list recent summaries
    """

    def __init__(self, session: Session) -> None:
        super().__init__(MemorySummaryModel, session)

    # ------------------------------------------------------------------
    # Mapping helpers
    # ------------------------------------------------------------------

    def _to_schema(self, model: MemorySummaryModel) -> MemorySummary:
        return MemorySummary(
            id=UUID(model.id),
            memory_id=UUID(model.memory_id) if model.memory_id else None,
            summary_text=model.summary_text,
            original_length=model.original_length,
            compressed_length=model.compressed_length,
            key_takeaways=model.key_takeaways or [],
            created_at=model.created_at,
        )

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def save_summary(self, summary: MemorySummary) -> MemorySummary:
        """
        Upsert a memory summary.
        If a summary already exists for memory_id, it is replaced; otherwise a new row is inserted.
        Standalone summaries (no memory_id) are always inserted fresh.
        """
        existing: MemorySummaryModel | None = None
        if summary.memory_id is not None:
            existing = (
                self.session.query(MemorySummaryModel)
                .filter_by(memory_id=str(summary.memory_id))
                .first()
            )

        if existing:
            existing.summary_text = summary.summary_text
            existing.original_length = summary.original_length
            existing.compressed_length = summary.compressed_length
            existing.key_takeaways = summary.key_takeaways or None
            self.session.flush()
            return self._to_schema(existing)

        model = MemorySummaryModel(
            id=str(summary.id),
            memory_id=str(summary.memory_id) if summary.memory_id else None,
            summary_text=summary.summary_text,
            original_length=summary.original_length,
            compressed_length=summary.compressed_length,
            key_takeaways=summary.key_takeaways or None,
            created_at=summary.created_at,
        )
        self.session.add(model)
        self.session.flush()
        return self._to_schema(model)

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_by_uuid(self, summary_id: UUID) -> MemorySummary | None:
        """Fetch a summary by its own UUID."""
        model = self.session.query(MemorySummaryModel).filter_by(id=str(summary_id)).first()
        return self._to_schema(model) if model else None

    def get_by_memory_id(self, memory_id: UUID) -> MemorySummary | None:
        """Fetch the summary associated with a source memory entry."""
        model = (
            self.session.query(MemorySummaryModel)
            .filter_by(memory_id=str(memory_id))
            .first()
        )
        return self._to_schema(model) if model else None

    def list_recent(self, limit: int = 50) -> Sequence[MemorySummary]:
        """Fetch most recent summaries ordered by created_at descending."""
        models = (
            self.session.query(MemorySummaryModel)
            .order_by(MemorySummaryModel.created_at.desc())
            .limit(limit)
            .all()
        )
        return [self._to_schema(m) for m in models]


# ---------------------------------------------------------------------------
# EmbeddingRepository
# ---------------------------------------------------------------------------


class EmbeddingRepository(BaseRepository[MemoryEmbeddingModel]):
    """
    Persists and retrieves memory embedding vector metadata.

    Stores dense float embeddings (e.g. 384 dimensions) as JSON alongside
    model provenance. One embedding per memory_id (primary key constraint).

    Supported operations:
    - save (upsert by memory_id)
    - get by memory_id
    - list all (for batch cosine similarity operations)
    - delete by memory_id
    """

    def __init__(self, session: Session) -> None:
        super().__init__(MemoryEmbeddingModel, session)

    # ------------------------------------------------------------------
    # Mapping helpers
    # ------------------------------------------------------------------

    def _to_schema(self, model: MemoryEmbeddingModel) -> MemoryEmbedding:
        return MemoryEmbedding(
            memory_id=UUID(model.memory_id),
            embedding=model.embedding,
            model_name=model.model_name,
            dimension=model.dimension,
            created_at=model.created_at,
        )

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def save_embedding(self, embedding: MemoryEmbedding) -> MemoryEmbedding:
        """
        Upsert an embedding for a memory entry.
        Replaces the existing embedding if one exists for memory_id.
        """
        model = (
            self.session.query(MemoryEmbeddingModel)
            .filter_by(memory_id=str(embedding.memory_id))
            .first()
        )
        if model:
            model.embedding = embedding.embedding
            model.model_name = embedding.model_name
            model.dimension = embedding.dimension
        else:
            model = MemoryEmbeddingModel(
                memory_id=str(embedding.memory_id),
                embedding=embedding.embedding,
                model_name=embedding.model_name,
                dimension=embedding.dimension,
                created_at=embedding.created_at,
            )
            self.session.add(model)
        self.session.flush()
        return self._to_schema(model)

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_by_memory_id(self, memory_id: UUID) -> MemoryEmbedding | None:
        """Fetch the embedding for a specific memory entry."""
        model = (
            self.session.query(MemoryEmbeddingModel)
            .filter_by(memory_id=str(memory_id))
            .first()
        )
        return self._to_schema(model) if model else None

    def list_all(self) -> Sequence[MemoryEmbedding]:
        """
        Fetch all stored embeddings.
        Used as the dense vector pool for batch cosine similarity ranking.
        """
        models = self.session.query(MemoryEmbeddingModel).all()
        return [self._to_schema(m) for m in models]

    def delete_by_memory_id(self, memory_id: UUID) -> bool:
        """Remove the embedding for a memory entry. Returns True if deleted."""
        model = (
            self.session.query(MemoryEmbeddingModel)
            .filter_by(memory_id=str(memory_id))
            .first()
        )
        if not model:
            return False
        self.session.delete(model)
        self.session.flush()
        return True


# ---------------------------------------------------------------------------
# RetrievalRepository
# ---------------------------------------------------------------------------


class RetrievalRepository:
    """
    Read-only query helper for composing retrieval candidate pools.

    This repository performs cross-query operations that span memory entries
    and their associated embeddings — returning only the IDs of entries that
    have embeddings available (required for similarity ranking).

    No write operations are defined here; persistence is handled by
    MemoryRepository and EmbeddingRepository.
    """

    def __init__(self, session: Session) -> None:
        self.session = session

    def get_embedded_entry_ids(self) -> Sequence[UUID]:
        """
        Return UUIDs of all memory entries that have associated embeddings.
        Used to narrow the candidate pool before similarity ranking.
        """
        rows = self.session.query(MemoryEmbeddingModel.memory_id).all()
        return [UUID(row.memory_id) for row in rows]

    def get_candidates_with_embeddings(
        self,
        *,
        domain: MemoryDomain | None = None,
        min_importance_score: float = 0.0,
        limit: int = 50,
    ) -> Sequence[tuple[MemoryEntry, MemoryEmbedding]]:
        """
        Return (MemoryEntry, MemoryEmbedding) pairs for the top-ranked candidates.

        Filters:
        - domain: restrict to a specific MemoryDomain
        - min_importance_score: minimum importance threshold
        - limit: max pairs returned

        Ordered by importance_score descending then created_at descending so
        the most significant and recent entries are evaluated first during retrieval.
        """
        query = (
            self.session.query(MemoryEntryModel, MemoryEmbeddingModel)
            .join(
                MemoryEmbeddingModel,
                MemoryEntryModel.id == MemoryEmbeddingModel.memory_id,
            )
        )
        if domain:
            query = query.filter(MemoryEntryModel.domain == domain.value)
        if min_importance_score > 0.0:
            query = query.filter(MemoryEntryModel.importance_score >= min_importance_score)

        rows = (
            query
            .order_by(
                MemoryEntryModel.importance_score.desc(),
                MemoryEntryModel.created_at.desc(),
            )
            .limit(limit)
            .all()
        )

        result: list[tuple[MemoryEntry, MemoryEmbedding]] = []
        for entry_model, emb_model in rows:
            entry = MemoryEntry(
                id=UUID(entry_model.id),
                content=entry_model.content,
                domain=MemoryDomain(entry_model.domain),
                memory_type=MemoryType(entry_model.memory_type),
                source=MemorySource(entry_model.source),
                importance_level=MemoryImportance(entry_model.importance_level),
                importance_score=entry_model.importance_score,
                tags=entry_model.tags or [],
                metadata=entry_model.metadata_json or {},
                created_at=entry_model.created_at,
                updated_at=entry_model.updated_at,
            )
            embedding = MemoryEmbedding(
                memory_id=UUID(emb_model.memory_id),
                embedding=emb_model.embedding,
                model_name=emb_model.model_name,
                dimension=emb_model.dimension,
                created_at=emb_model.created_at,
            )
            result.append((entry, embedding))
        return result
