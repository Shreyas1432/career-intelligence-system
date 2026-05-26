from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from src.modules.memory.embeddings import EmbeddingProvider, EmbeddingService
from src.modules.memory.repositories import EmbeddingRepository, MemoryRepository
from src.modules.memory.retrieval import (
    ContextAssembler,
    MemoryRetrievalService,
    RetrievalFilter,
    SemanticRetrievalEngine,
)
from src.modules.memory.schemas import (
    MemoryCreate,
    MemoryDomain,
    MemoryEmbedding,
    MemoryEntry,
    MemoryImportance,
    MemoryRetrievalResult,
    MemorySource,
    MemoryType,
)


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider returning deterministic mock vectors."""

    def __init__(self, dimension: int = 4) -> None:
        self._dimension = dimension

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        # Simple deterministic vector mapping
        results = []
        for text in texts:
            val = len(text) / 100.0
            if "query" in text.lower():
                val = 0.5
            elif "sqlite" in text.lower():
                val = 0.5  # matching SQLite queries closely
            elif "relationship" in text.lower():
                val = 0.1
            vector = [val, val * 2, val * 3, val * 4]
            results.append(vector[: self._dimension])
        return results

    @property
    def model_name(self) -> str:
        return "mock-model"

    @property
    def dimension(self) -> int:
        return self._dimension


def _make_entry(
    *,
    content: str = "SQLite WAL mode.",
    domain: MemoryDomain = MemoryDomain.ARCHITECTURE,
    memory_type: MemoryType = MemoryType.DECISION,
    source: MemorySource = MemorySource.USER,
    importance_level: MemoryImportance = MemoryImportance.HIGH,
    importance_score: float = 0.8,
    created_at: datetime | None = None,
    tags: list[str] | None = None,
) -> MemoryEntry:
    return MemoryEntry(
        id=uuid4(),
        content=content,
        domain=domain,
        memory_type=memory_type,
        source=source,
        importance_level=importance_level,
        importance_score=importance_score,
        created_at=created_at or datetime.now(UTC),
        updated_at=created_at or datetime.now(UTC),
        tags=tags or [],
    )


def test_retrieval_filter_basic() -> None:
    filter_layer = RetrievalFilter()
    ref_time = datetime.now(UTC)

    emb = MemoryEmbedding(memory_id=uuid4(), embedding=[0.5], model_name="mock", dimension=1)

    candidates = [
        (_make_entry(domain=MemoryDomain.ARCHITECTURE, importance_score=0.8, tags=["db"]), emb),
        (_make_entry(domain=MemoryDomain.RELATIONSHIP, importance_score=0.9, tags=["recruiter"]), emb),
        (_make_entry(domain=MemoryDomain.OPERATIONAL, importance_score=0.3, tags=["constraint"]), emb),
    ]

    # Domain filter
    res = filter_layer.filter_candidates(candidates, domain=MemoryDomain.ARCHITECTURE, reference_time=ref_time)
    assert len(res) == 1
    assert res[0][0].domain == MemoryDomain.ARCHITECTURE

    # Score filter
    res = filter_layer.filter_candidates(candidates, min_importance_score=0.5, reference_time=ref_time)
    assert len(res) == 2
    assert all(e.importance_score >= 0.5 for e, _ in res)

    # Tag filter
    res = filter_layer.filter_candidates(candidates, tag="recruiter", reference_time=ref_time)
    assert len(res) == 1
    assert "recruiter" in res[0][0].tags


def test_retrieval_filter_stale() -> None:
    filter_layer = RetrievalFilter()
    ref_time = datetime.now(UTC)
    emb = MemoryEmbedding(memory_id=uuid4(), embedding=[0.5], model_name="mock", dimension=1)

    # Create stale candidate: LOW importance and 100 days old
    stale_entry = _make_entry(
        importance_level=MemoryImportance.LOW,
        importance_score=0.15,
        created_at=ref_time - timedelta(days=100),
    )
    # Create active candidate: LOW importance and 5 days old
    active_entry = _make_entry(
        importance_level=MemoryImportance.LOW,
        importance_score=0.15,
        created_at=ref_time - timedelta(days=5),
    )

    candidates = [(stale_entry, emb), (active_entry, emb)]

    # Filter stale out
    res = filter_layer.filter_candidates(candidates, exclude_stale=True, reference_time=ref_time)
    assert len(res) == 1
    assert res[0][0].id == active_entry.id

    # Do not filter stale out
    res = filter_layer.filter_candidates(candidates, exclude_stale=False, reference_time=ref_time)
    assert len(res) == 2


def test_semantic_retrieval_engine() -> None:
    engine = SemanticRetrievalEngine()

    emb1 = [0.1, 0.2, 0.3, 0.4]
    emb2 = [0.1, 0.2, 0.3, 0.4]
    emb3 = [0.5, 0.5, 0.5, 0.5]

    # Identical
    assert engine.calculate_similarity(emb1, emb2) == pytest.approx(1.0)
    # Different
    assert engine.calculate_similarity(emb1, emb3) < 1.0

    candidates = [
        (_make_entry(content="SQLite database write WAL"), MemoryEmbedding(memory_id=uuid4(), embedding=emb1, model_name="mock", dimension=4)),
        (_make_entry(content="SQLite database write WAL"), MemoryEmbedding(memory_id=uuid4(), embedding=emb2, model_name="mock", dimension=4)),  # Duplicate
        (_make_entry(content="Relationship recruiter outreach"), MemoryEmbedding(memory_id=uuid4(), embedding=emb3, model_name="mock", dimension=4)),
    ]

    results = engine.rank_candidates(
        query_embedding=emb1,
        candidates=candidates,
        min_similarity=0.3,
        suppress_duplicates=True,
    )

    # Output should exclude the duplicate entry
    assert len(results) == 2
    assert results[0].entry.content == "SQLite database write WAL"
    assert results[1].entry.content == "Relationship recruiter outreach"
    assert results[0].similarity_score == pytest.approx(1.0)


def test_context_assembler() -> None:
    assembler = ContextAssembler()

    results = [
        MemoryRetrievalResult(
            entry=_make_entry(content="Hello A", domain=MemoryDomain.ARCHITECTURE, memory_type=MemoryType.DECISION, importance_level=MemoryImportance.HIGH),
            similarity_score=0.9,
        ),
        MemoryRetrievalResult(
            entry=_make_entry(content="Hello B", domain=MemoryDomain.RELATIONSHIP, memory_type=MemoryType.FACT, importance_level=MemoryImportance.MEDIUM),
            similarity_score=0.8,
        ),
    ]

    context = assembler.assemble(query="test", results=results, max_chars=100)

    # First entry length is around 80 chars. Adding Second entry would cross 100 chars, so it should be truncated.
    assert len(context.results) == 1
    assert context.results[0].entry.content == "Hello A"
    assert "Hello A" in context.assembled_context
    assert "Hello B" not in context.assembled_context
    assert context.total_tokens > 0


def test_memory_retrieval_service_db(db_session: Session) -> None:
    provider = MockEmbeddingProvider(dimension=4)
    emb_service = EmbeddingService(provider=provider)
    ret_service = MemoryRetrievalService(db_session, emb_service)

    mem_repo = MemoryRepository(db_session)
    emb_repo = EmbeddingRepository(db_session)

    # Persist entries
    entry1 = mem_repo.create_entry(MemoryCreate(
        content="SQLite WAL mode design decision.",
        domain=MemoryDomain.ARCHITECTURE,
        memory_type=MemoryType.DECISION,
        source=MemorySource.USER,
        importance_level=MemoryImportance.HIGH,
        importance_score=0.8,
    ))
    entry2 = mem_repo.create_entry(MemoryCreate(
        content="Recruiter contact relationship outreach.",
        domain=MemoryDomain.RELATIONSHIP,
        memory_type=MemoryType.FACT,
        source=MemorySource.SYSTEM,
        importance_level=MemoryImportance.MEDIUM,
        importance_score=0.5,
    ))

    # Save embeddings
    emb1 = emb_service.generate_embedding(entry1)
    emb2 = emb_service.generate_embedding(entry2)
    assert emb1 is not None
    assert emb2 is not None
    emb_repo.save_embedding(emb1)
    emb_repo.save_embedding(emb2)

    # Query matching sqlite
    context = ret_service.retrieve_context(
        query="sqlite",
        domain=None,
        min_similarity=0.4,
        limit=5,
    )

    assert len(context.results) == 2
    # entry1 matches MockEmbeddingProvider values for sqlite closely, so it should rank higher
    assert context.results[0].entry.content == "SQLite WAL mode design decision."
    assert "WAL mode" in context.assembled_context
