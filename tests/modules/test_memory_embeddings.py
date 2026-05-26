from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from src.modules.memory.embeddings import (
    EmbeddingEligibilityEvaluator,
    EmbeddingNormalizer,
    EmbeddingProvider,
    EmbeddingService,
)
from src.modules.memory.repositories import EmbeddingRepository, MemoryRepository
from src.modules.memory.schemas import (
    MemoryCreate,
    MemoryDomain,
    MemoryEntry,
    MemoryImportance,
    MemorySource,
    MemoryType,
)


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider returning deterministic mock vectors."""

    def __init__(self, dimension: int = 4) -> None:
        self._dimension = dimension

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            val = len(text) / 100.0
            vector = [val, val * 2, val * 3, val * 4]
            results.append(vector[: self._dimension])
        return results

    @property
    def model_name(self) -> str:
        return "mock-model"

    @property
    def dimension(self) -> int:
        return self._dimension


def test_embedding_normalizer() -> None:
    normalizer = EmbeddingNormalizer()

    # Simple L2 normalization check
    vector = [3.0, 4.0]
    normalized = normalizer.normalize(vector)
    assert normalized == [0.6, 0.8]  # sqrt(3^2 + 4^2) = 5.0, [3/5, 4/5] = [0.6, 0.8]

    # Validate dimension check
    assert normalizer.validate_dimension(normalized, 2) is True
    assert normalizer.validate_dimension(normalized, 3) is False


def _make_entry(
    *,
    content: str = "Architecture decision: WAL mode.",
    domain: MemoryDomain = MemoryDomain.ARCHITECTURE,
    memory_type: MemoryType = MemoryType.DECISION,
    source: MemorySource = MemorySource.USER,
    importance_level: MemoryImportance = MemoryImportance.HIGH,
    importance_score: float = 0.8,
) -> MemoryEntry:
    return MemoryEntry(
        id=uuid4(),
        content=content,
        domain=domain,
        memory_type=memory_type,
        source=source,
        importance_level=importance_level,
        importance_score=importance_score,
    )


def test_embedding_eligibility_evaluator() -> None:
    evaluator = EmbeddingEligibilityEvaluator()

    # Eligible: High importance architecture decision
    assert evaluator.is_eligible(_make_entry()) is True

    # Ineligible: Low importance
    assert evaluator.is_eligible(_make_entry(importance_level=MemoryImportance.LOW)) is False

    # Ineligible: Transcript content
    assert evaluator.is_eligible(_make_entry(content="Transcript: Speaker A: Hello.")) is False

    # Ineligible: Giant blob
    assert evaluator.is_eligible(_make_entry(content="x" * 8001)) is False


def test_embedding_service_generate_single() -> None:
    provider = MockEmbeddingProvider(dimension=4)
    service = EmbeddingService(provider=provider)

    entry = _make_entry(content="Hello World")
    emb = service.generate_embedding(entry)

    assert emb is not None
    assert emb.memory_id == entry.id
    assert emb.model_name == "mock-model"
    assert emb.dimension == 4
    # Check L2 normalization: vector sum of squares equals 1.0
    sum_sq = sum(x * x for x in emb.embedding)
    assert sum_sq == pytest.approx(1.0)


def test_embedding_service_generate_batch() -> None:
    provider = MockEmbeddingProvider(dimension=4)
    service = EmbeddingService(provider=provider)

    entries = [
        _make_entry(content="Hello A", importance_level=MemoryImportance.HIGH),
        _make_entry(content="Hello B", importance_level=MemoryImportance.LOW),  # Ineligible
        _make_entry(content="Hello C", importance_level=MemoryImportance.CRITICAL),
    ]

    embeddings = service.generate_embeddings_batch(entries)
    assert len(embeddings) == 2
    assert embeddings[0].memory_id == entries[0].id
    assert embeddings[1].memory_id == entries[2].id


def test_embedding_orchestration_db(db_session: Session) -> None:
    provider = MockEmbeddingProvider(dimension=4)
    service = EmbeddingService(provider=provider)

    mem_repo = MemoryRepository(db_session)
    emb_repo = EmbeddingRepository(db_session)

    # Persist memories first
    entry1 = mem_repo.create_entry(
        MemoryCreate(
            content="Architecture note",
            domain=MemoryDomain.ARCHITECTURE,
            memory_type=MemoryType.FACT,
            source=MemorySource.SYSTEM,
            importance_level=MemoryImportance.HIGH,
            importance_score=0.7,
        )
    )
    entry2 = mem_repo.create_entry(
        MemoryCreate(
            content="Low value temporary note",
            domain=MemoryDomain.OPERATIONAL,
            memory_type=MemoryType.FACT,
            source=MemorySource.SYSTEM,
            importance_level=MemoryImportance.LOW,
            importance_score=0.2,
        )
    )

    # Orchestrate
    orchestrated = service.orchestrate_embeddings(db_session, [entry1, entry2])
    assert len(orchestrated) == 1
    assert orchestrated[0].memory_id == entry1.id

    # Verify stored in DB
    fetched = emb_repo.get_by_memory_id(entry1.id)
    assert fetched is not None
    assert fetched.model_name == "mock-model"
    assert len(fetched.embedding) == 4

    # entry2 has no embedding stored
    assert emb_repo.get_by_memory_id(entry2.id) is None
