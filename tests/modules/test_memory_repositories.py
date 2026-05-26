from uuid import UUID, uuid4

import pytest
from sqlalchemy.orm import Session

from src.modules.memory.repositories import (
    EmbeddingRepository,
    MemoryRepository,
    MemorySummaryRepository,
    RetrievalRepository,
)
from src.modules.memory.schemas import (
    MemoryCreate,
    MemoryDomain,
    MemoryEmbedding,
    MemoryImportance,
    MemorySource,
    MemorySummary,
    MemoryType,
    MemoryUpdate,
)

# ---------------------------------------------------------------------------
# Shared factory helpers
# ---------------------------------------------------------------------------


def _make_create(
    *,
    content: str = "Operational memory note.",
    domain: MemoryDomain = MemoryDomain.OPERATIONAL,
    memory_type: MemoryType = MemoryType.FACT,
    source: MemorySource = MemorySource.SYSTEM,
    importance_level: MemoryImportance = MemoryImportance.MEDIUM,
    importance_score: float = 0.5,
    tags: list[str] | None = None,
) -> MemoryCreate:
    return MemoryCreate(
        content=content,
        domain=domain,
        memory_type=memory_type,
        source=source,
        importance_level=importance_level,
        importance_score=importance_score,
        tags=tags or [],
    )


def _make_embedding(memory_id: UUID, score: float = 0.5) -> MemoryEmbedding:
    return MemoryEmbedding(
        memory_id=memory_id,
        embedding=[score, score * 2, score * 3],
        model_name="all-MiniLM-L6-v2",
        dimension=3,
    )


# ---------------------------------------------------------------------------
# MemoryRepository — CRUD
# ---------------------------------------------------------------------------


def test_memory_create_and_get_by_uuid(db_session: Session) -> None:
    """Create a memory entry and fetch it by UUID."""
    repo = MemoryRepository(db_session)

    entry = repo.create_entry(_make_create(content="Architecture decision log."))

    assert isinstance(entry.id, UUID)
    assert entry.content == "Architecture decision log."
    assert entry.domain == MemoryDomain.OPERATIONAL
    assert entry.importance_score == 0.5

    fetched = repo.get_by_uuid(entry.id)
    assert fetched is not None
    assert fetched.id == entry.id
    assert fetched.content == "Architecture decision log."


def test_memory_get_by_uuid_returns_none_for_missing(db_session: Session) -> None:
    """get_by_uuid returns None when no entry exists for the given UUID."""
    repo = MemoryRepository(db_session)
    result = repo.get_by_uuid(uuid4())
    assert result is None


def test_memory_update_content_and_score(db_session: Session) -> None:
    """Update content and importance_score fields on an existing entry."""
    repo = MemoryRepository(db_session)

    entry = repo.create_entry(_make_create(content="Initial note."))
    updated = repo.update_entry(
        entry.id,
        MemoryUpdate(content="Updated note.", importance_score=0.9),
    )

    assert updated is not None
    assert updated.content == "Updated note."
    assert updated.importance_score == 0.9

    # Verify state held in DB
    refetched = repo.get_by_uuid(entry.id)
    assert refetched is not None
    assert refetched.content == "Updated note."
    assert refetched.importance_score == 0.9


def test_memory_update_returns_none_for_missing(db_session: Session) -> None:
    """update_entry returns None when the UUID does not exist."""
    repo = MemoryRepository(db_session)
    result = repo.update_entry(uuid4(), MemoryUpdate(content="Ghost"))
    assert result is None


def test_memory_update_partial_fields(db_session: Session) -> None:
    """Only the supplied fields are mutated; others stay unchanged."""
    repo = MemoryRepository(db_session)

    entry = repo.create_entry(
        _make_create(content="Stable content.", importance_score=0.6)
    )
    updated = repo.update_entry(entry.id, MemoryUpdate(domain=MemoryDomain.CODEBASE))

    assert updated is not None
    assert updated.domain == MemoryDomain.CODEBASE
    assert updated.content == "Stable content."       # unchanged
    assert updated.importance_score == 0.6            # unchanged


def test_memory_delete(db_session: Session) -> None:
    """delete_entry removes the row and returns True; second call returns False."""
    repo = MemoryRepository(db_session)

    entry = repo.create_entry(_make_create())
    assert repo.delete_entry(entry.id) is True
    assert repo.get_by_uuid(entry.id) is None
    assert repo.delete_entry(entry.id) is False


# ---------------------------------------------------------------------------
# MemoryRepository — domain filtering
# ---------------------------------------------------------------------------


def test_list_entries_domain_filter(db_session: Session) -> None:
    """list_entries restricts results to the given domain."""
    repo = MemoryRepository(db_session)

    repo.create_entry(_make_create(domain=MemoryDomain.ARCHITECTURE))
    repo.create_entry(_make_create(domain=MemoryDomain.ARCHITECTURE))
    repo.create_entry(_make_create(domain=MemoryDomain.RELATIONSHIP))

    arch = repo.list_entries(domain=MemoryDomain.ARCHITECTURE)
    rel = repo.list_entries(domain=MemoryDomain.RELATIONSHIP)

    assert all(e.domain == MemoryDomain.ARCHITECTURE for e in arch)
    assert all(e.domain == MemoryDomain.RELATIONSHIP for e in rel)
    # Both domains exist independently
    assert len(arch) >= 2
    assert len(rel) >= 1


# ---------------------------------------------------------------------------
# MemoryRepository — importance filtering
# ---------------------------------------------------------------------------


def test_list_entries_importance_level_filter(db_session: Session) -> None:
    """list_entries filters correctly by qualitative importance band."""
    repo = MemoryRepository(db_session)

    repo.create_entry(_make_create(importance_level=MemoryImportance.HIGH, importance_score=0.8))
    repo.create_entry(_make_create(importance_level=MemoryImportance.LOW, importance_score=0.1))

    high_entries = repo.list_entries(importance_level=MemoryImportance.HIGH)
    low_entries = repo.list_entries(importance_level=MemoryImportance.LOW)

    assert all(e.importance_level == MemoryImportance.HIGH for e in high_entries)
    assert all(e.importance_level == MemoryImportance.LOW for e in low_entries)


def test_list_entries_min_importance_score_filter(db_session: Session) -> None:
    """list_entries filters by minimum importance_score threshold."""
    repo = MemoryRepository(db_session)

    repo.create_entry(_make_create(importance_score=0.9))
    repo.create_entry(_make_create(importance_score=0.3))
    repo.create_entry(_make_create(importance_score=0.7))

    results = repo.list_entries(min_importance_score=0.7)

    assert all(e.importance_score >= 0.7 for e in results)
    scores = [e.importance_score for e in results]
    assert 0.9 in scores
    assert 0.7 in scores
    assert 0.3 not in scores


# ---------------------------------------------------------------------------
# MemoryRepository — tag filtering
# ---------------------------------------------------------------------------


def test_list_entries_tag_filter(db_session: Session) -> None:
    """list_entries post-filters by tag substring match (case-insensitive)."""
    repo = MemoryRepository(db_session)

    repo.create_entry(_make_create(tags=["SQLite", "architecture"]))
    repo.create_entry(_make_create(tags=["recruiter", "networking"]))
    repo.create_entry(_make_create(tags=[]))

    sqlite_results = repo.list_entries(tag="sqlite")
    recruiter_results = repo.list_entries(tag="Recruiter")
    unmatched = repo.list_entries(tag="nonexistent_xyz")

    assert len(sqlite_results) >= 1
    assert all(
        any("sqlite" in t.lower() for t in e.tags) for e in sqlite_results
    )
    assert len(recruiter_results) >= 1
    assert len(unmatched) == 0


# ---------------------------------------------------------------------------
# MemoryRepository — retrieval candidates
# ---------------------------------------------------------------------------


def test_get_retrieval_candidates_ordering(db_session: Session) -> None:
    """get_retrieval_candidates returns entries ordered by importance_score desc."""
    repo = MemoryRepository(db_session)

    repo.create_entry(_make_create(content="Low score", importance_score=0.2))
    repo.create_entry(_make_create(content="High score", importance_score=0.95))
    repo.create_entry(_make_create(content="Mid score", importance_score=0.6))

    candidates = repo.get_retrieval_candidates()

    scores = [c.importance_score for c in candidates]
    assert scores == sorted(scores, reverse=True), "Candidates must be sorted by score descending"


def test_get_retrieval_candidates_domain_filter(db_session: Session) -> None:
    """get_retrieval_candidates respects domain filter."""
    repo = MemoryRepository(db_session)

    repo.create_entry(_make_create(domain=MemoryDomain.CODEBASE, importance_score=0.8))
    repo.create_entry(_make_create(domain=MemoryDomain.RETRIEVAL, importance_score=0.9))

    codebase_candidates = repo.get_retrieval_candidates(domain=MemoryDomain.CODEBASE)

    assert all(c.domain == MemoryDomain.CODEBASE for c in codebase_candidates)


def test_get_retrieval_candidates_min_score(db_session: Session) -> None:
    """get_retrieval_candidates excludes entries below min_importance_score."""
    repo = MemoryRepository(db_session)

    repo.create_entry(_make_create(importance_score=0.1))
    repo.create_entry(_make_create(importance_score=0.85))

    candidates = repo.get_retrieval_candidates(min_importance_score=0.5)

    assert all(c.importance_score >= 0.5 for c in candidates)


# ---------------------------------------------------------------------------
# MemorySummaryRepository — CRUD
# ---------------------------------------------------------------------------


def test_summary_save_and_get_by_uuid(db_session: Session) -> None:
    """Save a standalone summary and retrieve it by UUID."""
    repo = MemorySummaryRepository(db_session)

    summary = MemorySummary(
        summary_text="Compressed operational note.",
        original_length=500,
        compressed_length=50,
        key_takeaways=["Use WAL mode", "Batch writes"],
    )
    saved = repo.save_summary(summary)

    assert isinstance(saved.id, UUID)
    assert saved.summary_text == "Compressed operational note."
    assert saved.compressed_length == 50
    assert "Use WAL mode" in saved.key_takeaways

    fetched = repo.get_by_uuid(saved.id)
    assert fetched is not None
    assert fetched.id == saved.id


def test_summary_upsert_by_memory_id(db_session: Session) -> None:
    """Saving a second summary with the same memory_id overwrites the first."""
    repo = MemorySummaryRepository(db_session)
    mem_id = uuid4()

    first = MemorySummary(
        memory_id=mem_id,
        summary_text="First summary.",
        original_length=300,
        compressed_length=30,
    )
    saved_first = repo.save_summary(first)

    second = MemorySummary(
        memory_id=mem_id,
        summary_text="Second summary — updated.",
        original_length=300,
        compressed_length=40,
    )
    saved_second = repo.save_summary(second)

    # Same memory_id should resolve to same row (upsert)
    fetched = repo.get_by_memory_id(mem_id)
    assert fetched is not None
    assert fetched.summary_text == "Second summary — updated."
    assert fetched.id == saved_first.id  # row identity preserved
    assert saved_second.id == saved_first.id


def test_summary_get_by_memory_id_returns_none(db_session: Session) -> None:
    """get_by_memory_id returns None when no summary exists for the UUID."""
    repo = MemorySummaryRepository(db_session)
    result = repo.get_by_memory_id(uuid4())
    assert result is None


def test_summary_standalone_no_memory_id(db_session: Session) -> None:
    """Summaries without a memory_id are stored without a parent reference."""
    repo = MemorySummaryRepository(db_session)

    summary = MemorySummary(
        summary_text="Standalone note.",
        original_length=100,
        compressed_length=20,
    )
    saved = repo.save_summary(summary)

    assert saved.memory_id is None


def test_summary_list_recent(db_session: Session) -> None:
    """list_recent returns summaries ordered newest-first."""
    repo = MemorySummaryRepository(db_session)

    for i in range(3):
        repo.save_summary(
            MemorySummary(
                summary_text=f"Summary {i}",
                original_length=100 * (i + 1),
                compressed_length=10 * (i + 1),
            )
        )

    recent = repo.list_recent(limit=10)
    assert len(recent) >= 3
    # Timestamps must be non-increasing (newest first)
    ts = [s.created_at for s in recent]
    assert ts == sorted(ts, reverse=True)


# ---------------------------------------------------------------------------
# EmbeddingRepository — CRUD
# ---------------------------------------------------------------------------


def test_embedding_save_and_get(db_session: Session) -> None:
    """Save an embedding and retrieve it by memory_id."""
    repo = EmbeddingRepository(db_session)
    mem_id = uuid4()

    emb = _make_embedding(mem_id, score=0.4)
    saved = repo.save_embedding(emb)

    assert saved.memory_id == mem_id
    assert saved.embedding == pytest.approx([0.4, 0.8, 1.2])
    assert saved.model_name == "all-MiniLM-L6-v2"
    assert saved.dimension == 3

    fetched = repo.get_by_memory_id(mem_id)
    assert fetched is not None
    assert fetched.memory_id == mem_id


def test_embedding_upsert_replaces_vector(db_session: Session) -> None:
    """Saving a second embedding for the same memory_id overwrites the vector."""
    repo = EmbeddingRepository(db_session)
    mem_id = uuid4()

    repo.save_embedding(_make_embedding(mem_id, score=0.1))

    new_emb = MemoryEmbedding(
        memory_id=mem_id,
        embedding=[9.9, 8.8, 7.7],
        model_name="all-MiniLM-L6-v2",
        dimension=3,
    )
    repo.save_embedding(new_emb)

    fetched = repo.get_by_memory_id(mem_id)
    assert fetched is not None
    assert fetched.embedding == [9.9, 8.8, 7.7]


def test_embedding_get_returns_none_for_missing(db_session: Session) -> None:
    """get_by_memory_id returns None when no embedding exists."""
    repo = EmbeddingRepository(db_session)
    result = repo.get_by_memory_id(uuid4())
    assert result is None


def test_embedding_list_all(db_session: Session) -> None:
    """list_all returns every stored embedding row."""
    repo = EmbeddingRepository(db_session)

    ids = [uuid4() for _ in range(3)]
    for mem_id in ids:
        repo.save_embedding(_make_embedding(mem_id))

    all_embs = repo.list_all()
    persisted_ids = {e.memory_id for e in all_embs}
    for mid in ids:
        assert mid in persisted_ids


def test_embedding_delete(db_session: Session) -> None:
    """delete_by_memory_id removes the row and returns True; second call returns False."""
    repo = EmbeddingRepository(db_session)
    mem_id = uuid4()

    repo.save_embedding(_make_embedding(mem_id))
    assert repo.delete_by_memory_id(mem_id) is True
    assert repo.get_by_memory_id(mem_id) is None
    assert repo.delete_by_memory_id(mem_id) is False


# ---------------------------------------------------------------------------
# RetrievalRepository — candidate pool queries
# ---------------------------------------------------------------------------


def test_retrieval_get_embedded_entry_ids(db_session: Session) -> None:
    """get_embedded_entry_ids returns UUIDs of entries with embeddings."""
    mem_repo = MemoryRepository(db_session)
    emb_repo = EmbeddingRepository(db_session)
    ret_repo = RetrievalRepository(db_session)

    entry_a = mem_repo.create_entry(_make_create())
    entry_b = mem_repo.create_entry(_make_create())
    entry_c = mem_repo.create_entry(_make_create())  # no embedding

    emb_repo.save_embedding(_make_embedding(entry_a.id))
    emb_repo.save_embedding(_make_embedding(entry_b.id))

    embedded_ids = ret_repo.get_embedded_entry_ids()

    assert entry_a.id in embedded_ids
    assert entry_b.id in embedded_ids
    assert entry_c.id not in embedded_ids


def test_retrieval_candidates_with_embeddings(db_session: Session) -> None:
    """get_candidates_with_embeddings returns (MemoryEntry, MemoryEmbedding) pairs."""
    mem_repo = MemoryRepository(db_session)
    emb_repo = EmbeddingRepository(db_session)
    ret_repo = RetrievalRepository(db_session)

    entry = mem_repo.create_entry(_make_create(importance_score=0.75))
    emb_repo.save_embedding(_make_embedding(entry.id, score=0.5))

    pairs = ret_repo.get_candidates_with_embeddings()

    assert len(pairs) >= 1
    pair_ids = {e.id for e, _ in pairs}
    assert entry.id in pair_ids


def test_retrieval_candidates_domain_filter(db_session: Session) -> None:
    """get_candidates_with_embeddings filters pairs by domain."""
    mem_repo = MemoryRepository(db_session)
    emb_repo = EmbeddingRepository(db_session)
    ret_repo = RetrievalRepository(db_session)

    arch_entry = mem_repo.create_entry(_make_create(domain=MemoryDomain.ARCHITECTURE))
    rel_entry = mem_repo.create_entry(_make_create(domain=MemoryDomain.RELATIONSHIP))

    emb_repo.save_embedding(_make_embedding(arch_entry.id))
    emb_repo.save_embedding(_make_embedding(rel_entry.id))

    arch_pairs = ret_repo.get_candidates_with_embeddings(domain=MemoryDomain.ARCHITECTURE)

    assert all(e.domain == MemoryDomain.ARCHITECTURE for e, _ in arch_pairs)
    pair_ids = {e.id for e, _ in arch_pairs}
    assert arch_entry.id in pair_ids
    assert rel_entry.id not in pair_ids


def test_retrieval_candidates_min_score_filter(db_session: Session) -> None:
    """get_candidates_with_embeddings excludes entries below min_importance_score."""
    mem_repo = MemoryRepository(db_session)
    emb_repo = EmbeddingRepository(db_session)
    ret_repo = RetrievalRepository(db_session)

    high_entry = mem_repo.create_entry(_make_create(importance_score=0.9))
    low_entry = mem_repo.create_entry(_make_create(importance_score=0.2))

    emb_repo.save_embedding(_make_embedding(high_entry.id))
    emb_repo.save_embedding(_make_embedding(low_entry.id))

    pairs = ret_repo.get_candidates_with_embeddings(min_importance_score=0.5)
    pair_ids = {e.id for e, _ in pairs}

    assert high_entry.id in pair_ids
    assert low_entry.id not in pair_ids


def test_retrieval_candidates_ordering(db_session: Session) -> None:
    """get_candidates_with_embeddings returns pairs ordered by importance_score desc."""
    mem_repo = MemoryRepository(db_session)
    emb_repo = EmbeddingRepository(db_session)
    ret_repo = RetrievalRepository(db_session)

    for score in [0.3, 0.9, 0.6]:
        entry = mem_repo.create_entry(_make_create(importance_score=score))
        emb_repo.save_embedding(_make_embedding(entry.id))

    pairs = ret_repo.get_candidates_with_embeddings()
    scores = [e.importance_score for e, _ in pairs]

    assert scores == sorted(scores, reverse=True)


def test_retrieval_candidates_empty_when_no_embeddings(db_session: Session) -> None:
    """get_candidates_with_embeddings returns empty list when no embeddings exist."""
    mem_repo = MemoryRepository(db_session)
    ret_repo = RetrievalRepository(db_session)

    mem_repo.create_entry(_make_create())

    pairs = ret_repo.get_candidates_with_embeddings()
    assert len(pairs) == 0
