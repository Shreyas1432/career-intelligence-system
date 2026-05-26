from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from src.modules.memory.ranking import (
    DiversityBalancer,
    FreshnessScorer,
    OperationalRelevanceScorer,
    RetrievalRankingService,
)
from src.modules.memory.repositories import MemoryRepository
from src.modules.memory.schemas import (
    MemoryCreate,
    MemoryDomain,
    MemoryEntry,
    MemoryImportance,
    MemoryRetrievalResult,
    MemorySource,
    MemoryType,
)


def _make_entry(
    *,
    content: str = "SQLite WAL mode.",
    domain: MemoryDomain = MemoryDomain.ARCHITECTURE,
    memory_type: MemoryType = MemoryType.DECISION,
    source: MemorySource = MemorySource.USER,
    importance_level: MemoryImportance = MemoryImportance.HIGH,
    importance_score: float = 0.5,
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


def test_operational_relevance_scorer_base() -> None:
    scorer = OperationalRelevanceScorer()

    # Entry with no boosts (domain: OPERATIONAL, type: FACT, no keywords)
    entry = _make_entry(
        domain=MemoryDomain.OPERATIONAL,
        memory_type=MemoryType.FACT,
        content="Normal text message.",
        importance_score=0.4,
    )
    assert scorer.score(entry) == pytest.approx(0.4)


def test_operational_relevance_scorer_domain_boosts() -> None:
    scorer = OperationalRelevanceScorer()

    # ARCHITECTURE boost (+0.20)
    entry_arch = _make_entry(
        domain=MemoryDomain.ARCHITECTURE,
        memory_type=MemoryType.FACT,
        content="Normal text message.",
        importance_score=0.4,
    )
    assert scorer.score(entry_arch) == pytest.approx(0.6)

    # RELATIONSHIP boost (+0.20)
    entry_rel = _make_entry(
        domain=MemoryDomain.RELATIONSHIP,
        memory_type=MemoryType.FACT,
        content="Normal text message.",
        importance_score=0.4,
    )
    assert scorer.score(entry_rel) == pytest.approx(0.6)

    # RETRIEVAL boost (+0.15)
    entry_ret = _make_entry(
        domain=MemoryDomain.RETRIEVAL,
        memory_type=MemoryType.FACT,
        content="Normal text message.",
        importance_score=0.4,
    )
    assert scorer.score(entry_ret) == pytest.approx(0.55)


def test_operational_relevance_scorer_type_boosts() -> None:
    scorer = OperationalRelevanceScorer()

    # DECISION boost (+0.20)
    entry_dec = _make_entry(
        domain=MemoryDomain.OPERATIONAL,
        memory_type=MemoryType.DECISION,
        content="Normal text message.",
        importance_score=0.4,
    )
    assert scorer.score(entry_dec) == pytest.approx(0.6)

    # SUMMARY boost (+0.10)
    entry_sum = _make_entry(
        domain=MemoryDomain.OPERATIONAL,
        memory_type=MemoryType.SUMMARY,
        content="Normal text message.",
        importance_score=0.4,
    )
    assert scorer.score(entry_sum) == pytest.approx(0.5)


def test_operational_relevance_scorer_keyword_boosts() -> None:
    scorer = OperationalRelevanceScorer()

    keywords = ["constraint", "deadline", "blocker", "milestone"]
    for keyword in keywords:
        # Lowercase keyword test
        entry = _make_entry(
            domain=MemoryDomain.OPERATIONAL,
            memory_type=MemoryType.FACT,
            content=f"This is a {keyword}.",
            importance_score=0.4,
        )
        assert scorer.score(entry) == pytest.approx(0.55)

        # Uppercase keyword test
        entry_upper = _make_entry(
            domain=MemoryDomain.OPERATIONAL,
            memory_type=MemoryType.FACT,
            content=f"THIS IS A {keyword.upper()}.",
            importance_score=0.4,
        )
        assert scorer.score(entry_upper) == pytest.approx(0.55)


def test_operational_relevance_scorer_bounds_and_clamping() -> None:
    scorer = OperationalRelevanceScorer()

    # Check max clamped to 1.0 (base 0.9 + ARCHITECTURE boost 0.2 = 1.1 -> clamped to 1.0)
    entry = _make_entry(
        domain=MemoryDomain.ARCHITECTURE,
        memory_type=MemoryType.FACT,
        importance_score=0.9,
    )
    assert scorer.score(entry) == pytest.approx(1.0)

    # Multiple boosts combined
    # ARCHITECTURE (+0.20) + DECISION (+0.20) + "blocker" keyword (+0.15) = +0.55 boost
    entry_combined = _make_entry(
        domain=MemoryDomain.ARCHITECTURE,
        memory_type=MemoryType.DECISION,
        content="This is a critical blocker.",
        importance_score=0.3,
    )
    assert scorer.score(entry_combined) == pytest.approx(0.85)


def test_freshness_scorer_decay() -> None:
    scorer = FreshnessScorer()
    ref_time = datetime(2026, 5, 26, 12, 0, 0, tzinfo=UTC)

    # 0 days old -> score should be 1.0
    entry_new = _make_entry(created_at=ref_time)
    assert scorer.score(entry_new, reference_time=ref_time) == pytest.approx(1.0)

    # 180 days old -> score should be exp(-1.0) ~ 0.3678794
    entry_180d = _make_entry(created_at=ref_time - timedelta(days=180))
    import math
    assert scorer.score(entry_180d, reference_time=ref_time) == pytest.approx(math.exp(-1.0))

    # Future dates -> score should be clamped to 1.0
    entry_future = _make_entry(created_at=ref_time + timedelta(days=10))
    assert scorer.score(entry_future, reference_time=ref_time) == pytest.approx(1.0)


def test_freshness_scorer_timezone_compatibility() -> None:
    scorer = FreshnessScorer()
    ref_time_naive = datetime(2026, 5, 26, 12, 0, 0)

    # Naive timezone
    entry_naive = _make_entry(created_at=datetime(2026, 5, 26, 12, 0, 0))
    assert scorer.score(entry_naive, reference_time=ref_time_naive) == pytest.approx(1.0)

    # Mixed timezone
    entry_aware = _make_entry(created_at=datetime(2026, 5, 26, 12, 0, 0, tzinfo=UTC))
    assert scorer.score(entry_aware, reference_time=ref_time_naive) == pytest.approx(1.0)


def test_diversity_balancer_basic() -> None:
    balancer = DiversityBalancer()

    cand1 = MemoryRetrievalResult(entry=_make_entry(domain=MemoryDomain.ARCHITECTURE), similarity_score=0.9)
    cand2 = MemoryRetrievalResult(entry=_make_entry(domain=MemoryDomain.RELATIONSHIP), similarity_score=0.8)

    candidates = [(cand1, 0.9), (cand2, 0.8)]

    # Check limit restriction
    res_limit_1 = balancer.balance(candidates, limit=1)
    assert len(res_limit_1) == 1
    assert res_limit_1[0].entry.id == cand1.entry.id

    res_limit_2 = balancer.balance(candidates, limit=5)
    assert len(res_limit_2) == 2


def test_diversity_balancer_penalization() -> None:
    balancer = DiversityBalancer()

    # Setup: 3 ARCHITECTURE candidates with high scores, 1 RELATIONSHIP candidate with lower score
    arch1 = MemoryRetrievalResult(entry=_make_entry(domain=MemoryDomain.ARCHITECTURE), similarity_score=0.9)
    arch2 = MemoryRetrievalResult(entry=_make_entry(domain=MemoryDomain.ARCHITECTURE), similarity_score=0.8)
    arch3 = MemoryRetrievalResult(entry=_make_entry(domain=MemoryDomain.ARCHITECTURE), similarity_score=0.7)
    rel1 = MemoryRetrievalResult(entry=_make_entry(domain=MemoryDomain.RELATIONSHIP), similarity_score=0.65)

    candidates = [
        (arch1, 0.9),
        (arch2, 0.8),
        (arch3, 0.7),
        (rel1, 0.65),
    ]

    selected = balancer.balance(candidates, limit=4)
    assert len(selected) == 4
    assert selected[0].entry.id == arch1.entry.id
    assert selected[1].entry.id == rel1.entry.id
    assert selected[2].entry.id == arch2.entry.id
    assert selected[3].entry.id == arch3.entry.id


def test_retrieval_ranking_service_empty() -> None:
    service = RetrievalRankingService()
    assert service.rerank([]) == []


def test_retrieval_ranking_service_blended_scoring() -> None:
    service = RetrievalRankingService()
    ref_time = datetime(2026, 5, 26, 12, 0, 0, tzinfo=UTC)

    # Test case: Isolate weights to verify math.
    entry1 = _make_entry(
        content="Unique entry content number one.",
        domain=MemoryDomain.OPERATIONAL,
        memory_type=MemoryType.FACT,
        importance_score=0.4,
        created_at=ref_time,
    )
    entry2 = _make_entry(
        content="Unique entry content number two.",
        domain=MemoryDomain.ARCHITECTURE,
        memory_type=MemoryType.DECISION,
        importance_score=0.4,
        created_at=ref_time - timedelta(days=180),
    )

    res1 = MemoryRetrievalResult(entry=entry1, similarity_score=0.9)
    res2 = MemoryRetrievalResult(entry=entry2, similarity_score=0.7)

    results = service.rerank(
        [res1, res2],
        limit=2,
        similarity_weight=0.4,
        relevance_weight=0.4,
        freshness_weight=0.2,
        reference_time=ref_time,
    )

    assert len(results) == 2
    assert results[0].entry.id == entry1.id
    assert results[1].entry.id == entry2.id

    results_alt = service.rerank(
        [res1, res2],
        limit=2,
        similarity_weight=0.1,
        relevance_weight=0.8,
        freshness_weight=0.1,
        reference_time=ref_time,
    )
    assert results_alt[0].entry.id == entry2.id
    assert results_alt[1].entry.id == entry1.id


def test_retrieval_ranking_service_duplicate_suppression() -> None:
    service = RetrievalRankingService()
    ref_time = datetime(2026, 5, 26, 12, 0, 0, tzinfo=UTC)

    # 1. Exact same UUID
    entry1 = _make_entry(content="First test entry content string.", importance_score=0.5)
    res1 = MemoryRetrievalResult(entry=entry1, similarity_score=0.8)
    res2 = MemoryRetrievalResult(entry=entry1, similarity_score=0.7)

    # With duplicate suppression enabled (default)
    results = service.rerank([res1, res2], limit=5, reference_time=ref_time)
    assert len(results) == 1
    assert results[0].entry.id == entry1.id

    # With duplicate suppression disabled
    results_no_sup = service.rerank([res1, res2], limit=5, suppress_duplicates=False, reference_time=ref_time)
    assert len(results_no_sup) == 2

    # 2. Similar content (Jaccard similarity >= 0.85)
    entry_similar_a = _make_entry(content="SQLite database WAL mode design decision.", importance_score=0.6)
    entry_similar_b = _make_entry(content="sqlite database wal mode design decisions.", importance_score=0.5)
    res_a = MemoryRetrievalResult(entry=entry_similar_a, similarity_score=0.9)
    res_b = MemoryRetrievalResult(entry=entry_similar_b, similarity_score=0.8)

    results_sim = service.rerank([res_a, res_b], limit=5, duplicate_threshold=0.7, reference_time=ref_time)
    assert len(results_sim) == 1
    assert results_sim[0].entry.id == entry_similar_a.id


def test_retrieval_ranking_service_deterministic_ordering() -> None:
    service = RetrievalRankingService()
    ref_time = datetime(2026, 5, 26, 12, 0, 0, tzinfo=UTC)

    candidates = [
        MemoryRetrievalResult(entry=_make_entry(content="Entry A", domain=MemoryDomain.ARCHITECTURE), similarity_score=0.9),
        MemoryRetrievalResult(entry=_make_entry(content="Entry B", domain=MemoryDomain.RELATIONSHIP), similarity_score=0.8),
        MemoryRetrievalResult(entry=_make_entry(content="Entry C", domain=MemoryDomain.RETRIEVAL), similarity_score=0.7),
        MemoryRetrievalResult(entry=_make_entry(content="Entry D", domain=MemoryDomain.OPERATIONAL), similarity_score=0.6),
    ]

    orders = []
    for _ in range(5):
        res = service.rerank(candidates, limit=4, reference_time=ref_time)
        orders.append([r.entry.id for r in res])

    # Assert all orders are identical
    for o in orders[1:]:
        assert o == orders[0]


def test_retrieval_ranking_service_token_priority_ordering() -> None:
    service = RetrievalRankingService()
    ref_time = datetime(2026, 5, 26, 12, 0, 0, tzinfo=UTC)

    # Check that output is ordered by selection priority (highest blended score + diversity selection first)
    # This allows downstream context assemblers to easily truncate from the end.
    entry1 = _make_entry(content="Priority 1 entry.", domain=MemoryDomain.ARCHITECTURE, importance_score=0.9)
    entry2 = _make_entry(content="Priority 2 entry.", domain=MemoryDomain.RELATIONSHIP, importance_score=0.8)

    res1 = MemoryRetrievalResult(entry=entry1, similarity_score=0.9)
    res2 = MemoryRetrievalResult(entry=entry2, similarity_score=0.5)

    # Weighted: res1 will be far higher in score
    results = service.rerank([res2, res1], limit=2, reference_time=ref_time)
    assert len(results) == 2
    # Highest priority element must be first
    assert results[0].entry.id == entry1.id
    assert results[1].entry.id == entry2.id


def test_retrieval_ranking_service_db_integration(db_session: Session) -> None:
    repo = MemoryRepository(db_session)
    service = RetrievalRankingService()
    ref_time = datetime(2026, 5, 26, 12, 0, 0, tzinfo=UTC)

    # Persist two entries via repository
    db_entry1 = repo.create_entry(MemoryCreate(
        content="System architecture overview.",
        domain=MemoryDomain.ARCHITECTURE,
        memory_type=MemoryType.FACT,
        source=MemorySource.USER,
        importance_level=MemoryImportance.HIGH,
        importance_score=0.7,
    ))

    db_entry2 = repo.create_entry(MemoryCreate(
        content="A simple team contact log.",
        domain=MemoryDomain.RELATIONSHIP,
        memory_type=MemoryType.FACT,
        source=MemorySource.USER,
        importance_level=MemoryImportance.MEDIUM,
        importance_score=0.4,
    ))

    # Fetch from db to verify ORM models
    fetched1 = repo.get_by_uuid(db_entry1.id)
    fetched2 = repo.get_by_uuid(db_entry2.id)
    assert fetched1 is not None
    assert fetched2 is not None

    res1 = MemoryRetrievalResult(entry=fetched1, similarity_score=0.5)
    res2 = MemoryRetrievalResult(entry=fetched2, similarity_score=0.8)

    # Rerank
    reranked = service.rerank(
        [res1, res2],
        limit=2,
        similarity_weight=0.5,
        relevance_weight=0.5,
        freshness_weight=0.0,
        reference_time=ref_time,
    )
    assert len(reranked) == 2

    reranked_custom = service.rerank(
        [res1, res2],
        limit=2,
        similarity_weight=0.1,
        relevance_weight=0.9,
        freshness_weight=0.0,
        reference_time=ref_time,
    )

    assert len(reranked_custom) == 2
    assert reranked_custom[0].entry.id == db_entry1.id
    assert reranked_custom[1].entry.id == db_entry2.id
