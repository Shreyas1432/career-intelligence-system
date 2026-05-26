from datetime import UTC, datetime, timedelta
from uuid import uuid4

from sqlalchemy.orm import Session

from src.modules.memory.persistence import (
    ArchivalClassification,
    MemoryDeduplicationCoordinator,
    MemoryPersistenceService,
    PersistenceEligibility,
    PersistencePolicyManager,
)
from src.modules.memory.repositories import MemoryRepository
from src.modules.memory.schemas import (
    MemoryCreate,
    MemoryDomain,
    MemoryEntry,
    MemoryImportance,
    MemorySource,
    MemoryType,
)
from src.modules.memory.significance import RetentionDecision, SignificanceScore


def _make_create(
    *,
    content: str = "Architecture design decision: adoption of local persistence.",
    domain: MemoryDomain = MemoryDomain.ARCHITECTURE,
    memory_type: MemoryType = MemoryType.DECISION,
    source: MemorySource = MemorySource.USER,
    importance_level: MemoryImportance = MemoryImportance.HIGH,
    importance_score: float = 0.75,
) -> MemoryCreate:
    return MemoryCreate(
        content=content,
        domain=domain,
        memory_type=memory_type,
        source=source,
        importance_level=importance_level,
        importance_score=importance_score,
    )


def test_persistence_policy_manager_allowed_profiles() -> None:
    policy = PersistencePolicyManager()

    # Allowed profiles
    assert policy.is_content_profile_allowed(MemoryDomain.ARCHITECTURE, MemoryType.DECISION) is True
    assert policy.is_content_profile_allowed(MemoryDomain.RELATIONSHIP, MemoryType.FACT) is True
    assert policy.is_content_profile_allowed(MemoryDomain.OPERATIONAL, MemoryType.SUMMARY) is True
    assert policy.is_content_profile_allowed(MemoryDomain.RETRIEVAL, MemoryType.SUMMARY) is True
    assert policy.is_content_profile_allowed(MemoryDomain.RETRIEVAL, MemoryType.METADATA) is True

    # Not allowed profile
    assert policy.is_content_profile_allowed(MemoryDomain.RETRIEVAL, MemoryType.DOCUMENT) is False


def test_persistence_policy_manager_eligibility() -> None:
    policy = PersistencePolicyManager()

    # RETAIN -> ELIGIBLE
    sig_retain = SignificanceScore(
        importance_score=0.6,
        importance_level=MemoryImportance.HIGH,
        retention_decision=RetentionDecision.RETAIN,
        is_duplicate=False,
        score_breakdown={},
    )
    eligibility, _ = policy.evaluate_eligibility(
        sig_retain, MemoryDomain.ARCHITECTURE, MemoryType.DECISION
    )
    assert eligibility == PersistenceEligibility.ELIGIBLE

    # REVIEW -> REVIEW_REQUIRED
    sig_review = SignificanceScore(
        importance_score=0.4,
        importance_level=MemoryImportance.MEDIUM,
        retention_decision=RetentionDecision.REVIEW,
        is_duplicate=False,
        score_breakdown={},
    )
    eligibility, _ = policy.evaluate_eligibility(
        sig_review, MemoryDomain.ARCHITECTURE, MemoryType.DECISION
    )
    assert eligibility == PersistenceEligibility.REVIEW_REQUIRED

    # REJECT -> INELIGIBLE
    sig_reject = SignificanceScore(
        importance_score=0.1,
        importance_level=MemoryImportance.LOW,
        retention_decision=RetentionDecision.REJECT,
        is_duplicate=False,
        score_breakdown={},
    )
    eligibility, _ = policy.evaluate_eligibility(
        sig_reject, MemoryDomain.ARCHITECTURE, MemoryType.DECISION
    )
    assert eligibility == PersistenceEligibility.INELIGIBLE


def test_persistence_policy_manager_archival_classification() -> None:
    policy = PersistencePolicyManager()
    ref_time = datetime.now(UTC)

    def _mock_entry(level: MemoryImportance, age_days: int) -> MemoryEntry:
        created = ref_time - timedelta(days=age_days)
        return MemoryEntry(
            id=uuid4(),
            content="Mock entry",
            domain=MemoryDomain.OPERATIONAL,
            memory_type=MemoryType.FACT,
            source=MemorySource.SYSTEM,
            importance_level=level,
            importance_score=0.5,
            created_at=created,
            updated_at=created,
        )

    # CRITICAL is always ACTIVE
    assert policy.classify_archival(_mock_entry(MemoryImportance.CRITICAL, 500), ref_time) == ArchivalClassification.ACTIVE

    # HIGH
    assert policy.classify_archival(_mock_entry(MemoryImportance.HIGH, 10), ref_time) == ArchivalClassification.ACTIVE
    assert policy.classify_archival(_mock_entry(MemoryImportance.HIGH, 200), ref_time) == ArchivalClassification.ARCHIVE

    # MEDIUM
    assert policy.classify_archival(_mock_entry(MemoryImportance.MEDIUM, 10), ref_time) == ArchivalClassification.ACTIVE
    assert policy.classify_archival(_mock_entry(MemoryImportance.MEDIUM, 100), ref_time) == ArchivalClassification.ARCHIVE
    assert policy.classify_archival(_mock_entry(MemoryImportance.MEDIUM, 200), ref_time) == ArchivalClassification.STALE

    # LOW
    assert policy.classify_archival(_mock_entry(MemoryImportance.LOW, 10), ref_time) == ArchivalClassification.ACTIVE
    assert policy.classify_archival(_mock_entry(MemoryImportance.LOW, 45), ref_time) == ArchivalClassification.ARCHIVE
    assert policy.classify_archival(_mock_entry(MemoryImportance.LOW, 100), ref_time) == ArchivalClassification.STALE


def test_deduplication_coordinator(db_session: Session) -> None:
    repo = MemoryRepository(db_session)
    coordinator = MemoryDeduplicationCoordinator(repo)

    # Create entries in two different domains
    repo.create_entry(_make_create(content="Arch 1", domain=MemoryDomain.ARCHITECTURE))
    repo.create_entry(_make_create(content="Arch 2", domain=MemoryDomain.ARCHITECTURE))
    repo.create_entry(_make_create(content="Rel 1", domain=MemoryDomain.RELATIONSHIP))

    arch_pool = coordinator.get_historical_pool(MemoryDomain.ARCHITECTURE)
    assert len(arch_pool) == 2
    assert all(e.domain == MemoryDomain.ARCHITECTURE for e in arch_pool)

    rel_pool = coordinator.get_historical_pool(MemoryDomain.RELATIONSHIP)
    assert len(rel_pool) == 1
    assert rel_pool[0].content == "Rel 1"


def test_persistence_service_successful_flow(db_session: Session) -> None:
    service = MemoryPersistenceService(db_session)

    result = service.persist(
        content="Architecture decision: use SQLite in WAL mode for local-first operations.",
        domain=MemoryDomain.ARCHITECTURE,
        memory_type=MemoryType.DECISION,
        source=MemorySource.USER,
        tags=["sqlite", "db"],
    )

    assert result.eligibility == PersistenceEligibility.ELIGIBLE
    assert result.entry is not None
    assert result.entry.content == "Architecture decision: use SQLite in WAL mode for local-first operations."
    assert "sqlite" in result.entry.tags

    # Verify db has it
    fetched = service.repository.get_by_uuid(result.entry.id)
    assert fetched is not None
    assert fetched.content == result.entry.content


def test_persistence_service_duplicate_rejection(db_session: Session) -> None:
    service = MemoryPersistenceService(db_session)

    # Persist first
    content = "Relationship update: Recruiter Jane from TechCorp likes LinkedIn DMs."
    res1 = service.persist(
        content=content,
        domain=MemoryDomain.RELATIONSHIP,
        memory_type=MemoryType.FACT,
        source=MemorySource.USER,
    )
    assert res1.eligibility == PersistenceEligibility.ELIGIBLE

    # Try to persist near-identical content
    res2 = service.persist(
        content=content,
        domain=MemoryDomain.RELATIONSHIP,
        memory_type=MemoryType.FACT,
        source=MemorySource.USER,
    )
    assert res2.eligibility == PersistenceEligibility.INELIGIBLE
    assert res2.entry is None
    assert res2.significance.is_duplicate is True


def test_persistence_service_noise_rejection(db_session: Session) -> None:
    service = MemoryPersistenceService(db_session)

    result = service.persist(
        content="debug: temporary breakpoint() for tracing orm session.",
        domain=MemoryDomain.OPERATIONAL,
        memory_type=MemoryType.FACT,
        source=MemorySource.SYSTEM,
    )

    assert result.eligibility == PersistenceEligibility.INELIGIBLE
    assert result.entry is None
    assert "temporary debug" in result.explanation or "debug" in result.explanation.lower()


def test_lifecycle_manager_sweep(db_session: Session) -> None:
    from src.modules.memory.models import MemoryEntryModel

    repo = MemoryRepository(db_session)
    service = MemoryPersistenceService(db_session)

    # Create one Low importance entry with noisy content so it gets rejected on re-evaluation
    entry1 = repo.create_entry(_make_create(
        content="debug: temporary task note.",
        domain=MemoryDomain.OPERATIONAL,
        memory_type=MemoryType.FACT,
        source=MemorySource.SYSTEM,
        importance_level=MemoryImportance.LOW,
        importance_score=0.1,
    ))

    # Create another medium entry
    entry2 = repo.create_entry(_make_create(
        content="Medium importance architecture note.",
        domain=MemoryDomain.ARCHITECTURE,
        memory_type=MemoryType.DECISION,
        source=MemorySource.USER,
        importance_level=MemoryImportance.MEDIUM,
        importance_score=0.4,
    ))


    # Update their created_at in the database to simulate age
    db_session.query(MemoryEntryModel).filter_by(id=str(entry1.id)).update(
        {"created_at": datetime.now(UTC) - timedelta(days=100)}
    )
    db_session.query(MemoryEntryModel).filter_by(id=str(entry2.id)).update(
        {"created_at": datetime.now(UTC) - timedelta(days=200)}
    )
    db_session.flush()

    # Verify they exist
    assert repo.get_by_uuid(entry1.id) is not None
    assert repo.get_by_uuid(entry2.id) is not None

    sweep_result = service.cleanup_stale_memories(
        low_staleness_days=90,
        medium_staleness_days=180,
    )

    assert sweep_result.evaluated == 2
    assert sweep_result.deleted == 1  # entry1 deleted
    assert sweep_result.retained == 1  # entry2 retained

    # Verify db state
    assert repo.get_by_uuid(entry1.id) is None
    assert repo.get_by_uuid(entry2.id) is not None
