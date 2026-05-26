from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from sqlalchemy.orm import Session

from src.modules.memory.export_sync import (
    ExportEligibilityEvaluator,
    MemoryExportService,
    ObsidianExportFormatter,
    VaultPathManager,
)
from src.modules.memory.repositories import MemoryRepository, MemorySummaryRepository
from src.modules.memory.schemas import (
    MemoryCreate,
    MemoryDomain,
    MemoryEntry,
    MemoryImportance,
    MemorySource,
    MemorySummary,
    MemoryType,
)


def _make_entry(
    *,
    content: str = "SQLite WAL mode.",
    domain: MemoryDomain = MemoryDomain.OPERATIONAL,
    memory_type: MemoryType = MemoryType.FACT,
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


def test_vault_path_manager_folders(tmp_path: Path) -> None:
    manager = VaultPathManager(tmp_path)

    # 1. 11-Issues blocker/bug/problem/incident
    e1 = _make_entry(content="Incident with the database server.")
    assert manager.get_vault_folder(e1).name == "11-Issues"

    e1_tag = _make_entry(content="Simple entry", tags=["blocker"])
    assert manager.get_vault_folder(e1_tag).name == "11-Issues"

    # 2. 04-Constraints
    e2 = _make_entry(content="Project milestone deadline is tomorrow.", domain=MemoryDomain.OPERATIONAL)
    assert manager.get_vault_folder(e2).name == "04-Constraints"

    e2_tag = _make_entry(content="Operational constraint.", tags=["constraint"])
    assert manager.get_vault_folder(e2_tag).name == "04-Constraints"

    # 3. 03-Decisions
    e3 = _make_entry(memory_type=MemoryType.DECISION, domain=MemoryDomain.OPERATIONAL)
    assert manager.get_vault_folder(e3).name == "03-Decisions"

    # 4. 00-Architecture
    e4 = _make_entry(domain=MemoryDomain.ARCHITECTURE, memory_type=MemoryType.FACT)
    assert manager.get_vault_folder(e4).name == "00-Architecture"

    # 5. 01-Relationships
    e5 = _make_entry(domain=MemoryDomain.RELATIONSHIP)
    assert manager.get_vault_folder(e5).name == "01-Relationships"

    e5_kw = _make_entry(content="Spoke with hiring manager at Stripe.")
    assert manager.get_vault_folder(e5_kw).name == "01-Relationships"

    # 6. 02-Outreach
    e6 = _make_entry(content="Sending cold email intro to tech team.")
    assert manager.get_vault_folder(e6).name == "02-Outreach"

    # 7. 05-Lessons (default)
    e7 = _make_entry(content="Just regular notes.", domain=MemoryDomain.OPERATIONAL, memory_type=MemoryType.FACT)
    assert manager.get_vault_folder(e7).name == "05-Lessons"


def test_vault_path_manager_slugify(tmp_path: Path) -> None:
    manager = VaultPathManager(tmp_path)

    # Standard clean slug
    slug1 = manager.slugify("SQLite WAL mode design decision.")
    assert slug1 == "sqlite-wal-mode-design-decision"

    # Long text truncated to 40 characters limit
    slug2 = manager.slugify("This is a very long memory entry content that should be truncated.")
    assert len(slug2) <= 40
    assert slug2.startswith("this-is-a-very-long")

    # Clean punctuation
    slug3 = manager.slugify("Hello! @World? #Tags.")
    assert slug3 == "hello-world-tags"


def test_vault_path_manager_file_path(tmp_path: Path) -> None:
    manager = VaultPathManager(tmp_path)
    entry = _make_entry(content="Standard note.")
    file_path = manager.get_vault_file_path(entry)

    assert file_path.suffix == ".md"
    assert file_path.parent.name == "05-Lessons"
    uuid_short = str(entry.id)[:8]
    assert file_path.name == f"standard-note-{uuid_short}.md"


def test_export_eligibility_evaluator() -> None:
    evaluator = ExportEligibilityEvaluator()

    # Ineligible retrieval logs
    e_ret = _make_entry(domain=MemoryDomain.RETRIEVAL)
    assert not evaluator.is_eligible(e_ret)

    # Ineligible transcripts
    e_trans = _make_entry(tags=["transcript"])
    assert not evaluator.is_eligible(e_trans)

    # Ineligible temporary logs
    e_temp = _make_entry(tags=["temporary-log"])
    assert not evaluator.is_eligible(e_temp)

    # Eligible: High importance
    e_high = _make_entry(importance_level=MemoryImportance.HIGH)
    assert evaluator.is_eligible(e_high)

    # Eligible: Critical importance score
    e_score = _make_entry(importance_score=0.85, importance_level=MemoryImportance.LOW)
    assert evaluator.is_eligible(e_score)

    # Eligible: Architecture domain
    e_arch = _make_entry(domain=MemoryDomain.ARCHITECTURE, importance_level=MemoryImportance.LOW)
    assert evaluator.is_eligible(e_arch)

    # Eligible: Relationship domain
    e_rel = _make_entry(domain=MemoryDomain.RELATIONSHIP, importance_level=MemoryImportance.LOW)
    assert evaluator.is_eligible(e_rel)

    # Eligible: Decision type
    e_dec = _make_entry(memory_type=MemoryType.DECISION, importance_level=MemoryImportance.LOW)
    assert evaluator.is_eligible(e_dec)

    # Eligible: Constraints
    e_const = _make_entry(content="Milestone deadline approach.", importance_level=MemoryImportance.LOW)
    assert evaluator.is_eligible(e_const)

    # Ineligible: Low significance general log
    e_low = _make_entry(
        content="General daily task notes with no important keywords.",
        domain=MemoryDomain.OPERATIONAL,
        memory_type=MemoryType.FACT,
        importance_level=MemoryImportance.LOW,
        importance_score=0.2,
    )
    assert not evaluator.is_eligible(e_low)



def test_obsidian_export_formatter() -> None:
    formatter = ObsidianExportFormatter()
    entry = _make_entry(
        content="SQLite WAL Mode implementation details.\nThis mode improves concurrency.",
        domain=MemoryDomain.ARCHITECTURE,
        memory_type=MemoryType.DECISION,
        source=MemorySource.USER,
        importance_level=MemoryImportance.HIGH,
        importance_score=0.8,
        tags=["sqlite", "db"],
    )

    formatted = formatter.format_entry(entry)
    assert "---" in formatted
    assert f"id: {entry.id}" in formatted
    assert "domain: architecture" in formatted
    assert "type: decision" in formatted
    assert "importance: high" in formatted
    assert "importance_score: 0.8" in formatted
    assert "- sqlite" in formatted
    assert "# SQLite WAL Mode implementation details." in formatted
    assert "This mode improves concurrency." in formatted

    # With summary & key takeaways
    summary = MemorySummary(
        id=uuid4(),
        memory_id=entry.id,
        summary_text="Concurrently writes SQLite pages in WAL format.",
        original_length=len(entry.content),
        compressed_length=47,
        key_takeaways=["Improves DB writes", "Saves transaction latency"],
    )

    formatted_with_sum = formatter.format_entry(entry, summary)
    assert "## Summary" in formatted_with_sum
    assert "Concurrently writes SQLite pages in WAL format." in formatted_with_sum
    assert "## Key Takeaways" in formatted_with_sum
    assert "- Improves DB writes" in formatted_with_sum
    assert "- Saves transaction latency" in formatted_with_sum


def test_memory_export_service_all(db_session: Session, tmp_path: Path) -> None:
    repo = MemoryRepository(db_session)
    sum_repo = MemorySummaryRepository(db_session)

    # 1. Seed database entries
    # Eligible architecture decision
    entry_arch = repo.create_entry(MemoryCreate(
        content="SQLite database WAL mode design.",
        domain=MemoryDomain.ARCHITECTURE,
        memory_type=MemoryType.DECISION,
        source=MemorySource.USER,
        importance_level=MemoryImportance.HIGH,
        importance_score=0.8,
        tags=["sqlite"],
    ))

    # Associated summary
    sum_repo.save_summary(MemorySummary(
        id=uuid4(),
        memory_id=entry_arch.id,
        summary_text="Compressed SQLite WAL summary.",
        original_length=32,
        compressed_length=30,
        key_takeaways=["Takeaway 1", "Takeaway 2"],
    ))

    # Ineligible transcript
    repo.create_entry(MemoryCreate(
        content="Raw transcript of meeting.",
        domain=MemoryDomain.OPERATIONAL,
        memory_type=MemoryType.FACT,
        source=MemorySource.INGESTION,
        importance_level=MemoryImportance.LOW,
        importance_score=0.2,
        tags=["transcript"],
    ))

    # Eligible blocker constraint
    entry_blocker = repo.create_entry(MemoryCreate(
        content="This issue is a blocker for launch.",
        domain=MemoryDomain.OPERATIONAL,
        memory_type=MemoryType.FACT,
        source=MemorySource.SYSTEM,
        importance_level=MemoryImportance.LOW,
        importance_score=0.4,
    ))

    service = MemoryExportService(db_session, vault_path=tmp_path)
    count = service.export_all()

    # Should export exactly 2 entries (architecture decision + blocker)
    assert count == 2

    # Check generated files in Vault
    path_manager = VaultPathManager(tmp_path)
    file_arch = path_manager.get_vault_file_path(entry_arch)
    file_blocker = path_manager.get_vault_file_path(entry_blocker)

    assert file_arch.exists()
    assert file_blocker.exists()

    # Check content in architecture file
    content_arch = file_arch.read_text(encoding="utf-8")
    assert "SQLite database WAL mode design." in content_arch
    assert "## Summary" in content_arch
    assert "Compressed SQLite WAL summary." in content_arch
    assert "- Takeaway 1" in content_arch


def test_memory_export_service_by_id(db_session: Session, tmp_path: Path) -> None:
    repo = MemoryRepository(db_session)
    service = MemoryExportService(db_session, vault_path=tmp_path)

    entry = repo.create_entry(MemoryCreate(
        content="Eligible standalone entry.",
        domain=MemoryDomain.ARCHITECTURE,
        memory_type=MemoryType.DECISION,
        source=MemorySource.USER,
        importance_level=MemoryImportance.HIGH,
        importance_score=0.9,
    ))

    # Export by ID
    success = service.export_entry_by_id(entry.id)
    assert success

    path_manager = VaultPathManager(tmp_path)
    file_path = path_manager.get_vault_file_path(entry)
    assert file_path.exists()

    # Try non-existent ID
    success_none = service.export_entry_by_id(uuid4())
    assert not success_none


def test_memory_export_service_duplicate_safety(db_session: Session, tmp_path: Path) -> None:
    repo = MemoryRepository(db_session)
    service = MemoryExportService(db_session, vault_path=tmp_path)

    entry = repo.create_entry(MemoryCreate(
        content="Duplicate safety test content.",
        domain=MemoryDomain.ARCHITECTURE,
        memory_type=MemoryType.DECISION,
        source=MemorySource.USER,
        importance_level=MemoryImportance.HIGH,
        importance_score=0.9,
    ))

    # 1. First sync - creates file
    count1 = service.export_all()
    assert count1 == 1

    path_manager = VaultPathManager(tmp_path)
    file_path = path_manager.get_vault_file_path(entry)
    assert file_path.exists()

    # 2. Second sync without modification - should skip write
    # We can inspect the internal `_write_entry` return value
    # Run export_all again, count should be 0 because it's skipped
    count2 = service.export_all()
    assert count2 == 0

    # 3. Modify the file manually in vault - next sync should overwrite back to DB state
    file_path.write_text("Corrupted manual text in vault.", encoding="utf-8")
    count3 = service.export_all()
    assert count3 == 1
    assert "Duplicate safety test content." in file_path.read_text(encoding="utf-8")


def test_memory_export_service_changes_since(db_session: Session, tmp_path: Path) -> None:
    repo = MemoryRepository(db_session)
    service = MemoryExportService(db_session, vault_path=tmp_path)

    # SQLite repository sets created_at/updated_at automatically on flush/commit.
    # To test since threshold, we can create one now, and sync with since=now - 1 hour,
    # or override/test filtering manually.

    entry_new = repo.create_entry(MemoryCreate(
        content="New entry created now.",
        domain=MemoryDomain.ARCHITECTURE,
        memory_type=MemoryType.DECISION,
        source=MemorySource.USER,
        importance_level=MemoryImportance.HIGH,
        importance_score=0.9,
    ))
    assert entry_new is not None

    # Export changes since 1 hour ago (should catch the new entry)
    count = service.export_changes_since(datetime.now(UTC) - timedelta(hours=1))
    assert count == 1

    # Export changes since 1 hour in the future (should catch 0 entries)
    count_future = service.export_changes_since(datetime.now(UTC) + timedelta(hours=1))
    assert count_future == 0
