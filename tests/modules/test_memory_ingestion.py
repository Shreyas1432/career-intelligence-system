from sqlalchemy.orm import Session

from src.modules.memory.ingestion import (
    IngestionSourceClassifier,
    IngestionStatus,
    MemoryCompressionCoordinator,
    MemoryIngestionService,
    MemoryPreprocessor,
)
from src.modules.memory.schemas import (
    MemoryDomain,
    MemoryImportance,
    MemorySource,
    MemoryType,
)


def test_ingestion_source_classifier() -> None:
    classifier = IngestionSourceClassifier()

    # Domain from path/name
    domain, memory_type, source = classifier.classify(
        content="Plain content",
        file_path="/users/shreyas/obsidian/vault/architecture_notes.md",
        file_name="architecture_notes.md",
    )
    assert domain == MemoryDomain.ARCHITECTURE
    assert memory_type == MemoryType.FACT  # Default fallback
    assert source == MemorySource.OBSIDIAN

    # Domain from content keywords
    domain, memory_type, source = classifier.classify(
        content="This is about recruiter outreach and LinkedIn interview process.",
        file_path="some_file.md",
        file_name="some_file.md",
    )
    assert domain == MemoryDomain.RELATIONSHIP
    assert source == MemorySource.INGESTION  # Default fallback

    # Type from content keywords
    domain, memory_type, source = classifier.classify(
        content="We decided to use SQLite in WAL mode.",
        file_path="note.md",
        file_name="note.md",
    )
    assert memory_type == MemoryType.DECISION


def test_memory_preprocessor() -> None:
    preprocessor = MemoryPreprocessor()

    raw_note = """---
domain: architecture
type: decision
source: user
tags: [sqlite, wal]
title: SQLite WAL mode
---
# WAL Mode
We adopt SQLite WAL mode.
- Better concurrency.
"""
    preprocessed = preprocessor.preprocess(raw_note)

    assert preprocessed.domain == MemoryDomain.ARCHITECTURE
    assert preprocessed.memory_type == MemoryType.DECISION
    assert preprocessed.source == MemorySource.USER
    assert preprocessed.tags == ["sqlite", "wal"]
    assert "WAL Mode" in preprocessed.clean_content
    assert preprocessed.extracted_metadata == {"title": "SQLite WAL mode"}


def test_memory_preprocessor_invalid_frontmatter() -> None:
    preprocessor = MemoryPreprocessor()

    raw_note = """---
invalid yaml
---
Body content here.
"""
    preprocessed = preprocessor.preprocess(raw_note)
    assert preprocessed.clean_content == "Body content here."
    assert preprocessed.extracted_metadata == {}


def test_memory_compression_coordinator() -> None:
    coordinator = MemoryCompressionCoordinator()

    content = """# Architecture Update
We are migrating to a modular bounded context pattern.

Key benefits:
- Loose coupling between services.
- Deterministic data flow.
- Maintainable components.

Action: Refactor memory module.
"""
    summary = coordinator.prepare_candidate_summary(content)

    assert len(summary.key_takeaways) == 3
    assert "Loose coupling between services." in summary.key_takeaways
    assert "migrating to a modular bounded context pattern" in summary.summary_text
    assert summary.original_length == len(content)
    assert summary.compressed_length == len(summary.summary_text)


def test_ingestion_service_success_flow(db_session: Session) -> None:
    service = MemoryIngestionService(db_session)

    raw_note = """---
domain: architecture
type: decision
source: user
tags: [modular, patterns]
---
# Bounded Domain Architecture
Adopt bounded domain modular architecture to organize modules cleanly.
- Keep dependencies explicit.
"""
    result = service.ingest_markdown(raw_note)

    assert result.status == IngestionStatus.SUCCESS
    assert result.entry is not None
    assert result.summary is not None
    assert result.entry.domain == MemoryDomain.ARCHITECTURE
    assert result.entry.memory_type == MemoryType.DECISION
    assert result.entry.importance_level == MemoryImportance.CRITICAL  # Architecture decision gets boosted to critical
    assert "Keep dependencies explicit." in result.summary.key_takeaways


    # Verify stored in DB
    fetched = service.persistence_service.repository.get_by_uuid(result.entry.id)
    assert fetched is not None
    assert fetched.content == result.entry.content

    fetched_summary = service.persistence_service.summary_repository.get_by_memory_id(result.entry.id)
    assert fetched_summary is not None
    assert fetched_summary.summary_text == result.summary.summary_text


def test_ingestion_service_empty_fails(db_session: Session) -> None:
    service = MemoryIngestionService(db_session)
    result = service.ingest_markdown("  \n  ")
    assert result.status == IngestionStatus.FAILED
    assert "empty content" in result.explanation


def test_ingestion_service_transcript_skipped(db_session: Session) -> None:
    service = MemoryIngestionService(db_session)
    transcript_note = """---
domain: relationship
type: document
---
Transcript:
Speaker A: Tell me about your project.
Speaker B: I built a python service.
"""
    result = service.ingest_markdown(transcript_note)
    assert result.status == IngestionStatus.SKIPPED
    assert "transcripts are not allowed" in result.explanation


def test_ingestion_service_duplicate_skipped(db_session: Session) -> None:
    service = MemoryIngestionService(db_session)

    note = """---
domain: relationship
type: fact
---
Recruiter Jane at techcorp prefers LinkedIn DMs.
- Response time is fast.
"""
    # First ingestion
    result1 = service.ingest_markdown(note)
    assert result1.status == IngestionStatus.SUCCESS

    # Duplicate ingestion
    result2 = service.ingest_markdown(note)
    assert result2.status == IngestionStatus.SKIPPED
    assert "duplicate" in result2.explanation
