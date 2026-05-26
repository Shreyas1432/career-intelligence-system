import uuid
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from src.modules.memory import (
    MemoryCreate,
    MemoryDomain,
    MemoryEmbedding,
    MemoryEntry,
    MemoryImportance,
    MemoryRetrievalResult,
    MemorySource,
    MemorySummary,
    MemoryType,
    MemoryUpdate,
    RetrievalContext,
)


def test_enum_behavior() -> None:
    """Validate string enum values."""
    assert MemoryDomain.OPERATIONAL.value == "operational"
    assert MemoryDomain.ARCHITECTURE.value == "architecture"
    assert MemoryDomain.RELATIONSHIP.value == "relationship"
    assert MemoryDomain.CODEBASE.value == "codebase"
    assert MemoryDomain.RETRIEVAL.value == "retrieval"

    assert MemoryImportance.LOW.value == "low"
    assert MemoryImportance.MEDIUM.value == "medium"
    assert MemoryImportance.HIGH.value == "high"
    assert MemoryImportance.CRITICAL.value == "critical"

    assert MemorySource.USER.value == "user"
    assert MemorySource.SYSTEM.value == "system"
    assert MemorySource.AI.value == "ai"
    assert MemorySource.OBSIDIAN.value == "obsidian"
    assert MemorySource.INGESTION.value == "ingestion"

    assert MemoryType.SUMMARY.value == "summary"
    assert MemoryType.DECISION.value == "decision"
    assert MemoryType.FACT.value == "fact"
    assert MemoryType.METADATA.value == "metadata"
    assert MemoryType.DOCUMENT.value == "document"


def test_memory_create_required_fields() -> None:
    """Validate that missing required fields raise a validation error."""
    with pytest.raises(ValidationError):
        MemoryCreate()  # type: ignore


def test_memory_create_defaults_and_validation() -> None:
    """Validate optional field handling and default values for MemoryCreate."""
    create = MemoryCreate(
        content="Testing memory operational info.",
        domain=MemoryDomain.OPERATIONAL,
        memory_type=MemoryType.SUMMARY,
        source=MemorySource.AI,
    )
    assert create.content == "Testing memory operational info."
    assert create.domain == MemoryDomain.OPERATIONAL
    assert create.memory_type == MemoryType.SUMMARY
    assert create.source == MemorySource.AI
    assert create.importance_level == MemoryImportance.MEDIUM
    assert create.importance_score == 0.5
    assert create.tags == []
    assert create.metadata == {}


def test_memory_create_constraints() -> None:
    """Validate min_length and max_length constraints on MemoryCreate content."""
    # min_length validation
    with pytest.raises(ValidationError):
        MemoryCreate(
            content="",
            domain=MemoryDomain.OPERATIONAL,
            memory_type=MemoryType.SUMMARY,
            source=MemorySource.AI,
        )

    # max_length validation
    long_content = "a" * 10001
    with pytest.raises(ValidationError) as exc_info:
        MemoryCreate(
            content=long_content,
            domain=MemoryDomain.OPERATIONAL,
            memory_type=MemoryType.SUMMARY,
            source=MemorySource.AI,
        )
    assert "String should have at most 10000 characters" in str(exc_info.value)


def test_memory_update_optionality() -> None:
    """Validate all update fields are fully optional."""
    update = MemoryUpdate()
    assert update.content is None
    assert update.domain is None
    assert update.importance_score is None

    # Verify updating with valid score
    update_score = MemoryUpdate(importance_score=0.8)
    assert update_score.importance_score == 0.8

    # Verify score validation bounds
    with pytest.raises(ValidationError):
        MemoryUpdate(importance_score=1.5)

    # Verify content max length constraint
    with pytest.raises(ValidationError):
        MemoryUpdate(content="a" * 10001)


def test_memory_entry_parsing_and_defaults() -> None:
    """Validate MemoryEntry fields and UUID/timestamp defaults."""
    entry = MemoryEntry(
        content="Codebase structured layout decision.",
        domain=MemoryDomain.CODEBASE,
        memory_type=MemoryType.DECISION,
        source=MemorySource.USER,
        importance_level=MemoryImportance.CRITICAL,
        importance_score=0.95,
    )

    assert isinstance(entry.id, uuid.UUID)
    assert isinstance(entry.created_at, datetime)
    assert entry.created_at.tzinfo == UTC
    assert entry.updated_at >= entry.created_at
    assert entry.content == "Codebase structured layout decision."
    assert entry.domain == MemoryDomain.CODEBASE
    assert entry.memory_type == MemoryType.DECISION
    assert entry.source == MemorySource.USER
    assert entry.importance_level == MemoryImportance.CRITICAL
    assert entry.importance_score == 0.95
    assert entry.tags == []
    assert entry.metadata == {}


def test_memory_summary_constraints() -> None:
    """Validate MemorySummary limits and fields."""
    summary = MemorySummary(
        summary_text="Quick compressed text.",
        original_length=2000,
        compressed_length=200,
        key_takeaways=["takeaway 1"],
    )

    assert isinstance(summary.id, uuid.UUID)
    assert summary.memory_id is None
    assert summary.summary_text == "Quick compressed text."
    assert summary.original_length == 2000
    assert summary.compressed_length == 200
    assert summary.key_takeaways == ["takeaway 1"]
    assert isinstance(summary.created_at, datetime)

    # Validate max summary length constraint
    with pytest.raises(ValidationError) as exc_info:
        MemorySummary(
            summary_text="a" * 2001,
            original_length=5000,
            compressed_length=2001,
        )
    assert "String should have at most 2000 characters" in str(exc_info.value)


def test_memory_embedding_structure() -> None:
    """Validate MemoryEmbedding format constraints."""
    memory_id = uuid.uuid4()
    emb = MemoryEmbedding(
        memory_id=memory_id,
        embedding=[0.1, -0.2, 0.35],
        model_name="all-MiniLM-L6-v2",
        dimension=3,
    )

    assert emb.memory_id == memory_id
    assert emb.embedding == [0.1, -0.2, 0.35]
    assert emb.model_name == "all-MiniLM-L6-v2"
    assert emb.dimension == 3
    assert isinstance(emb.created_at, datetime)


def test_retrieval_context_assembly() -> None:
    """Validate RetrievalContext structures and formatting elements."""
    entry = MemoryEntry(
        content="Remember to use SQLite WAL mode.",
        domain=MemoryDomain.ARCHITECTURE,
        memory_type=MemoryType.DECISION,
        source=MemorySource.SYSTEM,
        importance_level=MemoryImportance.HIGH,
        importance_score=0.85,
    )

    result = MemoryRetrievalResult(
        entry=entry,
        similarity_score=0.78,
        rerank_score=0.82,
    )

    context = RetrievalContext(
        query="database optimizations",
        results=[result],
        assembled_context="[ARCHITECTURE] [DECISION] Remember to use SQLite WAL mode.",
        total_tokens=10,
        domain_filters=[MemoryDomain.ARCHITECTURE],
    )

    assert context.query == "database optimizations"
    assert len(context.results) == 1
    assert context.results[0].entry.content == "Remember to use SQLite WAL mode."
    assert context.results[0].similarity_score == 0.78
    assert context.results[0].rerank_score == 0.82
    assert context.assembled_context == "[ARCHITECTURE] [DECISION] Remember to use SQLite WAL mode."
    assert context.total_tokens == 10
    assert context.domain_filters == [MemoryDomain.ARCHITECTURE]
    assert context.metadata == {}
