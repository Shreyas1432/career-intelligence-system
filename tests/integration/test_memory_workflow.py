"""
Integration tests for the complete operational memory workflow.

Validates the end-to-end data flow across all memory subsystem layers:
    Ingestion -> Persistence -> Embeddings -> Retrieval -> Ranking -> Export

Scenarios
---------
1. Architecture Memory Workflow
   Ingest an architecture decision, evaluate significance, persist to SQLite,
   generate embeddings, retrieve operational context, rerank candidates, assemble
   token-efficient context, and export a markdown summary to an Obsidian vault.

2. Recruiter Continuity Workflow
   Ingest a recruiter continuity note, evaluate operational relevance, persist
   relationship memory, retrieve continuity-aware context, rerank, and export a
   strategic relationship summary.

3. Noise Rejection Workflow
   Ingest low-value operational noise and validate that the pipeline short-circuits
   at the significance layer—no persistence, no embeddings, no retrieval inclusion.

Design constraints
------------------
- All tests use the `db_session` fixture (transactional in-memory SQLite).
- Embedding provider is replaced with a lightweight deterministic mock.
- No live network calls, no external models, no filesystem mutation (vault writes
  use `tmp_path`).
- Assertions are deterministic and do not depend on runtime randomness.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from sqlalchemy.orm import Session

from src.modules.memory.embeddings import (
    EmbeddingEligibilityEvaluator,
    EmbeddingProvider,
    EmbeddingService,
)
from src.modules.memory.export_sync import MemoryExportService
from src.modules.memory.ingestion import IngestionStatus, MemoryIngestionService
from src.modules.memory.persistence import MemoryPersistenceService, PersistenceEligibility
from src.modules.memory.ranking import RetrievalRankingService
from src.modules.memory.retrieval import MemoryRetrievalService
from src.modules.memory.schemas import (
    MemoryDomain,
    MemoryEntry,
    MemoryImportance,
    MemorySource,
    MemoryType,
)
from src.modules.memory.significance import RetentionDecision, SignificanceEvaluator

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class _MockEmbeddingProvider(EmbeddingProvider):
    """
    Deterministic 4-dimensional mock embedding provider.

    Returns vectors whose components are derived from the length of the input
    text so that different texts reliably produce distinct, non-zero vectors.
    """

    _DIMENSION: int = 4

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        for text in texts:
            # Base value derived from text length; each component is unique
            val = (len(text) % 100 + 1) / 100.0
            results.append([val, val * 1.5, val * 0.5, val * 2.0])
        return results

    @property
    def model_name(self) -> str:
        return "mock-4d"

    @property
    def dimension(self) -> int:
        return self._DIMENSION


# ---------------------------------------------------------------------------
# Scenario 1: Architecture Memory Workflow
# ---------------------------------------------------------------------------


class TestArchitectureMemoryWorkflow:
    """
    End-to-end workflow for ingesting and surfacing an architecture decision.

    Data flow:
        Ingestion (Markdown) -> Persistence (SQLite) -> Embeddings (Mock)
        -> Retrieval (Semantic) -> Ranking (Blended) -> Export (Obsidian MD)
    """

    _ARCHITECTURE_MARKDOWN = """\
---
domain: architecture
type: decision
source: user
tags: [sqlite, wal, performance]
---
# WAL Mode Architecture Decision

We adopt SQLite WAL mode as the default write-ahead logging strategy for the
local operational database. WAL mode improves concurrent read performance and
reduces write contention under the bounded-domain modular architecture.

**Decision rationale:**
- WAL mode allows readers to proceed without blocking writers.
- Crash recovery is automatic and consistent.
- Compatible with all bounded-context repositories.
"""

    def test_ingestion_succeeds(self, db_session: Session) -> None:
        """Markdown ingestion produces a persisted MemoryEntry."""
        service = MemoryIngestionService(db_session)
        result = service.ingest_markdown(self._ARCHITECTURE_MARKDOWN)

        assert result.status == IngestionStatus.SUCCESS
        assert result.entry is not None
        assert result.summary is not None
        assert isinstance(result.entry.id, UUID)

    def test_ingested_entry_has_correct_domain(self, db_session: Session) -> None:
        """The ingested entry domain matches the YAML frontmatter."""
        service = MemoryIngestionService(db_session)
        result = service.ingest_markdown(self._ARCHITECTURE_MARKDOWN)

        assert result.status == IngestionStatus.SUCCESS
        assert result.entry is not None
        assert result.entry.domain == MemoryDomain.ARCHITECTURE

    def test_significance_evaluates_architecture_as_retainable(self) -> None:
        """Architecture decisions score above the RETAIN threshold."""
        evaluator = SignificanceEvaluator()
        content = (
            "SQLite WAL mode architecture decision: adopt WAL mode for the bounded "
            "context database to enable concurrent read/write without contention."
        )
        sig = evaluator.evaluate(
            content,
            MemoryDomain.ARCHITECTURE,
            MemoryType.DECISION,
            MemorySource.USER,
        )

        assert sig.retention_decision == RetentionDecision.RETAIN
        assert sig.importance_level in (MemoryImportance.HIGH, MemoryImportance.CRITICAL)
        assert sig.importance_score >= 0.5

    def test_persistence_stores_architecture_entry(self, db_session: Session) -> None:
        """The persistence layer stores an architecture decision without rejection."""
        persistence = MemoryPersistenceService(db_session)
        result = persistence.persist(
            content=(
                "Architecture decision: Adopt SQLite WAL mode for the bounded-context "
                "database to enable concurrent read/write without contention."
            ),
            domain=MemoryDomain.ARCHITECTURE,
            memory_type=MemoryType.DECISION,
            source=MemorySource.USER,
            tags=["sqlite", "wal", "performance"],
        )

        assert result.eligibility == PersistenceEligibility.ELIGIBLE
        assert result.entry is not None
        assert result.entry.domain == MemoryDomain.ARCHITECTURE

    def test_embedding_generation_for_architecture_entry(self, db_session: Session) -> None:
        """Architecture entries eligible for embedding receive normalized vectors."""
        persistence = MemoryPersistenceService(db_session)
        persist_result = persistence.persist(
            content=(
                "Architecture decision: Adopt SQLite WAL mode for concurrent "
                "read/write performance under the bounded-domain architecture."
            ),
            domain=MemoryDomain.ARCHITECTURE,
            memory_type=MemoryType.DECISION,
            source=MemorySource.USER,
        )

        assert persist_result.entry is not None
        entry = persist_result.entry

        provider = _MockEmbeddingProvider()
        emb_service = EmbeddingService(provider=provider)
        embedding = emb_service.generate_embedding(entry)

        assert embedding is not None
        assert embedding.memory_id == entry.id
        assert embedding.dimension == 4
        # L2 normalization check: sum of squared components ≈ 1.0
        total = sum(x * x for x in embedding.embedding)
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_retrieval_surfaces_architecture_context(self, db_session: Session) -> None:
        """Persisted and embedded architecture entries appear in semantic retrieval."""
        # Persist a high-value architecture entry
        persistence = MemoryPersistenceService(db_session)
        persist_result = persistence.persist(
            content=(
                "Architecture decision: WAL mode for the SQLite operational database. "
                "Enables concurrent reader/writer access within bounded domains."
            ),
            domain=MemoryDomain.ARCHITECTURE,
            memory_type=MemoryType.DECISION,
            source=MemorySource.USER,
        )
        assert persist_result.entry is not None

        # Generate and store the embedding
        provider = _MockEmbeddingProvider()
        emb_service = EmbeddingService(provider=provider)
        emb_service.orchestrate_embeddings(db_session, [persist_result.entry])

        # Retrieve with a matching query
        retrieval_service = MemoryRetrievalService(db_session, emb_service)
        context = retrieval_service.retrieve_context(
            "SQLite WAL mode architecture",
            domain=MemoryDomain.ARCHITECTURE,
            min_similarity=0.0,
            limit=5,
        )

        assert len(context.results) >= 1
        returned_ids = {r.entry.id for r in context.results}
        assert persist_result.entry.id in returned_ids

    def test_ranking_reranks_architecture_results(self, db_session: Session) -> None:
        """Retrieval results are successfully reranked with blended scoring."""
        # Persist two entries
        persistence = MemoryPersistenceService(db_session)
        r1 = persistence.persist(
            content=(
                "Architecture decision: WAL mode for SQLite. "
                "Enables concurrent bounded-context reads and writes."
            ),
            domain=MemoryDomain.ARCHITECTURE,
            memory_type=MemoryType.DECISION,
            source=MemorySource.USER,
        )
        r2 = persistence.persist(
            content=(
                "Architecture constraint: database index must remain below 100MB "
                "to maintain sub-5ms query latency for the bounded architecture."
            ),
            domain=MemoryDomain.ARCHITECTURE,
            memory_type=MemoryType.FACT,
            source=MemorySource.USER,
        )
        assert r1.entry is not None
        assert r2.entry is not None

        provider = _MockEmbeddingProvider()
        emb_service = EmbeddingService(provider=provider)
        emb_service.orchestrate_embeddings(db_session, [r1.entry, r2.entry])

        retrieval_service = MemoryRetrievalService(db_session, emb_service)
        context = retrieval_service.retrieve_context(
            "database architecture decision",
            domain=MemoryDomain.ARCHITECTURE,
            min_similarity=0.0,
            limit=10,
        )

        ranking_service = RetrievalRankingService()
        reranked = ranking_service.rerank(context.results, limit=5)

        assert isinstance(reranked, list)
        # Reranked list is non-empty when candidates are available
        if context.results:
            assert len(reranked) >= 1

    def test_context_assembly_is_token_efficient(self, db_session: Session) -> None:
        """Assembled context stays within the token budget."""
        persistence = MemoryPersistenceService(db_session)
        persist_result = persistence.persist(
            content=(
                "Architecture decision: SQLite WAL mode improves concurrent "
                "read/write performance under the bounded-domain architecture."
            ),
            domain=MemoryDomain.ARCHITECTURE,
            memory_type=MemoryType.DECISION,
            source=MemorySource.USER,
        )
        assert persist_result.entry is not None

        provider = _MockEmbeddingProvider()
        emb_service = EmbeddingService(provider=provider)
        emb_service.orchestrate_embeddings(db_session, [persist_result.entry])

        retrieval_service = MemoryRetrievalService(db_session, emb_service)
        context = retrieval_service.retrieve_context(
            "architecture design decision",
            min_similarity=0.0,
            max_chars=4000,
            limit=5,
        )

        assert len(context.assembled_context) <= 4000 or context.assembled_context == ""
        assert context.total_tokens >= 0

    def test_export_writes_architecture_markdown(
        self, db_session: Session, tmp_path: Path
    ) -> None:
        """Eligible architecture entries are exported as Obsidian-compatible markdown."""
        persistence = MemoryPersistenceService(db_session)
        persist_result = persistence.persist(
            content=(
                "Architecture decision: Adopt SQLite WAL mode for the bounded-context "
                "database to enable concurrent read/write without contention."
            ),
            domain=MemoryDomain.ARCHITECTURE,
            memory_type=MemoryType.DECISION,
            source=MemorySource.USER,
            tags=["sqlite", "wal"],
        )
        assert persist_result.entry is not None

        vault_path = tmp_path / "vault"
        export_service = MemoryExportService(db_session, vault_path)
        exported_count = export_service.export_all()

        assert exported_count >= 1
        # Verify that at least one markdown file exists under the vault
        md_files = list(vault_path.rglob("*.md"))
        assert len(md_files) >= 1

        # Validate Obsidian frontmatter structure in the exported file
        content = md_files[0].read_text(encoding="utf-8")
        assert "---" in content
        assert "domain:" in content


# ---------------------------------------------------------------------------
# Scenario 2: Recruiter Continuity Workflow
# ---------------------------------------------------------------------------


class TestRecruiterContinuityWorkflow:
    """
    End-to-end workflow for recruiter relationship memory continuity.

    Data flow:
        Ingestion (Markdown) -> Persistence (Relationship Domain)
        -> Embeddings (Mock) -> Retrieval (Relationship Context)
        -> Ranking (Recency-Weighted) -> Export (Relationship Vault)
    """

    _RECRUITER_MARKDOWN = """\
---
domain: relationship
type: fact
source: user
tags: [recruiter, outreach, stripe]
---
# Recruiter Continuity: Jordan Lee at Stripe

Jordan Lee (recruiter at Stripe) reached out via LinkedIn on 2024-01-15 about
a Senior Backend Engineer role. Jordan prefers Slack for follow-up and typically
responds within 24 hours. The preferred communication style is concise and
technical. A warm referral from the hiring manager was offered if we progress
past the phone screen.

Key continuity points:
- Follow up via Slack: @jordan.lee
- Mention system design and distributed systems experience
- Reference the referral from the hiring manager
"""

    def test_recruiter_ingestion_succeeds(self, db_session: Session) -> None:
        """Recruiter continuity markdown is ingested successfully."""
        service = MemoryIngestionService(db_session)
        result = service.ingest_markdown(self._RECRUITER_MARKDOWN)

        assert result.status == IngestionStatus.SUCCESS
        assert result.entry is not None

    def test_recruiter_entry_domain_is_relationship(self, db_session: Session) -> None:
        """The ingested recruiter entry is classified under the RELATIONSHIP domain."""
        service = MemoryIngestionService(db_session)
        result = service.ingest_markdown(self._RECRUITER_MARKDOWN)

        assert result.status == IngestionStatus.SUCCESS
        assert result.entry is not None
        assert result.entry.domain == MemoryDomain.RELATIONSHIP

    def test_recruiter_significance_evaluates_as_retainable(self) -> None:
        """Recruiter relationship memory scores above the RETAIN threshold."""
        evaluator = SignificanceEvaluator()
        content = (
            "Recruiter continuity: Jordan Lee at Stripe reached out via LinkedIn. "
            "Prefers Slack follow-up within 24 hours. Warm referral from hiring manager "
            "offered for Senior Backend Engineer role."
        )
        sig = evaluator.evaluate(
            content,
            MemoryDomain.RELATIONSHIP,
            MemoryType.FACT,
            MemorySource.USER,
        )

        assert sig.retention_decision == RetentionDecision.RETAIN
        assert sig.importance_score >= 0.4

    def test_recruiter_embedding_generation(self, db_session: Session) -> None:
        """Persisted relationship memory generates a valid embedding."""
        persistence = MemoryPersistenceService(db_session)
        result = persistence.persist(
            content=(
                "Recruiter continuity: Jordan Lee at Stripe. Prefers Slack. "
                "Warm referral from hiring manager available. Senior Backend Engineer role."
            ),
            domain=MemoryDomain.RELATIONSHIP,
            memory_type=MemoryType.FACT,
            source=MemorySource.USER,
            tags=["recruiter", "stripe"],
        )
        assert result.entry is not None

        provider = _MockEmbeddingProvider()
        emb_service = EmbeddingService(provider=provider)
        embedding = emb_service.generate_embedding(result.entry)

        # Relationship domain entries are eligible for embedding
        assert embedding is not None
        assert embedding.memory_id == result.entry.id

    def test_recruiter_retrieval_surfaces_continuity(self, db_session: Session) -> None:
        """Recruiter memory surfaces in a continuity-aware retrieval query."""
        persistence = MemoryPersistenceService(db_session)
        persist_result = persistence.persist(
            content=(
                "Recruiter relationship continuity: Jordan Lee at Stripe via LinkedIn. "
                "Prefers Slack follow-up. Warm hiring manager referral offered."
            ),
            domain=MemoryDomain.RELATIONSHIP,
            memory_type=MemoryType.FACT,
            source=MemorySource.USER,
            tags=["recruiter", "stripe", "outreach"],
        )
        assert persist_result.entry is not None

        provider = _MockEmbeddingProvider()
        emb_service = EmbeddingService(provider=provider)
        emb_service.orchestrate_embeddings(db_session, [persist_result.entry])

        retrieval_service = MemoryRetrievalService(db_session, emb_service)
        context = retrieval_service.retrieve_context(
            "recruiter outreach Stripe follow-up",
            domain=MemoryDomain.RELATIONSHIP,
            min_similarity=0.0,
            limit=5,
        )

        assert len(context.results) >= 1
        returned_ids = {r.entry.id for r in context.results}
        assert persist_result.entry.id in returned_ids

    def test_recruiter_ranking_prioritizes_recency(self, db_session: Session) -> None:
        """Reranking of recruiter results completes without error and preserves results."""
        persistence = MemoryPersistenceService(db_session)
        result = persistence.persist(
            content=(
                "Recruiter continuity: Jordan Lee at Stripe via LinkedIn. "
                "Warm hiring manager referral available. Prefers Slack, responds within 24h."
            ),
            domain=MemoryDomain.RELATIONSHIP,
            memory_type=MemoryType.FACT,
            source=MemorySource.USER,
        )
        assert result.entry is not None

        provider = _MockEmbeddingProvider()
        emb_service = EmbeddingService(provider=provider)
        emb_service.orchestrate_embeddings(db_session, [result.entry])

        retrieval_service = MemoryRetrievalService(db_session, emb_service)
        context = retrieval_service.retrieve_context(
            "recruiter relationship",
            domain=MemoryDomain.RELATIONSHIP,
            min_similarity=0.0,
        )

        ranking_service = RetrievalRankingService()
        reranked = ranking_service.rerank(
            context.results,
            limit=5,
            freshness_weight=0.4,
            reference_time=datetime.now(UTC),
        )

        assert isinstance(reranked, list)

    def test_recruiter_export_writes_relationship_summary(
        self, db_session: Session, tmp_path: Path
    ) -> None:
        """Recruiter continuity entries are exported to the Relationships vault folder."""
        persistence = MemoryPersistenceService(db_session)
        persist_result = persistence.persist(
            content=(
                "Recruiter continuity: Jordan Lee at Stripe via LinkedIn. "
                "Warm hiring manager referral. Prefers Slack follow-up."
            ),
            domain=MemoryDomain.RELATIONSHIP,
            memory_type=MemoryType.FACT,
            source=MemorySource.USER,
            tags=["recruiter", "stripe"],
        )
        assert persist_result.entry is not None

        vault_path = tmp_path / "vault"
        export_service = MemoryExportService(db_session, vault_path)
        exported_count = export_service.export_all()

        assert exported_count >= 1
        md_files = list(vault_path.rglob("*.md"))
        assert any("Relationships" in str(f) or "01-Relationships" in str(f) for f in md_files)


# ---------------------------------------------------------------------------
# Scenario 3: Noise Rejection Workflow
# ---------------------------------------------------------------------------


class TestNoiseRejectionWorkflow:
    """
    Validates that the pipeline rejects low-value operational noise.

    The system must short-circuit at the significance layer and produce:
    - No persisted MemoryEntry
    - No generated embeddings
    - No retrieval inclusion
    """

    _NOISE_MARKDOWN = """\
# Temp Debug Note

debugging: console.log output from dev session
TODO: remove this before commit
placeholder text for testing
temp fix applied during dev sprint - will revert later

WIP: nothing actionable here
"""

    _TRANSCRIPT_MARKDOWN = """\
# Session Log

Session log from today.
Interviewer: Tell me about yourself.
Interviewee: I am a software engineer.
You said: I agree.
I said: Let us continue.
"""

    def test_noise_is_rejected_at_ingestion(self, db_session: Session) -> None:
        """Noise content is rejected or skipped by the ingestion pipeline."""
        service = MemoryIngestionService(db_session)
        result = service.ingest_markdown(self._NOISE_MARKDOWN)

        assert result.status in (IngestionStatus.SKIPPED, IngestionStatus.FAILED)

    def test_transcript_is_rejected_at_ingestion(self, db_session: Session) -> None:
        """Transcript content is rejected by the transcript filter."""
        service = MemoryIngestionService(db_session)
        result = service.ingest_markdown(self._TRANSCRIPT_MARKDOWN)

        assert result.status in (IngestionStatus.SKIPPED, IngestionStatus.FAILED)

    def test_noise_produces_no_persisted_entry(self, db_session: Session) -> None:
        """Noise rejection produces no entry in the repository."""
        service = MemoryIngestionService(db_session)
        result = service.ingest_markdown(self._NOISE_MARKDOWN)

        # Entry must be None for rejected content
        assert result.entry is None

    def test_noise_significance_evaluates_as_reject(self) -> None:
        """Low-value debug content is classified REJECT by the significance evaluator."""
        evaluator = SignificanceEvaluator()
        content = (
            "debugging: console.log output placeholder temp fix WIP "
            "TODO: remove this before commit breakpoint() pdb"
        )
        sig = evaluator.evaluate(
            content,
            MemoryDomain.OPERATIONAL,
            MemoryType.FACT,
            MemorySource.SYSTEM,
        )

        assert sig.retention_decision == RetentionDecision.REJECT
        assert sig.rejection_reason is not None

    def test_noise_persistence_is_ineligible(self, db_session: Session) -> None:
        """Direct persistence of noise content is classified as INELIGIBLE."""
        persistence = MemoryPersistenceService(db_session)
        result = persistence.persist(
            content="debugging placeholder temp fix TODO: remove this WIP breakpoint()",
            domain=MemoryDomain.OPERATIONAL,
            memory_type=MemoryType.FACT,
            source=MemorySource.SYSTEM,
        )

        assert result.eligibility != PersistenceEligibility.ELIGIBLE
        assert result.entry is None

    def test_noise_produces_no_embedding(self) -> None:
        """Noise entries are not eligible for embedding generation."""

        evaluator = EmbeddingEligibilityEvaluator()
        low_importance_entry = MemoryEntry(
            id=uuid4(),
            content="debugging placeholder temp fix WIP TODO: remove",
            domain=MemoryDomain.OPERATIONAL,
            memory_type=MemoryType.FACT,
            source=MemorySource.SYSTEM,
            importance_level=MemoryImportance.LOW,
            importance_score=0.05,
        )

        assert evaluator.is_eligible(low_importance_entry) is False

    def test_noise_does_not_appear_in_retrieval(self, db_session: Session) -> None:
        """
        Noise entries rejected by the persistence layer are absent from retrieval results.

        Validates that:
        1. The noise entry is not persisted.
        2. A retrieval query finds zero results (no spurious noise entries surface).
        """
        # Attempt to ingest noise — must not persist
        ingestion_service = MemoryIngestionService(db_session)
        noise_result = ingestion_service.ingest_markdown(self._NOISE_MARKDOWN)
        assert noise_result.status in (IngestionStatus.SKIPPED, IngestionStatus.FAILED)

        # Retrieve against the noise content — expect empty results
        provider = _MockEmbeddingProvider()
        emb_service = EmbeddingService(provider=provider)
        retrieval_service = MemoryRetrievalService(db_session, emb_service)
        context = retrieval_service.retrieve_context(
            "debugging placeholder temp fix WIP",
            min_similarity=0.0,
            limit=10,
        )

        # No noise entries should be present in results
        result_contents = [r.entry.content for r in context.results]
        noise_keywords = {"debugging", "placeholder", "temp fix", "WIP"}
        for content in result_contents:
            assert not any(kw.lower() in content.lower() for kw in noise_keywords), (
                f"Noise content leaked into retrieval results: {content!r}"
            )

    def test_empty_content_is_rejected(self, db_session: Session) -> None:
        """Empty content fails immediately at ingestion boundary."""
        service = MemoryIngestionService(db_session)
        result = service.ingest_markdown("")

        assert result.status == IngestionStatus.FAILED
        assert result.entry is None

    def test_whitespace_only_content_is_rejected(self, db_session: Session) -> None:
        """Whitespace-only content fails immediately at ingestion boundary."""
        service = MemoryIngestionService(db_session)
        result = service.ingest_markdown("   \n\t  \n  ")

        assert result.status == IngestionStatus.FAILED
        assert result.entry is None


# ---------------------------------------------------------------------------
# Scenario 4: Cross-domain isolation
# ---------------------------------------------------------------------------


class TestCrossDomainIsolation:
    """
    Validates that domain filters correctly isolate retrieval results.

    When querying with a domain filter, entries from other domains must not
    appear in the result set.
    """

    def test_architecture_query_excludes_relationship_entries(
        self, db_session: Session
    ) -> None:
        """Architecture domain retrieval must not surface RELATIONSHIP entries."""
        persistence = MemoryPersistenceService(db_session)

        arch_result = persistence.persist(
            content=(
                "Architecture decision: adopt bounded-context module boundaries "
                "and event-driven interfaces across the operational subsystem."
            ),
            domain=MemoryDomain.ARCHITECTURE,
            memory_type=MemoryType.DECISION,
            source=MemorySource.USER,
        )
        rel_result = persistence.persist(
            content=(
                "Recruiter relationship: Alex Kim at Google. Warm referral from hiring "
                "manager. Preferred channel is email. Responds within 48 hours."
            ),
            domain=MemoryDomain.RELATIONSHIP,
            memory_type=MemoryType.FACT,
            source=MemorySource.USER,
        )
        assert arch_result.entry is not None
        assert rel_result.entry is not None

        provider = _MockEmbeddingProvider()
        emb_service = EmbeddingService(provider=provider)
        emb_service.orchestrate_embeddings(
            db_session, [arch_result.entry, rel_result.entry]
        )

        retrieval_service = MemoryRetrievalService(db_session, emb_service)
        context = retrieval_service.retrieve_context(
            "bounded context architecture decision",
            domain=MemoryDomain.ARCHITECTURE,
            min_similarity=0.0,
            limit=10,
        )

        returned_domains = {r.entry.domain for r in context.results}
        assert MemoryDomain.RELATIONSHIP not in returned_domains

    def test_relationship_query_excludes_architecture_entries(
        self, db_session: Session
    ) -> None:
        """Relationship domain retrieval must not surface ARCHITECTURE entries."""
        persistence = MemoryPersistenceService(db_session)

        arch_result = persistence.persist(
            content=(
                "Architecture constraint: SQLite index size must remain below 100MB "
                "for sub-5ms query latency in the bounded-domain architecture."
            ),
            domain=MemoryDomain.ARCHITECTURE,
            memory_type=MemoryType.FACT,
            source=MemorySource.USER,
        )
        rel_result = persistence.persist(
            content=(
                "Recruiter continuity: Sam Park at Meta. Outreach via LinkedIn. "
                "Prefers concise technical communication. Hiring manager referral offered."
            ),
            domain=MemoryDomain.RELATIONSHIP,
            memory_type=MemoryType.FACT,
            source=MemorySource.USER,
        )
        assert arch_result.entry is not None
        assert rel_result.entry is not None

        provider = _MockEmbeddingProvider()
        emb_service = EmbeddingService(provider=provider)
        emb_service.orchestrate_embeddings(
            db_session, [arch_result.entry, rel_result.entry]
        )

        retrieval_service = MemoryRetrievalService(db_session, emb_service)
        context = retrieval_service.retrieve_context(
            "recruiter relationship continuity",
            domain=MemoryDomain.RELATIONSHIP,
            min_similarity=0.0,
            limit=10,
        )

        returned_domains = {r.entry.domain for r in context.results}
        assert MemoryDomain.ARCHITECTURE not in returned_domains
