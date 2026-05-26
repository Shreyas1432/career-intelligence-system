"""
Integration tests for the complete career intelligence workflow.

Validates the end-to-end data flow across:
    Relationship Intelligence -> Operational Memory -> Contextual Retrieval
    -> Continuity Tracking -> Follow-up Intelligence -> Communication Guidance

Scenarios
---------
1. Recruiter Relationship Workflow
   Create recruiter contact, track outreach, update state, generate follow-up
   recommendations, persist recruiter continuity memory, retrieve operational
   context via semantic search, rerank contextual memories, and generate
   communication guidance with continuity-aware recommendations.

2. Hiring Manager Workflow
   Create hiring manager contact, persist project/architecture continuity memory,
   retrieve architecture context, validate communication continuity, and validate
   retrieval prioritization by domain.

3. Stale Relationship Recovery Workflow
   Simulate a stale recruiter interaction, generate follow-up prioritization,
   retrieve historical continuity memory, and validate continuity-aware
   recommendations are surfaced correctly.

4. Cross-system Memory/Relationship Coordination
   Validate that the memory and relationship subsystems coordinate correctly
   within the same SQLite session without domain cross-contamination.

Design constraints
------------------
- All tests use the `db_session` fixture (transactional in-memory SQLite).
- Embedding provider replaced with a deterministic 4-dimensional mock.
- No live network calls, no external embedding APIs, no transcript persistence.
- All assertions are deterministic and do not depend on runtime randomness.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from src.modules.memory.embeddings import EmbeddingProvider, EmbeddingService
from src.modules.memory.export_sync import MemoryExportService
from src.modules.memory.ingestion import IngestionStatus, MemoryIngestionService
from src.modules.memory.persistence import MemoryPersistenceService, PersistenceEligibility
from src.modules.memory.ranking import RetrievalRankingService
from src.modules.memory.retrieval import MemoryRetrievalService
from src.modules.memory.schemas import (
    MemoryDomain,
    MemorySource,
    MemoryType,
)
from src.modules.relationship import (
    CommunicationProfileService,
    ContactCreate,
    ContactService,
    ContactType,
    FollowupRecommendationService,
    InteractionOutcome,
    OutreachEvent,
    OutreachStatus,
    OutreachTrackingService,
    RelationshipFreshnessEvaluator,
    RelationshipMemory,
    RelationshipStatus,
)
from src.modules.relationship.repositories import (
    OutreachEventRepository,
    RelationshipMemoryRepository,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class _MockEmbeddingProvider(EmbeddingProvider):
    """
    Deterministic 4-dimensional mock embedding provider.

    Vectors are derived from text length so different texts reliably produce
    distinct, non-zero, L2-normalizable vectors — no real model required.
    """

    _DIMENSION: int = 4

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        for text in texts:
            val = (len(text) % 100 + 1) / 100.0
            results.append([val, val * 1.5, val * 0.5, val * 2.0])
        return results

    @property
    def model_name(self) -> str:
        return "mock-4d"

    @property
    def dimension(self) -> int:
        return self._DIMENSION


def _build_emb_service() -> EmbeddingService:
    """Return an EmbeddingService backed by the deterministic mock provider."""
    return EmbeddingService(provider=_MockEmbeddingProvider())


def _build_retrieval_service(db_session: Session) -> MemoryRetrievalService:
    """Return a MemoryRetrievalService backed by the deterministic mock provider."""
    return MemoryRetrievalService(db_session, _build_emb_service())


def _orchestrate_embeddings(db_session: Session, *persist_results: object) -> None:
    """Orchestrate embeddings for one or more persistence results."""
    from src.modules.memory.persistence import PersistenceResult

    entries = []
    for result in persist_results:
        r: PersistenceResult = result  # type: ignore[assignment]
        if r.entry is not None:
            entries.append(r.entry)

    if entries:
        _build_emb_service().orchestrate_embeddings(db_session, entries)


# ---------------------------------------------------------------------------
# Scenario 1: Recruiter Relationship Workflow
# ---------------------------------------------------------------------------


class TestRecruiterRelationshipWorkflow:
    """
    End-to-end career intelligence workflow for a recruiter relationship.

    Data flow:
        ContactService (create) -> OutreachTrackingService (log events)
        -> RelationshipStateManager (state transition)
        -> FollowupRecommendationService (generate recommendation)
        -> MemoryIngestionService (persist continuity note)
        -> EmbeddingService (embed note)
        -> MemoryRetrievalService (retrieve context)
        -> RetrievalRankingService (rerank)
        -> CommunicationProfileService (generate guidance)
    """

    def test_contact_creation_and_normalization(self, db_session: Session) -> None:
        """Recruiter contact is created and fields are normalized deterministically."""
        service = ContactService(db_session)
        contact = service.create_contact(
            ContactCreate(
                first_name="  jordan  ",
                last_name="  lee  ",
                company="Stripe Inc.",
                contact_type=ContactType.RECRUITER,
                email="JORDAN@STRIPE.COM",
            )
        )

        assert contact.first_name == "Jordan"
        assert contact.last_name == "Lee"
        assert contact.company == "Stripe"
        assert contact.email == "jordan@stripe.com"
        assert contact.status == RelationshipStatus.NEW

    def test_outreach_event_transitions_state_to_contacted(self, db_session: Session) -> None:
        """Logging a sent outreach event transitions the contact to CONTACTED."""
        service = ContactService(db_session)
        tracking = OutreachTrackingService(db_session)

        contact = service.create_contact(
            ContactCreate(
                first_name="Jordan",
                last_name="Lee",
                company="Stripe",
                contact_type=ContactType.RECRUITER,
            )
        )

        tracking.log_outreach_event(
            contact_id=contact.id,
            method="LinkedIn",
            content="Hi Jordan, I'm interested in the Senior Backend Engineer role at Stripe.",
            completed_at=datetime.now(UTC) - timedelta(days=3),
        )

        updated = service.get_contact(contact.id)
        assert updated is not None
        assert updated.status == RelationshipStatus.CONTACTED

    def test_positive_response_transitions_state_to_active(self, db_session: Session) -> None:
        """A positive recruiter reply transitions state from CONTACTED to ACTIVE."""
        service = ContactService(db_session)
        tracking = OutreachTrackingService(db_session)

        contact = service.create_contact(
            ContactCreate(
                first_name="Jordan",
                last_name="Lee",
                company="Stripe",
                contact_type=ContactType.RECRUITER,
            )
        )

        tracking.log_outreach_event(
            contact_id=contact.id,
            method="LinkedIn",
            content="Hi Jordan, reaching out about the Backend role.",
            completed_at=datetime.now(UTC) - timedelta(days=2),
        )
        tracking.log_response(
            contact_id=contact.id,
            method="LinkedIn",
            content="Thanks for reaching out! Let's schedule a call.",
            outcome=InteractionOutcome.POSITIVE,
        )

        updated = service.get_contact(contact.id)
        assert updated is not None
        assert updated.status == RelationshipStatus.ACTIVE

    def test_followup_recommendation_generated(self, db_session: Session) -> None:
        """A follow-up recommendation is generated for a due recruiter contact."""
        service = ContactService(db_session)
        tracking = OutreachTrackingService(db_session)
        followup_service = FollowupRecommendationService(db_session)
        event_repo = OutreachEventRepository(db_session)

        contact = service.create_contact(
            ContactCreate(
                first_name="Jordan",
                last_name="Lee",
                company="Stripe",
                contact_type=ContactType.RECRUITER,
            )
        )

        # Positive reply 8 days ago — past the recruiter follow-up window
        event_repo.create_event(
            OutreachEvent(
                contact_id=contact.id,
                status=OutreachStatus.REPLIED,
                method="LinkedIn",
                outcome=InteractionOutcome.POSITIVE,
                completed_at=datetime.now(UTC) - timedelta(days=8),
            )
        )
        tracking._sync_relationship_state(contact.id)

        recommendations = followup_service.generate_recommendations()

        assert len(recommendations) >= 1
        rec = next(r for r in recommendations if r.contact_id == contact.id)
        assert rec.draft_message is not None
        assert len(rec.draft_message) > 0
        assert rec.priority in range(1, 6)

    def test_recruiter_continuity_memory_persisted(self, db_session: Session) -> None:
        """Recruiter continuity markdown is accepted by the ingestion pipeline."""
        ingestion = MemoryIngestionService(db_session)

        recruiter_note = """\
---
domain: relationship
type: fact
source: user
tags: [recruiter, stripe, continuity]
---
# Recruiter Continuity: Jordan Lee at Stripe

Jordan Lee (Senior Technical Recruiter at Stripe) made contact via LinkedIn.
Prefers Slack for follow-ups, responds within 24 hours. A warm referral from
the hiring manager is available. The role is Senior Backend Engineer.

Key continuity points:
- Follow-up via Slack preferred
- Mention distributed systems and high-availability architecture
- Referral from the hiring manager available
"""
        result = ingestion.ingest_markdown(recruiter_note)

        assert result.status == IngestionStatus.SUCCESS
        assert result.entry is not None
        assert result.entry.domain == MemoryDomain.RELATIONSHIP

    def test_retrieval_surfaces_recruiter_continuity(self, db_session: Session) -> None:
        """Persisted recruiter memory surfaces in a continuity-aware retrieval query."""
        persistence = MemoryPersistenceService(db_session)
        persist_result = persistence.persist(
            content=(
                "Recruiter continuity: Jordan Lee at Stripe via LinkedIn. "
                "Prefers Slack follow-up within 24 hours. "
                "Warm hiring manager referral available for Senior Backend Engineer role."
            ),
            domain=MemoryDomain.RELATIONSHIP,
            memory_type=MemoryType.FACT,
            source=MemorySource.USER,
            tags=["recruiter", "stripe", "continuity"],
        )
        assert persist_result.eligibility == PersistenceEligibility.ELIGIBLE
        assert persist_result.entry is not None

        _orchestrate_embeddings(db_session, persist_result)

        retrieval = _build_retrieval_service(db_session)
        context = retrieval.retrieve_context(
            "recruiter Jordan Stripe follow-up continuity",
            domain=MemoryDomain.RELATIONSHIP,
            min_similarity=0.0,
            limit=5,
        )

        returned_ids = {r.entry.id for r in context.results}
        assert persist_result.entry.id in returned_ids

    def test_retrieval_results_are_reranked(self, db_session: Session) -> None:
        """Retrieval results for recruiter context are successfully reranked."""
        persistence = MemoryPersistenceService(db_session)

        r1 = persistence.persist(
            content=(
                "Recruiter continuity: Jordan Lee at Stripe. "
                "Prefers Slack. Warm referral from hiring manager available."
            ),
            domain=MemoryDomain.RELATIONSHIP,
            memory_type=MemoryType.FACT,
            source=MemorySource.USER,
        )
        r2 = persistence.persist(
            content=(
                "Recruiter relationship: Alex Kim at Google via LinkedIn. "
                "Responsive via email. Senior SRE role — Kubernetes and Python required."
            ),
            domain=MemoryDomain.RELATIONSHIP,
            memory_type=MemoryType.FACT,
            source=MemorySource.USER,
        )
        assert r1.entry is not None
        assert r2.entry is not None

        _orchestrate_embeddings(db_session, r1, r2)

        retrieval = _build_retrieval_service(db_session)
        context = retrieval.retrieve_context(
            "recruiter relationship continuity follow-up",
            domain=MemoryDomain.RELATIONSHIP,
            min_similarity=0.0,
            limit=10,
        )

        ranker = RetrievalRankingService()
        reranked = ranker.rerank(
            context.results,
            limit=5,
            freshness_weight=0.3,
            reference_time=datetime.now(UTC),
        )

        assert isinstance(reranked, list)
        if context.results:
            assert len(reranked) >= 1

    def test_communication_guidance_generated(self, db_session: Session) -> None:
        """CommunicationProfileService produces recruiter-specific guidance."""
        service = ContactService(db_session)
        tracking = OutreachTrackingService(db_session)
        comm_service = CommunicationProfileService(db_session)

        contact = service.create_contact(
            ContactCreate(
                first_name="Jordan",
                last_name="Lee",
                company="Stripe",
                contact_type=ContactType.RECRUITER,
            )
        )

        tracking.log_outreach_event(
            contact_id=contact.id,
            method="LinkedIn",
            content="Hi Jordan, let us discuss the recruiter process at Stripe.",
            completed_at=datetime.now(UTC) - timedelta(days=1),
        )

        guidance = comm_service.generate_guidance(contact.id)

        assert guidance.contact_id == contact.id
        assert guidance.style_preference in ("concise", "detailed")
        assert guidance.orientation in ("business", "technical")
        assert any("Recruiter" in g for g in guidance.tone_guidance)
        assert len(guidance.continuity_recommendations) >= 1

    def test_communication_profile_reflects_engagement(self, db_session: Session) -> None:
        """Communication profile reflects updated engagement after a recruiter reply."""
        service = ContactService(db_session)
        comm_service = CommunicationProfileService(db_session)
        event_repo = OutreachEventRepository(db_session)

        contact = service.create_contact(
            ContactCreate(
                first_name="Jordan",
                last_name="Lee",
                company="Stripe",
                contact_type=ContactType.RECRUITER,
            )
        )

        event_repo.create_event(
            OutreachEvent(
                contact_id=contact.id,
                status=OutreachStatus.SENT,
                method="LinkedIn",
                content="Initial outreach to Jordan at Stripe.",
                completed_at=datetime.now(UTC) - timedelta(days=5),
            )
        )
        event_repo.create_event(
            OutreachEvent(
                contact_id=contact.id,
                status=OutreachStatus.REPLIED,
                method="LinkedIn",
                content="Great to hear from you. Let's schedule a call.",
                outcome=InteractionOutcome.POSITIVE,
                completed_at=datetime.now(UTC) - timedelta(days=4),
            )
        )

        profile = comm_service.analyze_and_sync_profile(contact.id)

        # 1 reply out of 2 events → engagement score == 0.5
        assert profile.engagement_score == pytest.approx(0.5)
        assert len(profile.insights) >= 1


# ---------------------------------------------------------------------------
# Scenario 2: Hiring Manager Workflow
# ---------------------------------------------------------------------------


class TestHiringManagerWorkflow:
    """
    End-to-end career intelligence workflow for a hiring manager relationship.

    Data flow:
        ContactService (create HM) -> MemoryPersistenceService (persist
        architecture/project memory) -> EmbeddingService (embed)
        -> MemoryRetrievalService (retrieve architecture context)
        -> CommunicationProfileService (guidance)
        -> Domain isolation validation
    """

    def test_hiring_manager_contact_created(self, db_session: Session) -> None:
        """Hiring manager contact is created with the correct type and NEW status."""
        service = ContactService(db_session)
        contact = service.create_contact(
            ContactCreate(
                first_name="Sam",
                last_name="Chen",
                company="Anthropic",
                title="Engineering Manager",
                contact_type=ContactType.HIRING_MANAGER,
                email="sam.chen@anthropic.com",
            )
        )

        assert contact.first_name == "Sam"
        assert contact.last_name == "Chen"
        assert contact.contact_type == ContactType.HIRING_MANAGER
        assert contact.status == RelationshipStatus.NEW

    def test_architecture_memory_persisted_for_hm_context(self, db_session: Session) -> None:
        """Architecture/project continuity memory is persisted for HM retrieval."""
        persistence = MemoryPersistenceService(db_session)
        result = persistence.persist(
            content=(
                "Architecture context for Anthropic HM interview: Sam Chen's team owns "
                "the inference serving layer. Key technical interests are distributed "
                "systems, low-latency inference pipelines, and bounded-domain architecture. "
                "Emphasis on system design and multi-tenant API reliability."
            ),
            domain=MemoryDomain.ARCHITECTURE,
            memory_type=MemoryType.DECISION,
            source=MemorySource.USER,
            tags=["hiring-manager", "anthropic", "architecture"],
        )

        assert result.eligibility == PersistenceEligibility.ELIGIBLE
        assert result.entry is not None
        assert result.entry.domain == MemoryDomain.ARCHITECTURE

    def test_architecture_context_retrieved_for_hm(self, db_session: Session) -> None:
        """Architecture memory for HM context surfaces in architecture retrieval."""
        persistence = MemoryPersistenceService(db_session)
        persist_result = persistence.persist(
            content=(
                "Architecture context for hiring manager interview: distributed inference "
                "pipeline, bounded-domain architecture, low-latency API reliability. "
                "Key project: multi-tenant serving layer with sub-10ms p99 latency."
            ),
            domain=MemoryDomain.ARCHITECTURE,
            memory_type=MemoryType.DECISION,
            source=MemorySource.USER,
            tags=["hiring-manager", "architecture"],
        )
        assert persist_result.entry is not None

        _orchestrate_embeddings(db_session, persist_result)

        retrieval = _build_retrieval_service(db_session)
        context = retrieval.retrieve_context(
            "architecture distributed inference pipeline hiring manager",
            domain=MemoryDomain.ARCHITECTURE,
            min_similarity=0.0,
            limit=5,
        )

        returned_ids = {r.entry.id for r in context.results}
        assert persist_result.entry.id in returned_ids

    def test_relationship_memory_saved_for_hm(self, db_session: Session) -> None:
        """Relationship memory (key facts, pain points) is persisted for HM continuity."""
        service = ContactService(db_session)
        memory_repo = RelationshipMemoryRepository(db_session)

        contact = service.create_contact(
            ContactCreate(
                first_name="Sam",
                last_name="Chen",
                company="Anthropic",
                contact_type=ContactType.HIRING_MANAGER,
            )
        )

        memory = RelationshipMemory(
            contact_id=contact.id,
            key_facts=[
                "Owns inference serving team at Anthropic",
                "Prefers technical depth discussions",
                "Values bounded-domain architecture patterns",
            ],
            shared_interests=["distributed systems", "LLM inference optimization"],
            pain_points=["latency spikes under multi-tenant load"],
            last_interaction_date=datetime.now(UTC) - timedelta(days=3),
        )
        saved = memory_repo.save_memory(memory)

        assert saved.contact_id == contact.id
        assert len(saved.key_facts) == 3
        assert len(saved.shared_interests) == 2
        assert len(saved.pain_points) == 1

    def test_hm_communication_guidance_is_technical(self, db_session: Session) -> None:
        """Hiring manager communication guidance emphasizes technical orientation."""
        service = ContactService(db_session)
        tracking = OutreachTrackingService(db_session)
        comm_service = CommunicationProfileService(db_session)
        event_repo = OutreachEventRepository(db_session)

        contact = service.create_contact(
            ContactCreate(
                first_name="Sam",
                last_name="Chen",
                company="Anthropic",
                contact_type=ContactType.HIRING_MANAGER,
            )
        )

        event_repo.create_event(
            OutreachEvent(
                contact_id=contact.id,
                status=OutreachStatus.REPLIED,
                method="Email",
                content=(
                    "Sam, I'd love to discuss distributed system architecture, "
                    "database engineering, backend system design, and API development "
                    "patterns used in the inference pipeline at Anthropic."
                ),
                outcome=InteractionOutcome.POSITIVE,
                completed_at=datetime.now(UTC) - timedelta(days=2),
            )
        )
        tracking._sync_relationship_state(contact.id)

        guidance = comm_service.generate_guidance(contact.id)

        assert guidance.contact_id == contact.id
        assert guidance.orientation == "technical"
        assert any("Hiring Manager" in g for g in guidance.tone_guidance)

    def test_hm_retrieval_excludes_relationship_domain(self, db_session: Session) -> None:
        """Architecture domain retrieval for HM context excludes RELATIONSHIP entries."""
        persistence = MemoryPersistenceService(db_session)

        arch_result = persistence.persist(
            content=(
                "Architecture context for HM interview: inference serving bounded-domain "
                "design, sub-10ms latency target, distributed system reliability patterns."
            ),
            domain=MemoryDomain.ARCHITECTURE,
            memory_type=MemoryType.DECISION,
            source=MemorySource.USER,
        )
        rel_result = persistence.persist(
            content=(
                "Recruiter contact: Alex Kim at Google. Prefers Slack. "
                "Warm referral available from hiring manager."
            ),
            domain=MemoryDomain.RELATIONSHIP,
            memory_type=MemoryType.FACT,
            source=MemorySource.USER,
        )
        assert arch_result.entry is not None
        assert rel_result.entry is not None

        _orchestrate_embeddings(db_session, arch_result, rel_result)

        retrieval = _build_retrieval_service(db_session)
        context = retrieval.retrieve_context(
            "architecture design bounded context inference HM",
            domain=MemoryDomain.ARCHITECTURE,
            min_similarity=0.0,
            limit=10,
        )

        returned_domains = {r.entry.domain for r in context.results}
        assert MemoryDomain.RELATIONSHIP not in returned_domains

    def test_hm_scores_higher_than_recruiter(self, db_session: Session) -> None:
        """Hiring managers receive a higher relationship priority score than recruiters."""
        service = ContactService(db_session)
        event_repo = OutreachEventRepository(db_session)

        hm_contact = service.create_contact(
            ContactCreate(
                first_name="Sam",
                company="Anthropic",
                contact_type=ContactType.HIRING_MANAGER,
            )
        )
        recruiter_contact = service.create_contact(
            ContactCreate(
                first_name="Jordan",
                company="Anthropic",
                contact_type=ContactType.RECRUITER,
            )
        )

        for cid in (hm_contact.id, recruiter_contact.id):
            event_repo.create_event(
                OutreachEvent(
                    contact_id=cid,
                    status=OutreachStatus.REPLIED,
                    method="LinkedIn",
                    outcome=InteractionOutcome.POSITIVE,
                    completed_at=datetime.now(UTC) - timedelta(days=5),
                )
            )

        hm_score = service.score_relationship(hm_contact.id, target_companies=["Anthropic"])
        recruiter_score = service.score_relationship(
            recruiter_contact.id, target_companies=["Anthropic"]
        )

        assert hm_score is not None
        assert recruiter_score is not None
        # HM role = 30 pts; recruiter role = 20 pts → HM must score higher
        assert hm_score.total_score > recruiter_score.total_score


# ---------------------------------------------------------------------------
# Scenario 3: Stale Relationship Recovery Workflow
# ---------------------------------------------------------------------------


class TestStaleRelationshipRecoveryWorkflow:
    """
    Validates the stale relationship recovery path through the career intelligence pipeline.

    Data flow:
        ContactService (create) -> OutreachEventRepository (stale event)
        -> OutreachTrackingService (state sync to STALE)
        -> FollowupRecommendationService (prioritize stale contacts)
        -> MemoryIngestionService (persist historical note)
        -> MemoryRetrievalService (retrieve historical context)
        -> CommunicationProfileService (continuity guidance)
    """

    def test_stale_relationship_detected(self, db_session: Session) -> None:
        """A contact with a 45-day-old interaction is classified as STALE."""
        service = ContactService(db_session)
        tracking = OutreachTrackingService(db_session)
        event_repo = OutreachEventRepository(db_session)

        contact = service.create_contact(
            ContactCreate(
                first_name="Alex",
                last_name="Rivera",
                company="OpenAI",
                contact_type=ContactType.RECRUITER,
            )
        )

        event_repo.create_event(
            OutreachEvent(
                contact_id=contact.id,
                status=OutreachStatus.SENT,
                method="LinkedIn",
                content="Initial outreach to Alex Rivera at OpenAI.",
                completed_at=datetime.now(UTC) - timedelta(days=45),
            )
        )

        tracking._sync_relationship_state(contact.id)
        updated = service.get_contact(contact.id)

        assert updated is not None
        assert updated.status == RelationshipStatus.STALE

    def test_stale_contact_is_followup_prioritized(self, db_session: Session) -> None:
        """Stale contacts appear in follow-up candidates with elevated urgency scores."""
        service = ContactService(db_session)
        followup_service = FollowupRecommendationService(db_session)
        event_repo = OutreachEventRepository(db_session)

        contact = service.create_contact(
            ContactCreate(
                first_name="Alex",
                last_name="Rivera",
                company="OpenAI",
                contact_type=ContactType.RECRUITER,
            )
        )

        event_repo.create_event(
            OutreachEvent(
                contact_id=contact.id,
                status=OutreachStatus.SENT,
                method="LinkedIn",
                completed_at=datetime.now(UTC) - timedelta(days=40),
            )
        )

        candidates = followup_service.retrieve_followup_candidates()
        alex_candidate = next(c for c in candidates if c.contact.id == contact.id)

        assert alex_candidate.is_stale is True
        assert alex_candidate.urgency_score > 50.0

    def test_stale_followup_recommendation_generated(self, db_session: Session) -> None:
        """A follow-up recommendation is generated for a stale recruiter contact."""
        service = ContactService(db_session)
        followup_service = FollowupRecommendationService(db_session)
        event_repo = OutreachEventRepository(db_session)

        contact = service.create_contact(
            ContactCreate(
                first_name="Alex",
                last_name="Rivera",
                company="OpenAI",
                contact_type=ContactType.RECRUITER,
            )
        )

        event_repo.create_event(
            OutreachEvent(
                contact_id=contact.id,
                status=OutreachStatus.SENT,
                method="LinkedIn",
                outcome=InteractionOutcome.NO_RESPONSE,
                completed_at=datetime.now(UTC) - timedelta(days=40),
            )
        )

        recs = followup_service.generate_recommendations()

        assert len(recs) >= 1
        rec = next(r for r in recs if r.contact_id == contact.id)
        assert rec.draft_message is not None
        assert any(
            kw in rec.draft_message.lower()
            for kw in ("nudge", "follow up", "chance to review", "check")
        )

    def test_historical_memory_retrieved_for_stale_contact(self, db_session: Session) -> None:
        """Historical continuity memory for a stale contact is surfaced by retrieval."""
        persistence = MemoryPersistenceService(db_session)
        persist_result = persistence.persist(
            content=(
                "Historical recruiter continuity: Alex Rivera at OpenAI — initial outreach "
                "45 days ago via LinkedIn. No response received. Role was Senior ML Engineer "
                "focusing on distributed inference and Python-based pipeline orchestration. "
                "Follow-up strategy: re-engage with a brief technical project update."
            ),
            domain=MemoryDomain.RELATIONSHIP,
            memory_type=MemoryType.FACT,
            source=MemorySource.USER,
            tags=["recruiter", "openai", "stale", "historical"],
        )
        assert persist_result.entry is not None

        _orchestrate_embeddings(db_session, persist_result)

        retrieval = _build_retrieval_service(db_session)
        context = retrieval.retrieve_context(
            "stale recruiter OpenAI historical continuity re-engage",
            domain=MemoryDomain.RELATIONSHIP,
            min_similarity=0.0,
            limit=5,
        )

        returned_ids = {r.entry.id for r in context.results}
        assert persist_result.entry.id in returned_ids

    def test_freshness_evaluator_classifies_stale(self) -> None:
        """RelationshipFreshnessEvaluator classifies a 45-day-old contact as stale."""
        stale_date = datetime.now(UTC) - timedelta(days=45)
        classification = RelationshipFreshnessEvaluator.classify_relationship(stale_date)
        freshness = RelationshipFreshnessEvaluator.calculate_freshness_score(stale_date)

        assert classification == "stale"
        assert freshness < 0.3

    def test_continuity_guidance_for_stale_contact(self, db_session: Session) -> None:
        """CommunicationProfileService generates a re-engagement recommendation for stale contacts."""
        service = ContactService(db_session)
        event_repo = OutreachEventRepository(db_session)
        comm_service = CommunicationProfileService(db_session)

        contact = service.create_contact(
            ContactCreate(
                first_name="Alex",
                last_name="Rivera",
                company="OpenAI",
                contact_type=ContactType.RECRUITER,
            )
        )

        event_repo.create_event(
            OutreachEvent(
                contact_id=contact.id,
                status=OutreachStatus.SENT,
                method="LinkedIn",
                content="Initial outreach to Alex Rivera at OpenAI about the ML Engineer role.",
                outcome=InteractionOutcome.NO_RESPONSE,
                completed_at=datetime.now(UTC) - timedelta(days=40),
            )
        )

        guidance = comm_service.generate_guidance(contact.id)

        assert guidance.contact_id == contact.id
        assert len(guidance.continuity_recommendations) >= 1
        combined = " ".join(guidance.continuity_recommendations).lower()
        assert any(
            kw in combined
            for kw in ("nudge", "unanswered", "follow up", "days ago", "check in")
        )

    def test_stale_relationship_memory_ingestion(self, db_session: Session) -> None:
        """Stale relationship continuity note is accepted by the memory pipeline."""
        ingestion = MemoryIngestionService(db_session)

        stale_note = """\
---
domain: relationship
type: fact
source: user
tags: [recruiter, openai, stale, re-engagement]
---
# Historical Recruiter Continuity: Alex Rivera at OpenAI

Alex Rivera (Senior Technical Recruiter at OpenAI) was contacted via LinkedIn 45 days ago
regarding the Senior ML Engineer role. No response was received. The role focuses on
distributed inference and Python-based pipeline orchestration.

Re-engagement strategy:
- Reference a recent technical blog post or system design update
- Keep the follow-up brief and non-intrusive
- Offer a new angle: a relevant project or open-source contribution
"""
        result = ingestion.ingest_markdown(stale_note)

        assert result.status == IngestionStatus.SUCCESS
        assert result.entry is not None
        assert result.entry.domain == MemoryDomain.RELATIONSHIP


# ---------------------------------------------------------------------------
# Scenario 4: Cross-system Memory / Relationship Coordination
# ---------------------------------------------------------------------------


class TestMemoryRelationshipCoordination:
    """
    Validates that the memory and relationship subsystems coordinate correctly.

    - Relationship contact records hold metadata and interaction history.
    - Memory entries hold semantic continuity notes for retrieval.
    - Both must remain consistent within the same SQLite transaction.
    - Domain filters must be mutually exclusive across retrieval.
    """

    def test_recruiter_memory_and_contact_coexist(self, db_session: Session) -> None:
        """A recruiter contact and a memory entry coexist within the same session."""
        contact_service = ContactService(db_session)
        persistence = MemoryPersistenceService(db_session)
        memory_repo = RelationshipMemoryRepository(db_session)

        contact = contact_service.create_contact(
            ContactCreate(
                first_name="Taylor",
                last_name="Park",
                company="DeepMind",
                contact_type=ContactType.RECRUITER,
            )
        )

        relationship_mem = RelationshipMemory(
            contact_id=contact.id,
            key_facts=["Recruiter at DeepMind", "Prefers email", "Warm HM referral"],
            shared_interests=["reinforcement learning", "agents"],
            pain_points=["long hiring pipeline"],
            last_interaction_date=datetime.now(UTC) - timedelta(days=2),
        )
        saved_rel_mem = memory_repo.save_memory(relationship_mem)

        mem_result = persistence.persist(
            content=(
                "Recruiter continuity: Taylor Park at DeepMind via email. "
                "Warm referral from hiring manager. Senior Research Engineer role. "
                "Prefers concise, technical communication."
            ),
            domain=MemoryDomain.RELATIONSHIP,
            memory_type=MemoryType.FACT,
            source=MemorySource.USER,
        )

        assert saved_rel_mem.contact_id == contact.id
        assert mem_result.eligibility == PersistenceEligibility.ELIGIBLE
        assert mem_result.entry is not None

    def test_relationship_memory_isolated_from_architecture_retrieval(
        self, db_session: Session
    ) -> None:
        """Operational memory for relationships does not pollute architecture retrieval."""
        persistence = MemoryPersistenceService(db_session)

        rel_result = persistence.persist(
            content=(
                "Recruiter continuity: Taylor Park at DeepMind. "
                "Warm referral from hiring manager. Prefers email."
            ),
            domain=MemoryDomain.RELATIONSHIP,
            memory_type=MemoryType.FACT,
            source=MemorySource.USER,
        )
        arch_result = persistence.persist(
            content=(
                "Architecture decision: adopt bounded-domain module boundaries "
                "for the inference serving layer at DeepMind research platform."
            ),
            domain=MemoryDomain.ARCHITECTURE,
            memory_type=MemoryType.DECISION,
            source=MemorySource.USER,
        )
        assert rel_result.entry is not None
        assert arch_result.entry is not None

        _orchestrate_embeddings(db_session, rel_result, arch_result)

        retrieval = _build_retrieval_service(db_session)
        context = retrieval.retrieve_context(
            "bounded domain architecture inference serving",
            domain=MemoryDomain.ARCHITECTURE,
            min_similarity=0.0,
            limit=10,
        )

        returned_domains = {r.entry.domain for r in context.results}
        assert MemoryDomain.RELATIONSHIP not in returned_domains

    def test_token_efficient_context_within_budget(self, db_session: Session) -> None:
        """Context assembly stays within the 4000-char budget across multiple entries."""
        persistence = MemoryPersistenceService(db_session)

        entries = []
        for i in range(3):
            r = persistence.persist(
                content=(
                    f"Recruiter relationship continuity note {i}: contact at company {i}. "
                    "Warm referral from hiring manager. Prefers Slack follow-up."
                ),
                domain=MemoryDomain.RELATIONSHIP,
                memory_type=MemoryType.FACT,
                source=MemorySource.USER,
            )
            if r.entry is not None:
                entries.append(r.entry)

        _build_emb_service().orchestrate_embeddings(db_session, entries)

        retrieval = _build_retrieval_service(db_session)
        context = retrieval.retrieve_context(
            "recruiter relationship continuity",
            domain=MemoryDomain.RELATIONSHIP,
            min_similarity=0.0,
            max_chars=4000,
            limit=5,
        )

        assert len(context.assembled_context) <= 4000 or context.assembled_context == ""
        assert context.total_tokens >= 0

    def test_contact_relationship_score_with_memory_context(
        self, db_session: Session
    ) -> None:
        """RelationshipScorer correctly uses memory + event context in scoring."""
        service = ContactService(db_session)
        event_repo = OutreachEventRepository(db_session)
        memory_repo = RelationshipMemoryRepository(db_session)

        contact = service.create_contact(
            ContactCreate(
                first_name="Taylor",
                last_name="Park",
                company="DeepMind",
                contact_type=ContactType.RECRUITER,
            )
        )

        event_repo.create_event(
            OutreachEvent(
                contact_id=contact.id,
                status=OutreachStatus.REPLIED,
                method="Email",
                outcome=InteractionOutcome.POSITIVE,
                completed_at=datetime.now(UTC) - timedelta(days=3),
            )
        )

        memory = RelationshipMemory(
            contact_id=contact.id,
            key_facts=["Senior Recruiter at DeepMind"],
            last_interaction_date=datetime.now(UTC) - timedelta(days=3),
        )
        memory_repo.save_memory(memory)

        score = service.score_relationship(contact.id, target_companies=["DeepMind"])

        assert score is not None
        assert score.total_score > 0.0
        # Recruiter (20) + Active status (25) + target company (20) + recent (15) ≥ 40
        assert score.total_score >= 40.0

    def test_vault_export_integrates_with_contact_workflow(
        self, db_session: Session, tmp_path: Path
    ) -> None:
        """Export service writes recruiter relationship memory to the Obsidian vault."""
        persistence = MemoryPersistenceService(db_session)
        persist_result = persistence.persist(
            content=(
                "Recruiter continuity: Taylor Park at DeepMind via email. "
                "Warm referral from hiring manager. Prefers concise technical communication."
            ),
            domain=MemoryDomain.RELATIONSHIP,
            memory_type=MemoryType.FACT,
            source=MemorySource.USER,
            tags=["recruiter", "deepmind"],
        )
        assert persist_result.entry is not None

        vault_path = tmp_path / "vault"
        export_service = MemoryExportService(db_session, vault_path)
        exported_count = export_service.export_all()

        assert exported_count >= 1
        md_files = list(vault_path.rglob("*.md"))
        assert len(md_files) >= 1
        assert any(
            "Relationships" in str(f) or "01-Relationships" in str(f)
            for f in md_files
        )

    def test_relationship_scorer_uses_comm_profile_engagement(
        self, db_session: Session
    ) -> None:
        """RelationshipScorer's engagement component incorporates communication profile data."""
        service = ContactService(db_session)
        comm_service = CommunicationProfileService(db_session)
        event_repo = OutreachEventRepository(db_session)

        contact = service.create_contact(
            ContactCreate(
                first_name="Taylor",
                last_name="Park",
                company="DeepMind",
                contact_type=ContactType.RECRUITER,
            )
        )

        # Log two sent events and one replied event
        for i in range(2):
            event_repo.create_event(
                OutreachEvent(
                    contact_id=contact.id,
                    status=OutreachStatus.SENT,
                    method="Email",
                    completed_at=datetime.now(UTC) - timedelta(days=10 - i),
                )
            )
        event_repo.create_event(
            OutreachEvent(
                contact_id=contact.id,
                status=OutreachStatus.REPLIED,
                method="Email",
                outcome=InteractionOutcome.POSITIVE,
                completed_at=datetime.now(UTC) - timedelta(days=3),
            )
        )

        # Sync communication profile so engagement score is stored
        comm_service.analyze_and_sync_profile(contact.id)

        score = service.score_relationship(contact.id)

        assert score is not None
        # Engagement score component must be > 0 after positive reply
        assert score.engagement_score > 0.0
