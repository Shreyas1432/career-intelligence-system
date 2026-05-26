from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from src.modules.relationship import (
    ContactCreate,
    ContactResponse,
    ContactService,
    ContactType,
    ContactUpdate,
    FollowupCandidate,
    FollowupPriorityScorer,
    FollowupRecommendationService,
    FollowupWindowCalculator,
    InteractionOutcome,
    OutreachEvent,
    OutreachStatus,
    RelationshipFreshnessEvaluator,
    RelationshipStatus,
)
from src.modules.relationship.repositories import OutreachEventRepository


def test_relationship_freshness_evaluator() -> None:
    """Validate that freshness score decays linearly and classifies active/stale correctly."""
    now_ts = datetime.now(UTC)

    # 1. No interaction
    assert RelationshipFreshnessEvaluator.calculate_freshness_score(None) == 0.0
    assert RelationshipFreshnessEvaluator.classify_relationship(None) == "stale"

    # 2. Fresh interaction (today)
    assert RelationshipFreshnessEvaluator.calculate_freshness_score(now_ts) == 1.0
    assert RelationshipFreshnessEvaluator.classify_relationship(now_ts) == "active"

    # 3. Halfway decayed (15 days ago)
    date_15_days_ago = now_ts - timedelta(days=15)
    assert pytest.approx(RelationshipFreshnessEvaluator.calculate_freshness_score(date_15_days_ago), abs=0.05) == 0.5
    assert RelationshipFreshnessEvaluator.classify_relationship(date_15_days_ago) == "active"

    # 4. Completely stale (31 days ago)
    date_31_days_ago = now_ts - timedelta(days=31)
    assert RelationshipFreshnessEvaluator.calculate_freshness_score(date_31_days_ago) == 0.0
    assert RelationshipFreshnessEvaluator.classify_relationship(date_31_days_ago) == "stale"


def test_followup_window_calculator() -> None:
    """Validate that follow-up windows are calculated based on role and outcome."""
    recruiter = ContactResponse(
        id=uuid4(),
        first_name="Jane",
        contact_type=ContactType.RECRUITER,
        status=RelationshipStatus.NEW,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    hm = ContactResponse(
        id=uuid4(),
        first_name="Bob",
        contact_type=ContactType.HIRING_MANAGER,
        status=RelationshipStatus.ACTIVE,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    peer = ContactResponse(
        id=uuid4(),
        first_name="Alice",
        contact_type=ContactType.PEER,
        status=RelationshipStatus.WARM,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )

    # Recruiter outcomes
    assert FollowupWindowCalculator.calculate_recommended_window(recruiter, InteractionOutcome.ACTION_REQUIRED) == (1, 2)
    assert FollowupWindowCalculator.calculate_recommended_window(recruiter, InteractionOutcome.POSITIVE) == (2, 4)
    assert FollowupWindowCalculator.calculate_recommended_window(recruiter, InteractionOutcome.NEUTRAL) == (5, 8)
    assert FollowupWindowCalculator.calculate_recommended_window(recruiter, InteractionOutcome.NO_RESPONSE) == (4, 7)

    # Hiring Manager outcomes
    assert FollowupWindowCalculator.calculate_recommended_window(hm, InteractionOutcome.ACTION_REQUIRED) == (1, 3)
    assert FollowupWindowCalculator.calculate_recommended_window(hm, InteractionOutcome.POSITIVE) == (4, 7)
    assert FollowupWindowCalculator.calculate_recommended_window(hm, InteractionOutcome.NEUTRAL) == (7, 12)

    # Peer outcomes
    assert FollowupWindowCalculator.calculate_recommended_window(peer, InteractionOutcome.POSITIVE) == (5, 10)


def test_followup_priority_scorer() -> None:
    """Validate the deterministic priority and urgency score calculation."""
    contact_recruiter = ContactResponse(
        id=uuid4(),
        first_name="Jane",
        contact_type=ContactType.RECRUITER,
        status=RelationshipStatus.NEW,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    now_ts = datetime.now(UTC)

    # 1. Base test: Recruiter, neutral, within normal bounds
    # Base: 50.0
    # Recruiter: +10.0
    # Not target company: +0.0
    # Neutral: +0.0
    # Responsiveness (0.5 engagement): +0.0
    # Recency: 5 days elapsed (min 3, max 6) -> no adjustment
    # Expected: 60.0, Priority: 2
    score, priority, explanation = FollowupPriorityScorer.calculate_urgency_score(
        contact=contact_recruiter,
        last_interaction_date=now_ts - timedelta(days=5),
        last_outcome=InteractionOutcome.NEUTRAL,
        engagement_score=0.5,
        is_target_company=False,
        min_days=3,
        max_days=6,
    )
    assert score == 60.0
    assert priority == 2
    assert "Recruiter role" in explanation

    # 2. High Urgency test: HM, target company, action required, highly responsive, overdue
    # Base: 50.0
    # HM: +15.0
    # Target Company: +15.0
    # Action Required: +25.0
    # Responsiveness (1.0 engagement): (1.0 - 0.5) * 20.0 = +10.0
    # Overdue: 10 days elapsed (min 1, max 3) -> overdue by 7 days -> min(20.0, 7 * 2.0) = +14.0
    # Total: 50 + 15 + 15 + 25 + 10 + 14 = 129.0 -> capped at 100.0, Priority: 1
    score, priority, explanation = FollowupPriorityScorer.calculate_urgency_score(
        contact=ContactResponse(
            id=uuid4(),
            first_name="Bob",
            contact_type=ContactType.HIRING_MANAGER,
            status=RelationshipStatus.ACTIVE,
            created_at=now_ts,
            updated_at=now_ts,
        ),
        last_interaction_date=now_ts - timedelta(days=10),
        last_outcome=InteractionOutcome.ACTION_REQUIRED,
        engagement_score=1.0,
        is_target_company=True,
        min_days=1,
        max_days=3,
    )
    assert score == 100.0
    assert priority == 1
    assert "Hiring manager role" in explanation
    assert "Target company alignment" in explanation
    assert "Action required outcome" in explanation
    assert "overdue" in explanation


def test_retrieve_followup_candidates(db_session: Session) -> None:
    """Validate that candidate retrieval retrieves and ranks contacts correctly."""
    contact_service = ContactService(db_session)
    event_repo = OutreachEventRepository(db_session)
    followup_service = FollowupRecommendationService(db_session)

    # 1. Create multiple contacts
    c_rec = contact_service.create_contact(
        ContactCreate(first_name="Jane", company="Google", contact_type=ContactType.RECRUITER)
    )
    c_hm = contact_service.create_contact(
        ContactCreate(first_name="Bob", company="Stripe", contact_type=ContactType.HIRING_MANAGER)
    )
    c_arch = contact_service.create_contact(
        ContactCreate(first_name="Old", company="Facebook", contact_type=ContactType.PEER)
    )
    # Correctly update status to ARCHIVED using ContactUpdate
    contact_service.update_contact(c_arch.id, ContactUpdate(status=RelationshipStatus.ARCHIVED))

    # 2. Add events directly with older completed dates
    event_repo.create_event(
        OutreachEvent(
            contact_id=c_hm.id,
            status=OutreachStatus.REPLIED,
            method="LinkedIn",
            outcome=InteractionOutcome.POSITIVE,
            completed_at=datetime.now(UTC) - timedelta(days=15),
        )
    )

    event_repo.create_event(
        OutreachEvent(
            contact_id=c_rec.id,
            status=OutreachStatus.SENT,
            method="Email",
            completed_at=datetime.now(UTC) - timedelta(days=10),
        )
    )

    # 3. Retrieve candidates
    candidates = followup_service.retrieve_followup_candidates(target_companies=["Stripe"])

    # Verify ARCHIVED contact is excluded
    assert not any(c.contact.id == c_arch.id for c in candidates)
    assert len(candidates) >= 2

    # Bob (HM, Stripe matches target companies, positive outcome) should be higher priority than Jane
    assert candidates[0].contact.id == c_hm.id
    assert isinstance(candidates[0], FollowupCandidate)
    assert candidates[0].freshness_score > 0.0


def test_generate_recommendations(db_session: Session) -> None:
    """Validate that recommendations are generated, drafted, and persisted correctly."""
    contact_service = ContactService(db_session)
    event_repo = OutreachEventRepository(db_session)
    followup_service = FollowupRecommendationService(db_session)

    contact = contact_service.create_contact(
        ContactCreate(first_name="Alice", company="Acme", contact_type=ContactType.HIRING_MANAGER)
    )

    # Add overdue action required event manually
    event_repo.create_event(
        OutreachEvent(
            contact_id=contact.id,
            status=OutreachStatus.REPLIED,
            method="LinkedIn",
            outcome=InteractionOutcome.ACTION_REQUIRED,
            completed_at=datetime.now(UTC) - timedelta(days=4),
        )
    )

    # Generate recommendations
    recs = followup_service.generate_recommendations()
    assert len(recs) == 1
    assert recs[0].contact_id == contact.id
    assert recs[0].priority == 1
    assert "Action required outcome" in recs[0].reasoning
    assert "requested details" in recs[0].draft_message

    # Verify recommendations are persisted in DB
    pending = followup_service.followup_repo.get_pending_followups()
    assert len(pending) >= 1
    assert any(p.contact_id == contact.id for p in pending)


def test_stable_followup_recommendations(db_session: Session) -> None:
    """Validate that follow-up prioritizations and candidates output list remain stable and deterministic."""
    contact_service = ContactService(db_session)
    event_repo = OutreachEventRepository(db_session)
    followup_service = FollowupRecommendationService(db_session)

    # Seed contact & history
    contact = contact_service.create_contact(
        ContactCreate(first_name="Jane", company="Apple", contact_type=ContactType.RECRUITER)
    )
    event_repo.create_event(
        OutreachEvent(
            contact_id=contact.id,
            status=OutreachStatus.SENT,
            method="Email",
            completed_at=datetime.now(UTC) - timedelta(days=5),
        )
    )

    # Candidate lists retrieval stability
    list1 = followup_service.retrieve_followup_candidates()
    list2 = followup_service.retrieve_followup_candidates()

    assert len(list1) == len(list2)
    for c1, c2 in zip(list1, list2, strict=True):
        assert c1.contact.id == c2.contact.id
        assert c1.urgency_score == c2.urgency_score
        assert c1.priority == c2.priority
        assert c1.explanation == c2.explanation
