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
    InteractionOutcome,
    OutreachEvent,
    OutreachStatus,
    RelationshipAnalyticsService,
    RelationshipStatus,
)
from src.modules.relationship.analytics import (
    EngagementMetricsCalculator,
    FollowupEffectivenessAnalyzer,
    NetworkingTrendAnalyzer,
    RelationshipAnalyticsSummary,
)
from src.modules.relationship.models import FollowupRecommendationModel
from src.modules.relationship.repositories import OutreachEventRepository


def test_engagement_metrics_calculator() -> None:
    """Validate recruiter, hiring manager, and company metrics aggregation calculations."""
    now_ts = datetime.now(UTC)

    # 1. Mock Contacts
    contacts = [
        ContactResponse(
            id=uuid4(), first_name="Rec1", contact_type=ContactType.RECRUITER,
            status=RelationshipStatus.NEW, created_at=now_ts, updated_at=now_ts, company="Google"
        ),
        ContactResponse(
            id=uuid4(), first_name="Rec2", contact_type=ContactType.RECRUITER,
            status=RelationshipStatus.CONTACTED, created_at=now_ts, updated_at=now_ts, company="Google"
        ),
        ContactResponse(
            id=uuid4(), first_name="HM1", contact_type=ContactType.HIRING_MANAGER,
            status=RelationshipStatus.ACTIVE, created_at=now_ts, updated_at=now_ts, company="Stripe"
        ),
    ]

    # 2. Mock Events
    events = [
        # Rec1: Sent but no response
        OutreachEvent(
            contact_id=contacts[0].id, status=OutreachStatus.SENT, method="Email",
            created_at=now_ts - timedelta(days=5), completed_at=now_ts - timedelta(days=5)
        ),
        # Rec2: Replied
        OutreachEvent(
            contact_id=contacts[1].id, status=OutreachStatus.REPLIED, method="Email",
            outcome=InteractionOutcome.NEUTRAL, created_at=now_ts - timedelta(days=3),
            completed_at=now_ts - timedelta(days=3)
        ),
        # HM1: Replied positive
        OutreachEvent(
            contact_id=contacts[2].id, status=OutreachStatus.REPLIED, method="LinkedIn",
            outcome=InteractionOutcome.POSITIVE, created_at=now_ts - timedelta(days=2),
            completed_at=now_ts - timedelta(days=2)
        ),
    ]

    # Recruiter Role Metrics: 2 contacts, 2 sent, 1 replied -> 50% response rate
    rec_metrics = EngagementMetricsCalculator.calculate_role_metrics(contacts, events, ContactType.RECRUITER)
    assert rec_metrics.total_contacts == 2
    assert rec_metrics.sent_count == 2
    assert rec_metrics.replied_count == 1
    assert rec_metrics.response_rate == 0.5

    # HM Role Metrics: 1 contact, 1 sent, 1 replied -> 100% response rate
    hm_metrics = EngagementMetricsCalculator.calculate_role_metrics(contacts, events, ContactType.HIRING_MANAGER)
    assert hm_metrics.total_contacts == 1
    assert hm_metrics.sent_count == 1
    assert hm_metrics.replied_count == 1
    assert hm_metrics.response_rate == 1.0

    # Company Metrics
    comp_metrics = EngagementMetricsCalculator.calculate_company_metrics(contacts, events)
    assert len(comp_metrics) == 2
    # Sorted by total_contacts count descending
    assert comp_metrics[0].company_name == "Google"
    assert comp_metrics[0].total_contacts == 2
    assert comp_metrics[0].total_outreach == 2
    assert comp_metrics[0].response_rate == 0.5
    assert comp_metrics[0].last_interaction is not None

    assert comp_metrics[1].company_name == "Stripe"
    assert comp_metrics[1].total_contacts == 1
    assert comp_metrics[1].total_outreach == 1
    assert comp_metrics[1].response_rate == 1.0

    # Progression Summary
    prog = EngagementMetricsCalculator.calculate_progression_summary(contacts)
    assert prog.active_count == 3
    assert prog.stale_count == 0
    assert prog.status_counts[RelationshipStatus.NEW.value] == 1
    assert prog.status_counts[RelationshipStatus.CONTACTED.value] == 1
    assert prog.status_counts[RelationshipStatus.ACTIVE.value] == 1


def test_followup_effectiveness_analyzer(db_session: Session) -> None:
    """Validate follow-up recommendations conversion rate calculations."""
    # 1. Create Follow-up Recommendations
    c1_id = uuid4()
    c2_id = uuid4()
    c3_id = uuid4()

    now_ts = datetime.now(UTC)

    # Recommendation 1: Acted upon and positive outcome
    rec1 = FollowupRecommendationModel(
        contact_id=str(c1_id),
        suggested_date=now_ts - timedelta(days=5),
        reasoning="Test 1",
        priority=1,
    )
    # Recommendation 2: Acted upon, neutral outcome
    rec2 = FollowupRecommendationModel(
        contact_id=str(c2_id),
        suggested_date=now_ts - timedelta(days=5),
        reasoning="Test 2",
        priority=2,
    )
    # Recommendation 3: Not acted upon
    rec3 = FollowupRecommendationModel(
        contact_id=str(c3_id),
        suggested_date=now_ts - timedelta(days=5),
        reasoning="Test 3",
        priority=3,
    )

    db_session.add_all([rec1, rec2, rec3])
    db_session.flush()

    # 2. Add events
    events = [
        # Contact 1: Outreach completed 3 days after recommendation suggested date -> within 7 days window (Acted upon, POSITIVE)
        OutreachEvent(
            contact_id=c1_id, status=OutreachStatus.REPLIED, method="Email",
            outcome=InteractionOutcome.POSITIVE, completed_at=now_ts - timedelta(days=2)
        ),
        # Contact 2: Outreach completed 10 days after suggestion -> outside 7 days window (NOT acted upon within window)
        OutreachEvent(
            contact_id=c2_id, status=OutreachStatus.REPLIED, method="Email",
            outcome=InteractionOutcome.NEUTRAL, completed_at=now_ts + timedelta(days=5)
        ),
    ]

    # 3. Analyze
    eff = FollowupEffectivenessAnalyzer.analyze_followup_effectiveness(db_session, events)
    assert eff.total_recommended == 3
    # Only c1_id is within the 7 day window
    assert eff.total_acted_upon == 1
    assert eff.action_rate == pytest.approx(1.0 / 3.0, abs=0.01)
    assert eff.positive_response_rate == 1.0


def test_networking_trend_analyzer() -> None:
    """Validate outreach consistency weekly indexing."""
    now_ts = datetime.now(UTC)

    # Outreach events logged across different weeks
    events = [
        # Today (this week)
        OutreachEvent(contact_id=uuid4(), status=OutreachStatus.SENT, method="Email", completed_at=now_ts),
        # 1 week ago
        OutreachEvent(contact_id=uuid4(), status=OutreachStatus.SENT, method="Email", completed_at=now_ts - timedelta(weeks=1)),
        # 2 weeks ago
        OutreachEvent(contact_id=uuid4(), status=OutreachStatus.SENT, method="Email", completed_at=now_ts - timedelta(weeks=2)),
        # 5 weeks ago
        OutreachEvent(contact_id=uuid4(), status=OutreachStatus.SENT, method="Email", completed_at=now_ts - timedelta(weeks=5)),
    ]

    # Over trailing 12 weeks: 4 active weeks -> consistency index: 4 / 12 = 33.3%
    metrics = NetworkingTrendAnalyzer.analyze_consistency(events, weeks_limit=12)
    assert metrics.weekly_consistency_index == pytest.approx(4.0 / 12.0, abs=0.01)
    assert len(metrics.outreach_by_week) >= 4


def test_relationship_analytics_service_summary(db_session: Session) -> None:
    """Validate orchestration summary generation and explainable insights list output."""
    contact_service = ContactService(db_session)
    event_repo = OutreachEventRepository(db_session)
    analytics_service = RelationshipAnalyticsService(db_session)

    # 1. Seed Contacts
    c_rec = contact_service.create_contact(
        ContactCreate(first_name="Jane", company="Google", contact_type=ContactType.RECRUITER)
    )
    c_hm = contact_service.create_contact(
        ContactCreate(first_name="Bob", company="Stripe", contact_type=ContactType.HIRING_MANAGER)
    )

    # Transition statuses to expected active statuses
    contact_service.update_contact(c_rec.id, ContactUpdate(status=RelationshipStatus.RESPONDED))
    contact_service.update_contact(c_hm.id, ContactUpdate(status=RelationshipStatus.CONTACTED))

    # 2. Seed Events
    event_repo.create_event(
        OutreachEvent(
            contact_id=c_rec.id, status=OutreachStatus.REPLIED, method="Email",
            outcome=InteractionOutcome.NEUTRAL, completed_at=datetime.now(UTC) - timedelta(days=2)
        )
    )
    event_repo.create_event(
        OutreachEvent(
            contact_id=c_hm.id, status=OutreachStatus.SENT, method="LinkedIn",
            completed_at=datetime.now(UTC) - timedelta(days=1)
        )
    )

    # 3. Generate Summary
    summary = analytics_service.generate_summary(weeks_limit=12)
    assert isinstance(summary, RelationshipAnalyticsSummary)

    # Verify overall metrics
    assert summary.overall_metrics.total_contacts == 2
    assert summary.overall_metrics.active_conversations == 2
    # Recruiter sent 1, replied 1. HM sent 1, replied 0. Overall sent 2, replied 1 -> 50%
    assert summary.overall_metrics.response_rate_percent == 50.0

    # Verify role metrics
    assert summary.recruiter_metrics.response_rate == 1.0
    assert summary.hiring_manager_metrics.response_rate == 0.0

    # Verify progression
    assert summary.progression.active_count == 2
    assert summary.progression.stale_count == 0

    # Verify insights output
    assert len(summary.explainable_insights) > 0
    assert any("response rate" in i for i in summary.explainable_insights)
