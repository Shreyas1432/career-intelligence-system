from datetime import UTC, datetime, timedelta

from sqlalchemy.orm import Session

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
    RelationshipAnalyticsService,
    RelationshipFreshnessEvaluator,
    RelationshipStatus,
)
from src.modules.relationship.repositories import OutreachEventRepository


def test_recruiter_relationship_lifecycle(db_session: Session) -> None:
    """
    Scenario 1: Recruiter Relationship Lifecycle E2E workflow integration test.
    - Create recruiter contact
    - Normalize name/company metadata
    - Log outreach event & response
    - Validate status updates
    - Generate communication guidance
    - Generate follow-up recommendation
    - Compile analytics summary
    """
    contact_service = ContactService(db_session)
    tracking_service = OutreachTrackingService(db_session)
    comm_service = CommunicationProfileService(db_session)
    followup_service = FollowupRecommendationService(db_session)
    analytics_service = RelationshipAnalyticsService(db_session)

    # 1. Create Recruiter Contact with messy inputs
    contact = contact_service.create_contact(
        ContactCreate(
            first_name="  jane  ",
            last_name="  doe  ",
            company="Google LLC",
            contact_type=ContactType.RECRUITER,
            email="JANE@GOOGLE.COM",
        )
    )

    # Validate normalization on create
    assert contact.first_name == "Jane"
    assert contact.last_name == "Doe"
    assert contact.company == "Google"  # Google LLC -> Google
    assert contact.email == "jane@google.com"
    assert contact.status == RelationshipStatus.NEW

    # 2. Log outreach event
    tracking_service.log_outreach_event(
        contact_id=contact.id,
        method="Email",
        content="Hi Jane, let's discuss recruiter options.",
        completed_at=datetime.now(UTC) - timedelta(days=2),
    )

    # Verify status transitioned to CONTACTED
    refetched = contact_service.get_contact(contact.id)
    assert refetched is not None
    assert refetched.status == RelationshipStatus.CONTACTED

    # 3. Log reply response
    tracking_service.log_response(
        contact_id=contact.id,
        method="Email",
        content="Hey Shreyas, thanks for reaching out. Let's schedule a Zoom call.",
        outcome=InteractionOutcome.NEUTRAL,
    )

    # Verify status transitioned to RESPONDED
    refetched = contact_service.get_contact(contact.id)
    assert refetched is not None
    assert refetched.status == RelationshipStatus.RESPONDED

    # 4. Generate Communication Guidance
    guidance = comm_service.generate_guidance(contact.id)
    assert guidance.style_preference == "concise"
    assert guidance.orientation == "business"
    assert any("Recruiter" in t for t in guidance.tone_guidance)
    assert any("Business Tone" in t for t in guidance.tone_guidance)

    # 5. Generate Follow-up Recommendations
    # Simulate that 6 days have elapsed since the last response by updating events in the DB
    from src.modules.relationship.models import OutreachEventModel
    db_session.query(OutreachEventModel).filter_by(contact_id=str(contact.id)).update(
        {OutreachEventModel.completed_at: datetime.now(UTC) - timedelta(days=6)}
    )
    db_session.flush()

    recs_due = followup_service.generate_recommendations()
    assert len(recs_due) >= 1
    assert recs_due[0].contact_id == contact.id
    assert "previous conversation" in recs_due[0].draft_message

    # 6. Generate Analytics Summary
    summary = analytics_service.generate_summary(weeks_limit=12)
    assert summary.overall_metrics.total_contacts >= 1
    assert summary.recruiter_metrics.total_contacts >= 1
    assert summary.recruiter_metrics.response_rate > 0.0


def test_hiring_manager_interaction_lifecycle(db_session: Session) -> None:
    """
    Scenario 2: Hiring Manager Interaction Lifecycle.
    - Create HM contact
    - Log positive interaction progression
    - Verify transition to ACTIVE
    - Check relationship freshness
    - Generate continuity-aware follow-up
    - Compute engagement metrics
    """
    contact_service = ContactService(db_session)
    tracking_service = OutreachTrackingService(db_session)
    comm_service = CommunicationProfileService(db_session)
    followup_service = FollowupRecommendationService(db_session)
    analytics_service = RelationshipAnalyticsService(db_session)
    event_repo = OutreachEventRepository(db_session)

    # 1. Create Hiring Manager
    contact = contact_service.create_contact(
        ContactCreate(
            first_name="Bob",
            company="Stripe",
            contact_type=ContactType.HIRING_MANAGER,
        )
    )

    # 2. Track interactions (Positive reply 4 days ago)
    event_repo.create_event(
        OutreachEvent(
            contact_id=contact.id,
            status=OutreachStatus.REPLIED,
            method="LinkedIn",
            outcome=InteractionOutcome.POSITIVE,
            completed_at=datetime.now(UTC) - timedelta(days=4),
        )
    )

    # Force sync state
    tracking_service._sync_relationship_state(contact.id)
    refetched = contact_service.get_contact(contact.id)
    assert refetched is not None
    assert refetched.status == RelationshipStatus.ACTIVE

    # 3. Check Freshness
    freshness = RelationshipFreshnessEvaluator.calculate_freshness_score(datetime.now(UTC) - timedelta(days=4))
    assert freshness > 0.8
    assert RelationshipFreshnessEvaluator.classify_relationship(datetime.now(UTC) - timedelta(days=4)) == "active"

    # 4. Generate follow-up recommendations
    # HM Positive window: (4, 7) days. 4 days elapsed -> due!
    recs = followup_service.generate_recommendations()
    assert len(recs) == 1
    assert recs[0].contact_id == contact.id
    assert "Bob" in recs[0].draft_message
    assert "updates" in recs[0].draft_message

    # 5. Compute engagement metrics
    comm_service.analyze_and_sync_profile(contact.id)
    summary = analytics_service.generate_summary()
    assert summary.hiring_manager_metrics.response_rate == 1.0
    assert summary.overall_metrics.meetings_booked >= 1


def test_stale_relationship_recovery(db_session: Session) -> None:
    """
    Scenario 3: Stale Relationship Recovery.
    - Simulate stale interaction history (40 days ago)
    - Validate stale detection
    - Validate follow-up prioritization
    - Validate analytics aggregation
    """
    contact_service = ContactService(db_session)
    event_repo = OutreachEventRepository(db_session)
    followup_service = FollowupRecommendationService(db_session)
    tracking_service = OutreachTrackingService(db_session)
    analytics_service = RelationshipAnalyticsService(db_session)

    # 1. Create contact
    contact = contact_service.create_contact(
        ContactCreate(first_name="Alice", company="Apple", contact_type=ContactType.PEER)
    )

    # 2. Simulate stale event (40 days ago)
    event_repo.create_event(
        OutreachEvent(
            contact_id=contact.id,
            status=OutreachStatus.SENT,
            method="Email",
            completed_at=datetime.now(UTC) - timedelta(days=40),
        )
    )

    # 3. Validate stale detection
    tracking_service._sync_relationship_state(contact.id)
    refetched = contact_service.get_contact(contact.id)
    assert refetched is not None
    assert refetched.status == RelationshipStatus.STALE

    # 4. Validate follow-up prioritization (overdue)
    candidates = followup_service.retrieve_followup_candidates()
    alice_cand = next(c for c in candidates if c.contact.id == contact.id)
    assert alice_cand.is_stale is True
    # Overdue by 40 - 14 = 26 days -> priority score is high
    assert alice_cand.urgency_score > 60.0

    # 5. Generate and check follow-up recommendation
    recs = followup_service.generate_recommendations()
    assert len(recs) == 1
    assert recs[0].contact_id == contact.id
    assert "nudge" in recs[0].draft_message or "follow up" in recs[0].draft_message

    # 6. Validate analytics aggregation (stale count should be 1)
    summary = analytics_service.generate_summary()
    assert summary.progression.stale_count == 1
    assert any("Stale" in i for i in summary.explainable_insights)
