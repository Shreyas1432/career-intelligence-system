from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from sqlalchemy.orm import Session

from src.modules.relationship import (
    ContactCreate,
    ContactResponse,
    ContactService,
    ContactType,
    FollowupRepository,
    InteractionOutcome,
    InteractionRecencyEvaluator,
    InteractionTimelineBuilder,
    OutreachEvent,
    OutreachStatus,
    OutreachTrackingService,
    RelationshipMemory,
    RelationshipStateManager,
    RelationshipStatus,
)
from src.modules.relationship.models import FollowupRecommendationModel


def test_recency_evaluator() -> None:
    """Validate interaction recency evaluation logic."""
    now_ts = datetime.now(UTC)

    # 1. No interactions
    assert InteractionRecencyEvaluator.get_last_interaction_date([], None) is None
    assert InteractionRecencyEvaluator.is_stale(None, threshold_days=30) is True

    # 2. Memory interaction date
    memory = RelationshipMemory(
        contact_id=uuid4(), last_interaction_date=now_ts - timedelta(days=10)
    )
    last_date = InteractionRecencyEvaluator.get_last_interaction_date([], memory)
    assert last_date == memory.last_interaction_date
    assert InteractionRecencyEvaluator.is_stale(last_date, threshold_days=15) is False
    assert InteractionRecencyEvaluator.is_stale(last_date, threshold_days=5) is True

    # 3. Outreach events interaction date (should take the latest)
    events = [
        OutreachEvent(
            contact_id=uuid4(),
            status=OutreachStatus.SENT,
            method="Email",
            completed_at=now_ts - timedelta(days=3),
        ),
        OutreachEvent(
            contact_id=uuid4(),
            status=OutreachStatus.SENT,
            method="LinkedIn",
            completed_at=now_ts - timedelta(days=7),
        ),
    ]

    last_date_events = InteractionRecencyEvaluator.get_last_interaction_date(events, memory)
    assert last_date_events == events[0].completed_at


def test_relationship_state_manager_logic() -> None:
    """Validate deterministic state transition machine logic."""
    contact_id = uuid4()
    now_ts = datetime.now(UTC)

    # Base contact mock
    contact = ContactResponse(
        id=contact_id,
        first_name="Jane",
        contact_type=ContactType.RECRUITER,
        status=RelationshipStatus.NEW,
        created_at=now_ts,
        updated_at=now_ts,
    )

    # 1. NEW state (no events)
    assert RelationshipStateManager.evaluate_state(contact, []) == RelationshipStatus.NEW

    # 2. CONTACTED state (outreach sent, no reply)
    event_sent = OutreachEvent(
        contact_id=contact_id,
        status=OutreachStatus.SENT,
        method="Email",
        completed_at=now_ts - timedelta(days=2),
    )
    assert (
        RelationshipStateManager.evaluate_state(contact, [event_sent])
        == RelationshipStatus.CONTACTED
    )

    # 3. RESPONDED state (outreach replied with neutral/negative outcome)
    event_replied_neutral = OutreachEvent(
        contact_id=contact_id,
        status=OutreachStatus.REPLIED,
        method="Email",
        outcome=InteractionOutcome.NEUTRAL,
        completed_at=now_ts - timedelta(days=1),
    )
    assert (
        RelationshipStateManager.evaluate_state(contact, [event_sent, event_replied_neutral])
        == RelationshipStatus.RESPONDED
    )

    # 4. ACTIVE state (replied with positive outcome)
    event_replied_positive = OutreachEvent(
        contact_id=contact_id,
        status=OutreachStatus.REPLIED,
        method="Email",
        outcome=InteractionOutcome.POSITIVE,
        completed_at=now_ts - timedelta(days=1),
    )
    assert (
        RelationshipStateManager.evaluate_state(contact, [event_sent, event_replied_positive])
        == RelationshipStatus.ACTIVE
    )

    # 5. STALE state (last completed event > 30 days ago)
    event_old = OutreachEvent(
        contact_id=contact_id,
        status=OutreachStatus.REPLIED,
        method="Email",
        outcome=InteractionOutcome.NEUTRAL,
        completed_at=now_ts - timedelta(days=31),
    )
    assert (
        RelationshipStateManager.evaluate_state(contact, [event_old], threshold_days=30)
        == RelationshipStatus.STALE
    )

    # 6. FOLLOWUP_PENDING state (suggested date passed, no newer outreach)
    event_recent = OutreachEvent(
        contact_id=contact_id,
        status=OutreachStatus.SENT,
        method="Email",
        completed_at=now_ts - timedelta(days=5),
    )
    followup_date = now_ts - timedelta(days=2)
    assert (
        RelationshipStateManager.evaluate_state(
            contact, [event_recent], followup_dates=[followup_date]
        )
        == RelationshipStatus.FOLLOWUP_PENDING
    )

    # If outreach occurred *after* the followup date, it's not pending anymore
    event_new = OutreachEvent(
        contact_id=contact_id,
        status=OutreachStatus.SENT,
        method="Email",
        completed_at=now_ts - timedelta(days=1),
    )
    assert (
        RelationshipStateManager.evaluate_state(
            contact, [event_recent, event_new], followup_dates=[followup_date]
        )
        == RelationshipStatus.CONTACTED
    )


def test_timeline_builder_generation() -> None:
    """Validate timeline builder gathers and sorts events correctly."""
    contact_id = uuid4()
    now_ts = datetime.now(UTC)

    contact = ContactResponse(
        id=contact_id,
        first_name="Jane",
        company="Stripe",
        status=RelationshipStatus.NEW,
        created_at=now_ts - timedelta(days=10),
        updated_at=now_ts - timedelta(days=10),
    )

    events = [
        OutreachEvent(
            contact_id=contact_id,
            status=OutreachStatus.REPLIED,
            method="Email",
            outcome=InteractionOutcome.POSITIVE,
            created_at=now_ts - timedelta(days=5),
            completed_at=now_ts - timedelta(days=4),
        )
    ]

    memory = RelationshipMemory(
        contact_id=contact_id,
        last_interaction_date=now_ts - timedelta(days=3),
        key_facts=["Likes Rust"],
    )

    timeline = InteractionTimelineBuilder.build_timeline(contact, events, memory)

    assert len(timeline) == 4
    # Event 1: Creation
    assert timeline[0].event_type == "state_change"
    assert timeline[0].description == "Contact created"
    # Event 2: Outreach Sent
    assert timeline[1].event_type == "outreach"
    assert "Outreach attempt" in timeline[1].description
    # Event 3: Response
    assert timeline[2].event_type == "response"
    assert "Response received" in timeline[2].description
    # Event 4: Memory Note
    assert timeline[3].event_type == "state_change"
    assert timeline[3].description == "Last memory state snapshot update"

    # Verify chronological order
    for i in range(len(timeline) - 1):
        assert timeline[i].timestamp.replace(tzinfo=None) <= timeline[i + 1].timestamp.replace(
            tzinfo=None
        )


def test_tracking_service_orchestration(db_session: Session) -> None:
    """Validate tracking service event logging and auto state-sync flows."""
    contact_service = ContactService(db_session)
    tracking_service = OutreachTrackingService(db_session)

    # 1. Create NEW contact
    contact = contact_service.create_contact(
        ContactCreate(first_name="Alice", company="Acme", email="alice@acme.com")
    )
    assert contact.status == RelationshipStatus.NEW

    # 2. Log Outreach Event -> Transitions to CONTACTED
    oe = tracking_service.log_outreach_event(
        contact_id=contact.id,
        method="LinkedIn",
        content="Hi Alice!",
        completed_at=datetime.now(UTC),
    )
    assert isinstance(oe.id, UUID)
    assert oe.status == OutreachStatus.SENT

    # Refetch contact and check updated status
    refetched = contact_service.get_contact(contact.id)
    assert refetched is not None
    assert refetched.status == RelationshipStatus.CONTACTED

    # 3. Log Response (Neutral) -> Transitions to RESPONDED
    tracking_service.log_response(
        contact_id=contact.id,
        method="LinkedIn",
        content="Thanks but no openings.",
        outcome=InteractionOutcome.NEUTRAL,
    )
    refetched = contact_service.get_contact(contact.id)
    assert refetched is not None
    assert refetched.status == RelationshipStatus.RESPONDED

    # 4. Log Response (Positive) -> Transitions to ACTIVE
    tracking_service.log_response(
        contact_id=contact.id,
        method="LinkedIn",
        content="Let's chat!",
        outcome=InteractionOutcome.POSITIVE,
    )
    refetched = contact_service.get_contact(contact.id)
    assert refetched is not None
    assert refetched.status == RelationshipStatus.ACTIVE


def test_stale_and_followup_detection_services(db_session: Session) -> None:
    """Validate state managers flag stale and pending follow-ups correctly."""
    contact_service = ContactService(db_session)
    tracking_service = OutreachTrackingService(db_session)
    followup_repo = FollowupRepository(db_session)

    # 1. Create Stale Contact
    stale_contact = contact_service.create_contact(
        ContactCreate(first_name="Stale User", company="Stale Corp")
    )

    # Set old completed event
    tracking_service.log_outreach_event(
        contact_id=stale_contact.id,
        method="Email",
        completed_at=datetime.now(UTC) - timedelta(days=35),
    )

    stale_list = tracking_service.get_stale_contacts(threshold_days=30)
    assert any(c.id == stale_contact.id for c in stale_list)

    # 2. Create Followup Pending Contact
    followup_contact = contact_service.create_contact(
        ContactCreate(first_name="Followup User", company="Followup Corp")
    )

    # Log an old outreach event
    tracking_service.log_outreach_event(
        contact_id=followup_contact.id,
        method="Email",
        completed_at=datetime.now(UTC) - timedelta(days=5),
    )

    # Persist a past followup recommendation
    followup_repo.session.add(
        FollowupRecommendationModel(
            contact_id=str(followup_contact.id),
            suggested_date=datetime.now(UTC) - timedelta(days=1),
            reasoning="Touch base",
        )
    )
    followup_repo.session.flush()

    pending_list = tracking_service.get_pending_followup_contacts()
    assert any(c.id == followup_contact.id for c in pending_list)


def test_stable_timeline_ordering() -> None:
    """Validate that timeline builder preserves relative input order for events with identical timestamps."""
    contact_id = uuid4()
    now_ts = datetime.now(UTC)

    contact = ContactResponse(
        id=contact_id,
        first_name="Jane",
        company="Stripe",
        status=RelationshipStatus.NEW,
        created_at=now_ts,
        updated_at=now_ts,
    )

    # Outreach events created with identical timestamps
    events = [
        OutreachEvent(
            id=uuid4(),
            contact_id=contact_id,
            status=OutreachStatus.SENT,
            method="Email",
            created_at=now_ts,
            completed_at=now_ts,
        ),
        OutreachEvent(
            id=uuid4(),
            contact_id=contact_id,
            status=OutreachStatus.SENT,
            method="LinkedIn",
            created_at=now_ts,
            completed_at=now_ts,
        ),
    ]

    timeline = InteractionTimelineBuilder.build_timeline(contact, events, None)

    # Check that Email outreach event precedes LinkedIn outreach event in the timeline (stable sort)
    outreach_types = [
        item.description for item in timeline if item.event_type == "outreach"
    ]
    assert outreach_types == [
        "Outreach attempt via Email",
        "Outreach attempt via Linkedin",
    ]

