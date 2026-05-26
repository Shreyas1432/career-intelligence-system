from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from src.modules.relationship import (
    ContactCreate,
    ContactResponse,
    ContactService,
    ContactType,
    InteractionOutcome,
    OutreachEvent,
    OutreachStatus,
    OutreachTrackingService,
    RelationshipStatus,
)
from src.modules.relationship.communication import (
    CommunicationGuidance,
    CommunicationProfileService,
    CommunicationStyleAnalyzer,
    OutreachContextBuilder,
    ToneRecommendationEngine,
)
from src.modules.relationship.schemas import CommunicationStyle


def test_style_analyzer_concise_vs_detailed() -> None:
    """Validate that the style analyzer correctly distinguishes between concise and detailed styles."""
    # 1. Concise default when no events exist
    contact = ContactResponse(
        id=uuid4(),
        first_name="Jane",
        contact_type=ContactType.RECRUITER,
        status=RelationshipStatus.NEW,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    style, style_pref, orientation, engagement, best_time, pref_channel = (
        CommunicationStyleAnalyzer.analyze_style_patterns(contact, [])
    )
    assert style == CommunicationStyle.UNKNOWN
    assert style_pref == "concise"
    assert orientation == "business"
    assert engagement == 0.0
    assert best_time is None
    assert pref_channel is None

    # 2. Detailed length (>= 150 characters average)
    events = [
        OutreachEvent(
            contact_id=contact.id,
            status=OutreachStatus.REPLIED,
            method="Email",
            content="Dear applicant, we reviewed your application and would appreciate scheduling a time to speak. "
            "Please review our corporate policies and select a suitable slot. Regards.",
            completed_at=datetime.now(UTC),
        )
    ]
    _, style_pref, orientation, engagement, _, pref_channel = (
        CommunicationStyleAnalyzer.analyze_style_patterns(contact, events)
    )
    assert style_pref == "detailed"
    assert orientation == "business"
    assert engagement == 1.0
    assert pref_channel == "Email"


def test_style_analyzer_keywords_and_orientation() -> None:
    """Validate that the analyzer classifies formal, casual, analytical styles and tech vs business orientation."""
    contact_hm = ContactResponse(
        id=uuid4(),
        first_name="Alice",
        contact_type=ContactType.HIRING_MANAGER,
        status=RelationshipStatus.ACTIVE,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )

    # 1. Technical + Analytical keywords
    events_tech = [
        OutreachEvent(
            contact_id=contact_hm.id,
            status=OutreachStatus.REPLIED,
            method="LinkedIn",
            content="We should talk about Python, AWS and database performance metrics. Let me know what data architecture you prefer.",
            completed_at=datetime.now(UTC),
        )
    ]
    _, style_pref, orientation, _, _, _ = (
        CommunicationStyleAnalyzer.analyze_style_patterns(contact_hm, events_tech)
    )
    assert orientation == "technical"
    # Content length: ~120 chars -> < 150 -> concise preference
    assert style_pref == "concise"

    # 2. Business + Formal keywords
    contact_recruiter = ContactResponse(
        id=uuid4(),
        first_name="Jane",
        contact_type=ContactType.RECRUITER,
        status=RelationshipStatus.NEW,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    events_bus = [
        OutreachEvent(
            contact_id=contact_recruiter.id,
            status=OutreachStatus.REPLIED,
            method="LinkedIn",
            content="Dear Shreyas, hope this email finds you well. I would appreciate if we could schedule a call to discuss the salary and contract role.",
            completed_at=datetime.now(UTC),
        )
    ]
    style, style_pref, orientation, _, _, _ = (
        CommunicationStyleAnalyzer.analyze_style_patterns(contact_recruiter, events_bus)
    )
    assert orientation == "business"
    assert style == CommunicationStyle.FORMAL


def test_style_analyzer_channel_and_time() -> None:
    """Validate the deterministic calculation of preferred channel and best time to contact."""
    contact_id = uuid4()
    # Create week weekday morning event
    dt_weekday_morning = datetime(2026, 5, 20, 10, 0, 0, tzinfo=UTC)  # Wednesday (weekday), 10:00 (morning)
    events = [
        OutreachEvent(
            contact_id=contact_id,
            status=OutreachStatus.REPLIED,
            method="LinkedIn",
            completed_at=dt_weekday_morning,
        )
    ]

    contact = ContactResponse(
        id=contact_id,
        first_name="Alice",
        contact_type=ContactType.PEER,
        status=RelationshipStatus.ACTIVE,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )

    _, _, _, _, best_time, pref_channel = CommunicationStyleAnalyzer.analyze_style_patterns(contact, events)
    assert pref_channel == "LinkedIn"
    assert best_time == "Weekday Mornings"


def test_tone_recommendation_engine() -> None:
    """Validate that tone recommendation engine outputs correct guidelines for recruiter/HM and concise/detailed style preferences."""
    contact_recruiter = ContactResponse(
        id=uuid4(),
        first_name="Recruiter Jane",
        contact_type=ContactType.RECRUITER,
        status=RelationshipStatus.NEW,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    tone_rec = ToneRecommendationEngine.generate_tone_guidance(contact_recruiter, "concise", "business")
    assert any("Recruiter Focus" in g for g in tone_rec)
    assert any("Structure: Prefer bullet points" in g for g in tone_rec)
    assert any("Business Tone" in g for g in tone_rec)

    contact_hm = ContactResponse(
        id=uuid4(),
        first_name="HM Bob",
        contact_type=ContactType.HIRING_MANAGER,
        status=RelationshipStatus.ACTIVE,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    tone_hm = ToneRecommendationEngine.generate_tone_guidance(contact_hm, "detailed", "technical")
    assert any("Hiring Manager Focus" in g for g in tone_hm)
    assert any("Structure: Provide complete background" in g for g in tone_hm)
    assert any("Technical Tone" in g for g in tone_hm)


def test_outreach_context_builder() -> None:
    """Validate that outreach context builder outputs correct recommendations and hints based on historical outcomes."""
    # 1. No history
    ctx = OutreachContextBuilder.build_continuity_context([])
    assert "Initiate first cold outreach" in ctx["continuity_recommendations"][0]

    # 2. Positive outcome
    e1 = OutreachEvent(
        contact_id=uuid4(),
        status=OutreachStatus.REPLIED,
        method="Email",
        outcome=InteractionOutcome.POSITIVE,
        completed_at=datetime.now(UTC),
    )
    ctx = OutreachContextBuilder.build_continuity_context([e1])
    assert "positive feedback" in ctx["continuity_recommendations"][0]

    # 3. Action required outcome
    e2 = OutreachEvent(
        contact_id=uuid4(),
        status=OutreachStatus.REPLIED,
        method="Email",
        outcome=InteractionOutcome.ACTION_REQUIRED,
        completed_at=datetime.now(UTC),
    )
    ctx = OutreachContextBuilder.build_continuity_context([e2])
    assert "pending action items" in ctx["continuity_recommendations"][0]

    # 4. Sent but no response (fresh, < 3 days)
    e3 = OutreachEvent(
        contact_id=uuid4(),
        status=OutreachStatus.SENT,
        method="Email",
        outcome=None,
        completed_at=datetime.now(UTC) - timedelta(days=1),
    )
    ctx = OutreachContextBuilder.build_continuity_context([e3])
    assert "Wait for a response" in ctx["continuity_recommendations"][0]

    # 5. Sent but no response (old, >= 3 days)
    e4 = OutreachEvent(
        contact_id=uuid4(),
        status=OutreachStatus.SENT,
        method="Email",
        outcome=None,
        completed_at=datetime.now(UTC) - timedelta(days=5),
    )
    ctx = OutreachContextBuilder.build_continuity_context([e4])
    assert "polite nudge" in ctx["continuity_recommendations"][0]


def test_communication_profile_service_sync_and_get(db_session: Session) -> None:
    """Validate CommunicationProfileService get, create, sync, and guidance generation via DB session."""
    contact_service = ContactService(db_session)
    tracking_service = OutreachTrackingService(db_session)
    comm_service = CommunicationProfileService(db_session)

    # 1. Create a Recruiter contact
    contact = contact_service.create_contact(
        ContactCreate(
            first_name="Jane",
            last_name="Doe",
            company="Google",
            contact_type=ContactType.RECRUITER,
            email="jane.doe@google.com",
        )
    )

    # 2. Fetch/create profile (should be empty default profile)
    profile = comm_service.get_or_create_profile(contact.id)
    assert profile.contact_id == contact.id
    assert profile.style == CommunicationStyle.UNKNOWN
    assert profile.engagement_score == 0.0

    # 3. Log outreach event and response
    tracking_service.log_outreach_event(
        contact_id=contact.id,
        method="Email",
        content="Dear Jane, I wanted to schedule an interview to discuss the technical recruitment processes. Hope this email finds you well. Regards.",
        completed_at=datetime.now(UTC) - timedelta(days=4),
    )
    tracking_service.log_response(
        contact_id=contact.id,
        method="Email",
        content="Hi Shreyas, let's connect on Zoom to review availability.",
        outcome=InteractionOutcome.POSITIVE,
    )

    # 4. Sync profile and verify persistence
    synced_profile = comm_service.analyze_and_sync_profile(contact.id)
    assert synced_profile.engagement_score == 1.0
    assert synced_profile.preferred_channel == "Email"
    assert len(synced_profile.insights) > 0

    # Verify db state
    refetched_profile = comm_service.get_or_create_profile(contact.id)
    assert refetched_profile.engagement_score == 1.0
    assert refetched_profile.preferred_channel == "Email"

    # 5. Generate guidance and verify schema
    guidance = comm_service.generate_guidance(contact.id)
    assert isinstance(guidance, CommunicationGuidance)
    assert guidance.contact_id == contact.id
    assert guidance.style_preference in {"concise", "detailed"}
    assert guidance.orientation == "business"
    assert len(guidance.tone_guidance) > 0
    assert len(guidance.continuity_recommendations) > 0
    assert len(guidance.context_hints) > 0


def test_communication_profile_service_invalid_contact(db_session: Session) -> None:
    """Validate that CommunicationProfileService raises ValueError when contact is missing."""
    comm_service = CommunicationProfileService(db_session)
    invalid_id = uuid4()

    with pytest.raises(ValueError, match="not found"):
        comm_service.analyze_and_sync_profile(invalid_id)

    with pytest.raises(ValueError, match="not found"):
        comm_service.generate_guidance(invalid_id)


def test_stable_recommendation_outputs() -> None:
    """Validate that the recommendation engines and helpers produce stable, deterministic outputs."""
    contact = ContactResponse(
        id=uuid4(),
        first_name="Alice",
        contact_type=ContactType.HIRING_MANAGER,
        status=RelationshipStatus.ACTIVE,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )

    # 1. Tone Recommendation Engine Stability
    guidance1 = ToneRecommendationEngine.generate_tone_guidance(contact, "detailed", "technical")
    guidance2 = ToneRecommendationEngine.generate_tone_guidance(contact, "detailed", "technical")
    assert guidance1 == guidance2
    assert len(guidance1) > 0

    # 2. Context Builder Stability
    events = [
        OutreachEvent(
            contact_id=contact.id,
            status=OutreachStatus.SENT,
            method="Email",
            outcome=None,
            completed_at=datetime.now(UTC) - timedelta(days=5),
        )
    ]
    ctx1 = OutreachContextBuilder.build_continuity_context(events)
    ctx2 = OutreachContextBuilder.build_continuity_context(events)
    assert ctx1 == ctx2
    assert len(ctx1["continuity_recommendations"]) > 0
