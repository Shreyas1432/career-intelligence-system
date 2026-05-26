import uuid
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from src.modules.relationship import (
    CommunicationProfile,
    CommunicationStyle,
    ContactBase,
    ContactCreate,
    ContactResponse,
    ContactType,
    ContactUpdate,
    FollowupRecommendation,
    InteractionOutcome,
    NetworkingMetrics,
    OutreachEvent,
    OutreachStatus,
    RelationshipMemory,
    RelationshipStatus,
)


def test_enum_behavior() -> None:
    """Validate string enum values."""
    assert ContactType.RECRUITER.value == "recruiter"
    assert RelationshipStatus.ACTIVE.value == "active"
    assert OutreachStatus.SENT.value == "sent"
    assert CommunicationStyle.DIRECT.value == "direct"
    assert InteractionOutcome.POSITIVE.value == "positive"


def test_contact_base_required_fields() -> None:
    """Validate that missing required fields raise a validation error."""
    with pytest.raises(ValidationError):
        ContactBase()  # type: ignore


def test_contact_base_optional_fields() -> None:
    """Validate optional field handling and defaults."""
    contact = ContactBase(first_name="Jane")
    assert contact.first_name == "Jane"
    assert contact.last_name is None
    assert contact.contact_type == ContactType.OTHER
    assert contact.metadata == {}


def test_invalid_enum_rejection() -> None:
    """Validate rejection of invalid enum values."""
    with pytest.raises(ValidationError) as exc_info:
        ContactBase(first_name="Jane", contact_type="invalid_type")  # type: ignore

    assert "Input should be" in str(exc_info.value)


def test_uuid_and_timestamps_generation() -> None:
    """Validate UUIDs and timestamps default behavior."""
    contact_id = uuid.uuid4()
    memory = RelationshipMemory(contact_id=contact_id)

    assert isinstance(memory.id, uuid.UUID)
    assert memory.contact_id == contact_id
    assert isinstance(memory.created_at, datetime)
    assert memory.created_at.tzinfo == UTC
    assert memory.updated_at >= memory.created_at
    assert memory.key_facts == []


def test_schema_serialization() -> None:
    """Validate complex serialization dumps."""
    contact_id = uuid.uuid4()
    outreach = OutreachEvent(
        contact_id=contact_id,
        status=OutreachStatus.SENT,
        method="Email",
        content="Hello!"
    )
    data = outreach.model_dump()

    assert data["contact_id"] == contact_id
    assert data["status"] == "sent"
    assert data["method"] == "Email"
    assert isinstance(data["id"], uuid.UUID)
    assert isinstance(data["created_at"], datetime)


def test_contact_update_optionality() -> None:
    """Validate all update fields are fully optional."""
    update = ContactUpdate()
    assert update.first_name is None
    assert update.status is None

    update_with_data = ContactUpdate(first_name="John", status=RelationshipStatus.ARCHIVED)
    assert update_with_data.first_name == "John"
    assert update_with_data.status == RelationshipStatus.ARCHIVED


def test_contact_response_enforcements() -> None:
    """Validate full response object enforcement constraints."""
    contact_id = uuid.uuid4()
    now_ts = datetime.now(UTC)

    response = ContactResponse(
        id=contact_id,
        first_name="Alice",
        status=RelationshipStatus.NEW,
        created_at=now_ts,
        updated_at=now_ts,
    )

    assert response.id == contact_id
    assert response.first_name == "Alice"
    assert response.status == RelationshipStatus.NEW


def test_followup_and_communication_profile() -> None:
    """Validate recommendation schemas."""
    contact_id = uuid.uuid4()
    now_ts = datetime.now(UTC)

    profile = CommunicationProfile(contact_id=contact_id)
    assert profile.style == CommunicationStyle.UNKNOWN
    assert profile.engagement_score == 0.0

    rec = FollowupRecommendation(
        contact_id=contact_id,
        suggested_date=now_ts,
        reasoning="Good time to follow up",
    )
    assert rec.priority == 1
    assert rec.draft_message is None


def test_metrics_defaults() -> None:
    """Validate numeric defaults across an analytics payload."""
    metrics = NetworkingMetrics()
    assert metrics.total_contacts == 0
    assert metrics.response_rate_percent == 0.0
    assert metrics.recent_activity_count == 0

def test_contact_create_validation() -> None:
    """Validate ContactCreate uses base configurations."""
    contact = ContactCreate(first_name="Bob")
    assert contact.first_name == "Bob"
