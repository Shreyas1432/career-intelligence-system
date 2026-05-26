from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


def _now() -> datetime:
    """Helper to get current time in UTC."""
    return datetime.now(UTC)


class ContactType(StrEnum):
    """Types of professional contacts."""
    RECRUITER = "recruiter"
    HIRING_MANAGER = "hiring_manager"
    PEER = "peer"
    MENTOR = "mentor"
    ALUMNI = "alumni"
    INDUSTRY_LEADER = "industry_leader"
    OTHER = "other"


class RelationshipStatus(StrEnum):
    """The current state of the relationship."""
    NEW = "new"
    WARM = "warm"
    ACTIVE = "active"
    DORMANT = "dormant"
    ARCHIVED = "archived"
    CONTACTED = "contacted"
    RESPONDED = "responded"
    FOLLOWUP_PENDING = "followup_pending"
    STALE = "stale"


class OutreachStatus(StrEnum):
    """Status of specific outreach events."""
    PLANNED = "planned"
    DRAFTED = "drafted"
    SENT = "sent"
    REPLIED = "replied"
    IGNORED = "ignored"
    BOUNCED = "bounced"


class CommunicationStyle(StrEnum):
    """Observed or preferred communication style."""
    FORMAL = "formal"
    CASUAL = "casual"
    DIRECT = "direct"
    ANALYTICAL = "analytical"
    UNKNOWN = "unknown"


class InteractionOutcome(StrEnum):
    """Outcome of an interaction."""
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    NO_RESPONSE = "no_response"
    ACTION_REQUIRED = "action_required"


class ContactBase(BaseModel):
    """Base schema for a professional contact."""
    first_name: str = Field(description="First name of the contact")
    last_name: str | None = Field(default=None, description="Last name of the contact")
    company: str | None = Field(default=None, description="Contact's current company")
    title: str | None = Field(default=None, description="Contact's job title")
    contact_type: ContactType = Field(default=ContactType.OTHER, description="Role or relationship type")
    linkedin_url: str | None = Field(default=None, description="LinkedIn profile URL")
    email: str | None = Field(default=None, description="Email address")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional context or links")


class ContactCreate(ContactBase):
    """Schema for creating a new contact."""
    pass


class ContactUpdate(BaseModel):
    """Schema for updating an existing contact."""
    first_name: str | None = None
    last_name: str | None = None
    company: str | None = None
    title: str | None = None
    contact_type: ContactType | None = None
    linkedin_url: str | None = None
    email: str | None = None
    status: RelationshipStatus | None = None
    metadata: dict[str, Any] | None = None


class ContactResponse(ContactBase):
    """Schema for retrieving a contact."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    status: RelationshipStatus
    created_at: datetime
    updated_at: datetime


class RelationshipMemory(BaseModel):
    """Stores persistent memory and context about the relationship."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID = Field(default_factory=uuid4)
    contact_id: UUID
    key_facts: list[str] = Field(default_factory=list, description="Important facts to remember")
    shared_interests: list[str] = Field(default_factory=list, description="Topics of mutual interest")
    pain_points: list[str] = Field(default_factory=list, description="Known professional challenges")
    last_interaction_date: datetime | None = None
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


class OutreachEvent(BaseModel):
    """Schema for an individual outreach attempt or interaction."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID = Field(default_factory=uuid4)
    contact_id: UUID
    status: OutreachStatus
    method: str = Field(description="Medium of outreach, e.g., LinkedIn, Email")
    content: str | None = Field(default=None, description="Message content or notes")
    scheduled_for: datetime | None = None
    completed_at: datetime | None = None
    outcome: InteractionOutcome | None = None
    created_at: datetime = Field(default_factory=_now)


class CommunicationProfile(BaseModel):
    """Inferred communication preferences from interaction history."""
    model_config = ConfigDict(from_attributes=True)

    contact_id: UUID
    style: CommunicationStyle = Field(default=CommunicationStyle.UNKNOWN)
    best_time_to_reach: str | None = Field(default=None, description="E.g., Tuesday mornings")
    preferred_channel: str | None = Field(default=None, description="E.g., LinkedIn, Email")
    engagement_score: float = Field(default=0.0, description="0.0 to 1.0 responsiveness rating")
    insights: list[str] = Field(default_factory=list, description="AI-generated insights on communication")


class FollowupRecommendation(BaseModel):
    """AI-generated recommendation for next steps."""
    contact_id: UUID
    suggested_date: datetime
    reasoning: str = Field(description="Why this follow-up is recommended")
    draft_message: str | None = Field(default=None, description="Suggested message content")
    priority: int = Field(default=1, description="1 (high) to 5 (low)")


class NetworkingMetrics(BaseModel):
    """Aggregated analytics for networking efforts."""
    total_contacts: int = 0
    active_conversations: int = 0
    response_rate_percent: float = 0.0
    meetings_booked: int = 0
    recent_activity_count: int = 0
