from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import JSON, DateTime, Float, String
from sqlalchemy.orm import Mapped, mapped_column

from src.core.database.models import Base


def _uuid_str() -> str:
    return str(uuid4())

def _now() -> datetime:
    return datetime.now(UTC)

class RelationshipContactModel(Base):
    __tablename__ = "relationship_contacts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    first_name: Mapped[str] = mapped_column(String(100), nullable=False)
    last_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    company: Mapped[str | None] = mapped_column(String(100), nullable=True)
    title: Mapped[str | None] = mapped_column(String(100), nullable=True)
    contact_type: Mapped[str] = mapped_column(String(50), nullable=False)
    linkedin_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    status: Mapped[str] = mapped_column(String(50), nullable=False)
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)

class RelationshipMemoryModel(Base):
    __tablename__ = "relationship_memories"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    contact_id: Mapped[str] = mapped_column(String(36), nullable=False)
    key_facts: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    shared_interests: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    pain_points: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    last_interaction_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)

class OutreachEventModel(Base):
    __tablename__ = "outreach_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    contact_id: Mapped[str] = mapped_column(String(36), nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False)
    method: Mapped[str] = mapped_column(String(100), nullable=False)
    content: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    scheduled_for: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    outcome: Mapped[str | None] = mapped_column(String(50), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

class CommunicationProfileModel(Base):
    __tablename__ = "communication_profiles"

    contact_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    style: Mapped[str] = mapped_column(String(50), nullable=False)
    best_time_to_reach: Mapped[str | None] = mapped_column(String(100), nullable=True)
    preferred_channel: Mapped[str | None] = mapped_column(String(100), nullable=True)
    engagement_score: Mapped[float] = mapped_column(Float, default=0.0)
    insights: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)

class FollowupRecommendationModel(Base):
    __tablename__ = "followup_recommendations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    contact_id: Mapped[str] = mapped_column(String(36), nullable=False)
    suggested_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    reasoning: Mapped[str] = mapped_column(String(1024), nullable=False)
    draft_message: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    priority: Mapped[int] = mapped_column(default=1)
