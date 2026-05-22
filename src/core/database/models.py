from datetime import datetime
from typing import Any, Optional

from sqlalchemy import JSON, DateTime, ForeignKey, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """
    SQLAlchemy Base class utilizing modern PEP 681 typed declarations.
    """

    pass


class UserProfile(Base):
    """
    User profile configuration details.
    """

    __tablename__ = "user_profile"

    id: Mapped[int] = mapped_column(primary_key=True)
    full_name: Mapped[str] = mapped_column(String(100), nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    target_roles: Mapped[str | None] = mapped_column(Text)  # Comma-separated target titles
    skills: Mapped[str | None] = mapped_column(Text)  # Comma-separated list of competencies
    experience_summary: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    def __repr__(self) -> str:
        return f"<UserProfile {self.email}>"


class Job(Base):
    """
    Target Job tracking definitions.
    """

    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    company: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    location: Mapped[str | None] = mapped_column(String(100))
    description: Mapped[str | None] = mapped_column(Text)
    salary_range: Mapped[str | None] = mapped_column(String(100))
    url: Mapped[str | None] = mapped_column(String(512))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    applications: Mapped[list["Application"]] = relationship(
        "Application", back_populates="job", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Job {self.title} at {self.company}>"


class Application(Base):
    """
    Application process tracking status.
    """

    __tablename__ = "applications"

    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="Applied", index=True)
    applied_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    notes: Mapped[str | None] = mapped_column(Text)
    resume_version: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    job: Mapped["Job"] = relationship("Job", back_populates="applications")
    interactions: Mapped[list["InteractionSummary"]] = relationship(
        "InteractionSummary", back_populates="application"
    )

    def __repr__(self) -> str:
        return f"<Application for Job {self.job_id} (Status: {self.status})>"


class Contact(Base):
    """
    Professional network contact cards.
    """

    __tablename__ = "contacts"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(150), nullable=False, index=True)
    role: Mapped[str | None] = mapped_column(String(100))
    company: Mapped[str | None] = mapped_column(String(100), index=True)
    email: Mapped[str | None] = mapped_column(String(255))
    phone: Mapped[str | None] = mapped_column(String(50))
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    interactions: Mapped[list["InteractionSummary"]] = relationship(
        "InteractionSummary", back_populates="contact", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Contact {self.name}>"


class InteractionSummary(Base):
    """
    Interaction logs tracking networking steps or interviews.
    """

    __tablename__ = "interaction_summaries"

    id: Mapped[int] = mapped_column(primary_key=True)
    contact_id: Mapped[int] = mapped_column(
        ForeignKey("contacts.id", ondelete="CASCADE"), nullable=False
    )
    application_id: Mapped[int | None] = mapped_column(
        ForeignKey("applications.id", ondelete="SET NULL")
    )
    date: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    interaction_type: Mapped[str] = mapped_column(
        String(100), default="Email"
    )  # Email, Call, LinkedIn
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    action_items: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    contact: Mapped["Contact"] = relationship("Contact", back_populates="interactions")
    application: Mapped[Optional["Application"]] = relationship(
        "Application", back_populates="interactions"
    )

    def __repr__(self) -> str:
        return f"<InteractionSummary type={self.interaction_type} on={self.date}>"


class StrategyInsight(Base):
    """
    Centralized repository of generated AI career plan suggestions.
    """

    __tablename__ = "strategy_insights"

    id: Mapped[int] = mapped_column(primary_key=True)
    topic: Mapped[str] = mapped_column(
        String(100), nullable=False, index=True
    )  # Resume Feedback, Target Skills
    insight: Mapped[str] = mapped_column(Text, nullable=False)
    action_plan: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<StrategyInsight topic={self.topic}>"


class JobIntelligence(Base):
    """
    Structured intelligence data captured from scraped/extracted job postings.
    """

    __tablename__ = "job_intelligence"

    id: Mapped[int] = mapped_column(primary_key=True)

    # URL hash for O(1) deduplication and fast indexing
    url_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    url: Mapped[str | None] = mapped_column(String(512))

    # Content hash for update detection (SHA-256 of raw text/description)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    # Raw job posting content (scraped description or HTML source)
    raw_content: Mapped[str | None] = mapped_column(Text)

    # Extraction Metadata
    title: Mapped[str | None] = mapped_column(String(255), index=True)
    company: Mapped[str | None] = mapped_column(String(255), index=True)
    location: Mapped[str | None] = mapped_column(String(255))
    experience_required: Mapped[str | None] = mapped_column(Text)
    salary_range: Mapped[str | None] = mapped_column(String(100))
    domain: Mapped[str | None] = mapped_column(String(100), index=True)
    employment_type: Mapped[str | None] = mapped_column(String(100))
    confidence_score: Mapped[float | None] = mapped_column()

    # ATS type (e.g. Workday, Greenhouse, Lever, Ashby, etc.)
    ats_type: Mapped[str | None] = mapped_column(String(100), index=True)

    # JSON lists/dicts mapped to SQLite Text columns for ease of storage in local single-user system
    normalized_skills: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    sponsorship_signals: Mapped[dict[str, Any] | None] = mapped_column(JSON)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    def __repr__(self) -> str:
        return f"<JobIntelligence {self.title} at {self.company} (URL Hash: {self.url_hash[:8]})>"

