import re
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy.orm import Session

from src.core.database.models import UserProfile
from src.core.database.repositories.profile import UserProfileRepository
from src.modules.scraping.normalization import canonicalize_skills

EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")


# ------------------------------------------------------------------------------
# Schemas
# ------------------------------------------------------------------------------

class PositioningSchema(BaseModel):
    """
    Structured positioning profile details (elevator pitch, headline, seniority level).
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    headline: str | None = Field(default=None, max_length=200)
    years_of_experience: int | None = Field(default=None, ge=0)
    seniority_level: str | None = Field(default=None, max_length=50)


class ExperienceItemSchema(BaseModel):
    """
    Structured professional experience entry.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    title: str = Field(min_length=1, max_length=150)
    company: str = Field(min_length=1, max_length=150)
    start_date: str = Field(min_length=1, max_length=50)  # e.g. "2020-01" or "Jan 2020"
    end_date: str | None = Field(default=None, max_length=50)  # None indicates "Present"
    description: str | None = Field(default=None, max_length=2000)


class CommunicationPreferencesSchema(BaseModel):
    """
    Structured communication preferences configuration.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    notifications_enabled: bool = Field(default=True)
    channels: list[str] = Field(default_factory=lambda: ["email"])
    digest_frequency: str = Field(default="weekly")  # "daily", "weekly", "never"

    @field_validator("digest_frequency")
    @classmethod
    def validate_digest_frequency(cls, v: str) -> str:
        freq = v.strip().lower()
        allowed = {"daily", "weekly", "never"}
        if freq not in allowed:
            raise ValueError(f"digest_frequency must be one of {allowed}")
        return freq


class RolePreferencesSchema(BaseModel):
    """
    Structured target role preferences (job types, remote policy, salary target).
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    job_types: list[str] = Field(default_factory=list)  # e.g., ["full_time", "contract"]
    work_modes: list[str] = Field(default_factory=list)  # e.g., ["remote", "hybrid", "onsite"]
    min_salary: float | None = Field(default=None, ge=0.0)
    locations: list[str] = Field(default_factory=list)


class AvoidRoleFiltersSchema(BaseModel):
    """
    Exclusion filters to automatically flag/filter jobs the user wants to avoid.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    avoid_titles: list[str] = Field(default_factory=list)
    avoid_companies: list[str] = Field(default_factory=list)
    avoid_keywords: list[str] = Field(default_factory=list)


class UserProfileCreate(BaseModel):
    """
    Schema for creating or updating a User Profile.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    full_name: str = Field(min_length=1, max_length=100)
    email: str = Field(max_length=255)
    target_roles: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    experience_summary: str | None = Field(default=None, max_length=1000)
    domains: list[str] = Field(default_factory=list)
    positioning: PositioningSchema = Field(default_factory=PositioningSchema)
    experience: list[ExperienceItemSchema] = Field(default_factory=list)
    target_industries: list[str] = Field(default_factory=list)
    communication_preferences: CommunicationPreferencesSchema = Field(
        default_factory=CommunicationPreferencesSchema
    )
    role_preferences: RolePreferencesSchema = Field(default_factory=RolePreferencesSchema)
    avoid_role_filters: AvoidRoleFiltersSchema = Field(default_factory=AvoidRoleFiltersSchema)
    additional_metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("full_name")
    @classmethod
    def validate_full_name(cls, v: str) -> str:
        name = v.strip()
        if not name:
            raise ValueError("full_name cannot be empty or whitespace only")
        return name

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        email = v.strip().lower()
        if not EMAIL_REGEX.match(email):
            raise ValueError("invalid email format")
        return email


class UserProfileResponse(BaseModel):
    """
    Validated user profile response structure representing normalized user intelligence.
    """

    model_config = ConfigDict(extra="ignore", from_attributes=True)

    id: int
    full_name: str
    email: str
    target_roles: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    experience_summary: str | None = None
    domains: list[str] = Field(default_factory=list)
    positioning: PositioningSchema = Field(default_factory=PositioningSchema)
    experience: list[ExperienceItemSchema] = Field(default_factory=list)
    target_industries: list[str] = Field(default_factory=list)
    communication_preferences: CommunicationPreferencesSchema = Field(
        default_factory=CommunicationPreferencesSchema
    )
    role_preferences: RolePreferencesSchema = Field(default_factory=RolePreferencesSchema)
    avoid_role_filters: AvoidRoleFiltersSchema = Field(default_factory=AvoidRoleFiltersSchema)
    additional_metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime

    @field_validator("target_roles", "skills", mode="before")
    @classmethod
    def parse_comma_separated_strings(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        if isinstance(v, list):
            return [str(s).strip() for s in v if str(s).strip()]
        return []

    @field_validator("domains", "target_industries", "experience", mode="before")
    @classmethod
    def parse_nullable_list(cls, v: Any) -> Any:
        if v is None:
            return []
        return v

    @field_validator(
        "positioning",
        "communication_preferences",
        "role_preferences",
        "avoid_role_filters",
        "additional_metadata",
        mode="before",
    )
    @classmethod
    def parse_nullable_dict(cls, v: Any) -> Any:
        if v is None:
            return {}
        return v


# ------------------------------------------------------------------------------
# Service
# ------------------------------------------------------------------------------

class UserProfileService:
    """
    Business logic and orchestration for User Profile operations.
    """

    @staticmethod
    def get_profile(session: Session) -> UserProfileResponse | None:
        """
        Fetch the active user profile, parsed into the normalized schema.
        """
        repo = UserProfileRepository(session)
        db_profile = repo.get_active_profile()
        if not db_profile:
            return None
        return UserProfileResponse.model_validate(db_profile)

    @staticmethod
    def create_or_update_profile(session: Session, data: UserProfileCreate) -> UserProfileResponse:
        """
        Create a new user profile or update the existing active profile.
        Normalizes input skills against the system's skill taxonomy.
        """
        repo = UserProfileRepository(session)
        db_profile = repo.get_active_profile()

        # Normalize skills using the SkillNormalizer
        normalized_skills = canonicalize_skills(data.skills)

        if not db_profile:
            db_profile = UserProfile(
                full_name=data.full_name,
                email=data.email,
                target_roles=", ".join(data.target_roles) if data.target_roles else None,
                skills=", ".join(normalized_skills) if normalized_skills else None,
                experience_summary=data.experience_summary,
                domains=data.domains,
                positioning=data.positioning.model_dump(),
                experience=[item.model_dump() for item in data.experience],
                target_industries=data.target_industries,
                communication_preferences=data.communication_preferences.model_dump(),
                role_preferences=data.role_preferences.model_dump(),
                avoid_role_filters=data.avoid_role_filters.model_dump(),
                additional_metadata=data.additional_metadata,
            )
            repo.create(db_profile)
        else:
            db_profile.full_name = data.full_name
            db_profile.email = data.email
            db_profile.target_roles = ", ".join(data.target_roles) if data.target_roles else None
            db_profile.skills = ", ".join(normalized_skills) if normalized_skills else None
            db_profile.experience_summary = data.experience_summary
            db_profile.domains = data.domains
            db_profile.positioning = data.positioning.model_dump()
            db_profile.experience = [item.model_dump() for item in data.experience]
            db_profile.target_industries = data.target_industries
            db_profile.communication_preferences = data.communication_preferences.model_dump()
            db_profile.role_preferences = data.role_preferences.model_dump()
            db_profile.avoid_role_filters = data.avoid_role_filters.model_dump()
            db_profile.additional_metadata = data.additional_metadata

        session.commit()
        session.refresh(db_profile)
        return UserProfileResponse.model_validate(db_profile)

    @staticmethod
    def should_avoid_job(
        profile: UserProfileResponse,
        job_title: str,
        company: str,
        job_description: str,
    ) -> bool:
        """
        Evaluate if a job matches any avoidance criteria defined in the user's profile.
        """
        filters = profile.avoid_role_filters
        title_lower = job_title.lower()
        company_lower = company.lower()
        desc_lower = job_description.lower()

        # 1. Check avoid_titles (case-insensitive substring match)
        for title in filters.avoid_titles:
            if title.strip() and title.lower() in title_lower:
                return True

        # 2. Check avoid_companies (case-insensitive substring match)
        for comp in filters.avoid_companies:
            if comp.strip() and comp.lower() in company_lower:
                return True

        # 3. Check avoid_keywords (case-insensitive substring match in title/description)
        for kw in filters.avoid_keywords:
            kw_clean = kw.strip()
            if kw_clean and (kw_clean.lower() in desc_lower or kw_clean.lower() in title_lower):
                return True

        return False
