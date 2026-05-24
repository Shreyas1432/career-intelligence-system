from sqlalchemy.orm import Session

from src.core.database.models import UserProfile
from src.core.database.repositories.profile import UserProfileRepository
from src.modules.skill_normalization import canonicalize_skills
from src.modules.user_profile.schemas import UserProfileCreate, UserProfileResponse


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
