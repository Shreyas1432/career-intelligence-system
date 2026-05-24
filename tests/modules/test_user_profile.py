import pytest
from sqlalchemy.orm import Session

from src.core.database.models import UserProfile
from src.modules.user_profile import (
    AvoidRoleFiltersSchema,
    CommunicationPreferencesSchema,
    ExperienceItemSchema,
    PositioningSchema,
    RolePreferencesSchema,
    UserProfileCreate,
    UserProfileResponse,
    UserProfileService,
)


def test_user_profile_schemas_validation() -> None:
    """
    Test validation and defaults for Pydantic user profile schemas.
    """
    # 1. Invalid email format
    with pytest.raises(ValueError, match="invalid email format"):
        UserProfileCreate(
            full_name="John Doe",
            email="invalid-email",
            skills=["Python"],
        )

    # 2. Empty full name
    with pytest.raises(ValueError, match="String should have at least 1 character"):
        UserProfileCreate(
            full_name="   ",
            email="john@example.com",
            skills=["Python"],
        )

    # 3. Invalid digest frequency
    with pytest.raises(ValueError, match="digest_frequency must be one of"):
        CommunicationPreferencesSchema(digest_frequency="hourly")

    # 4. Valid inputs and defaults
    profile_in = UserProfileCreate(
        full_name="John Doe",
        email="john@example.com",
        skills=["Python"],
        positioning=PositioningSchema(headline="AI Eng", years_of_experience=3),
    )
    assert profile_in.full_name == "John Doe"
    assert profile_in.email == "john@example.com"
    assert profile_in.communication_preferences.notifications_enabled is True
    assert profile_in.communication_preferences.digest_frequency == "weekly"
    assert profile_in.role_preferences.min_salary is None


def test_create_and_read_profile_persistence(db_session: Session) -> None:
    """
    Test persistence of UserProfile, including JSON columns and skill normalization.
    """
    profile_in = UserProfileCreate(
        full_name="Jane Doe",
        email="jane@example.com",
        target_roles=["Data Scientist", "ML Engineer"],
        skills=["PySpark", "Scikit-Learn", "Git"],
        experience_summary="Experienced data professional",
        domains=["Data & AI"],
        positioning=PositioningSchema(
            headline="Lead Data Scientist", years_of_experience=7, seniority_level="Lead"
        ),
        experience=[
            ExperienceItemSchema(
                title="Senior Data Scientist",
                company="Tech Corp",
                start_date="2021-01",
                end_date=None,
                description="Developing ML models",
            )
        ],
        target_industries=["Healthcare", "Finance"],
        communication_preferences=CommunicationPreferencesSchema(
            notifications_enabled=True,
            channels=["email", "slack"],
            digest_frequency="daily",
        ),
        role_preferences=RolePreferencesSchema(
            job_types=["full_time"],
            work_modes=["remote"],
            min_salary=150000.0,
            locations=["New York", "San Francisco"],
        ),
        avoid_role_filters=AvoidRoleFiltersSchema(
            avoid_titles=["Manager"],
            avoid_companies=["Old Company"],
            avoid_keywords=["legacy"],
        ),
        additional_metadata={"onboarding_completed": True},
    )

    # Create profile
    created = UserProfileService.create_or_update_profile(db_session, profile_in)

    assert created.id is not None
    assert created.full_name == "Jane Doe"
    assert created.email == "jane@example.com"
    # Verify skill normalization (PySpark -> Spark)
    assert "Spark" in created.skills
    assert "PySpark" not in created.skills
    assert created.target_roles == ["Data Scientist", "ML Engineer"]
    assert created.domains == ["Data & AI"]
    assert created.positioning.headline == "Lead Data Scientist"
    assert len(created.experience) == 1
    assert created.experience[0].title == "Senior Data Scientist"
    assert created.experience[0].end_date is None
    assert created.communication_preferences.digest_frequency == "daily"
    assert created.role_preferences.min_salary == 150000.0
    assert created.avoid_role_filters.avoid_titles == ["Manager"]
    assert created.additional_metadata.get("onboarding_completed") is True

    # Read profile
    fetched = UserProfileService.get_profile(db_session)
    assert fetched is not None
    assert fetched.id == created.id
    assert fetched.email == "jane@example.com"
    assert "Spark" in fetched.skills


def test_update_profile_singleton_behavior(db_session: Session) -> None:
    """
    Test that updating a profile modifies the existing singleton profile row in the database.
    """
    profile_1 = UserProfileCreate(
        full_name="Alice Smith",
        email="alice@example.com",
        skills=["Python"],
    )
    res_1 = UserProfileService.create_or_update_profile(db_session, profile_1)
    assert res_1.full_name == "Alice Smith"

    # Count profiles in database
    count_1 = db_session.query(UserProfile).count()
    assert count_1 == 1

    profile_2 = UserProfileCreate(
        full_name="Alice Jones",
        email="alice.jones@example.com",
        skills=["Python", "SQL"],
    )
    res_2 = UserProfileService.create_or_update_profile(db_session, profile_2)
    assert res_2.full_name == "Alice Jones"
    assert res_2.email == "alice.jones@example.com"

    # Verify that the row was updated, not added
    count_2 = db_session.query(UserProfile).count()
    assert count_2 == 1

    # Verify database state
    fetched = UserProfileService.get_profile(db_session)
    assert fetched is not None
    assert fetched.full_name == "Alice Jones"
    assert fetched.email == "alice.jones@example.com"


def test_should_avoid_job_filter_logic() -> None:
    """
    Test job avoidance filter matching on titles, companies, and keywords.
    """
    profile = UserProfileResponse(
        id=1,
        full_name="Test User",
        email="test@example.com",
        avoid_role_filters=AvoidRoleFiltersSchema(
            avoid_titles=["Manager", "Director"],
            avoid_companies=["Evil Corp", "Spam Inc"],
            avoid_keywords=["COBOL", "on-call support", "PHP"],
        ),
        created_at=pytest.importorskip("datetime").datetime.utcnow(),
        updated_at=pytest.importorskip("datetime").datetime.utcnow(),
    )

    # 1. Match on title
    assert (
        UserProfileService.should_avoid_job(
            profile,
            job_title="Engineering Manager",
            company="Good Co",
            job_description="We write Python.",
        )
        is True
    )

    # 2. Match on company
    assert (
        UserProfileService.should_avoid_job(
            profile,
            job_title="Software Engineer",
            company="Evil Corp Ltd",
            job_description="We write Python.",
        )
        is True
    )

    # 3. Match on keyword in description
    assert (
        UserProfileService.should_avoid_job(
            profile,
            job_title="Software Engineer",
            company="Good Co",
            job_description="We need 10 years of COBOL maintenance.",
        )
        is True
    )

    # 4. Match on keyword in title
    assert (
        UserProfileService.should_avoid_job(
            profile,
            job_title="PHP Developer",
            company="Good Co",
            job_description="We write modern web apps.",
        )
        is True
    )

    # 5. Clean / Safe job (no match)
    assert (
        UserProfileService.should_avoid_job(
            profile,
            job_title="Python/React Developer",
            company="Good Co",
            job_description="Modern web app development with zero on-call.",
        )
        is False
    )

    # 6. Empty filters do not trigger false positives
    empty_profile = UserProfileResponse(
        id=1,
        full_name="Test User",
        email="test@example.com",
        avoid_role_filters=AvoidRoleFiltersSchema(
            avoid_titles=[],
            avoid_companies=[],
            avoid_keywords=[],
        ),
        created_at=pytest.importorskip("datetime").datetime.utcnow(),
        updated_at=pytest.importorskip("datetime").datetime.utcnow(),
    )
    assert (
        UserProfileService.should_avoid_job(
            empty_profile,
            job_title="Engineering Manager",
            company="Evil Corp",
            job_description="We write COBOL.",
        )
        is False
    )
