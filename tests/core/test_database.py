from src.core.database.models import (
    Application,
    Contact,
    InteractionSummary,
    Job,
    StrategyInsight,
    UserProfile,
)
from src.core.database.repositories import (
    ApplicationRepository,
    ContactRepository,
    InteractionRepository,
    JobRepository,
    StrategyInsightRepository,
    UserProfileRepository,
)


def test_user_profile_repository(db_session):
    """
    Test basic CRUD operations on UserProfile.
    """
    repo = UserProfileRepository(db_session)

    # Create
    profile = UserProfile(
        full_name="Alice Smith",
        email="alice@example.com",
        target_roles="Data Engineer, AI Architect",
        skills="Python, SQL, PySpark",
    )
    repo.create(profile)
    db_session.commit()

    # Read
    fetched = repo.get_by_email("alice@example.com")
    assert fetched is not None
    assert fetched.full_name == "Alice Smith"

    # Read active singleton
    active = repo.get_active_profile()
    assert active.email == "alice@example.com"

    # Delete
    repo.delete(fetched.id)
    db_session.commit()
    assert repo.get_by_email("alice@example.com") is None


def test_job_application_cascade_deletion(db_session):
    """
    Test job repository operations and cascade deletion to applications.
    """
    job_repo = JobRepository(db_session)
    app_repo = ApplicationRepository(db_session)

    # 1. Create Job
    job = Job(
        title="Software Engineer",
        company="Tech Corp",
        location="Remote",
        salary_range="$120k-$150k",
    )
    job_repo.create(job)
    db_session.commit()
    assert job.id is not None

    # 2. Link Application
    app = Application(job_id=job.id, status="Applied", resume_version="v1.2_Tech")
    app_repo.create(app)
    db_session.commit()
    assert app.id is not None

    # Verify relationship
    assert len(job.applications) == 1
    assert job.applications[0].status == "Applied"

    # 3. Eager loading / Lookup
    fetched_app = app_repo.get_application_with_details(app.id)
    assert fetched_app.job.company == "Tech Corp"

    # 4. Cascade delete job -> application should be deleted automatically
    job_repo.delete(job.id)
    db_session.commit()

    # Application must no longer exist in the DB
    assert app_repo.get_by_id(app.id) is None


def test_contacts_interactions_mapping(db_session):
    """
    Test contacts repository operations and interaction summaries logic.
    """
    contact_repo = ContactRepository(db_session)
    interaction_repo = InteractionRepository(db_session)

    # Create Contact
    contact = Contact(
        name="Bob recruiter", role="Recruiting Lead", company="AI Labs", email="bob@ailabs.com"
    )
    contact_repo.create(contact)
    db_session.commit()

    # Create Interaction
    interaction = InteractionSummary(
        contact_id=contact.id,
        interaction_type="LinkedIn Message",
        summary="Sent initial connection request introducing self.",
    )
    interaction_repo.create(interaction)
    db_session.commit()

    # Verify mappings
    assert len(contact.interactions) == 1
    assert contact.interactions[0].interaction_type == "LinkedIn Message"

    recent = interaction_repo.get_recent_interactions(limit=1)
    assert len(recent) == 1
    assert recent[0].summary == "Sent initial connection request introducing self."


def test_strategy_insight_repository(db_session):
    """
    Test StrategyInsight retrieval logic.
    """
    repo = StrategyInsightRepository(db_session)

    insight = StrategyInsight(
        topic="Resume Feedback",
        insight="Increase visibility of lead engineer metrics.",
        action_plan="Rewrite past job summaries targeting key quantitative numbers.",
    )
    repo.create(insight)
    db_session.commit()

    fetched = repo.get_by_topic("Resume Feedback")
    assert len(fetched) == 1
    assert fetched[0].insight == "Increase visibility of lead engineer metrics."
