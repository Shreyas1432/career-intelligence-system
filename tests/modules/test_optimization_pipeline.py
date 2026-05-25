from typing import Any

import numpy as np
import pytest
from sqlalchemy.orm import Session

from src.core.database.models import JobIntelligence, UserProfile
from src.modules.automation import (
    OptimizationPipelineOrchestrator,
    OptimizationPipelineResponse,
)
from src.modules.positioning.profile import (
    CommunicationPreferencesSchema,
    ExperienceItemSchema,
    PositioningSchema,
    UserProfileCreate,
    UserProfileResponse,
    UserProfileService,
)
from src.modules.scraping.schemas import JobDomain


class MockSentenceTransformer:
    """
    Mock SentenceTransformer model that returns deterministic unit vectors
    of 384 dimensions to prevent internet hits or local cache loads in tests.
    """

    def encode(
        self, texts: str | list[str], *_args: Any, **_kwargs: Any
    ) -> np.ndarray[Any, Any]:
        if isinstance(texts, str):
            val = np.zeros(384, dtype=np.float32)
            val[0] = 1.0
            return val
        count = len(texts)
        val = np.zeros((count, 384), dtype=np.float32)
        val[:, 0] = 1.0
        return val


@pytest.fixture(autouse=True)
def mock_embedding_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Mock the underlying SentenceTransformer and EmbeddingPipeline
    so all embedding tasks remain fast and deterministic.
    """
    mock_model = MockSentenceTransformer()
    monkeypatch.setattr(
        "src.modules.matching.embeddings.SentenceTransformer", lambda _name: mock_model
    )

    async def mock_embed_profile(*_args: Any, **_kwargs: Any) -> list[float]:
        return [0.1] * 384

    async def mock_embed_job(*_args: Any, **_kwargs: Any) -> list[float]:
        return [0.2] * 384

    monkeypatch.setattr(
        "src.modules.matching.EmbeddingPipeline.embed_profile",
        mock_embed_profile,
    )
    monkeypatch.setattr(
        "src.modules.matching.EmbeddingPipeline.embed_job",
        mock_embed_job,
    )


@pytest.fixture
def rich_test_profile(db_session: Session) -> UserProfileResponse:
    profile_data = UserProfileCreate(
        full_name="Jane Doe",
        email="jane.doe@example.com",
        skills=["Python", "SQL", "Docker"],
        experience_summary="5 years experience",
        domains=["software_engineering"],
        target_industries=["Healthcare"],
        target_roles=["Senior Python Architect", "Software Engineer"],
        positioning=PositioningSchema(headline="Senior Software Engineer", years_of_experience=5),
        experience=[
            ExperienceItemSchema(
                title="Software Engineer",
                company="Tech Corp",
                start_date="2020-01",
                end_date="2023-01",
                description="Built Python backend services. Saved $50k in infrastructure costs.",
            )
        ],
        communication_preferences=CommunicationPreferencesSchema(
            channels=["email"], digest_frequency="weekly"
        ),
        additional_metadata={
            "projects": [
                {
                    "metadata": {
                        "title": "AI Pipeline",
                        "role": "Architect",
                        "description": "ML training pipelines",
                    },
                    "architecture": {
                        "design_patterns": ["Microservices"],
                        "database_setup": "Postgres",
                        "hosting_or_cloud": "AWS",
                    },
                    "technologies": ["Python", "PyTorch"],
                    "business_goals": ["Reduce training cost by 40%"],
                }
            ]
        },
    )
    return UserProfileService.create_or_update_profile(db_session, profile_data)


@pytest.fixture
def rich_test_job(db_session: Session) -> JobIntelligence:
    job = JobIntelligence(
        url_hash="job_opt_pipeline_hash",
        url="https://example.com/job/opt-pipeline",
        content_hash="content_opt_pipeline_hash",
        title="Python Engineer",
        company="TechCorp",
        location="Remote",
        normalized_skills=["Python", "SQL", "Docker"],
        domain=JobDomain.SOFTWARE_ENGINEERING,
        experience_required="3 years",
        sponsorship_signals={"status": "unknown", "confidence": 0.0},
    )
    db_session.add(job)
    db_session.flush()
    return job


@pytest.mark.asyncio
async def test_pipeline_success_flow(
    db_session: Session,
    rich_test_profile: UserProfileResponse,
    rich_test_job: JobIntelligence,
) -> None:
    orchestrator = OptimizationPipelineOrchestrator(db_session)

    outreach_rec = {
        "name": "Sarah",
        "title": "Recruiter",
        "company": "TechCorp",
        "role_type": "recruiter",
    }
    project = {
        "metadata": {
            "title": "Data Pipeline",
            "role": "Data Engineer",
            "description": "ETL pipelines",
        },
        "architecture": {
            "design_patterns": ["MVC"],
            "database_setup": "MySQL",
            "hosting_or_cloud": "GCP",
        },
        "technologies": ["Python", "Airflow"],
        "business_goals": ["Improve query performance by 25%"],
    }

    response = await orchestrator.run_pipeline(
        job_intelligence_id=rich_test_job.id,
        profile_id=rich_test_profile.id,
        outreach_recipient=outreach_rec,
        project_to_frame=project,
    )

    assert isinstance(response, OptimizationPipelineResponse)

    # Verify statuses
    assert len(response.step_statuses) == 7
    for step_name, status in response.step_statuses.items():
        assert status.status == "success", f"Step {step_name} failed: {status.error_message}"

    # Verify outputs are populated
    assert response.opportunity_ranking is not None
    assert response.positioning is not None
    assert response.resume_strategy is not None
    assert response.resume_tailoring is not None
    assert response.linkedin_optimization is not None
    assert response.outreach_draft is not None
    assert response.portfolio_framing is not None

    # Verify explanation contains summary details
    assert "opportunity_analysis" in response.explanation
    assert "TechCorp" in response.explanation


@pytest.mark.asyncio
async def test_pipeline_skips_optional_steps_if_no_input(
    db_session: Session,
    rich_test_profile: UserProfileResponse,
    rich_test_job: JobIntelligence,
) -> None:
    # Clear additional metadata of user profile directly in the database
    db_profile = db_session.query(UserProfile).filter_by(id=rich_test_profile.id).first()
    assert db_profile is not None
    db_profile.additional_metadata = {}
    db_session.add(db_profile)
    db_session.flush()

    orchestrator = OptimizationPipelineOrchestrator(db_session)

    response = await orchestrator.run_pipeline(
        job_intelligence_id=rich_test_job.id,
        profile_id=rich_test_profile.id,
    )

    assert response.step_statuses["outreach_preparation"].status == "skipped"
    assert response.step_statuses["portfolio_recommendations"].status == "skipped"

    assert response.outreach_draft is None
    assert response.portfolio_framing is None


@pytest.mark.asyncio
async def test_pipeline_partial_failure_propagation(
    db_session: Session,
    rich_test_profile: UserProfileResponse,
    rich_test_job: JobIntelligence,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Force StrategicPositioningEngine to raise an error
    def mock_generate_positioning(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("Positioning model computation failed")

    monkeypatch.setattr(
        "src.modules.automation.orchestration.StrategicPositioningEngine.generate_positioning",
        mock_generate_positioning,
    )

    orchestrator = OptimizationPipelineOrchestrator(db_session)

    response = await orchestrator.run_pipeline(
        job_intelligence_id=rich_test_job.id,
        profile_id=rich_test_profile.id,
    )

    # Fatal step opportunity_analysis succeeds, but positioning_generation fails
    assert response.step_statuses["opportunity_analysis"].status == "success"
    assert response.step_statuses["positioning_generation"].status == "failed"
    assert "Positioning model computation failed" in str(
        response.step_statuses["positioning_generation"].error_message
    )

    # Rest of the pipeline (resume_strategy, resume_tailoring, etc.) still completes successfully
    assert response.step_statuses["resume_strategy"].status == "success"
    assert response.step_statuses["resume_tailoring"].status == "success"
    assert response.step_statuses["linkedin_optimization"].status == "success"


@pytest.mark.asyncio
async def test_pipeline_fatal_failure_aborts(
    db_session: Session, rich_test_profile: UserProfileResponse
) -> None:
    orchestrator = OptimizationPipelineOrchestrator(db_session)

    # Invalid job ID triggers fatal error in opportunity_analysis
    with pytest.raises(RuntimeError, match="Fatal step 'opportunity_analysis' failed"):
        await orchestrator.run_pipeline(
            job_intelligence_id=999999,  # Non-existent ID
            profile_id=rich_test_profile.id,
        )
