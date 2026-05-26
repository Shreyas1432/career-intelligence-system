import pytest
from sqlalchemy.orm import Session

from src.core.database.models import JobIntelligence
from src.modules.matching import (
    OpportunityOrchestrator,
    OpportunityRankingResult,
)
from src.modules.positioning.profile import UserProfileCreate, UserProfileService
from src.modules.scraping.schemas import JobDomain


@pytest.fixture(autouse=True)
def mock_embedding_generation(monkeypatch):
    """
    Mock EmbeddingPipeline to return mock embeddings without calling HuggingFace.
    """

    async def mock_embed_profile(*_args, **_kwargs):
        return [0.1] * 384

    async def mock_embed_job(*_args, **_kwargs):
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
def test_profile(db_session: Session):
    profile_data = UserProfileCreate(
        full_name="Jane Doe",
        email="jane.doe@example.com",
        skills=["Python", "SQL"],
        experience_summary="5 years experience",
        domains=["software_engineering"],
        target_industries=["Healthcare"],
    )
    return UserProfileService.create_or_update_profile(db_session, profile_data)


@pytest.fixture
def test_job(db_session: Session):
    job = JobIntelligence(
        url_hash="job_orch_hash",
        url="https://example.com/job/orch",
        content_hash="content_orch_hash",
        title="Python Engineer",
        company="TechCorp",
        location="Remote",
        normalized_skills=["Python", "SQL"],
        domain=JobDomain.SOFTWARE_ENGINEERING,
        experience_required="3 years",
        sponsorship_signals={"status": "unknown", "confidence": 0.0},
    )
    db_session.add(job)
    db_session.flush()
    return job


@pytest.mark.asyncio
async def test_orchestrator_success_flow(db_session: Session, test_profile, test_job):
    from sqlalchemy import text

    db_session.execute(text("PRAGMA foreign_keys=ON;"))

    orchestrator = OpportunityOrchestrator(db_session)
    context = await orchestrator.run_pipeline(
        job_intelligence_id=test_job.id,
        profile_id=test_profile.id,
    )

    # 1. Verify all step statuses are successful
    assert len(context.step_statuses) == 8
    for step, details in context.step_statuses.items():
        assert details["status"] == "success", f"Step {step} failed: {details['error']}"

    # 2. Verify context models are populated
    assert context.profile is not None
    assert context.job_intelligence is not None
    assert context.profile_embedding == [0.1] * 384
    assert context.job_embedding == [0.2] * 384
    assert context.skill_match is not None
    assert context.domain_alignment is not None
    assert context.sponsorship is not None
    assert context.ranking is not None
    assert context.explainability is not None
    assert context.ranking_result_id is not None

    # 3. Verify database persistence
    db_result = (
        db_session.query(OpportunityRankingResult).filter_by(id=context.ranking_result_id).first()
    )
    assert db_result is not None
    assert db_result.overall_score == context.ranking.overall_score
    assert db_result.recommendation == context.ranking.recommendation.value
    assert db_result.run_number == 1


@pytest.mark.asyncio
async def test_orchestrator_partial_failure_fallback(
    db_session: Session, test_profile, test_job, monkeypatch
):
    from sqlalchemy import text

    db_session.execute(text("PRAGMA foreign_keys=ON;"))

    # Force DomainAlignmentEngine.align_domain to raise an error
    async def mock_align_domain(*_args, **_kwargs):
        raise RuntimeError("Mock Domain Engine Error")

    monkeypatch.setattr(
        "src.modules.matching.DomainAlignmentEngine.align_domain",
        mock_align_domain,
    )

    orchestrator = OpportunityOrchestrator(db_session)
    context = await orchestrator.run_pipeline(
        job_intelligence_id=test_job.id,
        profile_id=test_profile.id,
    )

    # 1. Check step statuses: domain_alignment should be failed, but explainability and persistence succeed
    assert context.step_statuses["domain_alignment"]["status"] == "failed"
    assert "Mock Domain Engine Error" in context.step_statuses["domain_alignment"]["error"]
    assert context.step_statuses["ranking"]["status"] == "success"
    assert context.step_statuses["explainability_generation"]["status"] == "success"
    assert context.step_statuses["persistence"]["status"] == "success"

    # 2. Verify fallback was applied (domain score is 50.0)
    assert context.domain_alignment.final_score == 50.0
    assert "fallback score" in context.domain_alignment.reasoning.explanation

    # 3. Verify database persistence still succeeded
    assert context.ranking_result_id is not None
    db_result = (
        db_session.query(OpportunityRankingResult).filter_by(id=context.ranking_result_id).first()
    )
    assert db_result is not None
    assert db_result.overall_score == context.ranking.overall_score


@pytest.mark.asyncio
async def test_orchestrator_fatal_failure_abort(db_session: Session, test_profile):
    orchestrator = OpportunityOrchestrator(db_session)

    # Run with a non-existent job ID -> should trigger fatal failure in profile_loading
    context = await orchestrator.run_pipeline(
        job_intelligence_id=99999,
        profile_id=test_profile.id,
    )

    # Verify profile_loading failed and subsequent steps were skipped/not run
    assert context.step_statuses["profile_loading"]["status"] == "failed"
    assert (
        "JobIntelligence with ID 99999 not found"
        in context.step_statuses["profile_loading"]["error"]
    )

    # Since it aborted early, subsequent steps like embedding generation and ranking should not exist in status map
    assert "embedding_generation" not in context.step_statuses
    assert "ranking" not in context.step_statuses
    assert context.ranking is None
    assert context.ranking_result_id is None
