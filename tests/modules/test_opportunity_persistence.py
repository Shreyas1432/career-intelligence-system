import pytest
from sqlalchemy.orm import Session

from src.core.database.models import JobIntelligence
from src.modules.opportunity_ranking import (
    FactorScores,
    OpportunityRankingRepository,
    OpportunityRankingResponse,
    OpportunityRankingResult,
    RankingReasoning,
    RankingWeights,
    RecommendationCategory,
)
from src.modules.user_profile import UserProfileCreate, UserProfileService


@pytest.fixture
def test_profile(db_session: Session):
    profile_data = UserProfileCreate(
        full_name="Jane Doe",
        email="jane.doe@example.com",
        skills=["Python", "SQL"],
        experience_summary="5 years experience",
    )
    return UserProfileService.create_or_update_profile(db_session, profile_data)


@pytest.fixture
def test_job(db_session: Session):
    job = JobIntelligence(
        url_hash="job_123_hash",
        url="https://example.com/job/123",
        content_hash="content_123_hash",
        title="Python Engineer",
        company="TechCorp",
        location="Remote",
        normalized_skills=["Python", "SQL"],
    )
    db_session.add(job)
    db_session.flush()
    return job


@pytest.fixture
def test_job_2(db_session: Session):
    job = JobIntelligence(
        url_hash="job_456_hash",
        url="https://example.com/job/456",
        content_hash="content_456_hash",
        title="Django Developer",
        company="StartupInc",
        location="New York",
        normalized_skills=["Python", "Django"],
    )
    db_session.add(job)
    db_session.flush()
    return job


def test_save_ranking_result_pydantic(db_session: Session, test_profile, test_job):
    repo = OpportunityRankingRepository(db_session)

    weights = RankingWeights(
        skill_matching=0.30,
        domain_alignment=0.20,
        sponsorship_probability=0.20,
        experience_relevance=0.15,
        enterprise_alignment=0.15,
    )
    factors = FactorScores(
        skill_matching=90.0,
        domain_alignment=80.0,
        sponsorship_probability=85.0,
        experience_relevance=95.0,
        enterprise_alignment=90.0,
    )
    reasoning = RankingReasoning(
        strengths=["Strong programming overlap", "Seniority matches"],
        gaps=["Domain overlap could be higher"],
        explanation="Highly aligned opportunity.",
    )

    ranking_response = OpportunityRankingResponse(
        overall_score=87.75,
        recommendation=RecommendationCategory.STRONG_APPLY,
        factors=factors,
        weights=weights,
        reasoning=reasoning,
    )

    db_result = repo.save_ranking_result(
        profile_id=test_profile.id,
        job_id=test_job.id,
        ranking_response=ranking_response,
    )
    db_session.flush()

    assert db_result.id is not None
    assert db_result.profile_id == test_profile.id
    assert db_result.job_id == test_job.id
    assert db_result.overall_score == 87.75
    assert db_result.recommendation == "strong_apply"
    assert db_result.run_number == 1

    # Check JSON structures
    assert db_result.factor_scores["skill_matching"] == 90.0
    assert db_result.weights["skill_matching"] == 0.30
    assert "Seniority matches" in db_result.reasoning_metadata["strengths"]


def test_save_ranking_result_dict(db_session: Session, test_profile, test_job):
    repo = OpportunityRankingRepository(db_session)

    raw_dict = {
        "overall_score": 60.5,
        "recommendation": "apply",
        "factors": {
            "skill_matching": 60.0,
            "domain_alignment": 50.0,
            "sponsorship_probability": 70.0,
            "experience_relevance": 65.0,
            "enterprise_alignment": 60.0,
        },
        "weights": {
            "skill_matching": 0.3,
            "domain_alignment": 0.2,
            "sponsorship_probability": 0.2,
            "experience_relevance": 0.15,
            "enterprise_alignment": 0.15,
        },
        "reasoning": {
            "strengths": ["Decent fit"],
            "gaps": ["Some missing skillsets"],
            "explanation": "Medium alignment.",
        },
    }

    db_result = repo.save_ranking_result(
        profile_id=test_profile.id,
        job_id=test_job.id,
        ranking_response=raw_dict,
    )
    db_session.flush()

    assert db_result.id is not None
    assert db_result.profile_id == test_profile.id
    assert db_result.job_id == test_job.id
    assert db_result.overall_score == 60.5
    assert db_result.recommendation == "apply"
    assert db_result.run_number == 1
    assert db_result.factor_scores["skill_matching"] == 60.0


def test_recalculation_tracking_and_latest(db_session: Session, test_profile, test_job):
    repo = OpportunityRankingRepository(db_session)

    res1_data = {
        "overall_score": 50.0,
        "recommendation": "weak_apply",
        "factors": dict.fromkeys(
            [
                "skill_matching",
                "domain_alignment",
                "sponsorship_probability",
                "experience_relevance",
                "enterprise_alignment",
            ],
            50.0,
        ),
        "weights": dict.fromkeys(
            [
                "skill_matching",
                "domain_alignment",
                "sponsorship_probability",
                "experience_relevance",
                "enterprise_alignment",
            ],
            0.2,
        ),
        "reasoning": {"strengths": [], "gaps": [], "explanation": "Neutral"},
    }

    res2_data = res1_data.copy()
    res2_data["overall_score"] = 75.0
    res2_data["recommendation"] = "apply"

    res3_data = res1_data.copy()
    res3_data["overall_score"] = 90.0
    res3_data["recommendation"] = "strong_apply"

    # Save multiple times (simulating recalculations)
    repo.save_ranking_result(test_profile.id, test_job.id, res1_data)
    repo.save_ranking_result(test_profile.id, test_job.id, res2_data)
    repo.save_ranking_result(test_profile.id, test_job.id, res3_data)
    db_session.flush()

    # Get latest
    latest = repo.get_latest_ranking_result(test_profile.id, test_job.id)
    assert latest is not None
    assert latest.run_number == 3
    assert latest.overall_score == 90.0
    assert latest.recommendation == "strong_apply"

    # Get history
    history = repo.get_ranking_history(test_profile.id, test_job.id)
    assert len(history) == 3
    assert history[0].run_number == 1
    assert history[0].overall_score == 50.0
    assert history[1].run_number == 2
    assert history[1].overall_score == 75.0
    assert history[2].run_number == 3
    assert history[2].overall_score == 90.0


def test_profile_comparison_history(db_session: Session, test_profile, test_job, test_job_2):
    repo = OpportunityRankingRepository(db_session)

    res_job1_run1 = {
        "overall_score": 60.0,
        "recommendation": "apply",
        "factors": dict.fromkeys(
            [
                "skill_matching",
                "domain_alignment",
                "sponsorship_probability",
                "experience_relevance",
                "enterprise_alignment",
            ],
            60.0,
        ),
        "weights": dict.fromkeys(
            [
                "skill_matching",
                "domain_alignment",
                "sponsorship_probability",
                "experience_relevance",
                "enterprise_alignment",
            ],
            0.2,
        ),
        "reasoning": {"strengths": [], "gaps": [], "explanation": "Neutral"},
    }

    res_job1_run2 = res_job1_run1.copy()
    res_job1_run2["overall_score"] = 80.0

    res_job2_run1 = res_job1_run1.copy()
    res_job2_run1["overall_score"] = 70.0

    # Save ranking results
    repo.save_ranking_result(test_profile.id, test_job.id, res_job1_run1)
    repo.save_ranking_result(test_profile.id, test_job_2.id, res_job2_run1)
    repo.save_ranking_result(test_profile.id, test_job.id, res_job1_run2)
    db_session.flush()

    # Query comparison history
    comparisons = repo.get_profile_comparison_history(test_profile.id)

    # Should only return 2 records: latest run for Job 1 (score 80.0) and Job 2 (score 70.0)
    assert len(comparisons) == 2

    # Map job ids to outcomes to verify correct ones returned
    scores_map = {c.job_id: c.overall_score for c in comparisons}
    runs_map = {c.job_id: c.run_number for c in comparisons}

    assert scores_map[test_job.id] == 80.0
    assert runs_map[test_job.id] == 2

    assert scores_map[test_job_2.id] == 70.0
    assert runs_map[test_job_2.id] == 1


def test_cascade_delete(db_session: Session):
    from sqlalchemy import text

    from src.core.database.models import UserProfile

    db_session.execute(text("PRAGMA foreign_keys=ON;"))
    repo = OpportunityRankingRepository(db_session)

    # Create profile manually
    profile = UserProfile(
        full_name="Jane Doe",
        email="jane.cascade@example.com",
        skills="Python, SQL",
    )
    db_session.add(profile)

    # Create job manually
    job = JobIntelligence(
        url_hash="job_cascade_hash",
        url="https://example.com/job/cascade",
        content_hash="content_cascade_hash",
        title="Python Engineer",
        company="TechCorp",
    )
    db_session.add(job)
    db_session.flush()

    res_data = {
        "overall_score": 85.0,
        "recommendation": "strong_apply",
        "factors": dict.fromkeys(
            [
                "skill_matching",
                "domain_alignment",
                "sponsorship_probability",
                "experience_relevance",
                "enterprise_alignment",
            ],
            85.0,
        ),
        "weights": dict.fromkeys(
            [
                "skill_matching",
                "domain_alignment",
                "sponsorship_probability",
                "experience_relevance",
                "enterprise_alignment",
            ],
            0.2,
        ),
        "reasoning": {"strengths": [], "gaps": [], "explanation": "Good"},
    }

    # 1. Test profile cascade delete
    repo.save_ranking_result(profile.id, job.id, res_data)
    db_session.flush()

    # Verify saved
    assert db_session.query(OpportunityRankingResult).count() == 1

    # Delete profile
    db_session.delete(profile)
    db_session.flush()

    # Ranking result should be automatically deleted
    assert db_session.query(OpportunityRankingResult).count() == 0

    # 2. Test job cascade delete
    # Re-create profile
    profile2 = UserProfile(
        full_name="John Smith",
        email="john.cascade@example.com",
        skills="Python",
    )
    db_session.add(profile2)

    # Re-create job
    job2 = JobIntelligence(
        url_hash="job_cascade2_hash",
        url="https://example.com/job/cascade2",
        content_hash="content_cascade2_hash",
        title="Django Developer",
        company="StartupInc",
    )
    db_session.add(job2)
    db_session.flush()

    repo.save_ranking_result(profile2.id, job2.id, res_data)
    db_session.flush()

    assert db_session.query(OpportunityRankingResult).count() == 1

    # Delete job
    db_session.delete(job2)
    db_session.flush()

    # Ranking result should be automatically deleted
    assert db_session.query(OpportunityRankingResult).count() == 0
