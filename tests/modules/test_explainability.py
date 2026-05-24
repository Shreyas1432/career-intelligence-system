import pytest

from src.modules.explainability.service import ExplainabilityService
from src.modules.sponsorship.types import SponsorshipStatus


@pytest.fixture
def explainability_service():
    return ExplainabilityService()


def test_generate_explanation_basic(explainability_service):
    # Mock SkillMatchResponse dict
    skill_match = {
        "matched_skills": [
            {"match_type": "exact", "matched_skill": "Python"},
            {"match_type": "semantic", "user_skill": "Azure", "job_skill": "AWS"},
        ],
        "missing_skills": [
            {"job_skill": "Kubernetes", "category": "cloud_infrastructure"},
        ],
    }

    # Mock DomainAlignmentResponse dict
    domain_align = {
        "reasoning": {
            "matched_keywords": ["python", "azure"],
        }
    }

    # Mock SponsorshipScoringResponse dict
    sponsorship = {
        "sponsorship_score": 86.0,
        "reasoning": {
            "historical_approved_petitions": 100,
            "historical_denied_petitions": 2,
            "extracted_job_status": SponsorshipStatus.UNKNOWN,
            "extracted_job_confidence": 0.0,
            "explanation": "Visa sponsorship is highly likely (Score: 86.0%).",
        },
    }

    # Mock OpportunityRankingResponse dict
    ranking = {
        "overall_score": 77.75,
        "recommendation": "apply",
        "weights": {
            "skill_matching": 0.1,
            "domain_alignment": 0.1,
            "sponsorship_probability": 0.6,
            "experience_relevance": 0.1,
            "enterprise_alignment": 0.1,
        },
        "factors": {
            "skill_matching": 100.0,
            "domain_alignment": 11.54,
            "sponsorship_probability": 86.0,
            "experience_relevance": 100.0,
            "enterprise_alignment": 50.0,
        },
        "reasoning": {
            "strengths": ["Strong technical skill overlap", "Strong experience fit"],
            "gaps": ["Low domain taxonomy alignment"],
        },
    }

    resp = explainability_service.generate_explanation(
        skill_match=skill_match,
        domain_align=domain_align,
        sponsorship=sponsorship,
        ranking=ranking,
        company="Google LLC",
        title="Python Developer",
    )

    # 1. Verify recruiter summary
    assert "Python Developer" in resp.recruiter_summary
    assert "Google LLC" in resp.recruiter_summary
    assert "Apply" in resp.recruiter_summary
    assert "Visa sponsorship is highly likely" in resp.recruiter_summary

    # 2. Verify score composition
    assert "77.75%" in resp.score_composition_explanation
    assert "Skill Matching: 100.0% score at 10%" in resp.score_composition_explanation
    assert "Visa Sponsorship: 86.0% score at 60%" in resp.score_composition_explanation

    # 3. Verify strengths and weaknesses
    assert "Technical fit: Exact match in 'Python'." in resp.strengths
    assert "Transferable skill: 'Azure' covers required 'AWS'." in resp.strengths
    assert (
        "Skill gap: Missing required 'Kubernetes' (Category: cloud_infrastructure)."
        in resp.weaknesses
    )

    # 4. Verify insights and recommendations
    assert any("Highly favorable sponsorship outlook" in ins for ins in resp.actionable_insights)
    assert any("python, azure" in ins for ins in resp.actionable_insights)
    assert (
        "Upskill in 'Kubernetes' to address a critical required skill."
        in resp.improvement_recommendations
    )
    assert (
        "Explicitly mention your experience with 'AWS' (currently matching via 'Azure')."
        in resp.improvement_recommendations
    )
