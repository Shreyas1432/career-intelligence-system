import pytest

from src.modules.resume.schemas import TailoredResumeResponse
from src.modules.resume.tailoring import (
    ResumeTailoringEngine,
    ResumeTailoringValidator,
    ResumeTransformationPipeline,
)


def test_prioritization_and_metric_preservation() -> None:
    """
    Verifies that the experience prioritizer reorders experience sections,
    replaces keywords according to job specs, and preserves quantified metrics.
    """
    pipeline = ResumeTransformationPipeline()

    experiences = [
        {
            "title": "Junior Developer",
            "company": "AppAgency",
            "start_date": "2019-01",
            "end_date": "2020-12",
            "description": "Coded database queries and software features.",
        },
        {
            "title": "Senior AI Architect",
            "company": "ScaleInc",
            "start_date": "2021-01",
            "end_date": "Present",
            "description": "Designed cloud systems saving 15% in AWS infrastructure costs with 3x faster processing.",
        },
    ]

    strategy_priorities = [
        {"title": "Senior AI Architect", "company": "ScaleInc", "priority_band": "HIGH"},
        {"title": "Junior Developer", "company": "AppAgency", "priority_band": "LOW"},
    ]

    # Test prioritization
    prioritized = pipeline.prioritize_experiences(experiences, strategy_priorities)
    assert len(prioritized) == 2
    assert prioritized[0]["title"] == "Senior AI Architect"
    assert prioritized[1]["title"] == "Junior Developer"

    # Test bullet points keyword alignment
    target_kws = ["SQL", "Python", "Cloud"]
    consolidated, clean_bullets = pipeline.refine_bullet_points(
        bullet_text="Coded database queries and software features.",
        target_keywords=target_kws,
    )
    # database -> database (SQL), software -> software engineering (Python)
    assert "database (SQL)" in consolidated
    assert "software engineering (Python)" in consolidated
    assert len(clean_bullets) == 1

    # Test metric preservation
    desc_with_metrics = (
        "Designed cloud systems saving 15% in AWS infrastructure costs with 3x faster processing."
    )
    consolidated_metrics, _ = pipeline.refine_bullet_points(
        bullet_text=desc_with_metrics,
        target_keywords=target_kws,
    )
    assert "15%" in consolidated_metrics
    assert "3x" in consolidated_metrics


def test_tailoring_validation_rules() -> None:
    """
    Verifies that the validator raises ValueError when structure, dates,
    metrics or companies/roles are altered or hallucinated.
    """
    validator = ResumeTailoringValidator()

    base_resume = {
        "full_name": "Alice Smith",
        "email": "alice@example.com",
        "experience": [
            {
                "title": "Data Engineer",
                "company": "CorpInc",
                "start_date": "2020-01",
                "end_date": "2022-12",
                "description": "Managed pipelines saving 20% storage.",
            }
        ],
    }

    # 1. Valid tailored resume should pass
    valid_tailored = {
        "full_name": "Alice Smith",
        "email": "alice@example.com",
        "experiences": [
            {
                "title": "Data Engineer",
                "company": "CorpInc",
                "start_date": "2020-01",
                "end_date": "2022-12",
                "description": "- Managed pipelines (SQL) saving 20% storage.",
                "bullets": ["Managed pipelines (SQL) saving 20% storage."],
            }
        ],
    }
    validator.validate_tailored_resume(base_resume, valid_tailored)

    # 2. Altered name should fail
    bad_name = dict(valid_tailored, full_name="Bob Smith")
    with pytest.raises(ValueError, match="name altered"):
        validator.validate_tailored_resume(base_resume, bad_name)

    # 3. New/hallucinated role/company should fail
    bad_company = {
        "full_name": "Alice Smith",
        "email": "alice@example.com",
        "experiences": [
            {
                "title": "Staff Architect",
                "company": "Google",
                "start_date": "2020-01",
                "end_date": "2022-12",
                "description": "Hallucinated experience.",
                "bullets": [],
            }
        ],
    }
    with pytest.raises(ValueError, match="Anti-Hallucination Violation"):
        validator.validate_tailored_resume(base_resume, bad_company)

    # 4. Lost metric (20%) should fail
    bad_metrics = {
        "full_name": "Alice Smith",
        "email": "alice@example.com",
        "experiences": [
            {
                "title": "Data Engineer",
                "company": "CorpInc",
                "start_date": "2020-01",
                "end_date": "2022-12",
                "description": "- Managed pipelines.",
                "bullets": [],
            }
        ],
    }
    with pytest.raises(ValueError, match="Quantified Impact Violation"):
        validator.validate_tailored_resume(base_resume, bad_metrics)

    # 5. Hallucinated new metric (e.g. 50% or $10k) should fail
    hallucinated_metric = {
        "full_name": "Alice Smith",
        "email": "alice@example.com",
        "experiences": [
            {
                "title": "Data Engineer",
                "company": "CorpInc",
                "start_date": "2020-01",
                "end_date": "2022-12",
                "description": "- Managed pipelines saving 20% storage and driving 50% profits.",
                "bullets": [],
            }
        ],
    }
    with pytest.raises(ValueError, match="Anti-Hallucination Violation: Unverified metric"):
        validator.validate_tailored_resume(base_resume, hallucinated_metric)

    # 6. Altered date should fail
    bad_date = {
        "full_name": "Alice Smith",
        "email": "alice@example.com",
        "experiences": [
            {
                "title": "Data Engineer",
                "company": "CorpInc",
                "start_date": "2020-01",
                "end_date": "2024-12",
                "description": "- Managed pipelines saving 20% storage.",
                "bullets": [],
            }
        ],
    }
    with pytest.raises(ValueError, match="Structure Violation: Dates altered"):
        validator.validate_tailored_resume(base_resume, bad_date)


def test_end_to_end_tailoring_engine() -> None:
    """
    Verifies end-to-end execution of ResumeTailoringEngine.
    """
    engine = ResumeTailoringEngine()

    user_profile = {
        "full_name": "John Doe",
        "email": "john.doe@example.com",
        "skills": ["Python", "SQL"],
        "experience": [
            {
                "title": "Data Analyst",
                "company": "LocalCorp",
                "start_date": "2020-01",
                "end_date": "2022-12",
                "description": "Wrote database queries and ran daily reports.",
            }
        ],
    }

    job_intel = {
        "title": "Senior Data Architect",
        "company": "ScaleCorp",
        "domain": "AI/Analytics",
        "normalized_skills": ["Python", "SQL", "Machine Learning"],
    }

    resume_strategy = {
        "positioning_recommendations": {
            "suggested_headline": "Senior Data Architect | AI Specialist",
            "positioning_pitch": "Experienced Data Analyst focusing on SQL database workflows.",
        },
        "prioritized_experiences": [
            {"title": "Data Analyst", "company": "LocalCorp", "priority_band": "HIGH"}
        ],
        "ats_optimization": {
            "target_keywords": ["Python", "SQL", "Machine Learning"],
            "missing_keywords_to_add": ["Machine Learning"],
        },
    }

    opp_scoring = {
        "overall_score": 85.0,
        "recommendation": "apply",
    }

    response = engine.tailor_resume(
        user_profile,
        job_intel,
        resume_strategy,
        opp_scoring,
    )

    assert isinstance(response, TailoredResumeResponse)
    assert response.tailored_resume.full_name == "John Doe"
    assert response.tailored_resume.suggested_headline == "Senior Data Architect | AI Specialist"
    # database -> database (SQL) should be triggered
    assert "database (SQL)" in response.tailored_resume.experiences[0].description
    assert response.ats_metadata.keyword_alignment_score > 0.0
    assert "Machine Learning" in response.missing_skill_suggestions
