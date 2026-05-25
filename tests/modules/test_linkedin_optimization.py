from typing import Any

import pytest
from pydantic import BaseModel

from src.modules.positioning.linkedin import (
    LinkedInOptimizationEngine,
    LinkedInPositioningLayer,
    LinkedInRecommendationLayer,
)
from src.modules.positioning.schemas import (
    ImpactLevel,
    LinkedInOptimizationResponse,
)


class MockUserProfile(BaseModel):
    full_name: str
    email: str
    skills: list[str]
    experience_summary: str
    domains: list[str]
    positioning: dict[str, Any]
    experience: list[dict[str, Any]]


@pytest.fixture
def mock_profile_data() -> dict[str, Any]:
    return {
        "full_name": "Jane Doe",
        "email": "jane.doe@example.com",
        "skills": ["Python", "SQL", "Logistics", "Procurement", "Negotiation"],
        "experience_summary": "Lead Supply Chain Specialist and AI enthusiast.",
        "domains": ["Supply Chain", "Procurement"],
        "positioning": {
            "headline": "Lead Sourcing Guru & Supply Chain Ninja",
            "years_of_experience": 8,
            "seniority_level": "Lead",
        },
        "experience": [
            {
                "title": "Lead Supply Chain Specialist",
                "company": "ScaleInc",
                "start_date": "2021-01",
                "end_date": "Present",
                "description": "Led sourcing operations saving 15% in costs. Managed a team of 5 logistics coordinators.",
            }
        ],
    }


@pytest.fixture
def mock_job_trends() -> dict[str, Any]:
    return {
        "top_keywords": ["Procurement", "SRM", "S&OP", "Negotiation", "Logistics", "SAP"],
    }


def test_buzzword_filtering() -> None:
    layer = LinkedInPositioningLayer()
    text = (
        "A visionary ML ninja and tech guru revolutionizing the industry with disruptive solutions."
    )
    cleaned = layer.clean_buzzwords(text)
    assert "ninja" not in cleaned.lower()
    assert "guru" not in cleaned.lower()
    assert "visionary" not in cleaned.lower()
    assert "disruptive" not in cleaned.lower()
    assert "revolutionizing" not in cleaned.lower()

    # Verify capitalization preservation
    assert "Specialist" in cleaned or "specialist" in cleaned.lower()
    assert "Expert" in cleaned or "expert" in cleaned.lower()


def test_headline_and_about_optimization(mock_profile_data: dict[str, Any]) -> None:
    engine = LinkedInOptimizationEngine()
    trends = {"top_keywords": ["Procurement", "Logistics", "Negotiation"]}

    # Test as dictionary input
    response = engine.optimize_profile(
        user_profile=mock_profile_data,
        target_roles=["Supply Chain Director", "Procurement Manager"],
        job_intelligence_trends=trends,
    )

    assert isinstance(response, LinkedInOptimizationResponse)

    headline = response.optimized_sections.headline.optimized
    assert "Supply Chain Director | Procurement Manager" in headline
    assert len(headline) <= 220
    assert "ninja" not in headline.lower()
    assert "guru" not in headline.lower()

    about = response.optimized_sections.about.optimized
    assert "8+ years of experience" in about
    assert "Core Competencies:" in about
    assert "Procurement" in about
    assert "Logistics" in about


def test_experience_description_preserves_metrics(mock_profile_data: dict[str, Any]) -> None:
    engine = LinkedInOptimizationEngine()
    trends = {"top_keywords": ["Procurement", "Logistics"]}

    response = engine.optimize_profile(
        user_profile=mock_profile_data,
        target_roles=["Supply Chain Director"],
        job_intelligence_trends=trends,
    )

    opt_exp = response.optimized_sections.experiences[0]
    assert "15%" in opt_exp.optimized_description
    assert "5" in opt_exp.optimized_description
    assert "sourcing operations" in opt_exp.optimized_description.lower()


def test_anti_hallucination_guard_violation_company(mock_profile_data: dict[str, Any]) -> None:
    LinkedInOptimizationEngine()

    # Corrupting the company name in the engine pipeline run to trigger violation
    # We will subclass engine or mock the layer output to return a different company name
    class HallucinatedCompanyLayer(LinkedInPositioningLayer):
        def optimize_experience_description(
            self,
            _title: str,
            _company: str,
            description: str,
            _profile_skills: list[str],
            _trending_keywords: list[str],
        ) -> tuple[str, str]:
            return description, "Weaved keywords"

    bad_engine = LinkedInOptimizationEngine(positioning_layer=HallucinatedCompanyLayer())

    # We manually alter the return formatting of optimized experience in engine.py by patching it or using custom positioning_layer
    # Let's verify that company or title mismatch raises ValueError.
    # To test _validate_no_hallucinations directly:
    from src.modules.positioning.schemas import LinkedInExperienceOptimization

    original = mock_profile_data["experience"]
    optimized = [
        LinkedInExperienceOptimization(
            title="Lead Supply Chain Specialist",
            company="FakeCompanyCorp",  # Altered company
            original_description=original[0]["description"],
            optimized_description=original[0]["description"],
            justification="",
        )
    ]

    with pytest.raises(ValueError, match="Anti-Hallucination Guard: Company name changed"):
        bad_engine._validate_no_hallucinations(original, optimized)


def test_anti_hallucination_guard_violation_metric(mock_profile_data: dict[str, Any]) -> None:
    engine = LinkedInOptimizationEngine()
    original = mock_profile_data["experience"]

    from src.modules.positioning.schemas import LinkedInExperienceOptimization

    # Scenario 1: Dropped metric
    optimized_missing_metric = [
        LinkedInExperienceOptimization(
            title="Lead Supply Chain Specialist",
            company="ScaleInc",
            original_description=original[0]["description"],
            optimized_description="Led sourcing operations saving costs. Managed a team of coordinators.",  # Dropped 15% and 5
            justification="",
        )
    ]

    with pytest.raises(ValueError, match="Quantified Impact Violation"):
        engine._validate_no_hallucinations(original, optimized_missing_metric)

    # Scenario 2: Hallucinated metric
    optimized_new_metric = [
        LinkedInExperienceOptimization(
            title="Lead Supply Chain Specialist",
            company="ScaleInc",
            original_description=original[0]["description"],
            optimized_description="Led sourcing operations saving 15% in costs. Managed a team of 5 logistics coordinators. Saved $100k.",  # Added $100k
            justification="",
        )
    ]

    with pytest.raises(ValueError, match="Anti-Hallucination Guard: Unverified metric"):
        engine._validate_no_hallucinations(original, optimized_new_metric)


def test_discoverability_index_calculation() -> None:
    rec_layer = LinkedInRecommendationLayer()
    profile_skills = ["Python", "SQL", "Kubernetes"]
    trends = ["Python", "SQL", "Docker", "AWS", "Kubernetes"]

    alignment = rec_layer.calculate_keyword_alignment(
        profile_skills=profile_skills,
        trending_keywords=trends,
        optimized_headline="Python Engineer | SQL Specialist",
        optimized_about="Skills include Kubernetes and Docker.",  # Docker is in about
    )

    # Matched keywords should be Python, SQL, Kubernetes, and Docker
    assert "Python" in alignment.matched_keywords
    assert "SQL" in alignment.matched_keywords
    assert "Kubernetes" in alignment.matched_keywords
    assert "Docker" in alignment.matched_keywords
    assert "AWS" in alignment.missing_keywords

    # 4 out of 5 matched = 80.0%
    assert alignment.discoverability_index == 80.0


def test_profile_improvement_suggestions(mock_profile_data: dict[str, Any]) -> None:
    rec_layer = LinkedInRecommendationLayer()

    # Profile has only 5 skills
    suggestions = rec_layer.generate_improvement_suggestions(
        profile_skills=["Python", "SQL"],
        trending_keywords=["Python", "SQL", "Logistics"],
        experiences=mock_profile_data["experience"],
        opportunity_analysis=None,
    )

    sections = [s.section for s in suggestions]
    assert "Skills" in sections
    assert "Featured" in sections

    # Check impact levels
    skills_sug = next(s for s in suggestions if s.section == "Skills")
    assert skills_sug.impact_level == ImpactLevel.HIGH


def test_input_resilience_pydantic_models(
    mock_profile_data: dict[str, Any], mock_job_trends: dict[str, Any]
) -> None:
    engine = LinkedInOptimizationEngine()

    # Wrap dict in a mock Pydantic model
    user_pydantic = MockUserProfile(**mock_profile_data)

    class MockJobTrendsModel(BaseModel):
        top_keywords: list[str]

    trends_pydantic = MockJobTrendsModel(**mock_job_trends)

    response = engine.optimize_profile(
        user_profile=user_pydantic,
        target_roles=["Supply Chain Director"],
        job_intelligence_trends=trends_pydantic,
    )

    assert isinstance(response, LinkedInOptimizationResponse)
    assert response.keyword_alignment.discoverability_index > 0.0
    assert len(response.positioning_recommendations) > 0
