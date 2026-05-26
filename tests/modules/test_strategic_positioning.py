from typing import Any

import pytest
from pydantic import BaseModel

from src.modules.positioning.schemas import (
    ProjectSchema,
    StrategicPositioningResponse,
)
from src.modules.positioning.strategy import StrategicNarrativeGenerator, StrategicPositioningEngine


class MockUserProfile(BaseModel):
    full_name: str
    skills: list[str]
    domains: list[str]
    target_roles: list[str]
    positioning: dict[str, Any]
    experience: list[dict[str, Any]]


@pytest.fixture
def mock_profile_data() -> dict[str, Any]:
    return {
        "full_name": "John Doe",
        "skills": ["Python", "Machine Learning", "Procurement", "Negotiation"],
        "domains": ["Data & AI", "Supply Chain"],
        "target_roles": ["Enterprise AI Engineer", "Operations Strategist"],
        "positioning": {
            "years_of_experience": 5,
        },
        "experience": [
            {
                "title": "Data Analyst",
                "company": "TechCorp",
                "description": "Built analytics pipelines saving 20% in AWS infrastructure runtime.",
            }
        ],
    }


@pytest.fixture
def mock_projects_data() -> list[dict[str, Any]]:
    return [
        {
            "title": "Strategic SRM Integration",
            "description": "Led sourcing integrations with vendors.",
            "technologies": ["Python", "SAP"],
            "outcome": "Reduced vendor turnaround time by 30%.",
        }
    ]


def test_anti_hype_filter() -> None:
    generator = StrategicNarrativeGenerator()
    hype_text = "I am a passionate thought leader, tech guru and disruptive visionary."
    cleaned = generator.clean_hype(hype_text)
    assert "thought leader" not in cleaned.lower()
    assert "guru" not in cleaned.lower()
    assert "visionary" not in cleaned.lower()
    assert "disruptive" not in cleaned.lower()
    assert "passionate" not in cleaned.lower()
    assert "Specialist" in cleaned or "specialist" in cleaned.lower()
    assert "Expert" in cleaned or "expert" in cleaned.lower()


def test_style_selection_and_narratives(
    mock_profile_data: dict[str, Any], mock_projects_data: list[dict[str, Any]]
) -> None:
    engine = StrategicPositioningEngine()

    # 1. Enterprise AI style via opportunity details
    opp_ai = {
        "title": "Machine Learning Architect",
        "description": "Seeking an AI engineer with strong Python background.",
    }
    response_ai = engine.generate_positioning(
        user_profile=mock_profile_data,
        projects=mock_projects_data,
        opportunity_intelligence=opp_ai,
    )
    assert isinstance(response_ai, StrategicPositioningResponse)
    assert "Enterprise AI" in response_ai.positioning_statements.headline
    assert "machine learning pipelines" in response_ai.positioning_statements.elevator_pitch

    # 2. Consulting style via opportunity details
    opp_consulting = {
        "title": "Management Consultant",
        "description": "Seeking a business strategist to advise leadership on roadmaps.",
    }
    response_cons = engine.generate_positioning(
        user_profile=mock_profile_data,
        projects=mock_projects_data,
        opportunity_intelligence=opp_consulting,
    )
    assert "Tech Consulting" in response_cons.positioning_statements.headline
    assert "roadmaps" in response_cons.positioning_statements.elevator_pitch.lower()


def test_cross_domain_differentiation_analysis(
    mock_profile_data: dict[str, Any], mock_projects_data: list[dict[str, Any]]
) -> None:
    engine = StrategicPositioningEngine()
    response = engine.generate_positioning(
        user_profile=mock_profile_data,
        projects=mock_projects_data,
        opportunity_intelligence=None,
    )

    diff = response.differentiation
    assert any("Enterprise AI" in c for c in diff.unique_skill_combinations)
    assert any("turnaround time by 30%" in d for d in diff.core_differentiators)
    assert any("saving 20%" in d for d in diff.core_differentiators)


def test_anti_hallucination_guard_preserves_metrics(
    mock_profile_data: dict[str, Any], mock_projects_data: list[dict[str, Any]]
) -> None:
    engine = StrategicPositioningEngine()

    # Should pass under normal conditions
    response = engine.generate_positioning(
        user_profile=mock_profile_data,
        projects=mock_projects_data,
        opportunity_intelligence=None,
    )
    assert response.differentiation.market_alignment_score == 80.0

    # Trigger violation: generated narrative contains unverified metric (e.g. 50%)
    with pytest.raises(ValueError, match="Anti-Hallucination Violation"):
        engine._validate_no_hallucinations(
            original_projects=mock_projects_data,
            original_experiences=mock_profile_data["experience"],
            generated_headline="Developer",
            generated_pitch="Led teams saving 50% in costs.",  # 50% is not in original projects/experiences
            generated_bio="Technology specialist",
            generated_synthesis="Delivered results",
            years_of_experience=5,
        )


def test_input_handling_resiliency(
    mock_profile_data: dict[str, Any], mock_projects_data: list[dict[str, Any]]
) -> None:
    engine = StrategicPositioningEngine()

    # Mix of Pydantic models and dictionaries
    profile_pydantic = MockUserProfile(**mock_profile_data)
    projects_pydantic = [ProjectSchema(**p) for p in mock_projects_data]

    response = engine.generate_positioning(
        user_profile=profile_pydantic,
        projects=projects_pydantic,
        opportunity_intelligence=None,
    )

    assert isinstance(response, StrategicPositioningResponse)
    assert response.differentiation.market_alignment_score == 80.0
    assert len(response.value_prop_recommendations) > 0
