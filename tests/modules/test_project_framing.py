from typing import Any

import pytest
from pydantic import BaseModel

from src.modules.positioning import ProjectFramingEngine, ProjectNarrativeGenerator
from src.modules.positioning.schemas import (
    ProjectFramingInput,
    ProjectFramingResponse,
)


class MockProjectInput(BaseModel):
    metadata: dict[str, Any]
    architecture: dict[str, Any]
    technologies: list[str]
    business_goals: list[str]


@pytest.fixture
def mock_project_data() -> dict[str, Any]:
    return {
        "metadata": {
            "title": "SRM Automation",
            "role": "Lead Data Engineer",
            "description": "Implemented database queries and writing scripts saving 25% in manual effort.",
        },
        "architecture": {
            "design_patterns": ["Microservices", "Event-Driven"],
            "database_setup": "PostgreSQL database containing vendor listings",
            "hosting_or_cloud": "AWS Cloud Platforms",
        },
        "technologies": ["Python", "PostgreSQL", "Docker"],
        "business_goals": ["Reduce backend processing latency by 40%"],
    }


def test_buzzword_cleaner() -> None:
    generator = ProjectNarrativeGenerator()
    hype = "A visionary ML ninja and tech guru built this game-changing project."
    cleaned = generator.clean_buzzwords(hype)
    assert "ninja" not in cleaned.lower()
    assert "guru" not in cleaned.lower()
    assert "visionary" not in cleaned.lower()
    assert "game-changing" not in cleaned.lower()
    assert "Specialist" in cleaned or "specialist" in cleaned.lower()
    assert "Expert" in cleaned or "expert" in cleaned.lower()


def test_technical_to_business_translation(mock_project_data: dict[str, Any]) -> None:
    engine = ProjectFramingEngine()
    response = engine.frame_project(mock_project_data)

    assert isinstance(response, ProjectFramingResponse)
    summary = response.recruiter_summary.summary_text
    # Check that query and scripting terms were translated
    assert "writing scripts" not in summary.lower()
    assert "database queries" not in summary.lower()
    assert "orchestrating automation workflows" in summary.lower()
    assert "data integrity systems" in summary.lower()


def test_engineering_reasoning_deep_dive(mock_project_data: dict[str, Any]) -> None:
    engine = ProjectFramingEngine()
    response = engine.frame_project(mock_project_data)

    tech_exp = response.technical_explanation
    assert "PostgreSQL" in tech_exp.architectural_decisions
    assert "Microservices" in tech_exp.architectural_decisions
    assert "caching boundaries" in tech_exp.problem_solving.lower()


def test_metric_preservation_validation(mock_project_data: dict[str, Any]) -> None:
    engine = ProjectFramingEngine()
    response = engine.frame_project(mock_project_data)

    # 25% from description and 40% from goals must be preserved in key outcomes
    outcomes = response.recruiter_summary.key_outcomes
    assert any("40%" in o for o in outcomes)

    # Triggering metric hallucination must raise ValueError
    with pytest.raises(ValueError, match="Anti-Hallucination Violation"):
        engine._validate_no_hallucinations(
            input_desc=mock_project_data["metadata"]["description"],
            input_goals=mock_project_data["business_goals"],
            generated_summary="Orchestrated workflows",
            generated_outcomes=[
                "Achieved 85% operational efficiency"
            ],  # 85% is not in original goals/desc
            generated_scalability="Scaled system",
            generated_integration="Integrated db",
            generated_impact="Reduced effort",
            generated_decisions="Chose Python",
            generated_solving="Fixed bottlenecks",
        )


def test_input_resiliency_models(mock_project_data: dict[str, Any]) -> None:
    engine = ProjectFramingEngine()

    # Test under Pydantic schema model
    pydantic_input = ProjectFramingInput(**mock_project_data)
    response = engine.frame_project(pydantic_input)

    assert isinstance(response, ProjectFramingResponse)
    assert response.recruiter_summary.key_outcomes[0].endswith("40%")
    assert len(response.portfolio_recommendations.readme_tips) > 0
