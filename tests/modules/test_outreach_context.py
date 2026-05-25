from typing import Any

import pytest
from pydantic import BaseModel

from src.modules.positioning import (
    OutreachContextEngine,
    OutreachContextResponse,
    OutreachInput,
)


class MockOutreachInput(BaseModel):
    recipient: dict[str, Any]
    relationship: dict[str, Any]
    preferences: dict[str, Any]
    opportunity: dict[str, Any] | None


@pytest.fixture
def mock_recruiter_cold_input() -> dict[str, Any]:
    return {
        "recipient": {
            "name": "Sarah",
            "title": "Technical Recruiter",
            "company": "ScaleInc",
            "role_type": "recruiter",
        },
        "relationship": {
            "connection_degree": "cold",
            "past_interactions": [],
        },
        "preferences": {
            "channel": "email",
            "preferred_tone": "formal",
        },
        "opportunity": {
            "role_title": "Senior Python Architect",
            "company": "ScaleInc",
            "key_requirements": ["Python", "AWS", "MLOps"],
        },
    }


@pytest.fixture
def mock_em_warm_input() -> dict[str, Any]:
    return {
        "recipient": {
            "name": "Robert",
            "title": "Director of Platform Engineering",
            "company": "CloudSoft",
            "role_type": "engineering_manager",
        },
        "relationship": {
            "connection_degree": "1st",
            "past_interactions": ["Meeting at PyCon 2024 to discuss MLOps architectures"],
        },
        "preferences": {
            "channel": "linkedin",
            "preferred_tone": "casual",
        },
        "opportunity": {
            "role_title": "Backend Engineering Lead",
            "company": "CloudSoft",
            "key_requirements": ["Python", "Scalability"],
        },
    }


def test_recruiter_cold_draft_generation(mock_recruiter_cold_input: dict[str, Any]) -> None:
    engine = OutreachContextEngine()
    response = engine.generate_outreach(mock_recruiter_cold_input)

    assert isinstance(response, OutreachContextResponse)
    draft = response.draft
    assert draft.subject == "Inquiry: Senior Python Architect Opportunities at ScaleInc"
    assert "Sarah" in draft.body
    assert "attached my resume" in draft.body
    assert "Python, AWS, MLOps" in draft.body
    assert "Dear" in draft.body  # Formal tone salutation

    assert response.outreach_recommendations.follow_up_cadence_days == 10
    assert "professional email" in response.outreach_recommendations.channel_advice.lower()


def test_em_warm_draft_generation(mock_em_warm_input: dict[str, Any]) -> None:
    engine = OutreachContextEngine()
    response = engine.generate_outreach(mock_em_warm_input)

    assert isinstance(response, OutreachContextResponse)
    draft = response.draft
    assert draft.subject is None  # LinkedIn channel removes subject line
    assert "Robert" in draft.body
    assert "Meeting at PyCon 2024" in draft.body
    assert "Hi" in draft.body  # Casual tone salutation

    assert response.outreach_recommendations.follow_up_cadence_days == 5
    assert "linkedin" in response.outreach_recommendations.channel_advice.lower()


def test_anti_manipulation_validator() -> None:
    engine = OutreachContextEngine()

    # Cold connection asserting prior meeting must trigger ValueError
    cold_deceptive_body = "Hi Robert, great connecting with you again regarding PyCon."

    with pytest.raises(ValueError, match="Anti-Manipulation Violation"):
        engine._validate_no_manipulation(
            connection_degree="cold",
            draft_body=cold_deceptive_body,
        )


def test_business_template_drafting() -> None:
    engine = OutreachContextEngine()
    bus_input = {
        "recipient": {
            "name": "Clara",
            "title": "Director of Procurement",
            "company": "LogiCorp",
            "role_type": "consultant_business",
        },
        "relationship": {
            "connection_degree": "cold",
            "past_interactions": [],
        },
        "preferences": {
            "channel": "email",
            "preferred_tone": "formal",
        },
        "opportunity": {
            "role_title": "Sourcing Specialist",
            "company": "LogiCorp",
            "key_requirements": ["SRM", "Negotiation"],
        },
    }

    response = engine.generate_outreach(bus_input)
    draft = response.draft
    assert "Clara" in draft.body
    assert "cost reduction, strategic sourcing, and process optimization" in draft.body.lower()
    assert draft.subject is not None
    assert "operational efficiency & strategy at logicorp" in draft.subject.lower()


def test_input_resiliency_models(mock_recruiter_cold_input: dict[str, Any]) -> None:
    engine = OutreachContextEngine()

    # Wrap in Pydantic schema model
    pydantic_input = OutreachInput(**mock_recruiter_cold_input)
    response = engine.generate_outreach(pydantic_input)

    assert isinstance(response, OutreachContextResponse)
    assert response.outreach_recommendations.follow_up_cadence_days == 10
    assert len(response.tone_recommendations) > 0
