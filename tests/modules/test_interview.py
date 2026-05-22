import pytest

from src.modules.interview.service import conduct_mock_interview


@pytest.mark.usefixtures("mock_ai_client")
def test_conduct_mock_interview_mocked():
    """
    Verifies that the interview service successfully communicates with the mock AI client.
    """
    role = "Software Engineer"
    question_type = "Technical / Coding"
    history = []

    result = conduct_mock_interview(role, question_type, history)

    assert "Mock" in result or "AI" in result
    assert isinstance(result, str)
