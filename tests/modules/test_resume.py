import pytest

from src.modules.resume.service import analyze_resume


@pytest.mark.usefixtures("mock_ai_client")
def test_analyze_resume_mocked():
    """
    Verifies that the resume analyzer successfully communicates with the mock AI client.
    """
    resume = "John Doe - Software Engineer with Python experience"
    job_desc = "Looking for a Python Developer who knows Django"

    result = analyze_resume(resume, job_desc)

    assert "Mock" in result or "AI" in result
    assert isinstance(result, str)
