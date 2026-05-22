import pytest

from src.modules.career_path.service import get_career_map


@pytest.mark.usefixtures("mock_ai_client")
def test_get_career_map_mocked():
    """
    Verifies that the career path service successfully communicates with the mock AI client.
    """
    current_role = "Junior Developer"
    target_role = "Senior Engineer"
    skills = "Python, Git"

    result = get_career_map(current_role, target_role, skills)

    assert "Mock" in result or "AI" in result
    assert isinstance(result, str)
