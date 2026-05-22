import pytest

from src.core.prompts import prompt_manager, prompt_registry
from src.core.prompts.models import PromptTemplate
from src.core.prompts.registry import (
    PromptRegistry,
    clean_name_and_extract_version,
    extract_variables,
    parse_frontmatter,
)


def test_clean_name_and_extract_version() -> None:
    # Test path normalization and stripping suffixes
    name, version = clean_name_and_extract_version("resume/tailor_prompt")
    assert name == "resume/tailor"
    assert version is None

    # Test filename based version parsing
    name, version = clean_name_and_extract_version("interview/coach_template_v2")
    assert name == "interview/coach"
    assert version == "2"

    name, version = clean_name_and_extract_version("career_path/map_v1.0.3")
    assert name == "career_path/map"
    assert version == "1.0.3"


def test_parse_frontmatter() -> None:
    # Test file content split with valid frontmatter
    raw = (
        "---\n"
        "version: '2.1'\n"
        "input_variables:\n"
        "  - name\n"
        "system_prompt: 'You are a bot'\n"
        "---\n"
        "Hello {{ name }}"
    )
    metadata, body = parse_frontmatter(raw)
    assert metadata["version"] == "2.1"
    assert metadata["input_variables"] == ["name"]
    assert metadata["system_prompt"] == "You are a bot"
    assert body == "Hello {{ name }}"

    # Test raw content split with no frontmatter
    raw_no_fm = "Hello {{ name }}"
    metadata, body = parse_frontmatter(raw_no_fm)
    assert metadata == {}
    assert body == "Hello {{ name }}"


def test_extract_variables() -> None:
    # Test auto-extraction of Jinja placeholders
    body = "Hello {{ user }}, welcome to {{ system }}!"
    vars_list = extract_variables(body)
    assert vars_list == ["system", "user"]


def test_prompt_template_rendering() -> None:
    template = PromptTemplate(
        name="test",
        version="1.0.0",
        input_variables=["a", "b"],
        body="A is {{ a }} and B is {{ b }}",
        raw_content="",
    )

    # Test successful render
    res = template.render({"a": "alpha", "b": "beta"})
    assert res == "A is alpha and B is beta"

    # Test failure due to missing inputs
    with pytest.raises(ValueError, match="Missing required input variables"):
        template.render({"a": "alpha"})


def test_prompt_template_messages() -> None:
    template = PromptTemplate(
        name="test",
        version="1.0.0",
        system_prompt="System instructions",
        input_variables=["a"],
        body="Value is {{ a }}",
        raw_content="",
    )

    messages = template.render_messages({"a": "val"})
    assert len(messages) == 2
    assert messages[0] == {"role": "system", "content": "System instructions"}
    assert messages[1] == {"role": "user", "content": "Value is val"}


def test_registry_registration() -> None:
    registry = PromptRegistry()
    raw = (
        "---\n"
        "version: '1.2.0'\n"
        "system_prompt: 'Act as reviewer'\n"
        "---\n"
        "Evaluate: {{ text }}"
    )

    # Dynamic registration
    template = registry.register_prompt_from_string("reviews/evaluator_v1.2.0", raw)
    assert template.name == "reviews/evaluator"
    assert template.version == "1.2.0"
    assert template.system_prompt == "Act as reviewer"
    assert template.input_variables == ["text"]

    # Verify retrieval
    retrieved = registry.get("reviews/evaluator", version="1.2.0")
    assert retrieved.name == "reviews/evaluator"
    assert retrieved.version == "1.2.0"


def test_registry_version_fallback() -> None:
    registry = PromptRegistry()

    # Register multiple versions
    registry.register_prompt_from_string("test/prompt_v1.0", "V1: {{ val }}")
    registry.register_prompt_from_string("test/prompt_v2.0", "V2: {{ val }}")
    registry.register_prompt_from_string("test/prompt_v1.5", "V1.5: {{ val }}")

    # Get latest fallback (should resolve to 2.0)
    latest = registry.get("test/prompt")
    assert latest.version == "2.0"
    assert latest.render({"val": "hello"}) == "V2: hello"

    # Get specific version
    v1_5 = registry.get("test/prompt", version="1.5")
    assert v1_5.version == "1.5"
    assert v1_5.render({"val": "hello"}) == "V1.5: hello"


def test_backward_compatible_manager() -> None:
    # Register mock prompt in the global registry
    prompt_registry.register_prompt_from_string(
        "mock/legacy_prompt", "---\nversion: '1.0.0'\n---\nHello {{ name }}!"
    )

    # Verify load_prompt works as before
    res = prompt_manager.load_prompt("mock/legacy_prompt.md", {"name": "World"})
    assert res == "Hello World!"

    # Verify load_prompt raises FileNotFoundError on missing files
    with pytest.raises(FileNotFoundError):
        prompt_manager.load_prompt("missing/file.md", {"name": "World"})
