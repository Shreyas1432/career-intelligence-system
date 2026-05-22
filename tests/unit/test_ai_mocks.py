import pytest
from pydantic import BaseModel

from tests.fixtures.ai import MockLLMClient
from tests.utils.helpers import create_mock_pydantic


class SimpleModel(BaseModel):
    name: str
    score: int
    tags: list[str]


@pytest.mark.asyncio
async def test_mock_llm_client_default(mock_llm_client: MockLLMClient) -> None:
    """
    Asserts default fallback text generation and telemetry.
    """
    prompt = "Hello"
    response = await mock_llm_client.generate(prompt)
    assert response == "AI Mock response for testing."
    assert len(mock_llm_client.calls) == 1
    assert mock_llm_client.calls[0]["prompt"] == prompt


@pytest.mark.asyncio
async def test_mock_llm_client_queue(mock_llm_client: MockLLMClient) -> None:
    """
    Asserts FIFO response queue behavior.
    """
    mock_llm_client.add_response("Response 1")
    mock_llm_client.add_response("Response 2")

    r1 = await mock_llm_client.generate("hello")
    r2 = await mock_llm_client.generate("world")
    r3 = await mock_llm_client.generate("fallback")

    assert r1 == "Response 1"
    assert r2 == "Response 2"
    assert r3 == "AI Mock response for testing."
    assert len(mock_llm_client.calls) == 3


@pytest.mark.asyncio
async def test_mock_llm_client_mapping(mock_llm_client: MockLLMClient) -> None:
    """
    Asserts prompt substring pattern mapping triggers correct mock response.
    """
    mock_llm_client.add_response_mapping("resume", "Resume advice")
    mock_llm_client.add_response_mapping("interview", "Interview prep")

    r1 = await mock_llm_client.generate("Please check my resume")
    r2 = await mock_llm_client.generate("Let's do an interview practice")
    r3 = await mock_llm_client.generate("other things")

    assert r1 == "Resume advice"
    assert r2 == "Interview prep"
    assert r3 == "AI Mock response for testing."


@pytest.mark.asyncio
async def test_mock_llm_client_structured_default(
    mock_llm_client: MockLLMClient,
) -> None:
    """
    Asserts structured validation defaults and creation using helpers.
    """
    mock_llm_client.default_structured_response = create_mock_pydantic(
        SimpleModel, name="Custom Default"
    )

    res = await mock_llm_client.generate_structured("give me structured", SimpleModel)
    assert res.name == "Custom Default"
    assert res.score == 1
    assert res.tags == ["mock_tags"]


@pytest.mark.asyncio
async def test_mock_llm_client_structured_queue_and_mapping(
    mock_llm_client: MockLLMClient,
) -> None:
    """
    Asserts queue and mapping precedence for structured outputs.
    """
    obj1 = create_mock_pydantic(SimpleModel, name="Alice")
    obj2 = create_mock_pydantic(SimpleModel, name="Bob")
    obj_mapped = create_mock_pydantic(SimpleModel, name="Mapped Charlie")

    mock_llm_client.add_structured_response(obj1)
    mock_llm_client.add_structured_response(obj2)
    mock_llm_client.add_structured_mapping("charlie", obj_mapped)

    r1 = await mock_llm_client.generate_structured("first", SimpleModel)
    r2 = await mock_llm_client.generate_structured("tell me about charlie", SimpleModel)
    r3 = await mock_llm_client.generate_structured("second", SimpleModel)

    assert r1.name == "Alice"
    assert r2.name == "Mapped Charlie"
    assert r3.name == "Bob"
