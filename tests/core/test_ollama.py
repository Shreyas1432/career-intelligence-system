from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from pydantic import BaseModel

from src.core.ai.ollama import OllamaClient
from src.core.ai.service import AIService
from src.core.config import settings


class MockResponseModel(BaseModel):
    name: str
    score: int


class MockResponse:
    def __init__(self, json_data: dict, status_code: int = 200):
        self._json_data = json_data
        self.status_code = status_code

    def json(self) -> dict:
        return self._json_data

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "Error", request=MagicMock(), response=MagicMock(status_code=self.status_code)
            )


@pytest.mark.asyncio
async def test_resolve_model() -> None:
    client = OllamaClient()
    # Test fallback to settings default
    assert client._resolve_model(None) == settings.ai.ollama.model

    # Test custom alias mapping
    assert client._resolve_model("fast") == settings.ai.ollama.model_mappings["fast"]

    # Test unknown tag mapping
    assert client._resolve_model("custom-llama-v3") == "custom-llama-v3"
    await client.close()


@pytest.mark.asyncio
async def test_generate_success() -> None:
    mock_client = MagicMock(spec=httpx.AsyncClient)
    mock_client.post = AsyncMock(return_value=MockResponse({"message": {"content": "Hello user!"}}))

    client = OllamaClient(client=mock_client)
    res = await client.generate(prompt="Hello", system_prompt="Sys")
    assert res == "Hello user!"

    # Verify endpoint call parameters
    mock_client.post.assert_called_once()
    args, kwargs = mock_client.post.call_args
    assert args[0].endswith("/api/chat")
    payload = kwargs["json"]
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][0]["content"] == "Sys"
    assert payload["messages"][1]["role"] == "user"
    assert payload["messages"][1]["content"] == "Hello"


@pytest.mark.asyncio
async def test_generate_structured_success() -> None:
    mock_client = MagicMock(spec=httpx.AsyncClient)
    mock_client.post = AsyncMock(
        return_value=MockResponse({"message": {"content": '{"name": "Alice", "score": 95}'}})
    )

    client = OllamaClient(client=mock_client)
    res = await client.generate_structured(prompt="Get user", response_model=MockResponseModel)
    assert isinstance(res, MockResponseModel)
    assert res.name == "Alice"
    assert res.score == 95


@pytest.mark.asyncio
async def test_retry_on_transient_error() -> None:
    mock_client = MagicMock(spec=httpx.AsyncClient)

    # First attempt fails with 503, second attempt succeeds
    mock_client.post = AsyncMock(
        side_effect=[
            httpx.HTTPStatusError(
                "Service Unavailable", request=MagicMock(), response=MagicMock(status_code=503)
            ),
            MockResponse({"message": {"content": "Retry works!"}}),
        ]
    )

    client = OllamaClient(client=mock_client)
    client.backoff_factor = 0.01

    res = await client.generate(prompt="Retry test")
    assert res == "Retry works!"
    assert mock_client.post.call_count == 2


@pytest.mark.asyncio
async def test_retry_failure_permanent() -> None:
    mock_client = MagicMock(spec=httpx.AsyncClient)
    mock_client.post = AsyncMock(
        side_effect=httpx.HTTPStatusError(
            "Bad Request", request=MagicMock(), response=MagicMock(status_code=400)
        )
    )

    client = OllamaClient(client=mock_client)
    # 400 Bad Request should not retry, should fail immediately
    with pytest.raises(RuntimeError):
        await client.generate(prompt="Fail test")

    assert mock_client.post.call_count == 1


@pytest.mark.asyncio
async def test_ai_service_generate_from_template() -> None:
    mock_llm = MagicMock()
    mock_llm.generate = AsyncMock(return_value="Output from template")

    service = AIService(llm_client=mock_llm)

    # Mock PromptManager.load_prompt
    with patch(
        "src.core.ai.service.prompt_manager.load_prompt", return_value="Rendered prompt"
    ) as mock_load:
        res = await service.generate_from_template(
            template_path="some/path.md",
            context={"var": "val"},
            system_prompt="sys",
        )
        assert res == "Output from template"
        mock_load.assert_called_once_with("some/path.md", {"var": "val"})
        mock_llm.generate.assert_called_once_with(
            prompt="Rendered prompt",
            system_prompt="sys",
            model=None,
            temperature=None,
            max_tokens=None,
        )


@pytest.mark.asyncio
async def test_ai_service_structured_retry() -> None:
    mock_llm = MagicMock()
    # First attempt: invalid JSON. Second attempt: valid JSON.
    mock_llm.generate_structured = AsyncMock(
        side_effect=[
            ValueError("Invalid fields"),
            MockResponseModel(name="Bob", score=80),
        ]
    )

    service = AIService(llm_client=mock_llm)

    with patch("src.core.ai.service.prompt_manager.load_prompt", return_value="Rendered prompt"):
        res = await service.generate_structured_from_template(
            template_path="some/path.md",
            context={"var": "val"},
            response_model=MockResponseModel,
            temperature=0.2,
        )
        assert res.name == "Bob"
        assert res.score == 80
        assert mock_llm.generate_structured.call_count == 2
