from typing import Any, TypeVar

import pytest
from pydantic import BaseModel

from src.core.ai.base import BaseLLMClient
from src.core.ai.service import AIService

T = TypeVar("T", bound=BaseModel)


class MockLLMClient(BaseLLMClient):
    """
    Highly configurable Test Double implementing the BaseLLMClient contract.
    Supports queue-based responses, substring mapping triggers, and call tracking.
    """

    def __init__(self) -> None:
        self.default_response = "AI Mock response for testing."
        self.response_queue: list[str] = []
        self.response_mappings: dict[str, str] = {}

        self.default_structured_response: BaseModel | None = None
        self.structured_queue: list[BaseModel] = []
        self.structured_mappings: dict[str, BaseModel] = {}

        # Call telemetry for assertions
        self.calls: list[dict[str, Any]] = []

    def add_response(self, text: str) -> None:
        self.response_queue.append(text)

    def add_response_mapping(self, substring: str, response: str) -> None:
        self.response_mappings[substring] = response

    def add_structured_response(self, obj: BaseModel) -> None:
        self.structured_queue.append(obj)

    def add_structured_mapping(self, substring: str, obj: BaseModel) -> None:
        self.structured_mappings[substring] = obj

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        self.calls.append(
            {
                "type": "generate",
                "prompt": prompt,
                "system_prompt": system_prompt,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "kwargs": kwargs,
            }
        )

        # Check mapping patterns
        for substr, resp in self.response_mappings.items():
            if substr in prompt or (system_prompt and substr in system_prompt):
                return resp

        # Check queue FIFO
        if self.response_queue:
            return self.response_queue.pop(0)

        return self.default_response

    async def generate_structured(
        self,
        prompt: str,
        response_model: type[T],
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> T:
        self.calls.append(
            {
                "type": "generate_structured",
                "prompt": prompt,
                "response_model": response_model,
                "system_prompt": system_prompt,
                "model": model,
                "temperature": temperature,
                "kwargs": kwargs,
            }
        )

        # Check mapping patterns
        for substr, resp in self.structured_mappings.items():
            if substr in prompt or (system_prompt and substr in system_prompt):
                return resp  # type: ignore

        # Check queue FIFO
        if self.structured_queue:
            return self.structured_queue.pop(0)  # type: ignore

        if self.default_structured_response:
            return self.default_structured_response  # type: ignore

        # Default fallback creation
        try:
            return response_model.model_validate({})
        except Exception as exc:
            raise ValueError(
                "No mock structured response matches payload, and default validation failed."
            ) from exc


@pytest.fixture
def mock_llm_client() -> MockLLMClient:
    """
    Fixture returning an isolated MockLLMClient.
    """
    return MockLLMClient()


@pytest.fixture
def mock_ai_service(mock_llm_client: MockLLMClient) -> AIService:
    """
    Fixture providing an AIService wrapper bound to the MockLLMClient.
    """
    return AIService(llm_client=mock_llm_client)


@pytest.fixture
def mock_litellm_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Fixture monkeypatching litellm.completion to intercept outbound requests globally.
    """
    import litellm

    class MockChoiceMessage:
        def __init__(self, content: str):
            self.content = content

    class MockChoice:
        def __init__(self, content: str):
            self.message = MockChoiceMessage(content)

    class MockResponse:
        def __init__(self, content: str):
            self.choices = [MockChoice(content)]

    def mock_completion(*_args: Any, **_kwargs: Any) -> MockResponse:
        return MockResponse("Globally mocked LiteLLM completion")

    monkeypatch.setattr(litellm, "completion", mock_completion)


@pytest.fixture
def mock_ai_client(monkeypatch: pytest.MonkeyPatch) -> Any:
    """
    Mocks the generative AI responses to avoid hitting API limits during unit tests.
    """
    from src.core.ai.client import ai_client

    def mock_generate_chat_response(*_args: Any, **_kwargs: Any) -> str:
        return "AI Mock response for testing."

    monkeypatch.setattr(ai_client, "generate_chat_response", mock_generate_chat_response)
    return ai_client
