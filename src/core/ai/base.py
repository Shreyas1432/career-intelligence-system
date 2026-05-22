from abc import ABC, abstractmethod
from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class BaseLLMClient(ABC):
    """
    Abstract Base Class defining the contract for LLM integration layers.
    Allows swapping Ollama with OpenAI, Anthropic, or Gemini.
    """

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Generates a text completion for the given prompt/messages.
        """
        pass

    @abstractmethod
    async def generate_structured(
        self,
        prompt: str,
        response_model: type[T],
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> T:
        """
        Generates a structured completion validated against a Pydantic model.
        """
        pass
