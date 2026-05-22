import logging
from typing import Any

import litellm

from src.core.config import settings

logger = logging.getLogger("src.core.ai")


class AIClient:
    """
    Unified client for LLM services, powered by LiteLLM.
    Supports OpenAI, Anthropic, and local Ollama integrations.
    """

    def __init__(self) -> None:
        # Configure LiteLLM configurations
        litellm.telemetry = False  # Keep private
        self.default_model = settings.ai.default_chat_model
        self.temperature = settings.ai.temperature
        self.max_tokens = settings.ai.max_tokens

        # Verify if API keys exist
        self.has_openai = bool(settings.openai_api_key)
        self.has_anthropic = bool(settings.anthropic_api_key)

        if not self.has_openai and not self.has_anthropic:
            logger.warning(
                "No cloud LLM API keys found. LLM queries will fall back "
                "to local Ollama or mock responses."
            )

    def generate_chat_response(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Sends chat queries to the configured LLM and returns string content.
        Automatically handles API routing and fallback strategies.
        """
        target_model = model or self.default_model
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens or self.max_tokens

        # Fallback to local Ollama if cloud credentials are absent and model is cloud-bound
        if not self.has_openai and "openai" in target_model:
            if settings.ollama_api_base:
                ollama_cfg = settings.ai.providers.get("ollama")
                ollama_model = ollama_cfg.model if ollama_cfg else "llama3"
                target_model = f"ollama/{ollama_model}"
                logger.info(f"Redirecting OpenAI request to Ollama: {target_model}")
            else:
                logger.error("No API key and Ollama not configured. Simulating response.")
                return "[MOCK RESPONSE] AI API Keys are not set. Configure OpenAI/Anthropic/Ollama in .env."

        try:
            response = litellm.completion(
                model=target_model,
                messages=messages,
                temperature=temp,
                max_tokens=tokens,
                api_base=settings.ollama_api_base if "ollama" in target_model else None,
                **kwargs,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"Error calling LLM provider: {e!s}")
            raise RuntimeError(f"AI Generation Failed: {e!s}") from e


# Global AI client instance
ai_client = AIClient()
