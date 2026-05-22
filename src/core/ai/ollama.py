import asyncio
from typing import Any, TypeVar, cast

import httpx
import structlog
from pydantic import BaseModel

from src.core.ai.base import BaseLLMClient
from src.core.config import settings

logger = structlog.get_logger("src.core.ai.ollama")

T = TypeVar("T", bound=BaseModel)


class OllamaClient(BaseLLMClient):
    """
    Production-grade local Ollama LLM integration wrapper using httpx.AsyncClient.
    Features strict timeouts, exponential backoff retries, and structured validation.
    """

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self.client = client or httpx.AsyncClient()
        self.base_url = settings.ai.ollama.base_url.rstrip("/")
        self.timeout = httpx.Timeout(float(settings.ai.ollama.timeout_seconds))
        self.max_retries = settings.ai.ollama.max_retries
        self.backoff_factor = settings.ai.ollama.backoff_factor

    def _resolve_model(self, model: str | None) -> str:
        """
        Maps a model alias to its real Ollama tag name, falling back to default configuration.
        """
        # Resolve target name or default
        target = model or settings.ai.ollama.model
        # Resolve mapping/alias if configured
        return settings.ai.ollama.model_mappings.get(target, target)

    async def _post_with_retry(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        """
        Makes a POST request with exponential backoff retry on connection errors and transient 5xx/429.
        """
        url = f"{self.base_url}{endpoint}"
        last_exception: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                logger.debug("Sending request to Ollama", url=url, attempt=attempt)
                response = await self.client.post(url, json=payload, timeout=self.timeout)
                response.raise_for_status()
                return cast(dict[str, Any], response.json())
            except (httpx.RequestError, httpx.HTTPStatusError) as exc:
                last_exception = exc
                should_retry = True

                # Disable retry for client errors (other than rate limits)
                if isinstance(exc, httpx.HTTPStatusError):
                    if exc.response.status_code not in (429, 500, 502, 503, 504):
                        should_retry = False

                if should_retry and attempt < self.max_retries:
                    delay = self.backoff_factor**attempt
                    logger.warning(
                        "Ollama request failed, retrying...",
                        error=str(exc),
                        attempt=attempt,
                        next_retry_delay=delay,
                    )
                    await asyncio.sleep(delay)
                    continue

                logger.error("Ollama request failed permanently", error=str(exc), attempt=attempt)
                break

        raise RuntimeError(f"Ollama API request failed: {last_exception}") from last_exception

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
        Generates completion content async using Ollama's /api/chat.
        """
        resolved_model = self._resolve_model(model)

        # Build chat message sequence
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Base Options overrides
        options: dict[str, Any] = {}
        if temperature is not None:
            options["temperature"] = temperature
        elif settings.ai.temperature is not None:
            options["temperature"] = settings.ai.temperature

        if max_tokens is not None:
            options["num_predict"] = max_tokens
        elif settings.ai.max_tokens is not None:
            options["num_predict"] = settings.ai.max_tokens

        # Merge parameter level options dict if supplied
        if "options" in kwargs:
            options.update(kwargs.pop("options"))

        payload = {
            "model": resolved_model,
            "messages": messages,
            "stream": False,
            "options": options,
            **kwargs,
        }

        result = await self._post_with_retry("/api/chat", payload)

        try:
            return cast(str, result["message"]["content"])
        except KeyError as exc:
            logger.error("Failed to parse completion response", response=result)
            raise RuntimeError("Invalid response structure from Ollama API") from exc

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
        Generates validation object constraint structured responses using Ollama and Pydantic.
        """
        schema_json = response_model.model_json_schema()

        # Enforce json formatting and provide structure hints
        json_hint = (
            "You MUST respond ONLY with a valid JSON object matching the JSON schema below.\n"
            "Do not include any markdown format tags, preambles, or post-explanations.\n"
            f"Schema:\n{schema_json}"
        )
        system_merged = f"{system_prompt}\n\n{json_hint}" if system_prompt else json_hint

        kwargs["format"] = "json"

        response_text = await self.generate(
            prompt=prompt,
            system_prompt=system_merged,
            model=model,
            temperature=temperature,
            **kwargs,
        )

        try:
            return response_model.model_validate_json(response_text)
        except Exception as exc:
            logger.error(
                "JSON verification failed against target schema",
                response_text=response_text,
                schema=schema_json,
            )
            raise ValueError(
                f"JSON response failed validation against Pydantic model: {exc!s}\n"
                f"Response: {response_text}"
            ) from exc

    async def close(self) -> None:
        """
        Closes the underlying HTTP client.
        """
        await self.client.aclose()
