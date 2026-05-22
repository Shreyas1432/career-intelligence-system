from typing import Any, TypeVar

import structlog
from pydantic import BaseModel

from src.core.ai.base import BaseLLMClient
from src.core.prompts import prompt_manager

logger = structlog.get_logger("src.core.ai.service")

T = TypeVar("T", bound=BaseModel)


class AIService:
    """
    Orchestration service decoupling feature flows from raw API formats.
    Coordinates jinja template rendering, client invocations, and format-validation recoveries.
    """

    def __init__(self, llm_client: BaseLLMClient) -> None:
        self.llm_client = llm_client

    async def generate_from_template(
        self,
        template_path: str,
        context: dict[str, Any],
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Loads prompt template from path, compiles it using Jinja2 context, and generates completion.
        """
        prompt = prompt_manager.load_prompt(template_path, context)
        return await self.llm_client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    async def generate_structured_from_template(
        self,
        template_path: str,
        context: dict[str, Any],
        response_model: type[T],
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_parse_retries: int = 2,
        **kwargs: Any,
    ) -> T:
        """
        Loads prompt template, compiles it, and validates generated JSON format.
        Retries query execution automatically if json validation fails.
        """
        prompt = prompt_manager.load_prompt(template_path, context)
        last_exc: Exception | None = None

        for attempt in range(max_parse_retries + 1):
            try:
                # Slightly adjust temperature on retries to nudge model to formatting correctness
                current_temp = temperature
                if attempt > 0 and current_temp is not None:
                    current_temp = min(current_temp + 0.1, 1.0)

                return await self.llm_client.generate_structured(
                    prompt=prompt,
                    response_model=response_model,
                    system_prompt=system_prompt,
                    model=model,
                    temperature=current_temp,
                    **kwargs,
                )
            except ValueError as exc:
                last_exc = exc
                logger.warning(
                    "Structured json validation failed, re-attempting completion generation",
                    attempt=attempt,
                    error=str(exc),
                )

        raise RuntimeError(
            f"Failed to generate structured response after {max_parse_retries} parser retries. "
            f"Validation breakdown: {last_exc}"
        ) from last_exc
