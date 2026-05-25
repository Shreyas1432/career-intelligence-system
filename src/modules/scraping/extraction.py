import asyncio
import logging
from typing import Any, Protocol, cast

from pydantic import BaseModel, ValidationError

from src.core.config import settings
from src.core.config.scrapegraphai import ScrapeGraphAIConfig
from src.core.prompts import prompt_manager
from src.modules.scraping.schemas import JobExtractionInput, JobExtractionResult

logger = logging.getLogger("src.modules.scraping.extraction")

DEFAULT_PROMPT_TEMPLATE = "job_extraction/extract_prompt.md"


class JobExtractionError(RuntimeError):
    """
    Base exception for intelligent job extraction failures.
    """


class ScrapeGraphAIUnavailableError(JobExtractionError):
    """
    Raised when ScrapeGraphAI is not installed or cannot be imported.
    """


class JobExtractionTimeoutError(JobExtractionError):
    """
    Raised when extraction exceeds the configured timeout.
    """


class JobExtractionValidationError(JobExtractionError):
    """
    Raised when extraction output cannot be validated.
    """


class StructuredExtractionAdapter(Protocol):
    """
    Adapter contract for schema-driven extraction engines.
    """

    def run(
        self,
        *,
        prompt: str,
        source: str,
        response_model: type[BaseModel],
    ) -> Any:
        """
        Run synchronous structured extraction.
        """


class ScrapeGraphAIAdapter:
    """
    Lazy ScrapeGraphAI adapter, isolated to avoid import cost at application startup.
    """

    def __init__(self, config: ScrapeGraphAIConfig | None = None):
        self.config = config or settings.scrapegraphai

    def run(
        self,
        *,
        prompt: str,
        source: str,
        response_model: type[BaseModel],
    ) -> Any:
        try:
            from scrapegraphai.graphs import SmartScraperGraph
        except Exception as exc:
            raise ScrapeGraphAIUnavailableError(
                "ScrapeGraphAI is not available. Install a compatible scrapegraphai package "
                "before running intelligent extraction."
            ) from exc

        graph = SmartScraperGraph(
            prompt=prompt,
            source=source,
            config=self.build_graph_config(),
            schema=response_model,
        )
        return graph.run()

    def build_graph_config(self) -> dict[str, Any]:
        base_url = self.config.base_url or settings.ai.ollama.base_url
        model_name = (
            self.config.model
            if self.config.model.startswith("ollama/")
            else f"ollama/{self.config.model}"
        )

        return {
            "llm": {
                "model": model_name,
                "base_url": base_url,
                "temperature": self.config.temperature,
                "format": "json",
                "model_tokens": self.config.model_tokens,
            },
            "verbose": self.config.verbose,
            "headless": self.config.headless,
        }


def validate_adapter_output(raw_output: Any, response_model: type[BaseModel]) -> BaseModel:
    """
    Normalize ScrapeGraphAI output into a Pydantic model.
    """
    if isinstance(raw_output, response_model):
        return raw_output

    if isinstance(raw_output, BaseModel):
        return response_model.model_validate(raw_output.model_dump())

    if isinstance(raw_output, dict):
        return response_model.model_validate(raw_output)

    if isinstance(raw_output, str):
        return response_model.model_validate_json(raw_output)

    return response_model.model_validate(cast(object, raw_output))


class JobExtractionService:
    """
    Async schema-driven job extraction service backed by ScrapeGraphAI.
    """

    def __init__(
        self,
        adapter: StructuredExtractionAdapter | None = None,
        config: ScrapeGraphAIConfig | None = None,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    ):
        self.config = config or settings.scrapegraphai
        self.adapter = adapter or ScrapeGraphAIAdapter(self.config)
        self.prompt_template = prompt_template

    async def extract(
        self,
        source: str,
        *,
        source_url: str | None = None,
    ) -> JobExtractionResult:
        extraction_input = JobExtractionInput(source=source, source_url=source_url)
        prepared_source = self._prepare_source(extraction_input.source)
        prompt = self._render_prompt(source_url=extraction_input.source_url)

        last_error: Exception | None = None
        for attempt in range(self.config.retry_attempts + 1):
            try:
                raw_output = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.adapter.run,
                        prompt=prompt,
                        source=prepared_source,
                        response_model=JobExtractionResult,
                    ),
                    timeout=self.config.timeout_seconds,
                )
                validated = validate_adapter_output(raw_output, JobExtractionResult)
                return JobExtractionResult.model_validate(validated)
            except TimeoutError as exc:
                last_error = exc
                if attempt >= self.config.retry_attempts:
                    raise JobExtractionTimeoutError(
                        f"Job extraction timed out after {self.config.timeout_seconds} seconds"
                    ) from exc
            except ValidationError as exc:
                last_error = exc
                if attempt >= self.config.retry_attempts:
                    raise JobExtractionValidationError(
                        f"Job extraction response failed schema validation: {exc}"
                    ) from exc
            except JobExtractionError:
                raise
            except Exception as exc:
                last_error = exc
                if attempt >= self.config.retry_attempts:
                    raise JobExtractionError(f"Job extraction failed: {exc}") from exc

            await self._sleep_before_retry(attempt)

        raise JobExtractionError("Job extraction failed after retries") from last_error

    def _prepare_source(self, source: str) -> str:
        if len(source) <= self.config.max_source_chars:
            return source

        logger.info(
            "Truncating extraction source for local-model token control",
            extra={
                "original_chars": len(source),
                "max_source_chars": self.config.max_source_chars,
            },
        )
        return source[: self.config.max_source_chars]

    def _render_prompt(self, source_url: str | None) -> str:
        try:
            return prompt_manager.load_prompt(
                self.prompt_template,
                {
                    "source_url": source_url or "unknown",
                    "max_skills": 20,
                },
            )
        except FileNotFoundError:
            logger.warning("Job extraction prompt template not found; using built-in fallback")
            return (
                "Extract only explicit job posting facts from the provided source. "
                "Return valid JSON matching the schema. Use null or [] when evidence is absent. "
                "Fields: company, title, skills, experience_required, location, "
                "visa_signal, employment_type, domain, confidence_score."
            )

    async def _sleep_before_retry(self, attempt: int) -> None:
        delay = self.config.retry_backoff_seconds * (2**attempt)
        if delay > 0:
            await asyncio.sleep(delay)


async def extract_job_information(
    source: str,
    *,
    source_url: str | None = None,
) -> JobExtractionResult:
    service = JobExtractionService()
    return await service.extract(source, source_url=source_url)
