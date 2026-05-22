from typing import Any, Protocol, cast

from pydantic import BaseModel

from src.core.config import settings
from src.core.config.scrapegraphai import ScrapeGraphAIConfig

from .exceptions import ScrapeGraphAIUnavailableError


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
