import time
from typing import Any

import pytest
from pydantic import BaseModel

from src.core.config.scrapegraphai import ScrapeGraphAIConfig
from src.modules.job_extraction import (
    EmploymentType,
    JobDomain,
    JobExtractionResult,
    JobExtractionService,
    JobExtractionTimeoutError,
    JobExtractionValidationError,
    JobIntelligenceSchema,
    VisaSignal,
)
from src.modules.job_extraction.scrapegraph_adapter import ScrapeGraphAIAdapter


class FakeAdapter:
    def __init__(self, outputs: list[Any]):
        self.outputs = outputs
        self.calls: list[dict[str, Any]] = []

    def run(
        self,
        *,
        prompt: str,
        source: str,
        response_model: type[BaseModel],
    ) -> Any:
        self.calls.append(
            {
                "prompt": prompt,
                "source": source,
                "response_model": response_model,
            }
        )
        return self.outputs.pop(0)


class SlowAdapter:
    def run(
        self,
        *,
        prompt: str,
        source: str,
        response_model: type[BaseModel],
    ) -> dict[str, str]:
        _ = (prompt, source, response_model)
        time.sleep(0.2)
        return {"title": "Late result"}


@pytest.mark.asyncio
async def test_extracts_validated_job_information() -> None:
    adapter = FakeAdapter(
        [
            {
                "company": " Acme AI ",
                "title": "Senior ML Engineer",
                "skills": ["Python", "python", " PySpark "],
                "experience_required": "5+ years",
                "location": "Dublin, Ireland",
                "visa_signal": "right to work required",
                "employment_type": "Full-time",
                "domain": "machine learning",
                "confidence_score": 0.88,
            }
        ]
    )
    service = JobExtractionService(
        adapter=adapter,
        config=ScrapeGraphAIConfig(retry_attempts=0),
    )

    result = await service.extract("<html>job posting</html>", source_url="https://example.com/job")

    assert result.company == "Acme AI"
    assert result.title == "Senior ML Engineer"
    assert result.skills == ["Python", "Spark"]
    assert result.visa_signal == VisaSignal.WORK_AUTH_REQUIRED
    assert result.employment_type == EmploymentType.FULL_TIME
    assert result.domain == JobDomain.DATA_AI
    assert result.confidence_score == 0.88
    assert adapter.calls[0]["response_model"] is JobExtractionResult
    assert "https://example.com/job" in adapter.calls[0]["prompt"]


@pytest.mark.asyncio
async def test_retries_validation_failures() -> None:
    adapter = FakeAdapter(
        [
            {},
            {"company": "Acme", "title": "Backend Engineer"},
        ]
    )
    service = JobExtractionService(
        adapter=adapter,
        config=ScrapeGraphAIConfig(retry_attempts=1, retry_backoff_seconds=0),
    )

    result = await service.extract("Job title Backend Engineer at Acme")

    assert result.company == "Acme"
    assert len(adapter.calls) == 2


@pytest.mark.asyncio
async def test_raises_validation_error_after_retry_exhaustion() -> None:
    service = JobExtractionService(
        adapter=FakeAdapter([{}]),
        config=ScrapeGraphAIConfig(retry_attempts=0),
    )

    with pytest.raises(JobExtractionValidationError):
        await service.extract("No usable job signal")


@pytest.mark.asyncio
async def test_enforces_timeout() -> None:
    service = JobExtractionService(
        adapter=SlowAdapter(),
        config=ScrapeGraphAIConfig(timeout_seconds=0.01, retry_attempts=0),
    )

    with pytest.raises(JobExtractionTimeoutError):
        await service.extract("Job title Backend Engineer at Acme")


@pytest.mark.asyncio
async def test_truncates_source_for_token_control() -> None:
    adapter = FakeAdapter([{"title": "Backend Engineer"}])
    service = JobExtractionService(
        adapter=adapter,
        config=ScrapeGraphAIConfig(max_source_chars=1000, retry_attempts=0),
    )

    await service.extract("x" * 1500)

    assert len(adapter.calls[0]["source"]) == 1000


def test_job_result_rejects_empty_extraction() -> None:
    with pytest.raises(ValueError, match="usable job signals"):
        JobExtractionResult.model_validate({})


def test_job_schema_normalizes_enums_and_estimates_confidence() -> None:
    result = JobIntelligenceSchema.model_validate(
        {
            "company": " Example Corp ",
            "title": " Backend Engineer ",
            "skills": [" Python ", "python", "Oracle Fusion"],
            "visa_signal": "No Sponsorship",
            "employment_type": "permanent",
            "domain": "backend",
        }
    )

    assert result.company == "Example Corp"
    assert result.title == "Backend Engineer"
    assert result.skills == ["Python", "ERP"]
    assert result.visa_signal == VisaSignal.NO_SPONSORSHIP
    assert result.employment_type == EmploymentType.FULL_TIME
    assert result.domain == JobDomain.SOFTWARE_ENGINEERING
    assert result.confidence_score is not None
    assert result.confidence_score > 0


@pytest.mark.parametrize(
    "payload",
    [
        {"title": "Engineer", "skills": "Python"},
        {"title": "Engineer", "skills": ["Python", 123]},
        {"title": "Engineer", "employment_type": "retainer plus equity"},
        {"title": "Engineer", "domain": "unmapped domain"},
        {"title": "Engineer", "confidence_score": "0.8"},
        {"title": "Engineer", "confidence_score": 1.2},
        {"title": "Engineer", "unexpected": "field"},
        {"title": "Engineer", "sponsorship_clues": ["visa"]},
        {"title": "Engineer", "domain_signals": ["AI"]},
    ],
)
def test_job_schema_rejects_invalid_outputs(payload: dict[str, object]) -> None:
    with pytest.raises(ValueError, match=r"validation error|Expected|Unsupported|extra"):
        JobIntelligenceSchema.model_validate(payload)


def test_scrapegraph_adapter_builds_ollama_json_config() -> None:
    adapter = ScrapeGraphAIAdapter(
        ScrapeGraphAIConfig(
            model="mistral",
            base_url="http://localhost:11434",
            temperature=0,
            model_tokens=2048,
        )
    )

    graph_config = adapter.build_graph_config()

    assert graph_config["llm"]["model"] == "ollama/mistral"
    assert graph_config["llm"]["base_url"] == "http://localhost:11434"
    assert graph_config["llm"]["format"] == "json"
    assert graph_config["llm"]["model_tokens"] == 2048
