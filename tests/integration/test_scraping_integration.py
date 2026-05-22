# tests/integration/test_scraping_integration.py

from typing import Any

import pytest
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from sqlalchemy.orm import Session

from src.core.database.models import JobIntelligence
from src.modules.scraping_pipeline import ScrapingPipeline
from tests.fixtures.scraping_extraction import MockScrapeGraphAdapter
from tests.utils.scraping_mocks import (
    MOCK_GREENHOUSE_EXTRACTION,
    MOCK_GREENHOUSE_HTML,
    MOCK_LEVER_EXTRACTION,
    MOCK_LEVER_HTML,
    MOCK_MALFORMED_HTML,
)


@pytest.mark.asyncio
async def test_greenhouse_scraping_and_extraction_integration(
    db_session: Session,
    mock_playwright_routes: dict[str, Any],
    mock_scrapegraph_adapter: MockScrapeGraphAdapter,
) -> None:
    """
    Test E2E Greenhouse job flow: Scraping -> ATS -> Cleaning -> AI -> DB
    """
    url = "https://boards.greenhouse.io/acme/jobs/12345"
    mock_playwright_routes[url] = MOCK_GREENHOUSE_HTML
    mock_scrapegraph_adapter.set_output(MOCK_GREENHOUSE_EXTRACTION)

    pipeline = ScrapingPipeline()
    result = await pipeline.run(url=url, session=db_session)

    # 1. Pipeline execution status
    assert result.status == "success"
    assert len(result.errors) == 0

    # 2. ATS Platform and Extraction validation
    assert result.steps["ats_detection_preliminary"].result.value == "greenhouse"
    assert result.extracted_data is not None
    assert result.extracted_data.company == "Acme Corp"
    assert result.extracted_data.skills == ["Python", "Spark", "SQL"]

    # 3. Database persistence validation
    persisted = db_session.query(JobIntelligence).filter_by(url=url).first()
    assert persisted is not None
    assert persisted.company == "Acme Corp"
    assert persisted.title == "Senior Python Engineer"
    assert persisted.ats_type == "greenhouse"

    sponsorship = persisted.sponsorship_signals
    assert isinstance(sponsorship, dict)
    assert sponsorship["status"] == "positive"
    assert persisted.normalized_skills == ["Python", "Spark", "SQL"]


@pytest.mark.asyncio
async def test_lever_scraping_and_extraction_integration(
    db_session: Session,
    mock_playwright_routes: dict[str, Any],
    mock_scrapegraph_adapter: MockScrapeGraphAdapter,
) -> None:
    """
    Test E2E Lever job flow: Scraping -> ATS -> Cleaning -> AI -> DB
    """
    url = "https://jobs.lever.co/leverinc/54321"
    mock_playwright_routes[url] = MOCK_LEVER_HTML
    mock_scrapegraph_adapter.set_output(MOCK_LEVER_EXTRACTION)

    pipeline = ScrapingPipeline()
    result = await pipeline.run(url=url, session=db_session)

    assert result.status == "success"
    assert result.steps["ats_detection_preliminary"].result.value == "lever"
    assert result.extracted_data is not None
    assert result.extracted_data.company == "Lever Inc"
    assert result.extracted_data.skills == ["Python", "FastAPI", "SQLAlchemy"]

    persisted = db_session.query(JobIntelligence).filter_by(url=url).first()
    assert persisted is not None
    assert persisted.company == "Lever Inc"
    assert persisted.ats_type == "lever"


@pytest.mark.asyncio
async def test_malformed_page_scraping_and_extraction_integration(
    db_session: Session,
    mock_playwright_routes: dict[str, Any],
    mock_scrapegraph_adapter: MockScrapeGraphAdapter,
) -> None:
    """
    Verify malformed HTML parsing is resilient and still extracts valid fields.
    """
    url = "https://jobs.lever.co/leverinc/malformed"
    mock_playwright_routes[url] = MOCK_MALFORMED_HTML
    mock_scrapegraph_adapter.set_output(
        {
            "company": "Broken Co",
            "title": "Software Developer",
            "skills": ["Python"],
            "confidence_score": 0.70,
        }
    )

    pipeline = ScrapingPipeline()
    result = await pipeline.run(url=url, session=db_session)

    assert result.status == "success"
    assert result.extracted_data is not None
    title = result.extracted_data.title
    assert title is not None
    assert "Software Developer" in title
    assert "noisy script" not in result.steps["content_cleaning"].result

    persisted = db_session.query(JobIntelligence).filter_by(url=url).first()
    assert persisted is not None
    assert persisted.company == "Broken Co"


@pytest.mark.asyncio
async def test_browser_timeout_handling(
    db_session: Session,
    mock_playwright_routes: dict[str, Any],
) -> None:
    """
    Verify browser rendering timeout causes pipeline failure gracefully.
    """
    url = "https://jobs.lever.co/leverinc/timeout"
    mock_playwright_routes[url] = PlaywrightTimeoutError("navigation timed out")

    pipeline = ScrapingPipeline()
    result = await pipeline.run(url=url, session=db_session)

    assert result.status == "failed"
    assert "browser_rendering" in result.steps
    assert result.steps["browser_rendering"].status == "failed"

    err = result.steps["browser_rendering"].error
    assert err is not None
    assert "Failed to load" in err
    assert "ai_extraction" not in result.steps


@pytest.mark.asyncio
async def test_ai_extraction_timeout_handling(
    db_session: Session,
    mock_playwright_routes: dict[str, Any],
    mock_scrapegraph_adapter: MockScrapeGraphAdapter,
) -> None:
    """
    Verify AI extraction timeout limits are handled as fatal errors.
    """
    url = "https://jobs.lever.co/leverinc/ai_delay"
    mock_playwright_routes[url] = MOCK_LEVER_HTML
    mock_scrapegraph_adapter.set_error(TimeoutError("LLM response timed out"))

    pipeline = ScrapingPipeline()
    result = await pipeline.run(url=url, session=db_session)

    assert result.status == "failed"
    assert result.steps["browser_rendering"].status == "success"
    assert result.steps["ai_extraction"].status == "failed"

    err = result.steps["ai_extraction"].error
    assert err is not None
    assert "timed out" in err


@pytest.mark.asyncio
async def test_invalid_ai_outputs_retry_success(
    db_session: Session,
    mock_playwright_routes: dict[str, Any],
    mock_scrapegraph_adapter: MockScrapeGraphAdapter,
) -> None:
    """
    Test retry logic: AI returns empty invalid output first, then valid output.
    """
    url = "https://jobs.lever.co/leverinc/retrysuccess"
    mock_playwright_routes[url] = MOCK_LEVER_HTML

    # First outputs an empty dict (fails validation), second outputs a valid payload
    mock_scrapegraph_adapter.set_outputs([{}, MOCK_LEVER_EXTRACTION])

    pipeline = ScrapingPipeline()
    result = await pipeline.run(url=url, session=db_session)

    assert result.status == "success"
    assert result.extracted_data is not None
    assert result.extracted_data.company == "Lever Inc"
    assert len(mock_scrapegraph_adapter.calls) == 2
