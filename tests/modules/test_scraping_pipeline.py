from typing import Any

import pytest
from sqlalchemy.orm import Session

from src.core.browser.types import PageSnapshot
from src.core.database.models import JobIntelligence
from src.modules.matching import DetectionResult, SponsorshipStatus
from src.modules.scraping import ScrapingPipeline, clean_html, clean_text
from src.modules.scraping.schemas import (
    EmploymentType,
    JobDomain,
    JobExtractionResult,
    VisaSignal,
)

# ------------------------------------------------------------------------------
# Mock / Double Definitions
# ------------------------------------------------------------------------------


class MockBrowserService:
    def __init__(self, html: str, text: str, should_fail: bool = False):
        self.html = html
        self.text = text
        self.should_fail = should_fail
        self.calls = 0

    async def capture_page(self, url: str, **_kwargs: Any) -> PageSnapshot:
        self.calls += 1
        if self.should_fail:
            raise RuntimeError("Browser failed to capture page")
        return PageSnapshot(
            requested_url=url,
            final_url=url,
            html=self.html,
            rendered_dom=self.html,
            text=self.text,
            screenshot=None,
        )


class MockExtractionService:
    def __init__(self, result: JobExtractionResult | None, should_fail: bool = False):
        self.result = result
        self.should_fail = should_fail
        self.calls = 0

    async def extract(self, source: str, source_url: str | None = None) -> JobExtractionResult:
        _ = (source, source_url)
        self.calls += 1
        if self.should_fail:
            raise ValueError("AI model validation failed")
        assert self.result is not None
        return self.result


class MockSponsorshipDetector:
    def __init__(self, result: DetectionResult | None, should_fail: bool = False):
        self.result = result
        self.should_fail = should_fail
        self.calls = 0

    async def detect(self, text: str, use_ai: bool = False) -> DetectionResult:
        _ = (text, use_ai)
        self.calls += 1
        if self.should_fail:
            raise RuntimeError("Sponsorship detector timeout")
        assert self.result is not None
        return self.result


# ------------------------------------------------------------------------------
# Content Cleaner Tests
# ------------------------------------------------------------------------------


def test_html_cleaner_strips_noise() -> None:
    html_content = """
    <html>
        <head><title>Job Posting</title></head>
        <body>
            <header>
                <nav><a href="/home">Home</a></nav>
            </header>
            <main>
                <h1>Software Engineer</h1>
                <p>We are looking for a <strong>Senior Engineer</strong>.</p>
                <ul>
                    <li>Python</li>
                    <li>SQL</li>
                </ul>
            </main>
            <aside>Related Jobs</aside>
            <footer>&copy; 2026 Company</footer>
            <script>console.log("noisy script");</script>
            <style>body { background: red; }</style>
        </body>
    </html>
    """
    cleaned = clean_html(html_content)

    assert "Home" not in cleaned
    assert "noisy script" not in cleaned
    assert "background: red" not in cleaned
    assert "Related Jobs" not in cleaned
    assert "Software Engineer" in cleaned
    assert "Senior Engineer" in cleaned
    assert "Python" in cleaned
    assert "SQL" in cleaned


def test_clean_text_normalizes_whitespace() -> None:
    text = "   Hello    World \n\n   This  is   a   test.   "
    cleaned = clean_text(text)
    assert cleaned == "Hello World\nThis is a test."


# ------------------------------------------------------------------------------
# Pipeline Orchestrator Tests
# ------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pipeline_success_flow(db_session: Session) -> None:
    # 1. Arrange mock services
    mock_browser = MockBrowserService(
        html="<html><body><h1>Staff Engineer</h1><p>Workday ATS link: company.myworkdayjobs.com/apply</p></body></html>",
        text="Staff Engineer at Google. Workday ATS.",
    )
    extracted_data = JobExtractionResult(
        company="Google",
        title="Staff Engineer",
        skills=["Python", "Go"],
        experience_required="5+ years",
        location="Remote",
        visa_signal=VisaSignal.SPONSORSHIP_AVAILABLE,
        employment_type=EmploymentType.FULL_TIME,
        domain=JobDomain.SOFTWARE_ENGINEERING,
        confidence_score=0.95,
    )
    mock_extraction = MockExtractionService(extracted_data)
    sponsorship_data = DetectionResult(
        status=SponsorshipStatus.POSITIVE,
        confidence=1.0,
        signals=[],
        explanation="Explicitly friendly",
    )
    mock_sponsorship = MockSponsorshipDetector(sponsorship_data)

    pipeline = ScrapingPipeline(
        browser_service=mock_browser,  # type: ignore[arg-type]
        extraction_service=mock_extraction,  # type: ignore[arg-type]
        sponsorship_detector=mock_sponsorship,  # type: ignore[arg-type]
    )

    url = "https://company.myworkdayjobs.com/careers/staff-engineer"

    # 2. Act
    run_result = await pipeline.run(url=url, session=db_session)

    # 3. Assert Overall Status
    assert run_result.status == "success"
    assert run_result.url == url
    assert run_result.run_id is not None
    assert len(run_result.errors) == 0

    # Assert Step-by-Step Outcomes
    assert run_result.steps["ats_detection_preliminary"].status == "success"
    assert run_result.steps["browser_rendering"].status == "success"
    assert run_result.steps["content_cleaning"].status == "success"
    assert run_result.steps["ats_detection_post_render"].status == "success"
    assert run_result.steps["ai_extraction"].status == "success"
    assert run_result.steps["sponsorship_detection"].status == "success"
    assert run_result.steps["persistence"].status == "success"

    # Assert correct integrations
    assert mock_browser.calls == 1
    assert mock_extraction.calls == 1
    assert mock_sponsorship.calls == 1

    # Assert DB Persistence
    persisted: JobIntelligence | None = db_session.query(JobIntelligence).filter_by(url=url).first()
    assert persisted is not None
    assert persisted.company == "Google"
    assert persisted.title == "Staff Engineer"
    assert persisted.ats_type == "workday"
    assert persisted.sponsorship_signals is not None
    assert persisted.sponsorship_signals["status"] == "positive"
    assert persisted.normalized_skills == ["Python", "Go"]


@pytest.mark.asyncio
async def test_pipeline_partial_failure_non_fatal_step(db_session: Session) -> None:
    # Arrange: Mock sponsorship detector to throw exception
    mock_browser = MockBrowserService(
        html="<html><body><h1>Engineer</h1></body></html>",
        text="Engineer",
    )
    extracted_data = JobExtractionResult(
        company="Meta",
        title="Engineer",
        skills=["Python"],
        experience_required="3+ years",
        location="Remote",
        visa_signal=VisaSignal.UNKNOWN,
        employment_type=EmploymentType.FULL_TIME,
        domain=JobDomain.SOFTWARE_ENGINEERING,
        confidence_score=0.8,
    )
    mock_extraction = MockExtractionService(extracted_data)
    mock_sponsorship = MockSponsorshipDetector(None, should_fail=True)  # Non-fatal step fails

    pipeline = ScrapingPipeline(
        browser_service=mock_browser,  # type: ignore[arg-type]
        extraction_service=mock_extraction,  # type: ignore[arg-type]
        sponsorship_detector=mock_sponsorship,  # type: ignore[arg-type]
    )

    url = "https://meta.com/careers/engineer"

    # Act
    run_result = await pipeline.run(url=url, session=db_session)

    # Assert
    assert run_result.status == "partial_success"
    assert "Step 'sponsorship_detection' failed" in run_result.errors[0]
    assert run_result.steps["sponsorship_detection"].status == "failed"
    assert run_result.steps["ai_extraction"].status == "success"
    assert run_result.steps["persistence"].status == "success"

    # Verify DB persistence still succeeded with default/null sponsorship details
    persisted: JobIntelligence | None = db_session.query(JobIntelligence).filter_by(url=url).first()
    assert persisted is not None
    assert persisted.company == "Meta"
    assert persisted.sponsorship_signals is not None
    assert persisted.sponsorship_signals["status"] == "unknown"  # Fails back gracefully


@pytest.mark.asyncio
async def test_pipeline_fatal_failure_halts_execution(db_session: Session) -> None:
    # Arrange: Mock browser service to fail (fatal step)
    mock_browser = MockBrowserService("", "", should_fail=True)
    mock_extraction = MockExtractionService(None)
    mock_sponsorship = MockSponsorshipDetector(None)

    pipeline = ScrapingPipeline(
        browser_service=mock_browser,  # type: ignore[arg-type]
        extraction_service=mock_extraction,  # type: ignore[arg-type]
        sponsorship_detector=mock_sponsorship,  # type: ignore[arg-type]
    )

    url = "https://broken-link.com"

    # Act
    run_result = await pipeline.run(url=url, session=db_session)

    # Assert
    assert run_result.status == "failed"
    assert any("browser_rendering" in err for err in run_result.errors)
    assert run_result.steps["browser_rendering"].status == "failed"

    # Extraction & persistence should not even be attempted
    assert "ai_extraction" not in run_result.steps
    assert "persistence" not in run_result.steps
    assert mock_extraction.calls == 0

    # DB must have no records
    count = db_session.query(JobIntelligence).filter_by(url=url).count()
    assert count == 0


@pytest.mark.asyncio
async def test_pipeline_with_pre_rendered_html(db_session: Session) -> None:
    # Arrange: Pass HTML directly to bypass browser rendering
    extracted_data = JobExtractionResult(
        company="Apple",
        title="iOS Developer",
        skills=["Swift"],
        experience_required="2+ years",
        location="Cupertino",
        visa_signal=VisaSignal.WORK_AUTH_REQUIRED,
        employment_type=EmploymentType.FULL_TIME,
        domain=JobDomain.SOFTWARE_ENGINEERING,
        confidence_score=0.9,
    )
    mock_extraction = MockExtractionService(extracted_data)
    mock_sponsorship = MockSponsorshipDetector(
        DetectionResult(
            status=SponsorshipStatus.NEUTRAL, confidence=0.8, signals=[], explanation=""
        )
    )

    # We do NOT pass a browser service, but if it gets called, it would fail
    mock_browser = MockBrowserService("", "", should_fail=True)

    pipeline = ScrapingPipeline(
        browser_service=mock_browser,  # type: ignore[arg-type]
        extraction_service=mock_extraction,  # type: ignore[arg-type]
        sponsorship_detector=mock_sponsorship,  # type: ignore[arg-type]
    )

    url = "https://apple.com/careers/ios-dev"
    html_source = "<html><body><h1>iOS Developer</h1></body></html>"

    # Act
    run_result = await pipeline.run(url=url, html=html_source, session=db_session)

    # Assert
    assert run_result.status == "success"
    assert run_result.steps["browser_rendering"].status == "skipped"
    assert run_result.steps["ai_extraction"].status == "success"
    assert mock_browser.calls == 0  # Browser was never called
    assert mock_extraction.calls == 1

    # Verify DB persistence succeeded
    persisted: JobIntelligence | None = db_session.query(JobIntelligence).filter_by(url=url).first()
    assert persisted is not None
    assert persisted.company == "Apple"


@pytest.mark.asyncio
async def test_pipeline_callbacks() -> None:
    # Arrange: Test callback invocation
    mock_browser = MockBrowserService("<html><body><h1>Title</h1></body></html>", "Title")
    extracted_data = JobExtractionResult(
        company="Test",
        title="Title",
        skills=[],
        experience_required=None,
        location=None,
        visa_signal=VisaSignal.UNKNOWN,
        employment_type=EmploymentType.UNKNOWN,
        domain=JobDomain.UNKNOWN,
        confidence_score=0.5,
    )
    mock_extraction = MockExtractionService(extracted_data)
    mock_sponsorship = MockSponsorshipDetector(
        DetectionResult(
            status=SponsorshipStatus.UNKNOWN, confidence=0.0, signals=[], explanation=""
        )
    )

    starts = []
    completes = []
    failures = []

    def on_start(step_name: str, _run_id: str) -> None:
        starts.append(step_name)

    def on_complete(step_name: str, _result: Any, _run_id: str) -> None:
        completes.append(step_name)

    def on_fail(step_name: str, _error: Exception, _run_id: str) -> None:
        failures.append(step_name)

    pipeline = ScrapingPipeline(
        browser_service=mock_browser,  # type: ignore[arg-type]
        extraction_service=mock_extraction,  # type: ignore[arg-type]
        sponsorship_detector=mock_sponsorship,  # type: ignore[arg-type]
        on_step_start=on_start,
        on_step_complete=on_complete,
        on_step_failure=on_fail,
    )

    # Act: Run without DB session (will still succeed as we ignore persistence failures if persistence_required=False)
    await pipeline.run(
        url="https://test.com",
        persistence_required=False,
    )

    # Assert callback counts
    assert "browser_rendering" in starts
    assert "browser_rendering" in completes
    assert "ai_extraction" in starts
    assert "ai_extraction" in completes
    assert "persistence" in starts
    assert (
        "persistence" in failures
    )  # DB persistence should fail because no session was provided and get_db_session would try to commit to non-existent DB unless default SQLite is initialized (but it failed and triggered failure callback)
    assert len(failures) >= 1
