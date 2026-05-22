import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import structlog
from sqlalchemy.orm import Session

from src.core.browser.service import BrowserScrapingService
from src.core.database.connection import get_db_session
from src.core.logging import get_correlation_id, set_correlation_id
from src.modules.ats_detector.service import detect_ats
from src.modules.ats_detector.types import ATSPlatform
from src.modules.job_extraction.schemas import JobExtractionResult
from src.modules.job_extraction.service import JobExtractionService
from src.modules.job_persistence.service import JobPersistenceService
from src.modules.sponsorship.engine import SponsorshipDetector
from src.modules.sponsorship.types import DetectionResult

from .cleaner import clean_html, clean_text
from .exceptions import PipelineFatalError

logger = structlog.get_logger("src.modules.scraping_pipeline.pipeline")


@dataclass
class PipelineStepResult:
    """
    Tracks the execution status of an individual pipeline step.
    """

    step_name: str
    status: str  # "success", "failed", "skipped"
    result: Any = None
    error: str | None = None
    duration_seconds: float = 0.0


@dataclass
class PipelineRunResult:
    """
    Aggregates the execution outcome of the entire scraping pipeline run.
    """

    run_id: str
    url: str
    status: str  # "success", "partial_success", "failed"
    steps: dict[str, PipelineStepResult] = field(default_factory=dict)
    extracted_data: JobExtractionResult | None = None
    sponsorship_result: DetectionResult | None = None
    persisted_record: Any = None
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0


class ScrapingPipeline:
    """
    Asynchronous scraping orchestration pipeline that sequences:
    ATS detection -> Browser rendering -> Cleaning -> AI extraction -> Sponsorship scan -> DB persistence.
    """

    def __init__(
        self,
        browser_service: BrowserScrapingService | None = None,
        extraction_service: JobExtractionService | None = None,
        sponsorship_detector: SponsorshipDetector | None = None,
        on_step_start: Callable[[str, str], None] | None = None,
        on_step_complete: Callable[[str, Any, str], None] | None = None,
        on_step_failure: Callable[[str, Exception, str], None] | None = None,
    ):
        self.browser_service = browser_service
        self.extraction_service = extraction_service or JobExtractionService()
        self.sponsorship_detector = sponsorship_detector or SponsorshipDetector()

        # Callbacks
        self._on_step_start = on_step_start
        self._on_step_complete = on_step_complete
        self._on_step_failure = on_step_failure

    async def run(
        self,
        url: str,
        *,
        html: str | None = None,
        session: Session | None = None,
        use_sponsorship_ai: bool = False,
        persistence_required: bool = True,
        run_id: str | None = None,
    ) -> PipelineRunResult:
        """
        Executes the scraping orchestration pipeline for a given job URL.
        """
        start_time = time.perf_counter()
        actual_run_id = run_id or str(uuid.uuid4())

        # Manage logging correlation context
        old_correlation_id = get_correlation_id()
        set_correlation_id(actual_run_id)

        logger.info("Starting scraping pipeline execution", url=url, run_id=actual_run_id)

        run_result = PipelineRunResult(run_id=actual_run_id, url=url, status="failed")

        try:
            # 1. Preliminary ATS Detection (URL-based, cheap check)
            ats_platform = await self._run_step(
                "ats_detection_preliminary",
                lambda: detect_ats(url),
                actual_run_id,
                run_result,
                is_fatal=False,
            )
            if ats_platform is None:
                ats_platform = ATSPlatform.UNKNOWN

            # 2. Browser Rendering (unless html is already supplied)
            raw_html = html
            raw_text = ""
            if not raw_html:
                snapshot = await self._run_step(
                    "browser_rendering",
                    lambda: self._render_page(url),
                    actual_run_id,
                    run_result,
                    is_fatal=True,
                )
                if snapshot:
                    raw_html = snapshot.html
                    raw_text = snapshot.text
            else:
                self._record_skipped_step("browser_rendering", run_result)

            # 3. Content Cleaning
            cleaned_text = ""
            if raw_html or raw_text:
                cleaned_text = await self._run_step(
                    "content_cleaning",
                    lambda: self._clean_content(raw_html, raw_text),
                    actual_run_id,
                    run_result,
                    is_fatal=True,
                )

            # 4. Post-render ATS Detection (HTML-based, precise check)
            if raw_html:
                refined_ats = await self._run_step(
                    "ats_detection_post_render",
                    lambda: detect_ats(url, html=raw_html),
                    actual_run_id,
                    run_result,
                    is_fatal=False,
                )
                if refined_ats and refined_ats != ATSPlatform.UNKNOWN:
                    ats_platform = refined_ats
            else:
                self._record_skipped_step("ats_detection_post_render", run_result)

            # 5. AI Extraction
            extracted_data = await self._run_step(
                "ai_extraction",
                lambda: self.extraction_service.extract(cleaned_text, source_url=url),
                actual_run_id,
                run_result,
                is_fatal=True,
            )
            run_result.extracted_data = extracted_data

            # 6. Sponsorship Signal Detection
            sponsorship_result = await self._run_step(
                "sponsorship_detection",
                lambda: self.sponsorship_detector.detect(cleaned_text, use_ai=use_sponsorship_ai),
                actual_run_id,
                run_result,
                is_fatal=False,
            )
            run_result.sponsorship_result = sponsorship_result

            # 7. Persistence
            persisted_record = await self._run_step(
                "persistence",
                lambda: self._persist_job(
                    session=session,
                    raw_content=raw_html or cleaned_text,
                    url=url,
                    extracted_data=extracted_data,
                    sponsorship_result=sponsorship_result,
                    ats_platform=ats_platform,
                ),
                actual_run_id,
                run_result,
                is_fatal=persistence_required,
            )
            run_result.persisted_record = persisted_record

            # Classify overall pipeline success status
            has_failures = any(s.status == "failed" for s in run_result.steps.values())
            run_result.status = "partial_success" if has_failures else "success"

        except PipelineFatalError as exc:
            logger.error("Scraping pipeline halted due to fatal step failure", error=str(exc))
            run_result.status = "failed"
            run_result.errors.append(str(exc))
        except Exception as exc:
            logger.exception("Scraping pipeline failed unexpectedly", error=str(exc))
            run_result.status = "failed"
            run_result.errors.append(f"Unexpected error: {exc}")
        finally:
            run_result.duration_seconds = round(time.perf_counter() - start_time, 4)
            set_correlation_id(old_correlation_id)
            logger.info(
                "Scraping pipeline execution finished",
                status=run_result.status,
                duration=run_result.duration_seconds,
            )

        return run_result

    async def _run_step(
        self,
        step_name: str,
        step_callable: Callable[[], Any],
        run_id: str,
        run_result: PipelineRunResult,
        is_fatal: bool,
    ) -> Any:
        """
        Helper method to run a step inside an error boundary with event notifications.
        """
        self._trigger_callback(self._on_step_start, step_name, run_id)
        start = time.perf_counter()

        try:
            # Execute step (handle both async and sync callbacks)
            res = step_callable()
            if hasattr(res, "__await__"):
                result = await res
            else:
                result = res

            duration = round(time.perf_counter() - start, 4)
            run_result.steps[step_name] = PipelineStepResult(
                step_name=step_name,
                status="success",
                result=result,
                duration_seconds=duration,
            )
            self._trigger_callback(self._on_step_complete, step_name, result, run_id)
            return result

        except Exception as exc:
            duration = round(time.perf_counter() - start, 4)
            logger.error(
                "Pipeline step execution failed",
                step=step_name,
                error=str(exc),
                is_fatal=is_fatal,
            )
            run_result.steps[step_name] = PipelineStepResult(
                step_name=step_name,
                status="failed",
                error=str(exc),
                duration_seconds=duration,
            )
            run_result.errors.append(f"Step '{step_name}' failed: {exc}")
            self._trigger_callback(self._on_step_failure, step_name, exc, run_id)

            if is_fatal:
                raise PipelineFatalError(f"Fatal failure in step '{step_name}': {exc}") from exc
            return None

    def _record_skipped_step(self, step_name: str, run_result: PipelineRunResult) -> None:
        run_result.steps[step_name] = PipelineStepResult(step_name=step_name, status="skipped")

    async def _render_page(self, url: str) -> Any:
        if self.browser_service:
            return await self.browser_service.capture_page(url)
        else:
            async with BrowserScrapingService() as service:
                return await service.capture_page(url)

    def _clean_content(self, html_content: str | None, text_content: str | None) -> str:
        if html_content:
            return clean_html(html_content)
        if text_content:
            return clean_text(text_content)
        raise ValueError("No content was retrieved to clean")

    async def _persist_job(
        self,
        session: Session | None,
        raw_content: str,
        url: str,
        extracted_data: JobExtractionResult | None,
        sponsorship_result: DetectionResult | None,
        ats_platform: ATSPlatform,
    ) -> Any:
        if not extracted_data:
            raise ValueError("Cannot persist job intelligence without extracted data")

        ats_str = ats_platform.value if hasattr(ats_platform, "value") else str(ats_platform)

        # Run inside standard DB session context manager if none provided
        if session:
            persist_service = JobPersistenceService(session)
            return await persist_service.persist_job(
                raw_content=raw_content,
                url=url,
                extracted_data=extracted_data,
                sponsorship_result=sponsorship_result,
                ats_type=ats_str,
            )
        else:
            with get_db_session() as new_session:
                persist_service = JobPersistenceService(new_session)
                return await persist_service.persist_job(
                    raw_content=raw_content,
                    url=url,
                    extracted_data=extracted_data,
                    sponsorship_result=sponsorship_result,
                    ats_type=ats_str,
                )

    def _trigger_callback(self, callback: Callable[..., None] | None, *args: Any) -> None:
        """
        Safely invokes step event callbacks.
        """
        if not callback:
            return
        try:
            callback(*args)
        except Exception as exc:
            logger.error("Error raised during step callback invocation", error=str(exc))
