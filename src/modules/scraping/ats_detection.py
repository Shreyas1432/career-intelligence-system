import logging
from collections.abc import Sequence
from typing import Protocol
from urllib.parse import urlparse

from src.modules.scraping.schemas import ATSPlatform, DetectionContext

logger = logging.getLogger("src.modules.scraping.ats_detection")


class ATSDetectionError(ValueError):
    """
    Raised when ATS detection input or strategy execution is invalid.
    """


class ATSDetectionStrategy(Protocol):
    """
    Contract for ATS-specific detection rules.
    """

    name: str

    def detect(self, context: DetectionContext) -> ATSPlatform | None:
        """
        Return an ATS platform when the strategy has enough evidence.
        """


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def _host_is(host: str, domains: tuple[str, ...]) -> bool:
    return any(host == domain or host.endswith(f".{domain}") for domain in domains)


class GreenhouseStrategy:
    name = "greenhouse"

    _HOSTS = ("boards.greenhouse.io", "job-boards.greenhouse.io", "grnh.se")
    _HTML_MARKERS = (
        "boards.greenhouse.io",
        "job-boards.greenhouse.io",
        "greenhouse.io/embed",
        "greenhouse-embed",
        "greenhouse job board",
        "grnh.se",
        "gh_jid",
        "app.greenhouse.io",
    )

    def detect(self, context: DetectionContext) -> ATSPlatform | None:
        if _host_is(context.host, self._HOSTS):
            return ATSPlatform.GREENHOUSE

        if _contains_any(context.html_text, self._HTML_MARKERS):
            return ATSPlatform.GREENHOUSE

        return None


class LeverStrategy:
    name = "lever"

    _HOSTS = ("jobs.lever.co",)
    _HTML_MARKERS = (
        "jobs.lever.co",
        "lever.co/apply",
        "lever-posting",
        "lever-job",
        "lever-application",
        "hosted by lever",
    )

    def detect(self, context: DetectionContext) -> ATSPlatform | None:
        if _host_is(context.host, self._HOSTS):
            return ATSPlatform.LEVER

        if _contains_any(context.html_text, self._HTML_MARKERS):
            return ATSPlatform.LEVER

        return None


class AshbyStrategy:
    name = "ashby"

    _HOSTS = ("jobs.ashbyhq.com",)
    _HTML_MARKERS = (
        "jobs.ashbyhq.com",
        "ashbyhq.com/embed",
        "ashby-embed",
        "ashby_job",
        "_ashby",
        "ashby application",
    )

    def detect(self, context: DetectionContext) -> ATSPlatform | None:
        if _host_is(context.host, self._HOSTS):
            return ATSPlatform.ASHBY

        if _contains_any(context.html_text, self._HTML_MARKERS):
            return ATSPlatform.ASHBY

        return None


class WorkdayStrategy:
    name = "workday"

    _HOSTS = ("myworkdayjobs.com",)
    _HTML_MARKERS = (
        "myworkdayjobs.com",
        "workdayjobs",
        "workday recruiting",
        "workday candidate",
        "workday-candidate",
        "wd-careers",
        "__workday",
    )

    def detect(self, context: DetectionContext) -> ATSPlatform | None:
        if _host_is(context.host, self._HOSTS):
            return ATSPlatform.WORKDAY

        if _contains_any(context.html_text, self._HTML_MARKERS):
            return ATSPlatform.WORKDAY

        return None


class GenericCustomStrategy:
    name = "generic_custom"

    _PATH_MARKERS = (
        "/careers",
        "/career",
        "/jobs",
        "/job/",
        "/join-us",
        "/open-positions",
        "/positions",
        "/vacancies",
        "/apply",
    )
    _SCHEMA_MARKERS = (
        '"@type":"jobposting"',
        '"@type": "jobposting"',
        "'@type':'jobposting'",
        "'@type': 'jobposting'",
    )
    _APPLY_MARKERS = (
        "apply now",
        "submit application",
        "start application",
        "application form",
    )
    _JOB_DETAIL_MARKERS = (
        "job description",
        "role description",
        "responsibilities",
        "requirements",
        "qualifications",
        "employment type",
        "requisition",
    )
    _FORM_MARKERS = (
        "upload resume",
        "upload cv",
        "cover letter",
        "candidate profile",
    )

    def detect(self, context: DetectionContext) -> ATSPlatform | None:
        score = 0

        if _contains_any(context.path, self._PATH_MARKERS):
            score += 2

        if _contains_any(context.html_text, self._SCHEMA_MARKERS):
            score += 3

        if _contains_any(context.html_text, self._APPLY_MARKERS):
            score += 1

        if _contains_any(context.html_text, self._JOB_DETAIL_MARKERS):
            score += 1

        if _contains_any(context.html_text, self._FORM_MARKERS):
            score += 1

        if score >= 2:
            return ATSPlatform.GENERIC_CUSTOM

        return None


DEFAULT_STRATEGIES: tuple[ATSDetectionStrategy, ...] = (
    GreenhouseStrategy(),
    LeverStrategy(),
    AshbyStrategy(),
    WorkdayStrategy(),
    GenericCustomStrategy(),
)


class ATSDetector:
    """
    Lightweight rule engine for identifying job board ATS platforms.
    """

    def __init__(self, strategies: Sequence[ATSDetectionStrategy] | None = None):
        self._strategies = tuple(strategies or DEFAULT_STRATEGIES)

    @property
    def strategies(self) -> tuple[ATSDetectionStrategy, ...]:
        return self._strategies

    async def detect(self, job_url: str, html: str | None = None) -> ATSPlatform:
        """
        Async-compatible detection entry point.
        """
        return self.detect_sync(job_url, html)

    def detect_sync(self, job_url: str, html: str | None = None) -> ATSPlatform:
        """
        Synchronous detection entry point for UI and tests.
        """
        try:
            context = self._build_context(job_url, html)
        except ATSDetectionError:
            logger.debug("ATS detection skipped for invalid input", exc_info=True)
            return ATSPlatform.UNKNOWN

        for strategy in self._strategies:
            try:
                platform = strategy.detect(context)
            except Exception:
                logger.exception("ATS detection strategy failed", extra={"strategy": strategy.name})
                continue

            if platform is not None:
                return platform

        return ATSPlatform.UNKNOWN

    def _build_context(self, job_url: str, html: str | None) -> DetectionContext:
        if not isinstance(job_url, str) or not job_url.strip():
            raise ATSDetectionError("job_url must be a non-empty string")

        if html is not None and not isinstance(html, str):
            raise ATSDetectionError("html must be a string when provided")

        raw_url = job_url.strip()
        parsed_url = urlparse(raw_url)

        if not parsed_url.scheme and not parsed_url.netloc:
            first_path_segment = parsed_url.path.split("/", maxsplit=1)[0]
            if "." in first_path_segment:
                parsed_url = urlparse(f"https://{raw_url}")

        if parsed_url.scheme.casefold() not in {"http", "https"}:
            raise ATSDetectionError("job_url must use http or https")

        host = parsed_url.hostname.casefold() if parsed_url.hostname else ""
        if not host:
            raise ATSDetectionError("job_url must include a host")

        return DetectionContext(
            job_url=raw_url,
            parsed_url=parsed_url,
            host=host,
            path=parsed_url.path.casefold(),
            query=parsed_url.query.casefold(),
            html=html,
            html_text=html.casefold() if html else "",
        )


_DEFAULT_DETECTOR = ATSDetector()


async def detect_ats(job_url: str, html: str | None = None) -> ATSPlatform:
    """
    Detect the ATS platform for a job URL and optional HTML content.
    """
    return await _DEFAULT_DETECTOR.detect(job_url, html)


def detect_ats_sync(job_url: str, html: str | None = None) -> ATSPlatform:
    """
    Detect the ATS platform without requiring an event loop.
    """
    return _DEFAULT_DETECTOR.detect_sync(job_url, html)
