import logging
from collections.abc import Sequence
from urllib.parse import urlparse

from .exceptions import ATSDetectionError
from .strategies import (
    AshbyStrategy,
    ATSDetectionStrategy,
    GenericCustomStrategy,
    GreenhouseStrategy,
    LeverStrategy,
    WorkdayStrategy,
)
from .types import ATSPlatform, DetectionContext

logger = logging.getLogger("src.modules.ats_detector.detector")

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
