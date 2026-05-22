from .detector import ATSDetector
from .types import ATSPlatform

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
