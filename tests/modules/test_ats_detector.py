import asyncio

import pytest

from src.modules.ats_detector import ATSDetector, ATSPlatform, detect_ats, detect_ats_sync
from src.modules.ats_detector.strategies import GreenhouseStrategy
from src.modules.ats_detector.types import DetectionContext


@pytest.mark.parametrize(
    ("job_url", "expected"),
    [
        ("https://boards.greenhouse.io/acme/jobs/123", ATSPlatform.GREENHOUSE),
        ("https://job-boards.greenhouse.io/acme/jobs/123", ATSPlatform.GREENHOUSE),
        ("https://grnh.se/abc123", ATSPlatform.GREENHOUSE),
        ("https://jobs.lever.co/acme/role-id", ATSPlatform.LEVER),
        ("https://jobs.ashbyhq.com/acme/role-id", ATSPlatform.ASHBY),
        ("https://acme.wd5.myworkdayjobs.com/en-US/careers/job/123", ATSPlatform.WORKDAY),
    ],
)
def test_detects_known_ats_from_url(job_url: str, expected: ATSPlatform) -> None:
    assert detect_ats_sync(job_url) == expected


@pytest.mark.parametrize(
    ("html", "expected"),
    [
        ('<div id="grnhse_app">Apply through boards.greenhouse.io</div>', ATSPlatform.GREENHOUSE),
        ('<section class="lever-posting">Hosted by Lever</section>', ATSPlatform.LEVER),
        ('<script src="https://jobs.ashbyhq.com/embed"></script>', ATSPlatform.ASHBY),
        (
            '<main data-automation-id="jobPosting">Workday Candidate Home</main>',
            ATSPlatform.WORKDAY,
        ),
    ],
)
def test_detects_known_ats_from_html(html: str, expected: ATSPlatform) -> None:
    assert detect_ats_sync("https://example.com/jobs/software-engineer", html) == expected


def test_detects_generic_custom_from_job_url_without_html() -> None:
    assert detect_ats_sync("https://example.com/careers/software-engineer") == (
        ATSPlatform.GENERIC_CUSTOM
    )


def test_detects_generic_custom_from_job_html() -> None:
    html = """
    <script type="application/ld+json">{"@type":"JobPosting"}</script>
    <h1>Senior Backend Engineer</h1>
    <p>Job description and requirements</p>
    <button>Apply now</button>
    """

    assert detect_ats_sync("https://example.com/opportunities/backend-engineer", html) == (
        ATSPlatform.GENERIC_CUSTOM
    )


@pytest.mark.parametrize(
    "job_url",
    [
        "",
        "   ",
        "not-a-url",
        "ftp://boards.greenhouse.io/acme/jobs/123",
        "https:///careers/software-engineer",
    ],
)
def test_invalid_or_unsupported_urls_return_unknown(job_url: str) -> None:
    assert detect_ats_sync(job_url) == ATSPlatform.UNKNOWN


def test_signal_free_url_without_html_returns_unknown() -> None:
    assert detect_ats_sync("https://example.com/about") == ATSPlatform.UNKNOWN


def test_accepts_url_without_scheme_when_host_is_clear() -> None:
    assert detect_ats_sync("jobs.lever.co/acme/role-id") == ATSPlatform.LEVER


def test_handles_mixed_case_inputs() -> None:
    html = '<DIV CLASS="LEVER-POSTING">Hosted By Lever</DIV>'

    assert detect_ats_sync("HTTPS://EXAMPLE.COM/JOBS/SOFTWARE-ENGINEER", html) == (
        ATSPlatform.LEVER
    )


def test_query_string_only_vendor_signals_do_not_trigger_detection() -> None:
    job_url = "https://example.com/?redirect=jobs.lever.co/acme&gh_jid=123"

    assert detect_ats_sync(job_url) == ATSPlatform.UNKNOWN


def test_known_ats_takes_priority_over_generic_signals() -> None:
    html = """
    <script src="https://jobs.ashbyhq.com/embed"></script>
    <button>Apply now</button>
    <p>Job description and requirements</p>
    """

    assert detect_ats_sync("https://example.com/careers/software-engineer", html) == (
        ATSPlatform.ASHBY
    )


def test_fixed_strategy_order_resolves_conflicting_known_markers() -> None:
    html = """
    <div>boards.greenhouse.io</div>
    <section class="lever-posting">Hosted by Lever</section>
    """

    assert detect_ats_sync("https://example.com/jobs/software-engineer", html) == (
        ATSPlatform.GREENHOUSE
    )


def test_async_public_api() -> None:
    result = asyncio.run(detect_ats("https://jobs.ashbyhq.com/acme/role-id"))

    assert result == ATSPlatform.ASHBY


def test_strategy_errors_are_logged_and_skipped() -> None:
    class BrokenStrategy:
        name = "broken"

        def detect(self, _context: DetectionContext) -> ATSPlatform | None:
            raise RuntimeError("strategy exploded")

    detector = ATSDetector(strategies=[BrokenStrategy(), GreenhouseStrategy()])

    assert detector.detect_sync("https://boards.greenhouse.io/acme/jobs/123") == (
        ATSPlatform.GREENHOUSE
    )
