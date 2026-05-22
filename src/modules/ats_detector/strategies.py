from typing import Protocol

from .types import ATSPlatform, DetectionContext


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
