from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Self
from urllib.parse import ParseResult

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ------------------------------------------------------------------------------
# ATS Detection Schemas & Enums
# ------------------------------------------------------------------------------

class ATSPlatform(StrEnum):
    """
    Normalized applicant tracking system identifiers.
    """

    GREENHOUSE = "greenhouse"
    LEVER = "lever"
    ASHBY = "ashby"
    WORKDAY = "workday"
    GENERIC_CUSTOM = "generic_custom"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class DetectionContext:
    """
    Pre-normalized inputs shared by detection strategies.
    """

    job_url: str
    parsed_url: ParseResult
    host: str
    path: str
    query: str
    html: str | None
    html_text: str


# ------------------------------------------------------------------------------
# Skill Normalization Schemas & Enums
# ------------------------------------------------------------------------------

class SkillCategory(StrEnum):
    """
    Broad taxonomy buckets for normalized job matching skills.
    """

    PROGRAMMING = "programming"
    DATA_AI = "data_ai"
    CLOUD_INFRASTRUCTURE = "cloud_infrastructure"
    ENTERPRISE_SYSTEMS = "enterprise_systems"
    SUPPLY_CHAIN = "supply_chain"
    PROCUREMENT = "procurement"
    ANALYTICS = "analytics"
    BUSINESS = "business"
    SECURITY = "security"
    PRODUCT = "product"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class CanonicalSkill:
    """
    Canonical skill entry plus taxonomy metadata.
    """

    name: str
    category: SkillCategory
    aliases: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class NormalizedSkill:
    """
    Result for one input skill after deterministic normalization.
    """

    original: str
    canonical: str
    category: SkillCategory
    matched_alias: str | None = None


# ------------------------------------------------------------------------------
# Job Extraction Schemas & Enums
# ------------------------------------------------------------------------------

class VisaSignal(StrEnum):
    """
    Normalized visa or work authorization signal.
    """

    SPONSORSHIP_AVAILABLE = "sponsorship_available"
    NO_SPONSORSHIP = "no_sponsorship"
    WORK_AUTH_REQUIRED = "work_auth_required"
    UNKNOWN = "unknown"


class EmploymentType(StrEnum):
    """
    Normalized employment arrangement.
    """

    FULL_TIME = "full_time"
    PART_TIME = "part_time"
    CONTRACT = "contract"
    INTERNSHIP = "internship"
    TEMPORARY = "temporary"
    FREELANCE = "freelance"
    UNKNOWN = "unknown"


class JobDomain(StrEnum):
    """
    Coarse job domain taxonomy for career intelligence grouping.
    """

    SOFTWARE_ENGINEERING = "software_engineering"
    DATA_AI = "data_ai"
    PRODUCT = "product"
    DESIGN = "design"
    SALES = "sales"
    MARKETING = "marketing"
    FINANCE = "finance"
    OPERATIONS = "operations"
    SECURITY = "security"
    INFRASTRUCTURE = "infrastructure"
    HEALTHCARE = "healthcare"
    EDUCATION = "education"
    LEGAL = "legal"
    OTHER = "other"
    UNKNOWN = "unknown"


def _normalize_text(value: str) -> str:
    return " ".join(value.strip().split())


def _normalize_key(value: str) -> str:
    normalized = _normalize_text(value).casefold()
    return normalized.replace("&", " and ").replace("/", " ").replace("-", " ").replace("_", " ")


def _normalize_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("Expected a string or null")

    cleaned = _normalize_text(value)
    return cleaned or None


def _normalize_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("Expected a list of strings")

    cleaned: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            raise ValueError("Expected every list item to be a string")

        normalized = _normalize_text(item)
        if not normalized:
            continue

        key = normalized.casefold()
        if key in seen:
            continue

        seen.add(key)
        cleaned.append(normalized)

    # Lazy import to avoid circular dependencies
    from src.modules.scraping.normalization import canonicalize_skills
    return canonicalize_skills(cleaned)


_VISA_ALIASES: dict[str, VisaSignal] = {
    "sponsorship available": VisaSignal.SPONSORSHIP_AVAILABLE,
    "visa sponsorship available": VisaSignal.SPONSORSHIP_AVAILABLE,
    "sponsors visas": VisaSignal.SPONSORSHIP_AVAILABLE,
    "sponsorship yes": VisaSignal.SPONSORSHIP_AVAILABLE,
    "no sponsorship": VisaSignal.NO_SPONSORSHIP,
    "visa sponsorship not available": VisaSignal.NO_SPONSORSHIP,
    "does not sponsor": VisaSignal.NO_SPONSORSHIP,
    "must not require sponsorship": VisaSignal.NO_SPONSORSHIP,
    "work authorization required": VisaSignal.WORK_AUTH_REQUIRED,
    "right to work required": VisaSignal.WORK_AUTH_REQUIRED,
    "must be authorized to work": VisaSignal.WORK_AUTH_REQUIRED,
    "unknown": VisaSignal.UNKNOWN,
}

_EMPLOYMENT_ALIASES: dict[str, EmploymentType] = {
    "full time": EmploymentType.FULL_TIME,
    "fulltime": EmploymentType.FULL_TIME,
    "permanent": EmploymentType.FULL_TIME,
    "part time": EmploymentType.PART_TIME,
    "parttime": EmploymentType.PART_TIME,
    "contract": EmploymentType.CONTRACT,
    "contractor": EmploymentType.CONTRACT,
    "intern": EmploymentType.INTERNSHIP,
    "internship": EmploymentType.INTERNSHIP,
    "temporary": EmploymentType.TEMPORARY,
    "temp": EmploymentType.TEMPORARY,
    "freelance": EmploymentType.FREELANCE,
    "unknown": EmploymentType.UNKNOWN,
}

_DOMAIN_ALIASES: dict[str, JobDomain] = {
    "software": JobDomain.SOFTWARE_ENGINEERING,
    "software engineering": JobDomain.SOFTWARE_ENGINEERING,
    "backend": JobDomain.SOFTWARE_ENGINEERING,
    "frontend": JobDomain.SOFTWARE_ENGINEERING,
    "full stack": JobDomain.SOFTWARE_ENGINEERING,
    "data": JobDomain.DATA_AI,
    "data science": JobDomain.DATA_AI,
    "machine learning": JobDomain.DATA_AI,
    "ml": JobDomain.DATA_AI,
    "ai": JobDomain.DATA_AI,
    "artificial intelligence": JobDomain.DATA_AI,
    "product": JobDomain.PRODUCT,
    "design": JobDomain.DESIGN,
    "ux": JobDomain.DESIGN,
    "sales": JobDomain.SALES,
    "marketing": JobDomain.MARKETING,
    "finance": JobDomain.FINANCE,
    "operations": JobDomain.OPERATIONS,
    "security": JobDomain.SECURITY,
    "cybersecurity": JobDomain.SECURITY,
    "infrastructure": JobDomain.INFRASTRUCTURE,
    "devops": JobDomain.INFRASTRUCTURE,
    "platform": JobDomain.INFRASTRUCTURE,
    "healthcare": JobDomain.HEALTHCARE,
    "education": JobDomain.EDUCATION,
    "legal": JobDomain.LEGAL,
    "other": JobDomain.OTHER,
    "unknown": JobDomain.UNKNOWN,
}


class JobIntelligenceSchema(BaseModel):
    """
    Strict, normalized schema for AI-extracted job intelligence.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        use_enum_values=False,
        validate_assignment=True,
    )

    company: str | None = Field(default=None, max_length=200)
    title: str | None = Field(default=None, max_length=200)
    skills: list[str] = Field(default_factory=list, max_length=40)
    experience_required: str | None = Field(default=None, max_length=500)
    location: str | None = Field(default=None, max_length=200)
    visa_signal: VisaSignal = VisaSignal.UNKNOWN
    employment_type: EmploymentType = EmploymentType.UNKNOWN
    domain: JobDomain = JobDomain.UNKNOWN
    confidence_score: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="before")
    @classmethod
    def reject_legacy_or_unknown_fields(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value

        legacy_fields = {"sponsorship_clues", "domain_signals"}
        present_legacy = sorted(legacy_fields.intersection(value))
        if present_legacy:
            raise ValueError(
                "Legacy extraction fields are not accepted by strict job intelligence schema: "
                f"{', '.join(present_legacy)}"
            )

        return value

    @field_validator("company", "title", "experience_required", "location", mode="before")
    @classmethod
    def normalize_optional_text(cls, value: Any) -> str | None:
        return _normalize_optional_text(value)

    @field_validator("skills", mode="before")
    @classmethod
    def normalize_skills(cls, value: Any) -> list[str]:
        return _normalize_text_list(value)

    @field_validator("visa_signal", mode="before")
    @classmethod
    def normalize_visa_signal(cls, value: Any) -> VisaSignal:
        if isinstance(value, VisaSignal):
            return value
        if value is None:
            return VisaSignal.UNKNOWN
        if not isinstance(value, str):
            raise ValueError("visa_signal must be a string enum value")

        key = _normalize_key(value)
        if key in _VISA_ALIASES:
            return _VISA_ALIASES[key]

        compact = key.replace(" ", "_")
        try:
            return VisaSignal(compact)
        except ValueError as exc:
            raise ValueError(f"Unsupported visa_signal: {value}") from exc

    @field_validator("employment_type", mode="before")
    @classmethod
    def normalize_employment_type(cls, value: Any) -> EmploymentType:
        if isinstance(value, EmploymentType):
            return value
        if value is None:
            return EmploymentType.UNKNOWN
        if not isinstance(value, str):
            raise ValueError("employment_type must be a string enum value")

        key = _normalize_key(value)
        if key in _EMPLOYMENT_ALIASES:
            return _EMPLOYMENT_ALIASES[key]

        compact = key.replace(" ", "_")
        try:
            return EmploymentType(compact)
        except ValueError as exc:
            raise ValueError(f"Unsupported employment_type: {value}") from exc

    @field_validator("domain", mode="before")
    @classmethod
    def normalize_domain(cls, value: Any) -> JobDomain:
        if isinstance(value, JobDomain):
            return value
        if value is None:
            return JobDomain.UNKNOWN
        if not isinstance(value, str):
            raise ValueError("domain must be a string enum value")

        key = _normalize_key(value)
        if key in _DOMAIN_ALIASES:
            return _DOMAIN_ALIASES[key]

        compact = key.replace(" ", "_")
        try:
            return JobDomain(compact)
        except ValueError as exc:
            raise ValueError(f"Unsupported domain: {value}") from exc

    @field_validator("confidence_score", mode="before")
    @classmethod
    def reject_non_numeric_confidence(cls, value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError("confidence_score must be a numeric value between 0.0 and 1.0")
        return float(value)

    @model_validator(mode="after")
    def normalize_and_validate_result(self) -> Self:
        if not self.has_job_signal():
            raise ValueError("Extraction did not contain any usable job signals")

        if self.confidence_score is None:
            self.confidence_score = self.estimate_confidence_score()

        if self.confidence_score <= 0:
            raise ValueError("confidence_score must be greater than 0 for accepted output")

        return self

    def has_job_signal(self) -> bool:
        scalar_signal = any(
            [
                self.company,
                self.title,
                self.experience_required,
                self.location,
            ]
        )
        enum_signal = any(
            [
                self.visa_signal != VisaSignal.UNKNOWN,
                self.employment_type != EmploymentType.UNKNOWN,
                self.domain != JobDomain.UNKNOWN,
            ]
        )
        return bool(scalar_signal or self.skills or enum_signal)

    def estimate_confidence_score(self) -> float:
        """
        Estimate confidence from populated evidence when the model omits a score.
        """
        score = 0.0
        score += 0.18 if self.company else 0.0
        score += 0.22 if self.title else 0.0
        score += min(len(self.skills), 8) * 0.035
        score += 0.1 if self.experience_required else 0.0
        score += 0.08 if self.location else 0.0
        score += 0.06 if self.visa_signal != VisaSignal.UNKNOWN else 0.0
        score += 0.08 if self.employment_type != EmploymentType.UNKNOWN else 0.0
        score += 0.08 if self.domain != JobDomain.UNKNOWN else 0.0
        return round(min(score, 1.0), 2)


class JobExtractionInput(BaseModel):
    """
    Input payload for a job extraction request.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    source: str = Field(min_length=1)
    source_url: str | None = None


JobExtractionResult = JobIntelligenceSchema


# ------------------------------------------------------------------------------
# Scraping Pipeline Orchestration Schemas
# ------------------------------------------------------------------------------

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
    sponsorship_result: Any = None  # DetectionResult | None (from sponsorship domain)
    persisted_record: Any = None
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
