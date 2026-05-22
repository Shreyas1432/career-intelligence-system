from dataclasses import dataclass, field
from enum import StrEnum


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
