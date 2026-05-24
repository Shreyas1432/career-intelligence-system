from src.modules.job_extraction.schemas import JobDomain
from src.modules.skill_normalization.types import SkillCategory

# Base weights for each SkillCategory bucket
BASE_SKILL_WEIGHTS: dict[SkillCategory, float] = {
    SkillCategory.PROGRAMMING: 1.2,
    SkillCategory.DATA_AI: 1.2,
    SkillCategory.CLOUD_INFRASTRUCTURE: 1.1,
    SkillCategory.ENTERPRISE_SYSTEMS: 1.1,
    SkillCategory.SUPPLY_CHAIN: 1.2,
    SkillCategory.PROCUREMENT: 1.2,
    SkillCategory.SECURITY: 1.1,
    SkillCategory.ANALYTICS: 1.0,
    SkillCategory.PRODUCT: 1.0,
    SkillCategory.BUSINESS: 1.0,
    SkillCategory.OTHER: 0.8,
}

# Domain-specific category multipliers
DOMAIN_MULTIPLIERS: dict[JobDomain, dict[SkillCategory, float]] = {
    JobDomain.SOFTWARE_ENGINEERING: {
        SkillCategory.PROGRAMMING: 1.5,
        SkillCategory.CLOUD_INFRASTRUCTURE: 1.3,
    },
    JobDomain.DATA_AI: {
        SkillCategory.DATA_AI: 1.5,
        SkillCategory.ANALYTICS: 1.3,
    },
    JobDomain.SECURITY: {
        SkillCategory.SECURITY: 1.5,
        SkillCategory.CLOUD_INFRASTRUCTURE: 1.2,
    },
    JobDomain.PRODUCT: {
        SkillCategory.PRODUCT: 1.5,
        SkillCategory.BUSINESS: 1.2,
    },
    JobDomain.OPERATIONS: {
        SkillCategory.SUPPLY_CHAIN: 1.8,
        SkillCategory.PROCUREMENT: 1.8,
        SkillCategory.BUSINESS: 1.2,
        SkillCategory.ANALYTICS: 1.1,
    },
    JobDomain.FINANCE: {
        SkillCategory.BUSINESS: 1.3,
        SkillCategory.ANALYTICS: 1.2,
    },
    JobDomain.SALES: {
        SkillCategory.BUSINESS: 1.4,
    },
    JobDomain.MARKETING: {
        SkillCategory.BUSINESS: 1.3,
        SkillCategory.ANALYTICS: 1.1,
    },
    JobDomain.INFRASTRUCTURE: {
        SkillCategory.CLOUD_INFRASTRUCTURE: 1.5,
        SkillCategory.SECURITY: 1.2,
    },
}


def calculate_skill_weight(category: SkillCategory, domain: JobDomain) -> float:
    """
    Calculate the deterministic weight of a skill based on its category and the job domain.
    """
    base_weight = BASE_SKILL_WEIGHTS.get(category, 1.0)
    multipliers = DOMAIN_MULTIPLIERS.get(domain, {})
    multiplier = multipliers.get(category, 1.0)
    return round(base_weight * multiplier, 2)


def calculate_procurement_bonus(
    job_domain: JobDomain,
    matched_categories: list[SkillCategory],
) -> float:
    """
    Calculate a bonus (up to 10 points) for matching procurement/supply chain skills
    against an operations/procurement role.

    Formula: +2.5 points per matched supply chain or procurement skill, capped at 10.0.
    Only active if job domain is OPERATIONS.
    """
    if job_domain != JobDomain.OPERATIONS:
        return 0.0

    relevant_count = sum(
        1
        for cat in matched_categories
        if cat in (SkillCategory.SUPPLY_CHAIN, SkillCategory.PROCUREMENT)
    )
    return min(10.0, relevant_count * 2.5)


def calculate_domain_alignment_bonus(
    job_domain: JobDomain,
    user_domains: list[str],
    job_title: str | None,
    user_target_industries: list[str],
) -> float:
    """
    Calculate an Enterprise Domain Alignment bonus (up to 10 points).

    - +5 points if the Job's domain matches any of the candidate's target domains (case-insensitive).
    - +5 points if there is a target industry match or domain keyword overlap.
    """
    bonus = 0.0

    # 1. Job domain matching user target domains
    normalized_user_domains = [d.strip().lower().replace("_", " ") for d in user_domains]
    job_domain_str = job_domain.value.lower().replace("_", " ")

    if job_domain_str in normalized_user_domains:
        bonus += 5.0

    # 2. Industry / Title keyword matching
    # Check if the job domain is in target industries, or job title contains target industry words
    if job_title:
        title_lower = job_title.lower()
        for industry in user_target_industries:
            ind_clean = industry.strip().lower()
            if ind_clean and (ind_clean in title_lower or ind_clean in job_domain_str):
                bonus += 5.0
                break

    return min(10.0, bonus)
