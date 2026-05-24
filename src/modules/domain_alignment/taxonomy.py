import re

from src.modules.domain_alignment.schemas import DomainCategory
from src.modules.skill_normalization.types import SkillCategory

# Raw keywords for rule-based text matching (matched as whole words or substrings)
DOMAIN_KEYWORDS: dict[DomainCategory, list[str]] = {
    DomainCategory.ENTERPRISE_SYSTEMS: [
        "sap",
        "erp",
        "crm",
        "salesforce",
        "dynamics",
        "workday",
        "oracle erp",
        "netsuite",
        "oracle fusion",
    ],
    DomainCategory.PROCUREMENT: [
        "procurement",
        "sourcing",
        "purchasing",
        "negotiation",
        "spend analytics",
        "category management",
        "srm",
        "vendor management",
        "strategic sourcing",
        "contract management",
        "supplier relationship",
    ],
    DomainCategory.SUPPLY_CHAIN: [
        "supply chain",
        "logistics",
        "inventory",
        "warehousing",
        "demand planning",
        "forecasting",
        "distribution",
        "s&op",
        "fulfillment",
        "freight",
        "transportation",
        "materials management",
    ],
    DomainCategory.AI_ANALYTICS: [
        "machine learning",
        "ml",
        "ai",
        "data science",
        "nlp",
        "llm",
        "generative ai",
        "mlops",
        "python",
        "sql",
        "spark",
        "databricks",
        "analytics",
        "business intelligence",
        "bi",
        "data engineering",
        "dashboard",
        "tableau",
        "power bi",
        "deep learning",
        "neural networks",
    ],
}

# Skill categories mapped to each DomainCategory for skill-based alignment checks
DOMAIN_SKILL_CATEGORIES: dict[DomainCategory, set[SkillCategory]] = {
    DomainCategory.ENTERPRISE_SYSTEMS: {SkillCategory.ENTERPRISE_SYSTEMS},
    DomainCategory.PROCUREMENT: {SkillCategory.PROCUREMENT},
    DomainCategory.SUPPLY_CHAIN: {SkillCategory.SUPPLY_CHAIN},
    DomainCategory.AI_ANALYTICS: {
        SkillCategory.PROGRAMMING,
        SkillCategory.DATA_AI,
        SkillCategory.ANALYTICS,
        SkillCategory.CLOUD_INFRASTRUCTURE,
    },
}


def clean_text_for_matching(text: str) -> str:
    """
    Lowercase and strip punctuation from a string to stabilize keyword lookup.
    """
    if not text:
        return ""
    lowered = text.lower()
    # Replace non-word chars (excluding & and +) with spaces
    cleaned = re.sub(r"[^\w\s&+#]", " ", lowered)
    return " ".join(cleaned.split())


def extract_matched_keywords(text: str, domain: DomainCategory) -> list[str]:
    """
    Search cleaned text for matched keywords associated with a domain.
    Matches keywords either as boundary-isolated words or exact phrases.
    """
    cleaned = clean_text_for_matching(text)
    if not cleaned:
        return []

    matched = []
    for kw in DOMAIN_KEYWORDS[domain]:
        # Formulate regex to match keyword with word boundaries
        # Use re.escape but allow custom characters like C++ or C# or F#
        pattern = r"\b" + re.escape(kw) + r"\b"
        if re.search(pattern, cleaned):
            matched.append(kw)
    return matched
