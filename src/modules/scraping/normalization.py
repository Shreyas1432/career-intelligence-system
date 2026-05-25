import re
from collections.abc import Iterable, Mapping, Sequence

from src.modules.scraping.schemas import (
    CanonicalSkill,
    NormalizedSkill,
    SkillCategory,
)

# ------------------------------------------------------------------------------
# Default Static Skill Taxonomy
# ------------------------------------------------------------------------------

DEFAULT_SKILL_TAXONOMY: tuple[CanonicalSkill, ...] = (
    CanonicalSkill(
        name="Python",
        category=SkillCategory.PROGRAMMING,
        aliases=("py", "python3"),
    ),
    CanonicalSkill(
        name="SQL",
        category=SkillCategory.DATA_AI,
        aliases=("structured query language", "sql querying", "sql queries"),
    ),
    CanonicalSkill(
        name="Spark",
        category=SkillCategory.DATA_AI,
        aliases=("apache spark", "pyspark", "spark sql", "spark streaming", "databricks spark"),
    ),
    CanonicalSkill(
        name="Databricks",
        category=SkillCategory.DATA_AI,
        aliases=("azure databricks", "databricks lakehouse"),
    ),
    CanonicalSkill(
        name="Machine Learning",
        category=SkillCategory.DATA_AI,
        aliases=("ml", "predictive modeling", "predictive modelling"),
    ),
    CanonicalSkill(
        name="Generative AI",
        category=SkillCategory.DATA_AI,
        aliases=("genai", "gen ai", "generative artificial intelligence"),
    ),
    CanonicalSkill(
        name="Large Language Models",
        category=SkillCategory.DATA_AI,
        aliases=("llm", "llms", "large language model"),
    ),
    CanonicalSkill(
        name="Natural Language Processing",
        category=SkillCategory.DATA_AI,
        aliases=("nlp", "text analytics", "text mining"),
    ),
    CanonicalSkill(
        name="MLOps",
        category=SkillCategory.DATA_AI,
        aliases=("ml ops", "model operations", "model deployment"),
    ),
    CanonicalSkill(
        name="Data Engineering",
        category=SkillCategory.DATA_AI,
        aliases=("data pipelines", "etl", "elt", "pipeline engineering"),
    ),
    CanonicalSkill(
        name="Data Visualization",
        category=SkillCategory.ANALYTICS,
        aliases=("data privatisation", "dashboarding", "dashboards"),
    ),
    CanonicalSkill(
        name="Power BI",
        category=SkillCategory.ANALYTICS,
        aliases=("powerbi", "microsoft power bi"),
    ),
    CanonicalSkill(
        name="Tableau",
        category=SkillCategory.ANALYTICS,
        aliases=("tableau desktop", "tableau server"),
    ),
    CanonicalSkill(
        name="ERP",
        category=SkillCategory.ENTERPRISE_SYSTEMS,
        aliases=(
            "enterprise resource planning",
            "oracle fusion",
            "oracle fusion cloud",
            "oracle erp cloud",
            "sap erp",
            "sap s4 hana",
            "sap s/4hana",
            "workday financials",
            "microsoft dynamics 365",
        ),
    ),
    CanonicalSkill(
        name="CRM",
        category=SkillCategory.ENTERPRISE_SYSTEMS,
        aliases=(
            "customer relationship management",
            "salesforce",
            "salesforce crm",
            "dynamics crm",
        ),
    ),
    CanonicalSkill(
        name="SAP",
        category=SkillCategory.ENTERPRISE_SYSTEMS,
        aliases=("sap ecc", "sap hana"),
    ),
    CanonicalSkill(
        name="Supply Chain Analytics",
        category=SkillCategory.SUPPLY_CHAIN,
        aliases=(
            "procurement analytics",
            "supply chain analysis",
            "supply chain reporting",
            "spend analytics",
            "supplier analytics",
            "category analytics",
        ),
    ),
    CanonicalSkill(
        name="Procurement",
        category=SkillCategory.PROCUREMENT,
        aliases=("strategic sourcing", "sourcing", "purchasing", "vendor management"),
    ),
    CanonicalSkill(
        name="Category Management",
        category=SkillCategory.PROCUREMENT,
        aliases=("category strategy", "category planning"),
    ),
    CanonicalSkill(
        name="Supplier Management",
        category=SkillCategory.PROCUREMENT,
        aliases=("supplier relationship management", "srm", "supplier performance"),
    ),
    CanonicalSkill(
        name="Demand Planning",
        category=SkillCategory.SUPPLY_CHAIN,
        aliases=("demand forecasting", "forecasting", "inventory forecasting"),
    ),
    CanonicalSkill(
        name="Inventory Management",
        category=SkillCategory.SUPPLY_CHAIN,
        aliases=("stock management", "inventory optimization", "inventory optimisation"),
    ),
    CanonicalSkill(
        name="AWS",
        category=SkillCategory.CLOUD_INFRASTRUCTURE,
        aliases=("amazon web services", "aws cloud"),
    ),
    CanonicalSkill(
        name="Azure",
        category=SkillCategory.CLOUD_INFRASTRUCTURE,
        aliases=("microsoft azure", "azure cloud"),
    ),
    CanonicalSkill(
        name="Google Cloud",
        category=SkillCategory.CLOUD_INFRASTRUCTURE,
        aliases=("gcp", "google cloud platform"),
    ),
    CanonicalSkill(
        name="Kubernetes",
        category=SkillCategory.CLOUD_INFRASTRUCTURE,
        aliases=("k8s", "kube"),
    ),
    CanonicalSkill(
        name="Docker",
        category=SkillCategory.CLOUD_INFRASTRUCTURE,
        aliases=("containerization", "containerisation", "containers"),
    ),
    CanonicalSkill(
        name="Cybersecurity",
        category=SkillCategory.SECURITY,
        aliases=("cyber security", "information security", "infosec"),
    ),
    CanonicalSkill(
        name="Product Management",
        category=SkillCategory.PRODUCT,
        aliases=("product strategy", "roadmapping", "product roadmap"),
    ),
    CanonicalSkill(
        name="Stakeholder Management",
        category=SkillCategory.BUSINESS,
        aliases=("stakeholder engagement", "executive communication"),
    ),
)


TOKEN_SEPARATOR_PATTERN = re.compile(r"[\s_\-/]+")
PUNCTUATION_PATTERN = re.compile(r"[^\w\s+#.]")


def normalize_lookup_key(value: str) -> str:
    """
    Convert skill text into a stable lowercase lookup key.
    """
    stripped = value.strip().casefold()
    without_punctuation = PUNCTUATION_PATTERN.sub(" ", stripped)
    normalized_separators = TOKEN_SEPARATOR_PATTERN.sub(" ", without_punctuation)
    return " ".join(normalized_separators.split())


def normalize_display_name(value: str) -> str:
    """
    Normalize unknown skills for display without inventing a taxonomy match.
    """
    return " ".join(value.strip().split())


class SkillNormalizer:
    """
    Deterministic skill synonym normalizer backed by an in-memory taxonomy.
    """

    def __init__(
        self,
        taxonomy: Sequence[CanonicalSkill] = DEFAULT_SKILL_TAXONOMY,
        extra_aliases: Mapping[str, str] | None = None,
    ):
        self._taxonomy = tuple(taxonomy)
        self._canonical_by_name = {entry.name: entry for entry in self._taxonomy}
        self._lookup: dict[str, CanonicalSkill] = {}
        self._load_taxonomy(self._taxonomy)
        self._load_extra_aliases(extra_aliases or {})

    @property
    def taxonomy(self) -> tuple[CanonicalSkill, ...]:
        return self._taxonomy

    def normalize(self, skill: str) -> NormalizedSkill | None:
        """
        Normalize one skill string to a canonical skill result.
        """
        if not isinstance(skill, str):
            raise TypeError("skill must be a string")

        original = normalize_display_name(skill)
        if not original:
            return None

        key = normalize_lookup_key(original)
        canonical = self._lookup.get(key)
        if canonical is None:
            return NormalizedSkill(
                original=original,
                canonical=original,
                category=SkillCategory.OTHER,
                matched_alias=None,
            )

        matched_alias = original if canonical.name != original else None
        return NormalizedSkill(
            original=original,
            canonical=canonical.name,
            category=canonical.category,
            matched_alias=matched_alias,
        )

    def normalize_many(self, skills: Iterable[str]) -> list[NormalizedSkill]:
        """
        Normalize a skill collection and dedupe by canonical skill key.
        """
        normalized: list[NormalizedSkill] = []
        seen: set[str] = set()

        for skill in skills:
            result = self.normalize(skill)
            if result is None:
                continue

            key = normalize_lookup_key(result.canonical)
            if key in seen:
                continue

            seen.add(key)
            normalized.append(result)

        return normalized

    def canonicalize_many(self, skills: Iterable[str]) -> list[str]:
        """
        Return only canonical skill names for downstream matching.
        """
        return [skill.canonical for skill in self.normalize_many(skills)]

    def _load_taxonomy(self, taxonomy: Sequence[CanonicalSkill]) -> None:
        for entry in taxonomy:
            self._register(entry.name, entry)
            for alias in entry.aliases:
                self._register(alias, entry)

    def _load_extra_aliases(self, extra_aliases: Mapping[str, str]) -> None:
        for alias, canonical_name in extra_aliases.items():
            canonical = self._canonical_by_name.get(canonical_name)
            if canonical is None:
                canonical = CanonicalSkill(
                    name=normalize_display_name(canonical_name),
                    category=SkillCategory.OTHER,
                    aliases=(alias,),
                )
                self._canonical_by_name[canonical.name] = canonical

            self._register(alias, canonical)

    def _register(self, alias: str, canonical: CanonicalSkill) -> None:
        key = normalize_lookup_key(alias)
        if key:
            self._lookup[key] = canonical


default_skill_normalizer = SkillNormalizer()


def normalize_skill(skill: str) -> NormalizedSkill | None:
    return default_skill_normalizer.normalize(skill)


def normalize_skills(skills: Iterable[str]) -> list[NormalizedSkill]:
    return default_skill_normalizer.normalize_many(skills)


def canonicalize_skills(skills: Iterable[str]) -> list[str]:
    return default_skill_normalizer.canonicalize_many(skills)
