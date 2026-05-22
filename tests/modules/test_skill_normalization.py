import pytest

from src.modules.skill_normalization import (
    CanonicalSkill,
    SkillCategory,
    SkillNormalizer,
    canonicalize_skills,
    normalize_lookup_key,
    normalize_skill,
    normalize_skills,
)


@pytest.mark.parametrize(
    ("raw_skill", "expected"),
    [
        ("PySpark", "Spark"),
        ("Apache Spark", "Spark"),
        ("Oracle Fusion", "ERP"),
        ("Oracle ERP Cloud", "ERP"),
        ("Procurement Analytics", "Supply Chain Analytics"),
        ("Spend Analytics", "Supply Chain Analytics"),
        ("GCP", "Google Cloud"),
        ("K8s", "Kubernetes"),
        ("LLMs", "Large Language Models"),
        ("Gen AI", "Generative AI"),
    ],
)
def test_normalizes_synonymous_skills(raw_skill: str, expected: str) -> None:
    result = normalize_skill(raw_skill)

    assert result is not None
    assert result.canonical == expected
    assert result.original == raw_skill


def test_preserves_unknown_skills_without_hallucinating_mapping() -> None:
    result = normalize_skill("Custom Internal Tool")

    assert result is not None
    assert result.canonical == "Custom Internal Tool"
    assert result.category == SkillCategory.OTHER
    assert result.matched_alias is None


def test_normalizes_and_dedupes_many_skills_by_canonical_name() -> None:
    skills = ["PySpark", "Spark", " Oracle Fusion ", "ERP", "", "Procurement Analytics"]

    assert canonicalize_skills(skills) == ["Spark", "ERP", "Supply Chain Analytics"]


def test_normalized_results_include_taxonomy_category_and_alias() -> None:
    results = normalize_skills(["PySpark", "Procurement Analytics", "Python"])

    assert [(result.canonical, result.category) for result in results] == [
        ("Spark", SkillCategory.DATA_AI),
        ("Supply Chain Analytics", SkillCategory.SUPPLY_CHAIN),
        ("Python", SkillCategory.PROGRAMMING),
    ]
    assert results[0].matched_alias == "PySpark"
    assert results[2].matched_alias is None


def test_lookup_key_handles_enterprise_punctuation_variants() -> None:
    assert normalize_lookup_key("SAP S/4HANA") == "sap s 4hana"
    assert normalize_lookup_key("Supply-Chain_Analytics") == "supply chain analytics"


def test_custom_aliases_extend_taxonomy_without_global_mutation() -> None:
    normalizer = SkillNormalizer(extra_aliases={"Oracle SCM Cloud": "Supply Chain Management"})

    result = normalizer.normalize("Oracle SCM Cloud")
    default_result = normalize_skill("Oracle SCM Cloud")

    assert result is not None
    assert result.canonical == "Supply Chain Management"
    assert result.category == SkillCategory.OTHER
    assert default_result is not None
    assert default_result.canonical == "Oracle SCM Cloud"


def test_custom_taxonomy_can_override_default_strategy() -> None:
    normalizer = SkillNormalizer(
        taxonomy=(
            CanonicalSkill(
                name="Enterprise Procurement Suite",
                category=SkillCategory.PROCUREMENT,
                aliases=("Oracle Fusion",),
            ),
        )
    )

    result = normalizer.normalize("Oracle Fusion")

    assert result is not None
    assert result.canonical == "Enterprise Procurement Suite"
    assert result.category == SkillCategory.PROCUREMENT


def test_rejects_non_string_single_skill() -> None:
    with pytest.raises(TypeError, match="skill must be a string"):
        normalize_skill(123)  # type: ignore[arg-type]
