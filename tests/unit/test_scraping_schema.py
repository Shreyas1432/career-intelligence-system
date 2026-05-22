# tests/unit/test_scraping_schema.py

import pytest
from pydantic import ValidationError

from src.modules.job_extraction import (
    EmploymentType,
    JobDomain,
    JobExtractionResult,
    VisaSignal,
)


def test_schema_rejects_empty_and_no_signal_extraction() -> None:
    """
    Ensure the schema rejects empty dicts or payloads that have no recognizable job signal.
    """
    with pytest.raises(ValidationError) as exc_info:
        JobExtractionResult.model_validate({})
    assert "usable job signals" in str(exc_info.value)

    with pytest.raises(ValidationError) as exc_info:
        JobExtractionResult.model_validate(
            {
                "company": None,
                "title": "",
                "skills": [],
                "visa_signal": "unknown",
                "employment_type": "unknown",
                "domain": "unknown",
            }
        )
    assert "usable job signals" in str(exc_info.value)


def test_schema_allows_minimal_valid_signal() -> None:
    """
    Verify the schema parses if at least one usable job signal is provided.
    """
    result = JobExtractionResult.model_validate({"title": "Staff Engineer"})
    assert result.title == "Staff Engineer"
    assert result.company is None
    assert result.skills == []
    assert result.confidence_score is not None
    assert result.confidence_score > 0.0


def test_schema_normalizes_aliases_for_enums() -> None:
    """
    Ensure raw string inputs map correctly to standard enums.
    """
    payload = {
        "title": "Backend Engineer",
        "visa_signal": "visa sponsorship available",
        "employment_type": "contractor",
        "domain": "ml",
    }
    result = JobExtractionResult.model_validate(payload)
    assert result.visa_signal == VisaSignal.SPONSORSHIP_AVAILABLE
    assert result.employment_type == EmploymentType.CONTRACT
    assert result.domain == JobDomain.DATA_AI


def test_schema_rejects_legacy_fields_explicitly() -> None:
    """
    Ensure legacy fields like 'sponsorship_clues' and 'domain_signals' are forbidden.
    """
    with pytest.raises(ValidationError) as exc_info:
        JobExtractionResult.model_validate(
            {"title": "Developer", "sponsorship_clues": ["needs sponsor"]}
        )
    assert "Legacy extraction fields are not accepted" in str(exc_info.value)

    with pytest.raises(ValidationError) as exc_info:
        JobExtractionResult.model_validate({"title": "Developer", "domain_signals": ["AI"]})
    assert "Legacy extraction fields are not accepted" in str(exc_info.value)


def test_schema_rejects_unmapped_extra_fields() -> None:
    """
    Verify model forbids unexpected input keys.
    """
    with pytest.raises(ValidationError) as exc_info:
        JobExtractionResult.model_validate({"title": "Developer", "random_field_xyz": "value"})
    assert "Extra inputs are not permitted" in str(exc_info.value)


def test_schema_confidence_estimation_logic() -> None:
    """
    Test manual confidence score estimation when omitted by AI model.
    """
    # 1. Very basic signals
    res_low = JobExtractionResult.model_validate({"title": "Developer"})
    assert res_low.confidence_score == 0.22

    # 2. Rich features
    res_high = JobExtractionResult.model_validate(
        {
            "company": "Google",
            "title": "Staff Engineer",
            "skills": ["Python", "Go", "C++"],
            "experience_required": "5 years",
            "location": "Mountain View, CA",
            "visa_signal": "sponsorship_available",
            "employment_type": "full_time",
            "domain": "software_engineering",
        }
    )
    # Max confidence should cap at 1.0
    assert (
        res_high.confidence_score == 0.9
    )  # calculated: 0.18 + 0.22 + (3 * 0.035 = 0.105) + 0.1 + 0.08 + 0.06 + 0.08 + 0.08 = 0.905 -> rounded to 0.9
    # Let's verify estimated logic matches Service behavior
    assert res_high.confidence_score > 0.5


def test_schema_enforces_maximum_lengths() -> None:
    """
    Ensure long text inputs are rejected according to validation rules.
    """
    long_title = "A" * 205
    with pytest.raises(ValidationError) as exc_info:
        JobExtractionResult.model_validate({"title": long_title})
    assert "String should have at most 200 characters" in str(exc_info.value)

    long_company = "C" * 205
    with pytest.raises(ValidationError) as exc_info:
        JobExtractionResult.model_validate({"company": long_company})
    assert "String should have at most 200 characters" in str(exc_info.value)


def test_skills_limit_and_canonicalization() -> None:
    """
    Verify that skill list is canonicalized and normalized.
    """
    res = JobExtractionResult.model_validate(
        {"title": "Data Analyst", "skills": ["PySpark", "Apache Spark", "pyspark", "pyspark"]}
    )
    # Duplicates should be pruned, normalized
    assert res.skills == ["Spark"]
