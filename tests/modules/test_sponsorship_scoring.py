import pytest
from sqlalchemy.orm import Session

from src.modules.sponsorship.engine import SponsorshipScoringEngine
from src.modules.sponsorship.persistence import (
    SponsorshipPersistenceService,
    normalize_company_name,
)
from src.modules.sponsorship.scoring import calculate_sponsorship_score
from src.modules.sponsorship.types import DetectionResult, SponsorshipStatus


def test_normalize_company_name():
    assert normalize_company_name("Google LLC") == "google"
    assert normalize_company_name("Amazon.com, Inc.") == "amazon com"
    assert normalize_company_name("Apple, incorporated") == "apple"
    assert normalize_company_name("Meta Platforms Group") == "meta platforms"
    assert normalize_company_name("Custom Solutions Co.") == "custom"
    assert normalize_company_name("") == ""
    assert normalize_company_name(None) == ""


def test_persistence_service_upserts_and_summaries(db_session: Session):
    service = SponsorshipPersistenceService(db_session)

    # Insert historical data
    r1 = service.upsert_sponsorship_record("Google LLC", 2024, 100, 5)
    assert r1.company_name == "Google LLC"
    assert r1.normalized_company_name == "google"
    assert r1.total_petitions == 105

    # Insert another year for same company
    r2 = service.upsert_sponsorship_record("Google Inc.", 2025, 200, 10)
    assert r2.normalized_company_name == "google"

    # Query summary
    summary = service.get_historical_summary("Google")
    assert summary["has_history"] is True
    assert summary["approved"] == 300
    assert summary["denied"] == 15
    assert summary["total"] == 315
    assert summary["company_name"] in ("Google LLC", "Google Inc.")

    # Update existing record (FY 2024)
    service.upsert_sponsorship_record("Google LLC", 2024, 150, 5)
    summary2 = service.get_historical_summary("Google")
    assert summary2["approved"] == 350


def test_persistence_service_empty_summary(db_session: Session):
    service = SponsorshipPersistenceService(db_session)
    summary = service.get_historical_summary("Unknown Company Corp")
    assert summary["has_history"] is False
    assert summary["approved"] == 0
    assert summary["denied"] == 0
    assert summary["total"] == 0


def test_calculate_sponsorship_score():
    # 1. Company with strong history, neutral job description
    history = {"has_history": True, "approved": 100, "denied": 2, "total": 102}
    score, conf, strengths, gaps, explanation = calculate_sponsorship_score(
        history, SponsorshipStatus.UNKNOWN, 0.0
    )
    # history_score = 90.0 (approved > 50), history_confidence = 0.9 (total > 10)
    # extract_score = 50.0, extract_confidence = 0.1
    # total_weight = 0.9 + 0.1 = 1.0
    # blended_score = (90 * 0.9 + 50 * 0.1) / 1.0 = 86.0
    assert score == 86.0
    assert conf == 0.91  # 1 - (1 - 0.9) * (1 - 0.1) = 1 - 0.1 * 0.9 = 0.91
    assert len(strengths) == 1
    assert len(gaps) == 1  # 1 historical denial gap
    assert "Highly Likely" in explanation

    # 2. Company with strong history, but job description explicitly excludes visa sponsorship
    # extract_score = 5.0, extract_confidence = 0.9, multiplier = 3.0 -> weight = 2.7
    # blended_score = (90 * 0.9 + 5 * 2.7) / (0.9 + 2.7) = (81 + 13.5) / 3.6 = 94.5 / 3.6 = 26.25
    score2, conf2, _strengths2, gaps2, explanation2 = calculate_sponsorship_score(
        history, SponsorshipStatus.NEGATIVE, 0.9
    )
    assert score2 == 26.25
    assert conf2 == 0.99  # 1 - 0.1 * 0.1 = 0.99
    assert len(gaps2) == 2  # 1 historical denial + 1 negative job signal
    assert "Unlikely" in explanation2

    # 3. Company with no history, but job description explicitly offers sponsorship
    # history_score = 50.0, history_confidence = 0.1
    # extract_score = 95.0, extract_confidence = 0.9, multiplier = 2.0 -> weight = 1.8
    # blended_score = (50 * 0.1 + 95 * 1.8) / (0.1 + 1.8) = (5 + 171) / 1.9 = 176 / 1.9 = 92.63
    no_history = {"has_history": False, "approved": 0, "denied": 0, "total": 0}
    score3, conf3, _strengths3, _gaps3, explanation3 = calculate_sponsorship_score(
        no_history, SponsorshipStatus.POSITIVE, 0.9
    )
    assert score3 == 92.63
    assert conf3 == 0.91
    assert "Highly Likely" in explanation3


@pytest.mark.asyncio
async def test_scoring_engine_flow(db_session: Session):
    persistence = SponsorshipPersistenceService(db_session)
    persistence.upsert_sponsorship_record("Amazon LLC", 2024, 200, 5)

    engine = SponsorshipScoringEngine(persistence)

    # Scenario 1: Using pre-extracted signals (dict)
    resp1 = await engine.evaluate_sponsorship(
        company="Amazon LLC",
        extracted_signals={"status": "negative", "confidence": 0.9},
    )
    assert resp1.sponsorship_score < 50.0
    assert resp1.reasoning.historical_approved_petitions == 200
    assert resp1.reasoning.extracted_job_status == SponsorshipStatus.NEGATIVE

    # Scenario 2: Using pre-extracted signals (DetectionResult object)
    det_res = DetectionResult(
        status=SponsorshipStatus.POSITIVE,
        confidence=0.8,
        signals=[],
        explanation="Positive test",
    )
    resp2 = await engine.evaluate_sponsorship(
        company="Amazon LLC",
        extracted_signals=det_res,
    )
    assert resp2.sponsorship_score > 80.0
    assert resp2.reasoning.extracted_job_status == SponsorshipStatus.POSITIVE

    # Scenario 3: Using raw job description text (which triggers the rule detector)
    resp3 = await engine.evaluate_sponsorship(
        company="Amazon LLC",
        job_description="Visa sponsorship is available for qualified applicants.",
    )
    # rule detector yields positive status (confidence 1.0)
    assert resp3.sponsorship_score > 80.0
    assert resp3.reasoning.extracted_job_status == SponsorshipStatus.POSITIVE
