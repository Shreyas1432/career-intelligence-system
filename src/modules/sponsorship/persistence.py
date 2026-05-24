import re
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.core.database.models import GovernmentSponsorship


def normalize_company_name(name: str) -> str:
    """
    Standardize corporate names for exact lookup by stripping suffixes, punctuation, and spaces.
    e.g. "Google LLC" -> "google", "Amazon.com, Inc." -> "amazon com"
    """
    if not name:
        return ""

    # Lowercase and replace non-alphanumeric (excluding whitespace) with spaces
    cleaned = re.sub(r"[^\w\s]", " ", name.lower())
    words = cleaned.split()

    corporate_suffixes = {
        "inc",
        "incorporated",
        "llc",
        "corp",
        "corporation",
        "ltd",
        "limited",
        "co",
        "company",
        "group",
        "solutions",
        "technologies",
        "services",
        "systems",
    }

    filtered_words = [w for w in words if w not in corporate_suffixes]
    return " ".join(filtered_words)


class SponsorshipPersistenceService:
    """
    Persistence service coordinating historical government sponsorship records stored in SQLite.
    """

    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert_sponsorship_record(
        self,
        company_name: str,
        fiscal_year: int,
        approved_petitions: int,
        denied_petitions: int,
    ) -> GovernmentSponsorship:
        """
        Create or update a historical government visa sponsorship record.
        """
        normalized = normalize_company_name(company_name)
        total = approved_petitions + denied_petitions

        stmt = select(GovernmentSponsorship).where(
            GovernmentSponsorship.normalized_company_name == normalized,
            GovernmentSponsorship.fiscal_year == fiscal_year,
        )
        record = self.session.scalars(stmt).first()

        if record:
            record.company_name = company_name
            record.approved_petitions = approved_petitions
            record.denied_petitions = denied_petitions
            record.total_petitions = total
        else:
            record = GovernmentSponsorship(
                company_name=company_name,
                normalized_company_name=normalized,
                fiscal_year=fiscal_year,
                approved_petitions=approved_petitions,
                denied_petitions=denied_petitions,
                total_petitions=total,
            )
            self.session.add(record)

        self.session.flush()
        return record

    def get_historical_summary(self, company_name: str) -> dict[str, Any]:
        """
        Query and sum historical sponsorship metrics for a given company.
        """
        normalized = normalize_company_name(company_name)

        stmt = select(GovernmentSponsorship).where(
            GovernmentSponsorship.normalized_company_name == normalized
        )
        records = self.session.scalars(stmt).all()

        if not records:
            return {
                "company_name": company_name,
                "approved": 0,
                "denied": 0,
                "total": 0,
                "has_history": False,
            }

        total_approved = sum(r.approved_petitions for r in records)
        total_denied = sum(r.denied_petitions for r in records)
        total_cases = sum(r.total_petitions for r in records)

        # Retrieve the most recently used casing for company name
        canonical_name = records[0].company_name

        return {
            "company_name": canonical_name,
            "approved": total_approved,
            "denied": total_denied,
            "total": total_cases,
            "has_history": True,
        }
