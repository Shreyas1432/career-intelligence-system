import re
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.modules.relationship.models import RelationshipContactModel
from src.modules.relationship.repositories import (
    CommunicationProfileRepository,
    ContactRepository,
    OutreachEventRepository,
    RelationshipMemoryRepository,
)
from src.modules.relationship.schemas import (
    CommunicationProfile,
    ContactCreate,
    ContactResponse,
    ContactType,
    ContactUpdate,
    InteractionOutcome,
    OutreachEvent,
    RelationshipMemory,
    RelationshipStatus,
)


class RelationshipScoreBreakdown(BaseModel):
    """Detailed deterministic score breakdown for a relationship."""
    total_score: float = Field(ge=0.0, le=100.0, description="Overall relationship priority score (0-100)")
    company_relevance: float = Field(description="Target company alignment score component")
    role_score: float = Field(description="Recruiter/Hiring Manager role priority score component")
    status_score: float = Field(description="Relationship status score component")
    recency_score: float = Field(description="Recency of interaction score component")
    engagement_score: float = Field(description="Engagement score component based on profile and outcomes")
    explanation: str = Field(description="Textual explanation of the scoring factors")


class ContactNormalizer:
    """Utilities for normalising contact fields."""

    @staticmethod
    def normalize_company(company: str | None) -> str:
        """Standardizes company names by converting to lowercase and stripping common suffixes."""
        if not company:
            return ""

        cleaned = re.sub(r"[^\w\s]", " ", company.lower())
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
        return " ".join(filtered_words).strip()

    @staticmethod
    def normalize_email(email: str | None) -> str:
        """Trims and lowercases emails."""
        if not email:
            return ""
        return email.strip().lower()

    @staticmethod
    def normalize_name(name: str | None) -> str:
        """Trims and titles name strings."""
        if not name:
            return ""
        return name.strip().title()


class DuplicateContactDetector:
    """Finds potential duplicate contacts in the database."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def _find_email_duplicates(self, email: str, query: Any) -> list[RelationshipContactModel]:
        norm_email = ContactNormalizer.normalize_email(email)
        return list(query.filter(RelationshipContactModel.email.ilike(norm_email)).all())

    def _find_linkedin_duplicates(self, linkedin_url: str, query: Any) -> list[RelationshipContactModel]:
        duplicates = []
        norm_li = linkedin_url.strip().lower().rstrip("/")
        li_clean = norm_li.replace("https://", "").replace("http://", "").replace("www.", "")

        all_contacts = query.filter(RelationshipContactModel.linkedin_url.isnot(None)).all()
        for c in all_contacts:
            if c.linkedin_url:
                c_li_clean = (
                    c.linkedin_url.strip()
                    .lower()
                    .rstrip("/")
                    .replace("https://", "")
                    .replace("http://", "")
                    .replace("www.", "")
                )
                if c_li_clean == li_clean:
                    duplicates.append(c)
        return duplicates

    def _find_name_company_duplicates(
        self, first_name: str, last_name: str | None, company: str, query: Any
    ) -> list[RelationshipContactModel]:
        duplicates = []
        norm_company = ContactNormalizer.normalize_company(company)
        if not norm_company:
            return []

        name_query = query.filter(RelationshipContactModel.first_name.ilike(first_name.strip()))
        if last_name:
            name_query = name_query.filter(RelationshipContactModel.last_name.ilike(last_name.strip()))
        else:
            name_query = name_query.filter(
                RelationshipContactModel.last_name.is_(None) | (RelationshipContactModel.last_name == "")
            )

        name_matches = name_query.all()
        for c in name_matches:
            if c.company:
                c_comp_norm = ContactNormalizer.normalize_company(c.company)
                if c_comp_norm == norm_company:
                    duplicates.append(c)
        return duplicates

    def find_duplicates(
        self,
        first_name: str,
        last_name: str | None,
        email: str | None,
        linkedin_url: str | None,
        company: str | None,
    ) -> list[RelationshipContactModel]:
        """
        Scan database for matching contacts using:
        1. Exact email match (case-insensitive)
        2. Exact linkedin URL match (case-insensitive, ignoring protocol/trailing slash)
        3. First name + Last name + Company match (using normalized company matching)
        """
        duplicates: list[RelationshipContactModel] = []
        query = self.session.query(RelationshipContactModel)

        if email:
            for c in self._find_email_duplicates(email, query):
                if c not in duplicates:
                    duplicates.append(c)

        if linkedin_url:
            for c in self._find_linkedin_duplicates(linkedin_url, query):
                if c not in duplicates:
                    duplicates.append(c)

        if company:
            for c in self._find_name_company_duplicates(first_name, last_name, company, query):
                if c not in duplicates:
                    duplicates.append(c)

        return duplicates


class RelationshipScorer:
    """Deterministic, explainable scoring engine for relationship prioritization."""

    @staticmethod
    def _score_company_relevance(
        company: str | None, target_companies: list[str] | None, explanations: list[str]
    ) -> float:
        if not company:
            explanations.append("Contact has no associated company.")
            return 0.0

        norm_contact_company = ContactNormalizer.normalize_company(company)
        if target_companies:
            for tc in target_companies:
                if ContactNormalizer.normalize_company(tc) == norm_contact_company:
                    explanations.append(f"Company '{company}' matches target companies (+20 pts).")
                    return 20.0

        explanations.append(f"Company '{company}' is not in the target company list.")
        return 0.0

    @staticmethod
    def _score_role(contact_type: ContactType, explanations: list[str]) -> float:
        if contact_type == ContactType.HIRING_MANAGER:
            explanations.append("Hiring manager role priority (+30 pts).")
            return 30.0
        if contact_type == ContactType.RECRUITER:
            explanations.append("Recruiter role priority (+20 pts).")
            return 20.0
        if contact_type in {ContactType.PEER, ContactType.ALUMNI, ContactType.MENTOR}:
            explanations.append(f"{contact_type.value.title()} role priority (+10 pts).")
            return 10.0

        explanations.append("Standard contact role tier (0 pts).")
        return 0.0

    @staticmethod
    def _score_status(status: RelationshipStatus, explanations: list[str]) -> float:
        if status == RelationshipStatus.ACTIVE:
            explanations.append("Active relationship status (+25 pts).")
            return 25.0
        if status == RelationshipStatus.WARM:
            explanations.append("Warm relationship status (+20 pts).")
            return 20.0
        if status == RelationshipStatus.NEW:
            explanations.append("New relationship status (+10 pts).")
            return 10.0
        if status == RelationshipStatus.DORMANT:
            explanations.append("Dormant relationship status (+5 pts).")
            return 5.0

        explanations.append("Archived/Other relationship status (0 pts).")
        return 0.0

    @staticmethod
    def _score_recency(
        outreach_events: list[OutreachEvent], memory: RelationshipMemory | None, explanations: list[str]
    ) -> float:
        last_date = None
        if memory and memory.last_interaction_date:
            last_date = memory.last_interaction_date

        for e in outreach_events:
            if e.completed_at:
                if last_date is None or e.completed_at > last_date:
                    last_date = e.completed_at

        if not last_date:
            explanations.append("No recorded past interactions.")
            return 0.0

        days_ago = (datetime.now(UTC).replace(tzinfo=None) - last_date.replace(tzinfo=None)).days
        if days_ago <= 14:
            explanations.append(f"Recent interaction within {days_ago} days (+15 pts).")
            return 15.0
        if days_ago <= 30:
            explanations.append(f"Interaction within {days_ago} days (+10 pts).")
            return 10.0
        if days_ago <= 90:
            explanations.append(f"Interaction within {days_ago} days (+5 pts).")
            return 5.0

        explanations.append(f"Last interaction was {days_ago} days ago (0 pts).")
        return 0.0

    @staticmethod
    def _score_engagement(
        comm_profile: CommunicationProfile | None, outreach_events: list[OutreachEvent], explanations: list[str]
    ) -> float:
        base_eng = 0.0
        if comm_profile:
            base_eng = comm_profile.engagement_score * 10.0

        pos_outcomes = sum(1 for e in outreach_events if e.outcome == InteractionOutcome.POSITIVE)
        neg_outcomes = sum(1 for e in outreach_events if e.outcome == InteractionOutcome.NEGATIVE)

        outcome_bonus = (pos_outcomes * 2.0) - (neg_outcomes * 3.0)
        engagement_score = min(max(base_eng + outcome_bonus, 0.0), 10.0)
        explanations.append(f"Engagement quality and outcomes (+{round(engagement_score, 1)} pts).")
        return round(engagement_score, 2)

    @staticmethod
    def calculate_score(
        contact: ContactResponse,
        outreach_events: list[OutreachEvent],
        memory: RelationshipMemory | None,
        comm_profile: CommunicationProfile | None,
        target_companies: list[str] | None = None,
    ) -> RelationshipScoreBreakdown:
        """
        Deterministically calculate relationship priority score (0-100) based on:
        1. target company relevance (max 20 pts)
        2. recruiter vs hiring manager role (max 30 pts)
        3. relationship status (max 25 pts)
        4. interaction recency (max 15 pts)
        5. engagement quality & outcomes (max 10 pts)
        """
        explanations: list[str] = []

        company_relevance = RelationshipScorer._score_company_relevance(
            contact.company, target_companies, explanations
        )
        role_score = RelationshipScorer._score_role(contact.contact_type, explanations)
        status_score = RelationshipScorer._score_status(contact.status, explanations)
        recency_score = RelationshipScorer._score_recency(outreach_events, memory, explanations)
        engagement_score = RelationshipScorer._score_engagement(comm_profile, outreach_events, explanations)

        total_score = company_relevance + role_score + status_score + recency_score + engagement_score
        explanation_str = " ".join(explanations)

        return RelationshipScoreBreakdown(
            total_score=round(total_score, 2),
            company_relevance=company_relevance,
            role_score=role_score,
            status_score=status_score,
            recency_score=recency_score,
            engagement_score=engagement_score,
            explanation=explanation_str,
        )


class ContactService:
    """Orchestrates relationship contact intelligence and lifecycle management."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.contact_repo = ContactRepository(session)
        self.event_repo = OutreachEventRepository(session)
        self.memory_repo = RelationshipMemoryRepository(session)
        self.profile_repo = CommunicationProfileRepository(session)

    @staticmethod
    def enrich_company_from_email(email: str | None, company: str | None) -> str | None:
        """Determines company name from corporate email domains if company name is missing."""
        if company:
            return company
        if not email:
            return None

        parts = email.split("@")
        if len(parts) != 2:
            return None

        domain = parts[1].lower()
        personal_domains = {
            "gmail.com",
            "yahoo.com",
            "hotmail.com",
            "outlook.com",
            "icloud.com",
            "aol.com",
            "zoho.com",
            "mail.com",
            "protonmail.com",
            "proton.me",
            "yandex.com",
            "live.com",
            "msn.com",
        }

        if domain in personal_domains:
            return None

        domain_parts = domain.split(".")
        if domain_parts:
            return domain_parts[0].title()
        return None

    def create_contact(self, create_schema: ContactCreate, raise_on_duplicate: bool = True) -> ContactResponse:
        """Orchestrates contact creation with normalization, enrichment, and duplicate checks."""
        # 1. Lightweight contact enrichment
        enriched_company = self.enrich_company_from_email(create_schema.email, create_schema.company)

        # 2. Field normalization
        norm_first = ContactNormalizer.normalize_name(create_schema.first_name)
        norm_last = ContactNormalizer.normalize_name(create_schema.last_name) if create_schema.last_name else None
        norm_company_raw = ContactNormalizer.normalize_company(enriched_company) if enriched_company else None
        norm_company = ContactNormalizer.normalize_name(norm_company_raw) if norm_company_raw else None
        norm_email = ContactNormalizer.normalize_email(create_schema.email) if create_schema.email else None

        # 3. Duplicate checks
        detector = DuplicateContactDetector(self.session)
        duplicates = detector.find_duplicates(
            first_name=norm_first,
            last_name=norm_last,
            email=norm_email,
            linkedin_url=create_schema.linkedin_url,
            company=norm_company,
        )

        if duplicates and raise_on_duplicate:
            raise ValueError(
                f"Duplicate contact detected: {norm_first} {norm_last or ''} at {norm_company or 'Unknown Company'}"
            )

        # 4. Construct normalized schema and save
        normalized_schema = ContactCreate(
            first_name=norm_first,
            last_name=norm_last,
            company=norm_company,
            title=create_schema.title.strip() if create_schema.title else None,
            contact_type=create_schema.contact_type,
            linkedin_url=create_schema.linkedin_url.strip() if create_schema.linkedin_url else None,
            email=norm_email,
            metadata=create_schema.metadata,
        )

        return self.contact_repo.create_contact(normalized_schema)

    def update_contact(self, contact_id: UUID, update_schema: ContactUpdate) -> ContactResponse | None:
        """Orchestrates contact updates with field normalization."""
        update_data = update_schema.model_dump(exclude_unset=True)

        if update_data.get("first_name"):
            update_data["first_name"] = ContactNormalizer.normalize_name(update_data["first_name"])
        if "last_name" in update_data:
            update_data["last_name"] = (
                ContactNormalizer.normalize_name(update_data["last_name"])
                if update_data["last_name"]
                else None
            )
        if "email" in update_data:
            update_data["email"] = (
                ContactNormalizer.normalize_email(update_data["email"]) if update_data["email"] else None
            )
        if "company" in update_data:
            norm_raw = ContactNormalizer.normalize_company(update_data["company"])
            update_data["company"] = (
                ContactNormalizer.normalize_name(norm_raw)
                if norm_raw
                else None
            )

        normalized_update = ContactUpdate(**update_data)
        return self.contact_repo.update_contact(contact_id, normalized_update)

    def get_contact(self, contact_id: UUID) -> ContactResponse | None:
        """Fetch contact by UUID."""
        return self.contact_repo.get_contact_by_uuid(contact_id)

    def list_contacts(
        self,
        skip: int = 0,
        limit: int = 100,
        status: RelationshipStatus | None = None,
        contact_type: ContactType | None = None,
        company: str | None = None,
    ) -> list[ContactResponse]:
        """Fetches contacts list with filtering helpers."""
        contacts = self.contact_repo.list_contacts(skip=skip, limit=limit, status=status)

        filtered = []
        for c in contacts:
            if contact_type and c.contact_type != contact_type:
                continue
            if company:
                c_norm = ContactNormalizer.normalize_company(c.company)
                target_norm = ContactNormalizer.normalize_company(company)
                if target_norm not in c_norm:
                    continue
            filtered.append(c)

        return filtered

    def score_relationship(
        self, contact_id: UUID, target_companies: list[str] | None = None
    ) -> RelationshipScoreBreakdown | None:
        """Evaluates and scores a relationship priority."""
        contact = self.contact_repo.get_contact_by_uuid(contact_id)
        if not contact:
            return None

        outreach_events = list(self.event_repo.get_events_for_contact(contact_id))
        memory = self.memory_repo.get_memory(contact_id)
        comm_profile = self.profile_repo.get_profile(contact_id)

        return RelationshipScorer.calculate_score(
            contact=contact,
            outreach_events=outreach_events,
            memory=memory,
            comm_profile=comm_profile,
            target_companies=target_companies,
        )

    def prioritize_relationships(
        self, target_companies: list[str] | None = None, limit: int = 50
    ) -> list[tuple[ContactResponse, RelationshipScoreBreakdown]]:
        """Fetch and rank contacts by priority score descending."""
        contacts = self.contact_repo.list_contacts(limit=limit)
        scored: list[tuple[ContactResponse, RelationshipScoreBreakdown]] = []

        for c in contacts:
            score = self.score_relationship(c.id, target_companies)
            if score:
                scored.append((c, score))

        scored.sort(key=lambda item: item[1].total_score, reverse=True)
        return scored
