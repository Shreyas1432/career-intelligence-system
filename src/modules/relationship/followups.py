from datetime import UTC, datetime, timedelta

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.modules.relationship.models import FollowupRecommendationModel
from src.modules.relationship.repositories import (
    CommunicationProfileRepository,
    ContactRepository,
    FollowupRepository,
    OutreachEventRepository,
)
from src.modules.relationship.schemas import (
    ContactResponse,
    ContactType,
    FollowupRecommendation,
    InteractionOutcome,
    RelationshipStatus,
)


class FollowupCandidate(BaseModel):
    """Encapsulates eligibility and priority assessment for a follow-up action."""
    contact: ContactResponse
    freshness_score: float = Field(description="Interaction freshness rating (0.0 to 1.0)")
    urgency_score: float = Field(description="Calculated urgency priority rating (0.0 to 100.0)")
    priority: int = Field(description="Priority tier (1 to 5, where 1 is highest priority)")
    recommended_window: tuple[int, int] = Field(description="Min and max recommended follow-up days since last contact")
    is_stale: bool = Field(description="True if last contact is older than the staleness threshold")
    explanation: str = Field(description="Textual breakdown of priority scoring factors")
    last_interaction_date: datetime | None = Field(default=None, description="Timestamp of most recent interaction")
    last_outcome: InteractionOutcome | None = Field(default=None, description="Outcome from last interaction")


class RelationshipFreshnessEvaluator:
    """Evaluates the freshness and staleness of relationships based on interaction recency."""

    @staticmethod
    def calculate_freshness_score(last_interaction_date: datetime | None) -> float:
        """
        Calculates a freshness score between 0.0 (completely stale) and 1.0 (extremely fresh).
        Linear decay over 30 days.
        """
        if not last_interaction_date:
            return 0.0
        now_ts = datetime.now(UTC).replace(tzinfo=None)
        diff = now_ts - last_interaction_date.replace(tzinfo=None)
        days = max(0, diff.days)
        return max(0.0, min(1.0, 1.0 - (days / 30.0)))

    @staticmethod
    def classify_relationship(last_interaction_date: datetime | None, threshold_days: int = 30) -> str:
        """Classifies a relationship as active or stale based on threshold."""
        if not last_interaction_date:
            return "stale"
        now_ts = datetime.now(UTC).replace(tzinfo=None)
        diff = now_ts - last_interaction_date.replace(tzinfo=None)
        return "stale" if diff.days > threshold_days else "active"


class FollowupWindowCalculator:
    """Calculates recommended follow-up timing windows based on contact role and history."""

    @staticmethod
    def _calculate_recruiter_window(last_outcome: InteractionOutcome | None) -> tuple[int, int]:
        if last_outcome == InteractionOutcome.ACTION_REQUIRED:
            return 1, 2
        if last_outcome == InteractionOutcome.POSITIVE:
            return 2, 4
        if last_outcome == InteractionOutcome.NEUTRAL:
            return 5, 8
        if last_outcome == InteractionOutcome.NO_RESPONSE:
            return 4, 7
        return 3, 6

    @staticmethod
    def _calculate_hm_window(last_outcome: InteractionOutcome | None) -> tuple[int, int]:
        if last_outcome == InteractionOutcome.ACTION_REQUIRED:
            return 1, 3
        if last_outcome == InteractionOutcome.POSITIVE:
            return 4, 7
        if last_outcome == InteractionOutcome.NEUTRAL:
            return 7, 12
        if last_outcome == InteractionOutcome.NO_RESPONSE:
            return 7, 10
        return 5, 10

    @staticmethod
    def _calculate_default_window(last_outcome: InteractionOutcome | None) -> tuple[int, int]:
        if last_outcome == InteractionOutcome.ACTION_REQUIRED:
            return 2, 4
        if last_outcome == InteractionOutcome.POSITIVE:
            return 5, 10
        if last_outcome == InteractionOutcome.NEUTRAL:
            return 10, 18
        return 7, 14

    @classmethod
    def calculate_recommended_window(
        cls, contact: ContactResponse, last_outcome: InteractionOutcome | None
    ) -> tuple[int, int]:
        """
        Calculates the recommended min and max days for follow-up.
        Returns:
            (min_days, max_days)
        """
        if contact.contact_type == ContactType.RECRUITER:
            return cls._calculate_recruiter_window(last_outcome)
        if contact.contact_type == ContactType.HIRING_MANAGER:
            return cls._calculate_hm_window(last_outcome)
        return cls._calculate_default_window(last_outcome)


class FollowupPriorityScorer:
    """Calculates follow-up urgency scores and priorities based on multiple deterministic factors."""

    @staticmethod
    def _get_role_adjustment(contact_type: ContactType) -> tuple[float, str | None]:
        if contact_type == ContactType.HIRING_MANAGER:
            return 15.0, "Hiring manager role (+15.0)"
        if contact_type == ContactType.RECRUITER:
            return 10.0, "Recruiter role (+10.0)"
        if contact_type == ContactType.PEER:
            return 5.0, "Peer role (+5.0)"
        return 0.0, None

    @staticmethod
    def _get_outcome_adjustment(last_outcome: InteractionOutcome | None) -> tuple[float, str | None]:
        if last_outcome == InteractionOutcome.ACTION_REQUIRED:
            return 25.0, "Action required outcome (+25.0)"
        if last_outcome == InteractionOutcome.POSITIVE:
            return 10.0, "Prior positive outcome (+10.0)"
        if last_outcome == InteractionOutcome.NEGATIVE:
            return -30.0, "Prior negative outcome (-30.0)"
        if last_outcome == InteractionOutcome.NO_RESPONSE:
            return 5.0, "No response on prior outreach (+5.0)"
        return 0.0, None

    @staticmethod
    def _get_recency_adjustment(
        last_interaction_date: datetime | None, min_days: int, max_days: int
    ) -> tuple[float, str | None]:
        if not last_interaction_date:
            days_elapsed = 999
        else:
            now_ts = datetime.now(UTC).replace(tzinfo=None)
            days_elapsed = (now_ts - last_interaction_date.replace(tzinfo=None)).days

        if days_elapsed > max_days:
            overdue_days = days_elapsed - max_days
            time_adj = min(20.0, overdue_days * 2.0)
            return time_adj, f"Follow-up is overdue by {overdue_days} days ({time_adj:+.1f})"
        if days_elapsed < min_days:
            return -30.0, f"Within warm-up window of {min_days} days (-30.0)"
        return 0.0, None

    @staticmethod
    def _map_priority(final_score: float) -> int:
        if final_score >= 80.0:
            return 1
        if final_score >= 60.0:
            return 2
        if final_score >= 40.0:
            return 3
        if final_score >= 20.0:
            return 4
        return 5

    @classmethod
    def calculate_urgency_score(
        cls,
        contact: ContactResponse,
        last_interaction_date: datetime | None,
        last_outcome: InteractionOutcome | None,
        engagement_score: float,
        is_target_company: bool,
        min_days: int,
        max_days: int,
    ) -> tuple[float, int, str]:
        """
        Calculates the urgency score (0-100), priority tier (1-5), and a list of explanatory factors.
        Returns:
            (urgency_score, priority, explanation)
        """
        score = 50.0
        factors = []

        # 1. Recruiter vs Hiring Manager role adaptation
        role_adj, role_factor = cls._get_role_adjustment(contact.contact_type)
        score += role_adj
        if role_factor:
            factors.append(role_factor)

        # 2. Target Company relevance
        if is_target_company:
            score += 15.0
            factors.append("Target company alignment (+15.0)")

        # 3. Prior engagement quality (outcome)
        outcome_adj, outcome_factor = cls._get_outcome_adjustment(last_outcome)
        score += outcome_adj
        if outcome_factor:
            factors.append(outcome_factor)

        # 4. Outreach responsiveness (engagement score)
        eng_adj = (engagement_score - 0.5) * 20.0
        score += eng_adj
        if eng_adj != 0.0:
            factors.append(f"Contact responsiveness adjustment ({eng_adj:+.1f})")

        # 5. Interaction recency / elapsed days vs recommended window
        recency_adj, recency_factor = cls._get_recency_adjustment(last_interaction_date, min_days, max_days)
        score += recency_adj
        if recency_factor:
            factors.append(recency_factor)

        # Capping the final score
        final_score = max(0.0, min(100.0, score))

        # Priority tier assignment (1 is highest, 5 is lowest)
        priority = cls._map_priority(final_score)

        explanation = "; ".join(factors) if factors else "Base scoring profile applied."
        return final_score, priority, explanation


class FollowupRecommendationService:
    """Orchestrates candidate collection and persistence of follow-up recommendations."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.contact_repo = ContactRepository(session)
        self.event_repo = OutreachEventRepository(session)
        self.profile_repo = CommunicationProfileRepository(session)
        self.followup_repo = FollowupRepository(session)

    def retrieve_followup_candidates(
        self, limit: int = 100, target_companies: list[str] | None = None
    ) -> list[FollowupCandidate]:
        """
        Retrieves contacts (excluding ARCHIVED) and calculates urgency profiles.
        Sorted by urgency_score descending.
        """
        contacts = self.contact_repo.list_contacts(limit=limit)
        target_set = {c.strip().lower() for c in target_companies} if target_companies else set()

        candidates = []
        for contact in contacts:
            if contact.status == RelationshipStatus.ARCHIVED:
                continue

            events = list(self.event_repo.get_events_for_contact(contact.id))

            # Resolve last interaction and outcome
            last_date = None
            last_outcome = None
            if events:
                completed = [e for e in events if e.completed_at or e.created_at]
                if completed:
                    completed.sort(key=lambda e: (e.completed_at or e.created_at).replace(tzinfo=None), reverse=True)
                    last_date = completed[0].completed_at or completed[0].created_at
                    last_outcome = completed[0].outcome

            # Freshness score
            freshness = RelationshipFreshnessEvaluator.calculate_freshness_score(last_date)
            is_stale = RelationshipFreshnessEvaluator.classify_relationship(last_date) == "stale"

            # Recommended window
            min_days, max_days = FollowupWindowCalculator.calculate_recommended_window(contact, last_outcome)

            # Target company match
            is_target_company = False
            if contact.company and target_set:
                from src.modules.relationship.contacts import ContactNormalizer
                norm_comp = ContactNormalizer.normalize_company(contact.company)
                for t in target_set:
                    if norm_comp in ContactNormalizer.normalize_company(t):
                        is_target_company = True
                        break

            # Responsiveness rating
            profile = self.profile_repo.get_profile(contact.id)
            engagement_score = profile.engagement_score if profile else 0.0

            # Urgency scoring
            urgency_score, priority, explanation = FollowupPriorityScorer.calculate_urgency_score(
                contact, last_date, last_outcome, engagement_score, is_target_company, min_days, max_days
            )

            candidates.append(
                FollowupCandidate(
                    contact=contact,
                    freshness_score=freshness,
                    urgency_score=urgency_score,
                    priority=priority,
                    recommended_window=(min_days, max_days),
                    is_stale=is_stale,
                    explanation=explanation,
                    last_interaction_date=last_date,
                    last_outcome=last_outcome,
                )
            )

        # Sort by urgency score descending (stable sort in python preserves DB retrieve order for equals)
        candidates.sort(key=lambda c: c.urgency_score, reverse=True)
        return candidates

    def _construct_draft_message(self, contact: ContactResponse, last_outcome: InteractionOutcome | None) -> str:
        """Constructs a professional, non-manipulative draft follow-up message."""
        name = contact.first_name
        company = contact.company or "your company"

        if last_outcome == InteractionOutcome.ACTION_REQUIRED:
            return (
                f"Hi {name},\n\nI hope you're doing well. Following up on our last conversation, "
                f"I've prepared the requested details. Please let me know the best time to share them."
            )
        if last_outcome == InteractionOutcome.POSITIVE:
            return (
                f"Hi {name},\n\nI hope you're having a great week. I'm following up to check if there are "
                f"any updates regarding the discussions we had last week. Let me know if there's any other "
                f"information I can provide on my end."
            )
        if last_outcome == InteractionOutcome.NO_RESPONSE:
            return (
                f"Hi {name},\n\nI hope you're having a good week. I wanted to check if you had a chance to "
                f"review my previous note. Let me know if you are free for a brief sync sometime soon."
            )

        return (
            f"Hi {name},\n\nI hope you're doing well. I wanted to follow up on our previous conversation "
            f"and see if there are any updates regarding potential opportunities at {company}."
        )

    def generate_recommendations(
        self, target_companies: list[str] | None = None
    ) -> list[FollowupRecommendation]:
        """
        Generates and persists follow-up recommendations for all contacts who are due.
        """
        candidates = self.retrieve_followup_candidates(target_companies=target_companies)
        recommendations = []

        now_ts = datetime.now(UTC).replace(tzinfo=None)

        for c in candidates:
            # A contact is due if days elapsed >= min_days, or outcome is ACTION_REQUIRED
            last_date = c.last_interaction_date
            min_days, _ = c.recommended_window

            if last_date:
                days_elapsed = (now_ts - last_date.replace(tzinfo=None)).days
            else:
                days_elapsed = 999

            is_due = (days_elapsed >= min_days) or (c.last_outcome == InteractionOutcome.ACTION_REQUIRED)

            # Filter out non-due or archived contacts
            if c.contact.status == RelationshipStatus.ARCHIVED:
                is_due = False

            if is_due:
                # 1. Clear any existing recommendations for this contact to avoid duplicates
                existing = self.session.query(FollowupRecommendationModel).filter_by(contact_id=str(c.contact.id)).all()
                for r in existing:
                    self.session.delete(r)

                # 2. Calculate suggested date: now + max(0, min_days - days_elapsed)
                suggested_days_offset = max(0, min_days - days_elapsed) if c.last_outcome != InteractionOutcome.ACTION_REQUIRED else 0
                suggested_date = datetime.now(UTC) + timedelta(days=suggested_days_offset)

                # 3. Construct draft message
                draft = self._construct_draft_message(c.contact, c.last_outcome)

                # 4. Save model
                rec_model = FollowupRecommendationModel(
                    contact_id=str(c.contact.id),
                    suggested_date=suggested_date,
                    reasoning=c.explanation,
                    draft_message=draft,
                    priority=c.priority,
                )
                self.session.add(rec_model)
                self.session.flush()

                recommendations.append(
                    FollowupRecommendation(
                        contact_id=c.contact.id,
                        suggested_date=suggested_date,
                        reasoning=c.explanation,
                        draft_message=draft,
                        priority=c.priority,
                    )
                )

        return recommendations
