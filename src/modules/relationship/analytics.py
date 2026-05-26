from datetime import UTC, datetime, timedelta

from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.modules.relationship.models import (
    FollowupRecommendationModel,
    OutreachEventModel,
    RelationshipContactModel,
)
from src.modules.relationship.repositories import (
    ContactRepository,
    FollowupRepository,
    OutreachEventRepository,
)
from src.modules.relationship.schemas import (
    ContactResponse,
    ContactType,
    InteractionOutcome,
    NetworkingMetrics,
    OutreachEvent,
    OutreachStatus,
    RelationshipStatus,
)


class RecruiterHMAnalytics(BaseModel):
    """Analytics for a specific role category."""
    total_contacts: int
    response_rate: float
    replied_count: int
    sent_count: int


class FollowupEffectiveness(BaseModel):
    """Aggregated outcome effectiveness for follow-ups."""
    total_recommended: int
    total_acted_upon: int
    action_rate: float
    positive_response_rate: float


class CompanyEngagement(BaseModel):
    """Engagement metrics aggregated at corporate level."""
    company_name: str
    total_contacts: int
    total_outreach: int
    response_rate: float
    last_interaction: datetime | None = None


class ConsistencyMetrics(BaseModel):
    """Weekly outreach consistency trends."""
    outreach_by_week: dict[str, int]
    weekly_consistency_index: float


class ProgressionSummary(BaseModel):
    """State transition and health summaries of all relationships."""
    status_counts: dict[str, int]
    active_count: int
    stale_count: int


class RelationshipAnalyticsSummary(BaseModel):
    """Unified networking and relationship intelligence report."""
    overall_metrics: NetworkingMetrics
    recruiter_metrics: RecruiterHMAnalytics
    hiring_manager_metrics: RecruiterHMAnalytics
    followup_effectiveness: FollowupEffectiveness
    top_companies: list[CompanyEngagement]
    consistency: ConsistencyMetrics
    progression: ProgressionSummary
    explainable_insights: list[str]


class EngagementMetricsCalculator:
    """Calculates recruiter, hiring manager, and company-level response rates and engagement metrics."""

    @staticmethod
    def calculate_role_metrics(
        contacts: list[ContactResponse], events: list[OutreachEvent], role: ContactType
    ) -> RecruiterHMAnalytics:
        """Calculates engagement analytics for a specific contact role (e.g. Recruiter, HM)."""
        role_contacts = {c.id for c in contacts if c.contact_type == role}
        role_events = [e for e in events if e.contact_id in role_contacts]

        sent_count = sum(
            1 for e in role_events
            if e.status in {OutreachStatus.SENT, OutreachStatus.REPLIED, OutreachStatus.IGNORED}
            or e.completed_at is not None
        )
        replied_count = sum(
            1 for e in role_events
            if e.status == OutreachStatus.REPLIED
            or (e.outcome is not None and e.outcome != InteractionOutcome.NO_RESPONSE)
        )

        response_rate = float(replied_count) / sent_count if sent_count > 0 else 0.0
        return RecruiterHMAnalytics(
            total_contacts=len(role_contacts),
            response_rate=response_rate,
            replied_count=replied_count,
            sent_count=sent_count,
        )

    @staticmethod
    def calculate_company_metrics(
        contacts: list[ContactResponse], events: list[OutreachEvent]
    ) -> list[CompanyEngagement]:
        """Groups engagement metrics by company name, normalized."""
        company_groups: dict[str, list[ContactResponse]] = {}
        for c in contacts:
            if c.company:
                comp = c.company.strip()
                if comp:
                    company_groups.setdefault(comp, []).append(c)

        company_engagement = []
        for company, comp_contacts in company_groups.items():
            contact_ids = {c.id for c in comp_contacts}
            comp_events = [e for e in events if e.contact_id in contact_ids]

            sent = sum(
                1 for e in comp_events
                if e.status in {OutreachStatus.SENT, OutreachStatus.REPLIED, OutreachStatus.IGNORED}
                or e.completed_at is not None
            )
            replies = sum(
                1 for e in comp_events
                if e.status == OutreachStatus.REPLIED
                or (e.outcome is not None and e.outcome != InteractionOutcome.NO_RESPONSE)
            )

            response_rate = float(replies) / sent if sent > 0 else 0.0

            # Resolve last interaction
            last_date = None
            if comp_events:
                dates = [e.completed_at or e.created_at for e in comp_events if e.completed_at or e.created_at]
                if dates:
                    last_date = max(dates)

            company_engagement.append(
                CompanyEngagement(
                    company_name=company,
                    total_contacts=len(comp_contacts),
                    total_outreach=len(comp_events),
                    response_rate=response_rate,
                    last_interaction=last_date,
                )
            )

        # Sort top companies by contacts count, then outreach volume descending
        company_engagement.sort(key=lambda c: (c.total_contacts, c.total_outreach), reverse=True)
        return company_engagement

    @staticmethod
    def calculate_progression_summary(contacts: list[ContactResponse]) -> ProgressionSummary:
        """Summarizes current state distribution and active vs stale classifications."""
        status_counts: dict[str, int] = {}
        active_count = 0
        stale_count = 0

        for c in contacts:
            status_counts[c.status.value] = status_counts.get(c.status.value, 0) + 1
            if c.status == RelationshipStatus.STALE:
                stale_count += 1
            elif c.status != RelationshipStatus.ARCHIVED:
                active_count += 1

        return ProgressionSummary(
            status_counts=status_counts,
            active_count=active_count,
            stale_count=stale_count,
        )


class FollowupEffectivenessAnalyzer:
    """Analyzes the correlation between recommended follow-up dates and subsequent user interaction events."""

    @staticmethod
    def analyze_followup_effectiveness(
        session: Session, events: list[OutreachEvent]
    ) -> FollowupEffectiveness:
        """Determines what percentage of follow-ups were acted upon and resulted in positive outcomes."""
        recs = session.query(FollowupRecommendationModel).all()
        if not recs:
            return FollowupEffectiveness(
                total_recommended=0,
                total_acted_upon=0,
                action_rate=0.0,
                positive_response_rate=0.0,
            )

        events_by_contact: dict[str, list[OutreachEvent]] = {}
        for e in events:
            events_by_contact.setdefault(str(e.contact_id), []).append(e)

        acted_upon_count = 0
        positive_count = 0

        for r in recs:
            c_events = events_by_contact.get(r.contact_id, [])
            acted_upon = False
            positive_outcome = False

            # Check if an outreach occurred within 7 days after the recommended suggested_date
            limit_date = r.suggested_date.replace(tzinfo=None) + timedelta(days=7)
            for e in c_events:
                e_date = (e.completed_at or e.created_at).replace(tzinfo=None)
                if r.suggested_date.replace(tzinfo=None) <= e_date <= limit_date:
                    acted_upon = True
                    if e.outcome == InteractionOutcome.POSITIVE:
                        positive_outcome = True
                    break

            if acted_upon:
                acted_upon_count += 1
            if positive_outcome:
                positive_count += 1

        action_rate = float(acted_upon_count) / len(recs)
        positive_rate = float(positive_count) / acted_upon_count if acted_upon_count > 0 else 0.0

        return FollowupEffectiveness(
            total_recommended=len(recs),
            total_acted_upon=acted_upon_count,
            action_rate=action_rate,
            positive_response_rate=positive_rate,
        )


class NetworkingTrendAnalyzer:
    """Analyzes weekly outreach consistency index over a trailing timeline window."""

    @staticmethod
    def analyze_consistency(events: list[OutreachEvent], weeks_limit: int = 12) -> ConsistencyMetrics:
        """Computes weekly outreach distribution and trailing consistency index."""
        outreach_by_week: dict[str, int] = {}
        for e in events:
            dt = e.completed_at or e.created_at
            if dt:
                year, week, _ = dt.isocalendar()
                week_key = f"{year}-W{week:02d}"
                outreach_by_week[week_key] = outreach_by_week.get(week_key, 0) + 1

        # CheckTrailing Weeks
        now_dt = datetime.now(UTC).replace(tzinfo=None)
        active_weeks = 0
        for i in range(weeks_limit):
            dt = now_dt - timedelta(weeks=i)
            year, week, _ = dt.isocalendar()
            week_key = f"{year}-W{week:02d}"
            if outreach_by_week.get(week_key, 0) > 0:
                active_weeks += 1

        weekly_consistency_index = float(active_weeks) / weeks_limit
        return ConsistencyMetrics(
            outreach_by_week=outreach_by_week,
            weekly_consistency_index=weekly_consistency_index,
        )


class RelationshipAnalyticsService:
    """Orchestrates structured, deterministic reports on professional relationship health."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.contact_repo = ContactRepository(session)
        self.event_repo = OutreachEventRepository(session)
        self.followup_repo = FollowupRepository(session)

    def _generate_insights(
        self,
        recruiter: RecruiterHMAnalytics,
        hm: RecruiterHMAnalytics,
        consistency: ConsistencyMetrics,
        progression: ProgressionSummary,
    ) -> list[str]:
        """Generates list of explainable text insights based on aggregated statistics."""
        insights = []

        # 1. Response Rate comparison
        if recruiter.sent_count > 0 or hm.sent_count > 0:
            insights.append(
                f"Recruiter response rate is {recruiter.response_rate:.0%} (on {recruiter.sent_count} sent) "
                f"vs Hiring Manager response rate of {hm.response_rate:.0%} (on {hm.sent_count} sent)."
            )

        # 2. Staleness insight
        total = progression.active_count + progression.stale_count
        if total > 0:
            stale_pct = float(progression.stale_count) / total
            if stale_pct > 0.4:
                insights.append(
                    f"Warning: {stale_pct:.0%} of your active contacts are currently classified as Stale. "
                    "Prioritize follow-ups to warm up cold relationships."
                )
            else:
                insights.append(f"Relationship health is solid: only {stale_pct:.0%} of contacts are currently Stale.")

        # 3. Consistency index insight
        ci = consistency.weekly_consistency_index
        if ci >= 0.8:
            insights.append(f"Excellent networking consistency! You were active in {ci:.0%} of trailing trailing weeks.")
        elif ci >= 0.5:
            insights.append(f"Moderate consistency index of {ci:.0%}. Try establishing a regular weekly outreach routine.")
        else:
            insights.append(f"Outreach frequency is low: active in only {ci:.0%} of trailing weeks. Increase weekly outreach volume.")

        return insights

    def generate_summary(self, weeks_limit: int = 12) -> RelationshipAnalyticsSummary:
        """Compiles the complete relationship metrics report in exactly two SQL queries."""
        contacts_models = self.session.query(RelationshipContactModel).all()
        contacts = [self.contact_repo._to_schema(c) for c in contacts_models]

        events_models = self.session.query(OutreachEventModel).all()
        events = [self.event_repo._to_schema(e) for e in events_models]

        # 1. Role engagement metrics
        recruiter_metrics = EngagementMetricsCalculator.calculate_role_metrics(
            contacts, events, ContactType.RECRUITER
        )
        hm_metrics = EngagementMetricsCalculator.calculate_role_metrics(
            contacts, events, ContactType.HIRING_MANAGER
        )

        # 2. Company engagement metrics
        top_companies = EngagementMetricsCalculator.calculate_company_metrics(contacts, events)

        # 3. Progression summary
        progression = EngagementMetricsCalculator.calculate_progression_summary(contacts)

        # 4. Followup effectiveness
        followup_eff = FollowupEffectivenessAnalyzer.analyze_followup_effectiveness(self.session, events)

        # 5. Consistency
        consistency = NetworkingTrendAnalyzer.analyze_consistency(events, weeks_limit)

        # 6. Overall NetworkingMetrics
        active_statuses = {
            RelationshipStatus.ACTIVE,
            RelationshipStatus.WARM,
            RelationshipStatus.RESPONDED,
            RelationshipStatus.CONTACTED,
            RelationshipStatus.FOLLOWUP_PENDING,
        }
        active_conversations = sum(1 for c in contacts if c.status in active_statuses)

        sent_all = sum(
            1 for e in events
            if e.status in {OutreachStatus.SENT, OutreachStatus.REPLIED, OutreachStatus.IGNORED}
            or e.completed_at is not None
        )
        replies_all = sum(
            1 for e in events
            if e.status == OutreachStatus.REPLIED
            or (e.outcome is not None and e.outcome != InteractionOutcome.NO_RESPONSE)
        )
        response_rate = (float(replies_all) / sent_all * 100.0) if sent_all > 0 else 0.0

        meetings_booked = sum(1 for e in events if e.outcome == InteractionOutcome.POSITIVE)

        fourteen_days_ago = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=14)
        recent_activity_count = sum(
            1 for e in events
            if (e.completed_at or e.created_at).replace(tzinfo=None) >= fourteen_days_ago
        )

        overall_metrics = NetworkingMetrics(
            total_contacts=len(contacts),
            active_conversations=active_conversations,
            response_rate_percent=response_rate,
            meetings_booked=meetings_booked,
            recent_activity_count=recent_activity_count,
        )

        # Generate Explainable Insights
        explainable_insights = self._generate_insights(
            recruiter_metrics, hm_metrics, consistency, progression
        )

        return RelationshipAnalyticsSummary(
            overall_metrics=overall_metrics,
            recruiter_metrics=recruiter_metrics,
            hiring_manager_metrics=hm_metrics,
            followup_effectiveness=followup_eff,
            top_companies=top_companies,
            consistency=consistency,
            progression=progression,
            explainable_insights=explainable_insights,
        )
