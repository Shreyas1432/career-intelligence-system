import re
from datetime import UTC, datetime
from uuid import UUID

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.modules.relationship.repositories import (
    CommunicationProfileRepository,
    ContactRepository,
    OutreachEventRepository,
)
from src.modules.relationship.schemas import (
    CommunicationProfile,
    CommunicationStyle,
    ContactResponse,
    ContactType,
    InteractionOutcome,
    OutreachEvent,
    OutreachStatus,
)


class CommunicationGuidance(BaseModel):
    """Professional communication guidance compiled from style, type, and history."""
    contact_id: UUID
    style_preference: str = Field(description="Communication style preference (concise or detailed)")
    orientation: str = Field(description="Communication orientation (technical or business)")
    tone_guidance: list[str] = Field(default_factory=list, description="Tone adaptation guidelines")
    continuity_recommendations: list[str] = Field(default_factory=list, description="Suggestions for interaction continuity")
    context_hints: list[str] = Field(default_factory=list, description="Contextual outreach hints")


class CommunicationStyleAnalyzer:
    """Deterministic analyzer for communication preferences from history."""

    @staticmethod
    def _calculate_engagement(events: list[OutreachEvent]) -> tuple[float, int, int]:
        """Calculates engagement score and outreach metrics."""
        # Total attempts: status is SENT, REPLIED, IGNORED, or completed_at is set
        total = sum(
            1 for e in events
            if e.status in {OutreachStatus.SENT, OutreachStatus.REPLIED, OutreachStatus.IGNORED}
            or e.completed_at is not None
        )
        if total == 0:
            return 0.0, 0, 0

        replies = sum(
            1 for e in events
            if e.status == OutreachStatus.REPLIED
            or (e.outcome is not None and e.outcome != InteractionOutcome.NO_RESPONSE)
        )
        score = float(replies) / float(total)
        return min(max(score, 0.0), 1.0), total, replies

    @staticmethod
    def _determine_style_and_pref(events: list[OutreachEvent]) -> tuple[CommunicationStyle, str]:
        """Analyzes event content lengths and keywords to determine style and preference."""
        contents = [e.content for e in events if e.content]
        if not contents:
            return CommunicationStyle.UNKNOWN, "concise"

        total_len = sum(len(c) for c in contents)
        avg_len = float(total_len) / len(contents)
        pref = "concise" if avg_len < 150.0 else "detailed"

        combined = " ".join(contents).lower()
        formal_words = ["dear", "sincerely", "regards", "respectfully", "hope this email finds you well", "would appreciate", "mr.", "ms."]
        casual_words = ["hey", "thanks", "cheers", "cool", "awesome", "great", "chat"]
        analytical_words = ["analytics", "metrics", "data", "performance", "numbers", "percent", "architecture", "scale", "system", "statistics"]

        formal_count = sum(combined.count(w) for w in formal_words)
        casual_count = sum(combined.count(w) for w in casual_words)
        casual_count += len(re.findall(r"\bhi\b", combined))
        analytical_count = sum(combined.count(w) for w in analytical_words)

        if avg_len < 100.0:
            return CommunicationStyle.DIRECT, pref

        if analytical_count > formal_count and analytical_count > casual_count:
            return CommunicationStyle.ANALYTICAL, pref
        if casual_count > formal_count:
            return CommunicationStyle.CASUAL, pref
        if formal_count > 0 or casual_count > 0 or analytical_count > 0:
            return CommunicationStyle.FORMAL, pref

        return CommunicationStyle.DIRECT, pref

    @staticmethod
    def _determine_orientation(contact: ContactResponse, events: list[OutreachEvent]) -> str:
        """Determines orientation (technical vs business) from content keywords or contact type."""
        contents = [e.content for e in events if e.content]
        if not contents:
            if contact.contact_type == ContactType.HIRING_MANAGER:
                return "technical"
            return "business"

        combined = " ".join(contents).lower()
        tech_keywords = [
            "python", "java", "c++", "rust", "typescript", "javascript", "react", "aws", "gcp",
            "azure", "docker", "kubernetes", "sql", "api", "database", "backend", "frontend",
            "engineering", "technical", "stack", "system", "architecture", "ci/cd", "git",
            "developer", "development"
        ]
        bus_keywords = [
            "schedule", "call", "zoom", "meet", "meeting", "salary", "compensation", "benefits",
            "hiring", "process", "resume", "recruit", "recruiter", "interview", "availability",
            "time", "rate", "role", "position", "offer", "contract", "logistics", "hr", "talent"
        ]

        tech_count = sum(len(re.findall(r"\b" + re.escape(w) + r"\b", combined)) for w in tech_keywords)
        bus_count = sum(len(re.findall(r"\b" + re.escape(w) + r"\b", combined)) for w in bus_keywords)

        if tech_count > bus_count:
            return "technical"
        if bus_count > tech_count:
            return "business"

        if contact.contact_type == ContactType.HIRING_MANAGER:
            return "technical"
        return "business"

    @staticmethod
    def _determine_channel_and_time(events: list[OutreachEvent]) -> tuple[str | None, str | None]:
        """Infers preferred channel and best time to reach from interaction history."""
        if not events:
            return None, None

        replies = [
            e for e in events
            if e.status == OutreachStatus.REPLIED
            or (e.outcome is not None and e.outcome != InteractionOutcome.NO_RESPONSE)
        ]

        # Preferred Channel
        channel_candidates = [r.method for r in replies if r.method] if replies else [e.method for e in events if e.method]
        preferred_channel = max(set(channel_candidates), key=channel_candidates.count) if channel_candidates else None

        # Best Time to Reach
        time_candidates = replies if replies else events
        times = []
        for e in time_candidates:
            dt = e.completed_at or e.created_at
            if dt:
                day = "Weekday" if dt.weekday() < 5 else "Weekend"
                hour = dt.hour
                if 9 <= hour < 12:
                    tod = "Mornings"
                elif 12 <= hour < 17:
                    tod = "Afternoons"
                else:
                    tod = "Evenings"
                times.append(f"{day} {tod}")

        best_time = max(set(times), key=times.count) if times else None
        return preferred_channel, best_time

    @classmethod
    def analyze_style_patterns(
        cls, contact: ContactResponse, outreach_events: list[OutreachEvent]
    ) -> tuple[CommunicationStyle, str, str, float, str | None, str | None]:
        """
        Runs deterministic rule-based analysis of style patterns.
        Returns:
            (style, style_preference, orientation, engagement_score, best_time, preferred_channel)
        """
        style, style_pref = cls._determine_style_and_pref(outreach_events)
        orientation = cls._determine_orientation(contact, outreach_events)
        engagement_score, _, _ = cls._calculate_engagement(outreach_events)
        pref_channel, best_time = cls._determine_channel_and_time(outreach_events)

        return style, style_pref, orientation, engagement_score, best_time, pref_channel


class ToneRecommendationEngine:
    """Generates professional tone guidelines adapted to context."""

    @staticmethod
    def generate_tone_guidance(contact: ContactResponse, style_pref: str, orientation: str) -> list[str]:
        """Generates list of explainable tone recommendations based on deterministic rules."""
        guidance = []

        # 1. Contact Type adaptation
        if contact.contact_type == ContactType.RECRUITER:
            guidance.append(
                "Recruiter Focus: Emphasize availability, timeline constraints, and alignment on recruiting process parameters."
            )
            guidance.append(
                "Tone: Warm, enthusiastic, and highly responsive. Confirm logistics and calendar links promptly."
            )
        elif contact.contact_type == ContactType.HIRING_MANAGER:
            guidance.append(
                "Hiring Manager Focus: Highlight domain expertise, direct team contributions, and past project impact."
            )
            guidance.append(
                "Tone: Objective, value-first, and highly professional. Avoid generic greetings and focus directly on role alignment."
            )
        else:
            guidance.append(
                "Professional Focus: Keep updates high-level, clear, and structured around mutual professional context."
            )

        # 2. Style Preference adaptation
        if style_pref == "concise":
            guidance.append(
                "Structure: Prefer bullet points and short paragraphs (2-3 sentences max) to ensure easy scannability."
            )
        else:
            guidance.append(
                "Structure: Provide complete background context and back up claims with brief project references or metrics."
            )

        # 3. Orientation adaptation
        if orientation == "technical":
            guidance.append(
                "Technical Tone: Incorporate precise technical terms, frameworks, and architecture patterns relevant to the role."
            )
        else:
            guidance.append(
                "Business Tone: Frame achievements in terms of business impact, schedule delivery, and cross-functional coordination."
            )

        return guidance


class OutreachContextBuilder:
    """Helper to structure context-aware continuity highlights from history."""

    @staticmethod
    def build_continuity_context(outreach_events: list[OutreachEvent]) -> dict[str, list[str]]:
        """Looks at interaction history to suggest context-aware continuity recommendations."""
        recommendations = []
        hints = []

        if not outreach_events:
            recommendations.append("Initiate first cold outreach focusing on establishing connection and introducing your background.")
            hints.append("No previous interaction history found. Keep the introduction simple and clear.")
            return {"continuity_recommendations": recommendations, "context_hints": hints}

        # Sort chronologically to get the latest completed event
        completed_events = sorted(
            [e for e in outreach_events if e.completed_at or e.created_at],
            key=lambda e: (e.completed_at or e.created_at).replace(tzinfo=None),
            reverse=True
        )

        if not completed_events:
            recommendations.append("Review planned outreach draft and schedule message transmission.")
            hints.append("No completed outreach event found. Set up initial contact parameters.")
            return {"continuity_recommendations": recommendations, "context_hints": hints}

        latest = completed_events[0]
        date_str = (latest.completed_at or latest.created_at).strftime("%Y-%m-%d")

        if latest.outcome == InteractionOutcome.POSITIVE:
            recommendations.append(f"Acknowledge and build upon the positive feedback received on {date_str}.")
            hints.append("Express appreciation for their responsiveness and move to the next clear action item.")
        elif latest.outcome == InteractionOutcome.ACTION_REQUIRED:
            recommendations.append(f"Address the pending action items or questions raised in the conversation on {date_str}.")
            hints.append("Double-check all requested information is fully answered before sending.")
        elif latest.outcome == InteractionOutcome.NEUTRAL:
            recommendations.append(f"Maintain rapport by following up on the neutral exchange on {date_str}.")
            hints.append("Check if they require any additional materials, resumes, or context about your profile.")
        elif latest.outcome == InteractionOutcome.NEGATIVE:
            recommendations.append(f"Respectfully pivot the discussion or check in at a later date based on feedback on {date_str}.")
            hints.append("Acknowledge any rejection/redirection professionally, leaving a positive final impression.")
        else:
            # If outcome is NO_RESPONSE or None (meaning sent but ignored)
            delta = datetime.now(UTC).replace(tzinfo=None) - (latest.completed_at or latest.created_at).replace(tzinfo=None)
            if delta.days >= 3:
                recommendations.append(f"Send a polite nudge following up on the unanswered message sent on {date_str} ({delta.days} days ago).")
                hints.append("Remind them briefly of the initial request without expressing frustration.")
            else:
                recommendations.append("Wait for a response before sending additional follow-up messages.")
                hints.append("Outreach is still fresh. Avoid over-communicating too quickly.")

        return {"continuity_recommendations": recommendations, "context_hints": hints}


class CommunicationProfileService:
    """Orchestrates communication intelligence profile synchronization and guidance generation."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.contact_repo = ContactRepository(session)
        self.profile_repo = CommunicationProfileRepository(session)
        self.outreach_repo = OutreachEventRepository(session)

    def get_or_create_profile(self, contact_id: UUID) -> CommunicationProfile:
        """Fetches or initializes a communication profile for the contact."""
        profile = self.profile_repo.get_profile(contact_id)
        if not profile:
            profile = CommunicationProfile(
                contact_id=contact_id,
                style=CommunicationStyle.UNKNOWN,
                best_time_to_reach=None,
                preferred_channel=None,
                engagement_score=0.0,
                insights=[]
            )
            profile = self.profile_repo.update_profile(profile)
        return profile

    def analyze_and_sync_profile(self, contact_id: UUID) -> CommunicationProfile:
        """Orchestrates the rules-based style and engagement calculations, syncing the profile."""
        contact = self.contact_repo.get_contact_by_uuid(contact_id)
        if not contact:
            raise ValueError(f"Contact with ID {contact_id} not found.")

        events = list(self.outreach_repo.get_events_for_contact(contact_id))

        # Style Pattern Analysis
        style, style_pref, orientation, engagement_score, best_time, pref_channel = (
            CommunicationStyleAnalyzer.analyze_style_patterns(contact, events)
        )

        # Engagement score calculation details
        total = sum(
            1 for e in events
            if e.status in {OutreachStatus.SENT, OutreachStatus.REPLIED, OutreachStatus.IGNORED}
            or e.completed_at is not None
        )
        replies = sum(
            1 for e in events
            if e.status == OutreachStatus.REPLIED
            or (e.outcome is not None and e.outcome != InteractionOutcome.NO_RESPONSE)
        )

        # Generate Explainable Insights
        insights = []
        insights.append(f"Inferred communication style: {style.value.upper()}")
        insights.append(f"Prefers {style_pref} communication structures.")
        insights.append(f"Primary message orientation: {orientation}.")
        insights.append(f"Engagement score of {engagement_score:.0%} based on {replies} replies out of {total} events.")

        if pref_channel:
            insights.append(f"Preferred channel: {pref_channel.title()}.")
        if best_time:
            insights.append(f"Best time to contact: {best_time}.")

        # Add profile-specific rules
        if contact.contact_type == ContactType.RECRUITER:
            insights.append("Logistics-heavy: Focus outreach on scheduling availability and compensation.")
        elif contact.contact_type == ContactType.HIRING_MANAGER:
            insights.append("Engineering-heavy: Emphasize technology alignment and problem-solving capabilities.")

        # Update and save profile
        profile = CommunicationProfile(
            contact_id=contact_id,
            style=style,
            best_time_to_reach=best_time,
            preferred_channel=pref_channel,
            engagement_score=engagement_score,
            insights=insights
        )
        return self.profile_repo.update_profile(profile)

    def generate_guidance(self, contact_id: UUID) -> CommunicationGuidance:
        """Computes and returns the complete, explainable CommunicationGuidance."""
        contact = self.contact_repo.get_contact_by_uuid(contact_id)
        if not contact:
            raise ValueError(f"Contact with ID {contact_id} not found.")

        # Sync profile to ensure up-to-date analysis
        self.analyze_and_sync_profile(contact_id)

        events = list(self.outreach_repo.get_events_for_contact(contact_id))

        # Analyze style preference and orientation
        _, style_pref, orientation, _, _, _ = (
            CommunicationStyleAnalyzer.analyze_style_patterns(contact, events)
        )

        # Tone recommendations
        tone_guidance = ToneRecommendationEngine.generate_tone_guidance(contact, style_pref, orientation)

        # Continuity recommendations
        continuity = OutreachContextBuilder.build_continuity_context(events)

        return CommunicationGuidance(
            contact_id=contact_id,
            style_preference=style_pref,
            orientation=orientation,
            tone_guidance=tone_guidance,
            continuity_recommendations=continuity["continuity_recommendations"],
            context_hints=continuity["context_hints"]
        )
