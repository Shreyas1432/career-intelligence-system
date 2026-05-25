from typing import Any

from src.modules.positioning.schemas import (
    CommunicationDraft,
    OutreachContextResponse,
    OutreachRecommendation,
)

# ------------------------------------------------------------------------------
# Outreach Adaptation Layer
# ------------------------------------------------------------------------------

class OutreachAdaptationLayer:
    """
    Formulates email and messaging drafts tailored to recipient role types
    and connection degrees, enforcing professional, honest communication.
    """

    def generate_draft(
        self,
        recipient: dict[str, Any],
        relationship: dict[str, Any],
        preferences: dict[str, Any],
        opportunity: dict[str, Any] | None,
    ) -> CommunicationDraft:
        """
        Creates communication subject lines and message bodies.
        """
        name = recipient.get("name", "there")
        title = recipient.get("title", "Professional")
        company = recipient.get("company", "your company")
        role_type = recipient.get("role_type", "other").lower()

        degree = relationship.get("connection_degree", "cold").lower()
        past_interactions = relationship.get("past_interactions", []) or []

        channel = preferences.get("channel", "email").lower()
        tone = preferences.get("preferred_tone", "formal").lower()

        role_title = "Opportunities"
        opp_company = company
        skills_str = "relevant technical skills"

        if opportunity:
            role_title = opportunity.get("role_title", "Opportunities")
            opp_company = opportunity.get("company", company)
            reqs = opportunity.get("key_requirements", []) or []
            if reqs:
                skills_str = ", ".join(reqs[:3])

        is_warm = degree in ("1st", "warm") and len(past_interactions) > 0
        past_ref = past_interactions[0] if is_warm else ""

        subject = None
        body = ""

        if role_type == "recruiter":
            subject, body = self._draft_recruiter(
                name, role_title, opp_company, skills_str, is_warm, past_ref, tone
            )
        elif role_type == "engineering_manager":
            subject, body = self._draft_em(
                name, role_title, opp_company, skills_str, is_warm, past_ref, tone
            )
        elif role_type == "consultant_business":
            subject, body = self._draft_business(
                name, role_title, opp_company, skills_str, is_warm, past_ref, tone
            )
        else:
            subject, body = self._draft_generic(name, title, opp_company, is_warm, past_ref, tone)

        if channel == "linkedin":
            subject = None

        return CommunicationDraft(subject=subject, body=body, channel=channel)

    def _draft_recruiter(
        self,
        name: str,
        role_title: str,
        company: str,
        skills_str: str,
        is_warm: bool,
        past_ref: str,
        tone: str,
    ) -> tuple[str | None, str]:
        if is_warm:
            subject = f"Following Up: {role_title} Roles at {company}"
            salutation = "Hi" if tone == "casual" else "Dear"
            body = (
                f"{salutation} {name},\n\n"
                f"Hope you're doing well. It was great connecting with you previously regarding {past_ref}. "
                f"I'm reaching out because I saw a {role_title} opening at {company} that aligns with my background in {skills_str}. "
                f"I wanted to follow up and see if we could connect briefly to discuss if my profile matches your team's current hiring needs.\n\n"
                f"Best regards,\n[Candidate Name]"
            )
        else:
            subject = f"Inquiry: {role_title} Opportunities at {company}"
            salutation = "Hi" if tone == "casual" else "Dear"
            body = (
                f"{salutation} {name},\n\n"
                f"I hope you're having a good week. I'm reaching out because I'm interested in the {role_title} opportunities at {company}. "
                f"I have professional experience in {skills_str} and would love to share my background with you. "
                f"I've attached my resume for your review. Please let me know if you are open to a brief call next week to discuss potential alignment.\n\n"
                f"Best regards,\n[Candidate Name]"
            )
        return subject, body

    def _draft_em(
        self,
        name: str,
        role_title: str,
        company: str,
        skills_str: str,
        is_warm: bool,
        past_ref: str,
        tone: str,
    ) -> tuple[str | None, str]:
        if is_warm:
            subject = f"Connecting: Engineering and Scaling at {company}"
            salutation = "Hi" if tone == "casual" else "Dear"
            body = (
                f"{salutation} {name},\n\n"
                f"Hope you're doing well. Great connecting with you again—I recall our conversation regarding {past_ref}. "
                f"I'm following up as I'm exploring new opportunities in the {skills_str} space. "
                f"I saw your team is hiring for {role_title} and would love to catch up briefly to learn about your current engineering roadmaps.\n\n"
                f"Best regards,\n[Candidate Name]"
            )
        else:
            subject = f"Discussion: Scaling {skills_str} Systems at {company}"
            salutation = "Hi" if tone == "casual" else "Dear"
            body = (
                f"{salutation} {name},\n\n"
                f"I'm reaching out to connect with other engineering leaders. I've been following {company}'s work, "
                f"particularly your development in {skills_str}. I have hands-on experience building scale systems and I'm interested in "
                f"learning about the technical scaling challenges your team is currently solving. "
                f"I would appreciate the chance to ask a few questions about your engineering practices. Please let me know if you are open to a brief virtual coffee.\n\n"
                f"Best regards,\n[Candidate Name]"
            )
        return subject, body

    def _draft_business(
        self,
        name: str,
        role_title: str,
        company: str,
        _skills_str: str,
        is_warm: bool,
        past_ref: str,
        tone: str,
    ) -> tuple[str | None, str]:
        if is_warm:
            subject = f"Operational Sourcing & Process Strategy at {company}"
            salutation = "Hi" if tone == "casual" else "Dear"
            body = (
                f"{salutation} {name},\n\n"
                f"Hope you're doing well. It was great connecting previously regarding {past_ref}. "
                f"I'm reaching out to see if we could follow up. I'm currently looking at operations and strategic management openings "
                f"similar to {role_title} at {company}, and would value your guidance on how your group structures process delivery policies.\n\n"
                f"Best regards,\n[Candidate Name]"
            )
        else:
            subject = f"Operational Efficiency & Strategy at {company}"
            salutation = "Hi" if tone == "casual" else "Dear"
            body = (
                f"{salutation} {name},\n\n"
                f"I hope this email finds you well. I'm reaching out because I'm interested in the operations and strategy team at {company}. "
                f"I specialize in cost reduction, strategic sourcing, and process optimization. I've been following your company's market expansion "
                f"and would love to learn more about your operational goals and frameworks. Please let me know if you are open to a brief chat next week.\n\n"
                f"Best regards,\n[Candidate Name]"
            )
        return subject, body

    def _draft_generic(
        self, name: str, title: str, company: str, is_warm: bool, past_ref: str, tone: str
    ) -> tuple[str | None, str]:
        salutation = "Hi" if tone == "casual" else "Dear"
        if is_warm:
            subject = "Reconnecting: Professional Networking"
            body = (
                f"{salutation} {name},\n\n"
                f"Hope you're doing well. Great connecting with you again—I remember our discussion regarding {past_ref}. "
                f"I wanted to follow up and see how things are going at {company}. I'm currently exploring new directions in my career "
                f"and would love to catch up briefly if you have time next week.\n\n"
                f"Best regards,\n[Candidate Name]"
            )
        else:
            subject = f"Connecting: {title} Opportunities"
            body = (
                f"{salutation} {name},\n\n"
                f"I hope you're having a good week. I'm reaching out to expand my professional network. "
                f"I've been following your work at {company} and would love to connect to learn more about your career journey. "
                f"Please let me know if you have 10 minutes for a brief chat sometime.\n\n"
                f"Best regards,\n[Candidate Name]"
            )
        return subject, body


# ------------------------------------------------------------------------------
# Outreach Recommendation Layer
# ------------------------------------------------------------------------------

class OutreachRecommendationLayer:
    """
    Formulates strategic outreach advice, follow-up cadence, and tone guidelines.
    """

    def generate_recommendations(
        self,
        recipient: dict[str, Any],
        relationship: dict[str, Any],
        preferences: dict[str, Any],
    ) -> tuple[OutreachRecommendation, list[str], list[str]]:
        """
        Generates platform timing, follow-up days, and tone guidelines.
        """
        recipient.get("role_type", "other").lower()
        degree = relationship.get("connection_degree", "cold").lower()
        channel = preferences.get("channel", "email").lower()

        if degree in ("1st", "warm"):
            cadence_days = 5
        else:
            cadence_days = 10

        if channel == "linkedin":
            channel_advice = (
                "Use LinkedIn messaging for a direct connection request. Ensure your profile "
                "is updated prior to sending, as the recipient will review it."
            )
        else:
            channel_advice = (
                "Send via professional email. This is best for formal recruitment inquiries "
                "and sharing attachments such as your resume."
            )

        best_time = "Tuesday or Thursday between 9:00 AM and 11:00 AM (recipient local time)"

        rec = OutreachRecommendation(
            channel_advice=channel_advice,
            best_time_to_send=best_time,
            follow_up_cadence_days=cadence_days,
        )

        tone_guidelines = [
            "Keep the outreach concise and under 150 words.",
            "Avoid manipulative phrasing; state your objective honestly and directly.",
            "Maintain professional respect and avoid casual language (unless connection is close).",
        ]

        follow_ups = [
            f"If there is no response, wait at least {cadence_days} days before sending a single polite follow-up.",
            "Limit your communication to a maximum of two follow-ups. Stop if there is no response to prevent spamming.",
            "Provide a low-friction call-to-action, such as asking for a brief 10-minute conversation.",
        ]

        return rec, tone_guidelines, follow_ups


# ------------------------------------------------------------------------------
# Outreach Context Engine
# ------------------------------------------------------------------------------

def _get_val(obj: Any, field: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(field, default)
    return getattr(obj, field, default)


class OutreachContextEngine:
    """
    Core engine orchestrating networking communications. Adapts messages based on recipient role,
    opportunity details, and enforces professional, non-manipulative communication.
    """

    def __init__(
        self,
        adaptation_layer: OutreachAdaptationLayer | None = None,
        recommendation_layer: OutreachRecommendationLayer | None = None,
    ) -> None:
        self.adaptation_layer = adaptation_layer or OutreachAdaptationLayer()
        self.recommendation_layer = recommendation_layer or OutreachRecommendationLayer()

    def generate_outreach(self, outreach_input: Any) -> OutreachContextResponse:
        """
        Orchestrates the outreach generation process.
        Accepts dicts, Pydantic models, or database ORMs.
        """
        recipient = _get_val(outreach_input, "recipient", {})
        rec_dict = {
            "name": _get_val(recipient, "name", "there"),
            "title": _get_val(recipient, "title", "Professional"),
            "company": _get_val(recipient, "company", "your company"),
            "role_type": _get_val(recipient, "role_type", "other"),
        }

        relationship = _get_val(outreach_input, "relationship", {})
        rel_dict = {
            "connection_degree": _get_val(relationship, "connection_degree", "cold"),
            "past_interactions": _get_val(relationship, "past_interactions", []) or [],
        }

        preferences = _get_val(outreach_input, "preferences", {})
        pref_dict = {
            "channel": _get_val(preferences, "channel", "email"),
            "preferred_tone": _get_val(preferences, "preferred_tone", "formal"),
        }

        opportunity = _get_val(outreach_input, "opportunity", None)
        opp_dict = None
        if opportunity:
            opp_dict = {
                "role_title": _get_val(opportunity, "role_title", "Opportunities"),
                "company": _get_val(opportunity, "company", ""),
                "key_requirements": _get_val(opportunity, "key_requirements", []) or [],
            }

        draft = self.adaptation_layer.generate_draft(
            recipient=rec_dict,
            relationship=rel_dict,
            preferences=pref_dict,
            opportunity=opp_dict,
        )

        self._validate_no_manipulation(
            connection_degree=rel_dict["connection_degree"],
            draft_body=draft.body,
        )

        rec, tone_guidelines, follow_ups = self.recommendation_layer.generate_recommendations(
            recipient=rec_dict,
            relationship=rel_dict,
            preferences=pref_dict,
        )

        explanation = (
            f"The outreach draft was customized for a {rec_dict['role_type']} recipient at {rec_dict['company']}. "
            f"Tone was set to {pref_dict['preferred_tone']} and formatted for {pref_dict['channel']}. "
            f"Follow-up cadence is recommended at {rec.follow_up_cadence_days} days to ensure respectful communication gaps."
        )

        return OutreachContextResponse(
            draft=draft,
            outreach_recommendations=rec,
            tone_recommendations=tone_guidelines,
            follow_up_recommendations=follow_ups,
            explanation=explanation,
        )

    def _validate_no_manipulation(self, connection_degree: str, draft_body: str) -> None:
        """
        Validates that cold outreach drafts do not claim prior familiarity or meetings.
        """
        if connection_degree.lower() == "cold":
            deceptive_phrases = (
                "great connecting",
                "connecting again",
                "good seeing you",
                "remember our meeting",
                "nice to reconnect",
                "we previously discussed",
                "as we discussed",
                "our conversation",
                "reconnect",
            )
            body_lower = draft_body.lower()
            for phrase in deceptive_phrases:
                if phrase in body_lower:
                    raise ValueError(
                        f"Anti-Manipulation Violation: Cold outreach draft contains deceptive phrase "
                        f"'{phrase}' which falsely implies a pre-existing relationship."
                    )
