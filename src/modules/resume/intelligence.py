import logging
import re
import typing
from datetime import datetime
from typing import Any

from src.core.ai.client import ai_client
from src.core.prompts import prompt_manager
from src.modules.resume.schemas import (
    ATSOptimization,
    EmphasizedSkill,
    PositioningRecommendation,
    PrioritizedExperience,
    TailoringStrategyResponse,
)

logger = logging.getLogger("src.modules.resume.intelligence")

def _get_val(obj: Any, field: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(field, default)
    return getattr(obj, field, default)

class ExperiencePrioritizer:
    """Prioritization logic for candidate work experiences.
    Computes scores based on recency, tenure, enterprise alignment, and job domain keywords.
    """
    @staticmethod
    def parse_year(date_str: str | None) -> int | None:
        """Extracts a 4-digit year from a date string (e.g. '2024-05' or 'May 2020' or '2020')."""
        if not date_str:
            return None
        date_clean = str(date_str).strip().lower()
        if date_clean in ('present', 'current', 'none', ''):
            return datetime.utcnow().year
        match = re.search(r'\b(19\d{2}|20\d{2})\b', date_clean)
        if match:
            return int(match.group(1))
        return None

    def _score_experience(self, title: str, company: str, description: str, start_date: str, end_date: str | None, job_title: str, job_domain: str) -> tuple[float, list[str]]:
        """Calculates the relevance score and justifications for a single work experience entry."""
        score = 100.0
        justifications = []

        end_year = self.parse_year(end_date)
        current_year = datetime.utcnow().year

        if not end_date or str(end_date).strip().lower() in ('present', 'current', 'none', ''):
            score += 30.0
            justifications.append('Current experience (maximum recency)')
        elif end_year is not None:
            years_ago = max(0, current_year - end_year)
            penalty = years_ago * 5.0
            score -= penalty
            if penalty > 0:
                justifications.append(f'Completed {years_ago} year(s) ago (recency adjustment applied)')
        else:
            score -= 10.0
            justifications.append('Older experience (default recency adjustment)')

        start_year = self.parse_year(start_date)
        end_year_val = end_year if end_year else current_year
        if start_year is not None:
            tenure = max(1, end_year_val - start_year)
            tenure_bonus = min(tenure * 5.0, 20.0)
            score += tenure_bonus
            justifications.append(f'Solid tenure of {tenure} year(s)')

        enterprise_keywords = ('enterprise', 'corp', 'corporation', 'global', 'sap', 'oracle', 'salesforce', 'cloud', 'saas', 'integration', 'scale', 'infrastructure')
        title_lower = title.lower()
        company_lower = company.lower()
        desc_lower = description.lower()

        ent_matches = [kw for kw in enterprise_keywords if kw in company_lower or kw in title_lower or kw in desc_lower]
        if ent_matches:
            score += 20.0
            justifications.append(f"Enterprise scale alignment (matches: {', '.join(ent_matches[:2])})")

        domain_keywords_map = {
            'Technology': ['software', 'developer', 'engineering', 'python', 'javascript', 'code', 'programming', 'system', 'architecture'],
            'AI/Analytics': ['ai', 'machine learning', 'data', 'analytics', 'statistics', 'model', 'python', 'nlp', 'llm', 'intelligence'],
            'Procurement': ['procurement', 'sourcing', 'purchasing', 'vendor', 'contract', 'negotiation', 'rfp', 'buyer'],
            'Supply Chain': ['supply chain', 'logistics', 'warehouse', 'inventory', 'operations', 'shipping', 'distribution', 'planning']
        }
        domain_kws = domain_keywords_map.get(job_domain, domain_keywords_map['Technology'])
        matched_domain_kws = [kw for kw in domain_kws if kw in title_lower or kw in desc_lower]
        if matched_domain_kws:
            domain_bonus = min(len(matched_domain_kws) * 15.0, 45.0)
            score += domain_bonus
            justifications.append(f"Domain keywords matched: {', '.join(matched_domain_kws[:3])}")

        job_title_words = set(re.findall(r'\w+', job_title.lower()))
        stops = {'coordinator', 'senior', 'engineer', 'junior', 'specialist', 'manager', 'developer', 'lead'}
        job_keywords = job_title_words - stops
        role_keywords = set(re.findall(r'\w+', title_lower))
        matched_title_kws = job_keywords.intersection(role_keywords)
        if matched_title_kws:
            score += 15.0
            justifications.append(f"Title alignment on keyword(s): {', '.join(matched_title_kws)}")

        return score, justifications

    def prioritize_work_experience(self, experiences: list[typing.Any], job_title: str, job_domain: str) -> list[PrioritizedExperience]:
        """Analyzes a list of user experiences against the job specifications and domain,
        scores them, and sorts/classifies them into relevance priority bands.
        """
        prioritized = []
        for exp in experiences:
            title = _get_val(exp, 'title', '')
            company = _get_val(exp, 'company', '')
            description = _get_val(exp, 'description', '') or ''
            start_date = _get_val(exp, 'start_date', '')
            end_date = _get_val(exp, 'end_date', None)

            score, justifications = self._score_experience(
                title=title, company=company, description=description,
                start_date=start_date, end_date=end_date,
                job_title=job_title, job_domain=job_domain
            )
            final_score = max(0.0, min(score, 200.0))
            if final_score >= 135.0:
                band = 'HIGH'
            elif final_score >= 110.0:
                band = 'MEDIUM'
            else:
                band = 'LOW'

            justification_str = '; '.join(justifications) if justifications else 'Standard work experience entry'
            prioritized.append(
                PrioritizedExperience(
                    title=title,
                    company=company,
                    priority_score=round(final_score, 2),
                    priority_band=band,
                    justification=justification_str
                )
            )

        prioritized.sort(key=lambda x: x.priority_score, reverse=True)
        return prioritized

class ResumeExplanationLayer:
    """Generates human-readable, recruiter-style summaries, tailoring strategy descriptions,
    and structured justifications for tailoring decisions.
    """
    def generate_overall_summary(self, job_title: str, company: str, overall_score: float, recommendation: str) -> str:
        """Synthesizes a short recruiter-style overview of candidate fit."""
        rec_label = str(recommendation).replace('_', ' ').title()
        return f"The candidate has a '{rec_label}' fit for the '{job_title}' role at {company} (Opportunity Score: {overall_score}%). Tailoring the resume is recommended to emphasize enterprise alignment and address specific skill gaps."

    def generate_tailoring_strategy_text(self, job_domain: str, prioritized_exps: list[PrioritizedExperience], emphasized_skills: list[EmphasizedSkill]) -> str:
        """Builds a cohesive strategy description detailing resume modification steps."""
        high_priority = [e.title for e in prioritized_exps if e.priority_band == 'HIGH']
        critical_skills = [s.skill_name for s in emphasized_skills if s.importance == 'CRITICAL' and s.user_possesses]
        missing_skills = [s.skill_name for s in emphasized_skills if not s.user_possesses]

        strategy = f"To optimize alignment for the '{job_domain}' domain:\n"
        if high_priority:
            strategy += f"1. Prioritize details and bullets for your roles as: {', '.join(high_priority[:2])}.\n"
        else:
            strategy += "1. Revise professional experience bullets to highlight enterprise scale and project tenure.\n"

        if critical_skills:
            strategy += f"2. Highlight key matching skills: {', '.join(critical_skills[:3])} in your summary and experience highlights.\n"
        if missing_skills:
            strategy += f"3. Address critical skill gaps: {', '.join(missing_skills[:3])} by listing adjacent projects or active learning sections.\n"

        strategy += "4. Format resume layout utilizing clean headers, standard margins, and clear section dividers to ensure maximum ATS readability."
        return strategy

    def generate_detailed_explanation(self, job_title: str, company: str, overall_score: float, prioritized_exps: list[PrioritizedExperience], emphasized_skills: list[EmphasizedSkill]) -> str:
        """Builds a comprehensive, human-readable justification of all tailoring decisions."""
        high_priority = [e for e in prioritized_exps if e.priority_band == 'HIGH']
        missing_skills = [s.skill_name for s in emphasized_skills if not s.user_possesses]

        explanation_parts = [
            '### Resume Tailoring Justification Report',
            f"We analyzed your profile compatibility against the '{job_title}' position at {company} (Score: {overall_score}%).",
            '\n#### Experience Prioritization:'
        ]

        if high_priority:
            explanation_parts.append(
                f"We identified {len(high_priority)} high-priority role(s) to highlight. Specifically, your experience as '{high_priority[0].title}' at '{high_priority[0].company}' is highly relevant because: {high_priority[0].justification}."
            )
        else:
            explanation_parts.append(
                'No experience entries exceeded the high-relevance threshold. We recommend expanding current/recent role descriptions to incorporate enterprise keyword contexts.'
            )

        explanation_parts.append('\n#### Skill Alignment:')
        if missing_skills:
            explanation_parts.append(
                f"The job requires specific skills missing from your profile: {', '.join(missing_skills[:4])}. Highlighting transferable skills (e.g. adjacent framework experience) is key to passing ATS screening."
            )
        else:
            explanation_parts.append(
                'Your profile contains matches for all core technical skills listed in the job intelligence. Focus on optimizing formatting and headline positioning.'
            )

        explanation_parts.append(
            '\n#### ATS & Positioning Strategy:\n- Structure your professional summary with the suggested headline and elevator pitch.\n- Focus experience bullet descriptions on business impacts, using metrics where possible (e.g., efficiency, scope, budget savings).'
        )

        return '\n'.join(explanation_parts)

class ResumeStrategyLayer:
    """Resume strategy generation. Identifies matching and missing skills,
    builds positioning headlines/pitches by domain, and recommends ATS placements.
    """
    def generate_strategy(self, profile_skills: list[str], job_skills: list[str], job_title: str, job_domain: str, job_description: str, positioning_seniority: str | None = None) -> tuple[list[EmphasizedSkill], PositioningRecommendation, ATSOptimization]:
        """Synthesizes the skills matching profile, positioning pitch, and ATS optimization details."""
        emphasized = []
        user_skills_lower = {s.lower() for s in profile_skills}

        matched_skills = [s for s in job_skills if s.lower() in user_skills_lower]
        for skill in matched_skills:
            importance = 'CRITICAL' if skill.lower() in ('python', 'procurement', 'supply chain', 'sql', 'ai') else 'HIGH'
            emphasized.append(
                EmphasizedSkill(
                    skill_name=skill, importance=importance, user_possesses=True,
                    rationale='Direct requirement match. Highlight prominently in the resume skills section and experience bullet points.'
                )
            )

        missing_skills = [s for s in job_skills if s.lower() not in user_skills_lower]
        for skill in missing_skills:
            importance = 'HIGH' if skill.lower() in ('django', 'docker', 'kubernetes', 'sourcing', 'logistics') else 'MEDIUM'
            emphasized.append(
                EmphasizedSkill(
                    skill_name=skill, importance=importance, user_possesses=False,
                    rationale='Core job requirement missing from your profile. Highlight adjacent skills or add target training details.'
                )
            )

        seniority = positioning_seniority if positioning_seniority else 'Experienced'
        seniority_title = seniority.strip().capitalize()

        if job_domain == 'AI/Analytics':
            headline = f"{seniority_title} AI & Advanced Analytics Professional | Specializing in Machine Learning & Enterprise Solutions"
            focus_areas = ['Machine Learning & LLMs', 'Predictive Analytics & Modeling', 'Big Data & Python Engineering']
            pitch = f"A highly specialized {seniority_title.lower()} professional with strong capabilities in Python, machine learning, and enterprise data analytics, focused on implementing scalable AI solutions to drive business automation and intelligence."
        elif job_domain == 'Procurement':
            headline = f"{seniority_title} Procurement & Strategic Sourcing Leader | Optimizing Vendor Management & Spend Analytics"
            focus_areas = ['Strategic Sourcing & RFPs', 'Contract Negotiation & SLAs', 'Vendor Relationship Management']
            pitch = f"A results-driven {seniority_title.lower()} procurement strategist specializing in optimizing strategic sourcing pipelines, contract management, and vendor relationship systems to maximize cost savings and compliance."
        elif job_domain == 'Supply Chain':
            headline = f"{seniority_title} Supply Chain & Operations Strategist | Orchestrating Global Logistics & Inventory Control"
            focus_areas = ['Logistics & Warehouse Operations', 'Inventory Optimization & Planning', 'Supply Chain Risk Management']
            pitch = f"An analytical {seniority_title.lower()} operations leader focused on supply chain optimization, logistics execution, and inventory control systems to build resilient and cost-effective distribution workflows."
        else:
            headline = f"{seniority_title} {job_title} | Enterprise Technology & Systems Architecture Specialist"
            focus_areas = ['Software Systems Design', 'Cloud Infrastructure & SaaS', 'Agile Software Development']
            pitch = f"An experienced {seniority_title.lower()} technology professional skilled in software engineering, system architecture, and modern development frameworks to deliver robust and high-performing enterprise applications."

        pos_rec = PositioningRecommendation(suggested_headline=headline, recommended_focus_areas=focus_areas, positioning_pitch=pitch)

        job_desc_terms = set(re.findall(r'\b[A-Za-z0-9#\-\.+]{2,}\b', job_description))
        possible_keywords = {'SaaS', 'SQL', 'KPIs', 'React', 'Agile', 'Docker', 'Sourcing', 'Kubernetes', 'Java', 'Django', 'Logistics', 'AWS', 'Python', 'SAP', 'ERP', 'Procurement', 'Operations', 'Kafka'}
        extracted_kws = {term for term in job_desc_terms if term in possible_keywords}

        all_target_kws = sorted(set(job_skills) | extracted_kws)
        missing_kws = [kw for kw in all_target_kws if kw.lower() not in user_skills_lower]

        section_recs = {}
        section_recs['Skills Section'] = [
            f"Add matches directly: {', '.join(matched_skills[:5])}" if matched_skills else "List core domain tools.",
            f"Incorporate missing core skills: {', '.join(missing_kws[:3])}" if missing_kws else "No critical skills missing."
        ]
        section_recs['Professional Summary'] = [
            f"Adopt headline: '{headline}'",
            f"Integrate key keywords like: {', '.join(all_target_kws[:3])}"
        ]
        section_recs['Professional Experience'] = [
            f"Mention the job's domain terms ({job_domain}) in your highest priority role description.",
            "Detail project milestones using metrics (e.g. cost savings, performance speedups)."
        ]

        ats_opt = ATSOptimization(
            target_keywords=all_target_kws,
            missing_keywords_to_add=missing_kws,
            resume_section_recommendations=section_recs
        )

        return emphasized, pos_rec, ats_opt

class ResumeIntelligenceEngine:
    """Core engine orchestrating resume tailoring strategy, work experience prioritization,
    ATS keyword optimization, and recruiter-style feedback explanations.
    """
    def __init__(self, prioritizer: ExperiencePrioritizer | None = None, strategy_layer: ResumeStrategyLayer | None = None, explanation_layer: ResumeExplanationLayer | None = None) -> None:
        self.prioritizer = prioritizer or ExperiencePrioritizer()
        self.strategy_layer = strategy_layer or ResumeStrategyLayer()
        self.explanation_layer = explanation_layer or ResumeExplanationLayer()

    def generate_tailoring_strategy(self, user_profile: Any, job_intelligence: Any, opportunity_ranking: Any) -> TailoringStrategyResponse:
        """Orchestrates the resume tailoring analysis. Accepts database ORM models,
        Pydantic schemas, or dictionaries for input structures.
        """
        job_title = _get_val(job_intelligence, 'title', 'Target Role')
        job_company = _get_val(job_intelligence, 'company', 'Target Company')
        job_domain = _get_val(job_intelligence, 'domain', 'Technology')
        job_description = _get_val(job_intelligence, 'raw_content', '') or _get_val(job_intelligence, 'description', '') or ''
        job_skills = _get_val(job_intelligence, 'normalized_skills', []) or []

        raw_skills = _get_val(user_profile, 'skills', [])
        if isinstance(raw_skills, str):
            profile_skills = [s.strip() for s in raw_skills.split(',') if s.strip()]
        elif isinstance(raw_skills, list):
            profile_skills = [str(s).strip() for s in raw_skills if str(s).strip()]
        else:
            profile_skills = []

        experiences = _get_val(user_profile, 'experience', []) or []
        positioning = _get_val(user_profile, 'positioning', {})
        seniority = _get_val(positioning, 'seniority_level', 'Experienced')

        overall_score = _get_val(opportunity_ranking, 'overall_score', 70.0)
        recommendation = _get_val(opportunity_ranking, 'recommendation', 'apply')

        prioritized_exps = self.prioritizer.prioritize_work_experience(experiences, job_title, job_domain)
        emphasized_skills, positioning_rec, ats_opt = self.strategy_layer.generate_strategy(
            profile_skills, job_skills, job_title, job_domain, job_description, seniority
        )

        overall_summary = self.explanation_layer.generate_overall_summary(job_title, job_company, overall_score, recommendation)
        tailoring_strategy = self.explanation_layer.generate_tailoring_strategy_text(job_domain, prioritized_exps, emphasized_skills)
        detailed_explanation = self.explanation_layer.generate_detailed_explanation(job_title, job_company, overall_score, prioritized_exps, emphasized_skills)

        return TailoringStrategyResponse(
            overall_alignment_summary=overall_summary,
            resume_tailoring_strategy=tailoring_strategy,
            prioritized_experiences=prioritized_exps,
            emphasized_skills=emphasized_skills,
            positioning_recommendations=positioning_rec,
            ats_optimization=ats_opt,
            explanation=detailed_explanation
        )

def analyze_resume(resume_text: str, job_description: str) -> str:
    """Parses resume text, loads tailor prompt templates, and runs LLM alignment scores."""
    logger.info("Executing resume alignment analysis...")
    try:
        prompt = prompt_manager.load_prompt(
            "resume/tailor_prompt.md", {"resume": resume_text, "job_description": job_description}
        )
    except FileNotFoundError:
        logger.warning(
            "Resume tailor template prompt file not found. Falling back to default string."
        )
        prompt = (
            f"Analyze the following resume against the job description:\n\n"
            f"Resume:\n{resume_text}\n\n"
            f"Job Description:\n{job_description}\n\n"
            f"Provide matching scores, keyword gaps, and suggestions."
        )

    messages = [
        {
            "role": "system",
            "content": "You are an expert career consultant specializing in resume alignment and optimization.",
        },
        {"role": "user", "content": prompt},
    ]

    response = ai_client.generate_chat_response(messages)
    return response
