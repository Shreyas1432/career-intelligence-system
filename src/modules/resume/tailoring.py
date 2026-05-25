import re
import typing
from typing import Any

from src.modules.resume.schemas import (
    ATSMatchMetadata,
    TailoredExperienceItem,
    TailoredResumeResponse,
    TailoredResumeStructure,
)


def _get_val(obj: Any, field: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(field, default)
    return getattr(obj, field, default)

class ResumeTransformationPipeline:
    """Pipeline that prioritizes resume sections, refines bullet points to align with job keywords,
    and guarantees the preservation of all quantified impact metrics.
    """
    @staticmethod
    def extract_metrics(text: str) -> set[str]:
        """Extracts all numeric, monetary, percentage, and multiplier metrics from a string.
        Matches values like: 15%, $50k, 3x, 100,000, 3 years.
        """
        pattern = r'\b\d+[\d,\.]*(?:%|\s*x|\s*k|\s*m|\b)?|\$\d+[\d,\.]*(?:k|m|b)?'
        matches = re.findall(pattern, text, re.IGNORECASE)
        res = set()
        for m in matches:
            val = m.strip()
            if val and not val.startswith('0'):
                res.add(val.lower())
        return res

    def prioritize_experiences(self, base_experiences: list[typing.Any], prioritized_strategy: list[typing.Any]) -> list[typing.Any]:
        """Sorts the base experiences according to their priority band in the resume strategy.
        Priority order: HIGH first, then MEDIUM, then LOW. Unlisted roles default to LOW.
        """
        band_mapping = {}
        for item in prioritized_strategy:
            title = _get_val(item, 'title', '')
            company = _get_val(item, 'company', '')
            band = _get_val(item, 'priority_band', 'LOW')
            key = f"{company.lower()}|{title.lower()}"
            band_mapping[key] = band

        def get_priority_weight(exp: typing.Any) -> int:
            title = _get_val(exp, 'title', '')
            company = _get_val(exp, 'company', '')
            key = f"{company.lower()}|{title.lower()}"
            band = band_mapping.get(key, 'LOW')
            if band == 'HIGH':
                return 0
            elif band == 'MEDIUM':
                return 1
            else:
                return 2

        return sorted(base_experiences, key=get_priority_weight)

    def refine_bullet_points(self, bullet_text: str, target_keywords: list[str]) -> tuple[str, list[str]]:
        """Refines bullet points to include target keywords while avoiding keyword stuffing
        and strictly preserving all quantified metrics.
        """
        original_bullets = []
        for b in bullet_text.split('\n'):
            val = b.strip()
            if val:
                original_bullets.append(b.strip().lstrip('-*• ').strip())

        refined_bullets = []
        keywords_inserted = set()

        rephrasings = {
            'database': 'database (SQL)',
            'querying': 'querying (SQL)',
            'coding': 'software development (Python)',
            'software': 'software engineering (Python)',
            'sourcing': 'strategic sourcing',
            'logistics': 'logistics operations',
            'warehouse': 'warehouse planning',
            'contracts': 'vendor contracts',
            'negotiation': 'commercial negotiation',
            'modeling': 'statistical modeling (Machine Learning)',
            'analytics': 'analytics (Machine Learning)',
        }

        for bullet in original_bullets:
            refined = bullet
            original_metrics = self.extract_metrics(bullet)
            insert_count = 0

            for pattern, replacement in rephrasings.items():
                replacement_kws = [kw for kw in target_keywords if kw.lower() in replacement.lower()]
                if not replacement_kws:
                    continue

                if re.search(r'\b' + re.escape(pattern) + r'\b', refined, re.IGNORECASE):
                    if insert_count < 2:
                        refined_new = re.sub(r'\b' + re.escape(pattern) + r'\b', replacement, refined, flags=re.IGNORECASE)
                        new_metrics = self.extract_metrics(refined_new)
                        if original_metrics.issubset(new_metrics):
                            refined = refined_new
                            insert_count += 1
                            keywords_inserted.update(replacement_kws)

            final_metrics = self.extract_metrics(refined)
            if not original_metrics.issubset(final_metrics):
                refined = bullet

            refined_bullets.append(f"- {refined}")

        consolidated_desc = '\n'.join(refined_bullets)
        clean_bullets = [b.lstrip('-*• ').strip() for b in refined_bullets]
        return consolidated_desc, clean_bullets

class ResumeTailoringValidator:
    """Validates tailored resumes to prevent hallucinated experience/metrics,
    ensure quantified impact integrity, and guard structure boundaries.
    """
    def __init__(self) -> None:
        self.pipeline = ResumeTransformationPipeline()

    def validate_tailored_resume(self, base_resume: Any, tailored_resume: Any) -> None:
        """Validates the tailored resume structure against the original base resume profile.
        Raises ValueError if validation constraints are violated.
        """
        base_name = _get_val(base_resume, 'full_name', '')
        base_email = _get_val(base_resume, 'email', '')

        tailored_name = _get_val(tailored_resume, 'full_name', '')
        tailored_email = _get_val(tailored_resume, 'email', '')

        if base_name != tailored_name:
            raise ValueError(f"Candidate name altered during tailoring: expected '{base_name}', got '{tailored_name}'")

        if base_email != tailored_email:
            raise ValueError(f"Candidate email altered during tailoring: expected '{base_email}', got '{tailored_email}'")

        base_exps = _get_val(base_resume, 'experience', []) or []
        tailored_exps = _get_val(tailored_resume, 'experiences', []) or []

        base_by_key = {}
        for exp in base_exps:
            title = _get_val(exp, 'title', '')
            company = _get_val(exp, 'company', '')
            key = f"{company.lower()}|{title.lower()}"
            base_by_key[key] = exp

        for t_exp in tailored_exps:
            t_title = _get_val(t_exp, 'title', '')
            t_company = _get_val(t_exp, 'company', '')
            key = f"{t_company.lower()}|{t_title.lower()}"

            if key not in base_by_key:
                raise ValueError(f"Anti-Hallucination Violation: New role/company introduced: '{t_title}' at '{t_company}'")

            b_exp = base_by_key[key]
            b_desc = _get_val(b_exp, 'description', '') or ''
            t_desc = _get_val(t_exp, 'description', '') or ''

            b_metrics = self.pipeline.extract_metrics(b_desc)
            t_metrics = self.pipeline.extract_metrics(t_desc)

            missing_metrics = b_metrics - t_metrics
            if missing_metrics:
                raise ValueError(
                    f"Quantified Impact Violation: The metrics {missing_metrics} from role '{t_title}' at '{t_company}' were lost during tailoring."
                )

            new_metrics = t_metrics - b_metrics
            for metric in new_metrics:
                if re.match(r"^\b(19\d{2}|20\d{2})\b$", metric):
                    continue
                raise ValueError(
                    f"Anti-Hallucination Violation: Unverified metric '{metric}' introduced in role '{t_title}' at '{t_company}' during tailoring."
                )

            b_start = _get_val(b_exp, 'start_date', '')
            b_end = _get_val(b_exp, 'end_date', '')
            t_start = _get_val(t_exp, 'start_date', '')
            t_end = _get_val(t_exp, 'end_date', '')

            if b_start != t_start or b_end != t_end:
                raise ValueError(f"Structure Violation: Dates altered for role '{t_title}' at '{t_company}'")

class ResumeTailoringEngine:
    """Orchestrating engine for resume tailoring.
    Assembles, refines, optimizes, and validates tailored resumes based on strategy and scoring.
    """
    def __init__(self, pipeline: ResumeTransformationPipeline | None = None, validator: ResumeTailoringValidator | None = None) -> None:
        self.pipeline = pipeline or ResumeTransformationPipeline()
        self.validator = validator or ResumeTailoringValidator()

    def tailor_resume(self, user_profile: Any, _job_intelligence: Any, resume_strategy: Any, _opportunity_scoring: Any) -> TailoredResumeResponse:
        """Coordinates the transformation pipeline and returns a validated, structured tailored resume.
        Supports database ORM objects, Pydantic schemas, and dictionary inputs.
        """
        full_name = _get_val(user_profile, 'full_name', '')
        email = _get_val(user_profile, 'email', '')
        raw_skills = _get_val(user_profile, 'skills', [])

        if isinstance(raw_skills, str):
            profile_skills = [s.strip() for s in raw_skills.split(',') if s.strip()]
        elif isinstance(raw_skills, list):
            profile_skills = [str(s).strip() for s in raw_skills if str(s).strip()]
        else:
            profile_skills = []

        experiences = _get_val(user_profile, 'experience', []) or []

        headline = _get_val(resume_strategy, 'positioning_recommendations', {})
        suggested_headline = _get_val(headline, 'suggested_headline', 'Professional')
        professional_summary = _get_val(headline, 'positioning_pitch', '')

        prioritized_strategy_list = _get_val(resume_strategy, 'prioritized_experiences', []) or []
        ats_opt = _get_val(resume_strategy, 'ats_optimization', {})
        target_keywords = _get_val(ats_opt, 'target_keywords', []) or []
        missing_skills_list = _get_val(ats_opt, 'missing_keywords_to_add', []) or []

        sorted_experiences = self.pipeline.prioritize_experiences(
            base_experiences=experiences,
            prioritized_strategy=prioritized_strategy_list
        )

        tailored_experiences = []
        for exp in sorted_experiences:
            title = _get_val(exp, 'title', '')
            company = _get_val(exp, 'company', '')
            start_date = _get_val(exp, 'start_date', '')
            end_date = _get_val(exp, 'end_date', None)
            description = _get_val(exp, 'description', '') or ''

            consolidated_desc, clean_bullets = self.pipeline.refine_bullet_points(
                bullet_text=description,
                target_keywords=target_keywords
            )

            tailored_experiences.append(
                TailoredExperienceItem(
                    title=title,
                    company=company,
                    start_date=start_date,
                    end_date=end_date,
                    description=consolidated_desc,
                    bullets=clean_bullets
                )
            )

        profile_skills_lower = {s.lower() for s in profile_skills}
        matched_skills = [kw for kw in target_keywords if kw.lower() in profile_skills_lower]
        other_skills = [s for s in profile_skills if s.lower() not in {m.lower() for m in matched_skills}]
        prioritized_skills = matched_skills + other_skills

        phone = _get_val(user_profile, 'phone', None)
        github = _get_val(user_profile, 'github', None)
        linkedin = _get_val(user_profile, 'linkedin', None)

        tailored_resume = TailoredResumeStructure(
            full_name=full_name,
            email=email,
            suggested_headline=suggested_headline,
            professional_summary=professional_summary,
            experiences=tailored_experiences,
            skills=prioritized_skills,
            phone=phone,
            github=github,
            linkedin=linkedin
        )

        self.validator.validate_tailored_resume(user_profile, tailored_resume)

        total_kws = len(target_keywords)
        matched_kws = sorted(set(matched_skills))
        missing_kws = sorted(set(missing_skills_list))

        if total_kws > 0:
            alignment_ratio = len(matched_kws) / total_kws
            alignment_score = alignment_ratio * 100.0
        else:
            alignment_ratio = 1.0
            alignment_score = 100.0

        ats_metadata = ATSMatchMetadata(
            keyword_alignment_score=round(alignment_score, 2),
            matched_keywords=matched_kws,
            missing_keywords=missing_kws,
            keyword_alignment_ratio=round(alignment_ratio, 4)
        )

        return TailoredResumeResponse(
            tailored_resume=tailored_resume,
            ats_metadata=ats_metadata,
            missing_skill_suggestions=missing_kws
        )
