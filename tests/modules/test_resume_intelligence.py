from src.modules.resume_intelligence.engine import ResumeIntelligenceEngine
from src.modules.resume_intelligence.explanation import ResumeExplanationLayer
from src.modules.resume_intelligence.prioritization import ExperiencePrioritizer
from src.modules.resume_intelligence.schemas import TailoringStrategyResponse
from src.modules.resume_intelligence.strategy import ResumeStrategyLayer


def test_experience_prioritization_rules() -> None:
    """
    Verifies that the experience prioritizer scores and categorizes work experience
    based on recency, tenure, enterprise alignment, and domain keyword matches.
    """
    prioritizer = ExperiencePrioritizer()

    experiences = [
        {
            "title": "Senior AI Engineer",
            "company": "TechCorp LLC",
            "start_date": "2022-01",
            "end_date": "Present",
            "description": "Building cloud-based SaaS LLM integrations and enterprise machine learning scale platforms.",
        },
        {
            "title": "Junior Developer",
            "company": "Local Agency",
            "start_date": "2018-01",
            "end_date": "2020-12",
            "description": "Worked on small WordPress code projects.",
        },
    ]

    prioritized = prioritizer.prioritize_work_experience(
        experiences=experiences,
        job_title="Senior Python AI Developer",
        job_domain="AI/Analytics",
    )

    assert len(prioritized) == 2
    # The first role should score highly and be in the HIGH band
    high_role = prioritized[0]
    assert high_role.title == "Senior AI Engineer"
    assert high_role.priority_band == "HIGH"
    assert high_role.priority_score >= 135.0
    assert "maximum recency" in high_role.justification
    assert "Enterprise scale alignment" in high_role.justification
    assert "Domain keywords matched" in high_role.justification

    # The second role should be LOW band due to recency and low domain/enterprise relevance
    low_role = prioritized[1]
    assert low_role.title == "Junior Developer"
    assert low_role.priority_band == "LOW"


def test_strategy_layer_domain_positioning() -> None:
    """
    Verifies that the strategy layer correctly maps headlines, focus areas,
    and pitches based on the target job domain.
    """
    strategy = ResumeStrategyLayer()

    profile_skills = ["Python", "SQL", "Git"]
    job_skills = ["Python", "Machine Learning", "NLP"]
    job_title = "Data Scientist"

    # 1. AI/Analytics domain
    emphasized, pos, ats = strategy.generate_strategy(
        profile_skills=profile_skills,
        job_skills=job_skills,
        job_title=job_title,
        job_domain="AI/Analytics",
        job_description="Seeking a Data Scientist skilled in NLP and Machine Learning.",
        positioning_seniority="senior",
    )

    assert any(s.skill_name == "Python" and s.user_possesses for s in emphasized)
    assert any(s.skill_name == "Machine Learning" and not s.user_possesses for s in emphasized)
    assert "AI & Advanced Analytics" in pos.suggested_headline
    assert "Machine Learning & LLMs" in pos.recommended_focus_areas
    assert "senior" in pos.positioning_pitch
    assert "Machine Learning" in ats.target_keywords

    # 2. Procurement domain
    _, pos_proc, _ = strategy.generate_strategy(
        profile_skills=profile_skills,
        job_skills=["Sourcing", "Negotiation"],
        job_title="Buyer",
        job_domain="Procurement",
        job_description="RFP sourcing vendor contracts negotiation",
        positioning_seniority="mid",
    )
    assert "Procurement & Strategic Sourcing" in pos_proc.suggested_headline
    assert "Contract Negotiation & SLAs" in pos_proc.recommended_focus_areas

    # 3. Supply Chain domain
    _, pos_sc, _ = strategy.generate_strategy(
        profile_skills=profile_skills,
        job_skills=["Logistics"],
        job_title="Logistics Analyst",
        job_domain="Supply Chain",
        job_description="Warehouse inventory logistics operations",
        positioning_seniority="lead",
    )
    assert "Supply Chain & Operations" in pos_sc.suggested_headline
    assert "Logistics & Warehouse Operations" in pos_sc.recommended_focus_areas


def test_explanation_layer_rendering() -> None:
    """
    Verifies that the explanation layer constructs coherent summaries and justifications.
    """
    expl_layer = ResumeExplanationLayer()

    summary = expl_layer.generate_overall_summary(
        job_title="AI Engineer",
        company="Google",
        overall_score=88.5,
        recommendation="strong_apply",
    )
    assert "Google" in summary
    assert "AI Engineer" in summary
    assert "Strong Apply" in summary
    assert "88.5%" in summary

    strategy_text = expl_layer.generate_tailoring_strategy_text(
        job_domain="AI/Analytics",
        prioritized_exps=[],
        emphasized_skills=[],
    )
    assert "AI/Analytics" in strategy_text


def test_orchestration_engine_runs() -> None:
    """
    Verifies the end-to-end execution of ResumeIntelligenceEngine.
    """
    engine = ResumeIntelligenceEngine()

    user_profile = {
        "skills": "Python, SQL, PySpark",
        "experience": [
            {
                "title": "Data Engineer",
                "company": "BigCorp",
                "start_date": "2021-01",
                "end_date": "Present",
                "description": "Managed cloud data pipelines and SaaS database integrations.",
            }
        ],
        "positioning": {"seniority_level": "senior"},
    }

    job_intel = {
        "title": "Senior AI Architect",
        "company": "TechInc",
        "domain": "AI/Analytics",
        "normalized_skills": ["Python", "Machine Learning", "SQL"],
        "raw_content": "We need a Senior AI Architect to build Python machine learning models and scale cloud infrastructures.",
    }

    opp_ranking = {
        "overall_score": 92.5,
        "recommendation": "strong_apply",
    }

    response = engine.generate_tailoring_strategy(
        user_profile=user_profile,
        job_intelligence=job_intel,
        opportunity_ranking=opp_ranking,
    )

    assert isinstance(response, TailoringStrategyResponse)
    assert "TechInc" in response.overall_alignment_summary
    assert len(response.prioritized_experiences) == 1
    assert response.prioritized_experiences[0].priority_band == "HIGH"
    assert len(response.emphasized_skills) > 0
    assert response.positioning_recommendations.suggested_headline is not None
    assert "Skills Section" in response.ats_optimization.resume_section_recommendations
