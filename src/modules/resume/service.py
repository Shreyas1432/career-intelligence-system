import logging

from src.core.ai.client import ai_client
from src.core.prompts import prompt_manager

logger = logging.getLogger("src.modules.resume.service")


def analyze_resume(resume_text: str, job_description: str) -> str:
    """
    Parses resume text, loads tailor prompt templates, and runs LLM alignment scores.
    """
    logger.info("Executing resume alignment analysis...")

    # 1. Load prompt template
    try:
        prompt = prompt_manager.load_prompt(
            "resume/tailor_prompt.md", {"resume": resume_text, "job_description": job_description}
        )
    except FileNotFoundError:
        # Graceful fallback prompt in case file-based templates are not yet populated
        logger.warning(
            "Resume tailor template prompt file not found. Falling back to default string."
        )
        prompt = (
            f"Analyze the following resume against the job description:\n\n"
            f"Resume:\n{resume_text}\n\n"
            f"Job Description:\n{job_description}\n\n"
            f"Provide matching scores, keyword gaps, and suggestions."
        )

    # 2. Query AI client
    messages = [
        {
            "role": "system",
            "content": "You are an expert career consultant specializing in resume alignment and optimization.",
        },
        {"role": "user", "content": prompt},
    ]

    response = ai_client.generate_chat_response(messages)
    return response
