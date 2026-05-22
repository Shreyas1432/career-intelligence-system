import logging

from src.core.ai.client import ai_client
from src.core.prompts import prompt_manager

logger = logging.getLogger("src.modules.career_path.service")


def get_career_map(current_role: str, target_role: str, skills: str) -> str:
    """
    Computes a transition roadmap from a current role to a target role, highlighting skill gaps.
    """
    logger.info(f"Generating career transition map from {current_role} to {target_role}...")

    try:
        prompt = prompt_manager.load_prompt(
            "career_path/mapping_prompt.md",
            {"current_role": current_role, "target_role": target_role, "skills": skills},
        )
    except FileNotFoundError:
        logger.warning(
            "Career mapping template prompt file not found. Falling back to default string."
        )
        prompt = (
            f"Map the transition path from '{current_role}' to '{target_role}'. "
            f"Current Skills: {skills}. Identify skills gap, learning path, and estimated timeframes."
        )

    messages = [
        {
            "role": "system",
            "content": "You are a professional career path strategist and tech labor economist.",
        },
        {"role": "user", "content": prompt},
    ]

    response = ai_client.generate_chat_response(messages)
    return response
