import logging

from src.core.ai.client import ai_client
from src.core.prompts import prompt_manager

logger = logging.getLogger("src.modules.interview.service")


def conduct_mock_interview(role: str, question_type: str, history: list[dict[str, str]]) -> str:
    """
    Handles conversational mock interview generation and feedback evaluation.
    """
    logger.info(f"Generating interview response for {role} (mode: {question_type})...")

    try:
        prompt = prompt_manager.load_prompt(
            "interview/coach_prompt.md",
            {"role": role, "question_type": question_type, "history": str(history)},
        )
    except FileNotFoundError:
        logger.warning("Interview template prompt file not found. Falling back to default string.")
        prompt = (
            f"You are interviewing a candidate for a '{role}' position. "
            f"Context: {question_type}. Generate the next interview statement or feedback response."
        )

    messages = [
        {
            "role": "system",
            "content": "You are a professional, rigorous tech industry interviewer conducting a screen.",
        },
        {"role": "user", "content": prompt},
    ]

    response = ai_client.generate_chat_response(messages)
    return response
