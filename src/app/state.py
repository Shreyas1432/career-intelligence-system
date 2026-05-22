from typing import Any, cast

import streamlit as st

from src.core.database.models import UserProfile


class AppStateManager:
    """
    Unified manager for Streamlit Session State keys.
    Prevents key collisions and enforces static type annotations for single-user profile flows.
    """

    @staticmethod
    def initialize_state() -> None:
        """
        Ensures default session state fields exist at application startup.
        """
        defaults: dict[str, Any] = {
            "current_profile": None,
            "selected_job_id": None,
            "interview_history": [],
            "current_question": None,
            "last_error": None,
            "is_sidebar_collapsed": False,
        }
        for key, val in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = val

    @property
    def current_profile(self) -> UserProfile | None:
        return st.session_state.get("current_profile")

    @current_profile.setter
    def current_profile(self, profile: UserProfile | None) -> None:
        st.session_state["current_profile"] = profile

    @property
    def selected_job_id(self) -> int | None:
        return st.session_state.get("selected_job_id")

    @selected_job_id.setter
    def selected_job_id(self, job_id: int | None) -> None:
        st.session_state["selected_job_id"] = job_id

    @property
    def interview_history(self) -> list[dict[str, str]]:
        return cast(list[dict[str, str]], st.session_state.get("interview_history", []))

    @interview_history.setter
    def interview_history(self, history: list[dict[str, str]]) -> None:
        st.session_state["interview_history"] = history

    @property
    def current_question(self) -> str | None:
        return st.session_state.get("current_question")

    @current_question.setter
    def current_question(self, question: str | None) -> None:
        st.session_state["current_question"] = question

    def clear_session(self) -> None:
        """
        Resets session state variables back to defaults.
        """
        st.session_state["current_profile"] = None
        st.session_state["selected_job_id"] = None
        st.session_state["interview_history"] = []
        st.session_state["current_question"] = None
        st.session_state["last_error"] = None


# Global state manager instance
state_manager = AppStateManager()
