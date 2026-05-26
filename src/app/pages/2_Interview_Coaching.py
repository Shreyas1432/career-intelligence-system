import streamlit as st

from src.app.components.ui import apply_custom_theme, render_header
from src.app.state import state_manager

st.set_page_config(page_title="Interview Coaching", page_icon="🎙️", layout="wide")

# Apply custom styling theme
apply_custom_theme()

# Render title header
render_header(
    title="🎙️ AI Interview Simulator",
    subtitle="Simulate realistic interviews and receive tailored quality feedback.",
)

st.write("---")

# Ensure state is initialized
state_manager.initialize_state()

role = st.text_input(
    "What role are you practicing for?", placeholder="e.g., Senior Python Engineer"
)
question_type = st.selectbox(
    "Select question focus", ["Behavioral (STAR method)", "Technical / Coding", "System Design"]
)

if st.button("Generate Interview Question", type="secondary"):
    if not role:
        st.warning("Please enter a target role to generate relevant questions.")
    else:
        with st.spinner("Generating mock question..."):
            try:
                # Lazy loading backend logic to decrease memory footprint
                from src.modules.automation import conduct_mock_interview

                question = conduct_mock_interview(
                    role, question_type, state_manager.interview_history
                )

                # Update history and current question
                history = state_manager.interview_history
                history.append({"role": "assistant", "content": question})
                state_manager.interview_history = history
                state_manager.current_question = question
            except Exception as e:
                st.error(f"Failed to generate question: {e!s}")

# Display current question
if state_manager.current_question:
    st.info(f"**AI Interviewer**: {state_manager.current_question}")

# Input answer
answer = st.text_area("Your Response", height=150)

if st.button("Submit Response", type="primary"):
    if not answer:
        st.warning("Please type your response before submitting.")
    elif not state_manager.current_question:
        st.warning("Please generate a question first.")
    else:
        # Save user response to history
        history = state_manager.interview_history
        history.append({"role": "user", "content": answer})
        state_manager.interview_history = history

        with st.spinner("Evaluating your response..."):
            try:
                # Lazy loading backend logic to decrease memory footprint
                from src.modules.automation import conduct_mock_interview

                # Get feedback using assistant's question and user's answer
                feedback_prompt = f"Analyze response for: '{state_manager.current_question}'. User answered: '{answer}'"
                feedback = conduct_mock_interview(
                    role, "Feedback", [{"role": "user", "content": feedback_prompt}]
                )

                st.success("Feedback Generated!")
                st.write("### AI Feedback & Coaching Advice")
                st.markdown(feedback)

                # Reset history/current question for next session
                state_manager.interview_history = []
                state_manager.current_question = None
            except Exception as e:
                st.error(f"Evaluation error: {e!s}")
