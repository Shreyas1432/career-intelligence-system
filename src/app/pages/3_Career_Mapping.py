import streamlit as st

from src.app.components.ui import apply_custom_theme, render_header
from src.app.state import state_manager

st.set_page_config(page_title="Career Mapping", page_icon="📊", layout="wide")

# Apply custom styling theme
apply_custom_theme()

# Render title header
render_header(
    title="📊 Career Mapping & Analytics",
    subtitle="Plan and track transition pathways to unlock next-level roles.",
)

st.write("---")

# Ensure state is initialized
state_manager.initialize_state()

col1, col2 = st.columns(2)

with col1:
    current_role = st.text_input("Current Job Title", placeholder="e.g., Software Engineer")
    target_role = st.text_input("Target Job Title", placeholder="e.g., Lead AI Engineer")

with col2:
    skills = st.text_area(
        "Your Core Skills (comma-separated)", placeholder="e.g., Python, SQL, REST APIs, Git"
    )

if st.button("Generate Transition Roadmap", type="primary"):
    if not current_role or not target_role:
        st.warning("Please specify both your current role and your target role.")
    else:
        with st.spinner("Generating labor market map and gap analysis..."):
            try:
                # Lazy loading backend logic to decrease memory footprint
                from src.modules.career_path import get_career_map

                roadmap = get_career_map(current_role, target_role, skills)
                st.success("Roadmap generated successfully!")
                st.write("### AI Transition Strategy Roadmap")
                st.markdown(roadmap)
            except Exception as e:
                st.error(f"Failed to generate roadmap: {e!s}")
