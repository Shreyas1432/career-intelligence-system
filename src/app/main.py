import streamlit as st

from src.app.components.ui import apply_custom_theme, render_card, render_header, render_kpi_card
from src.app.state import state_manager
from src.core.config import settings
from src.core.logging import configure_logging

# Initialize structured logging configuration at boot
configure_logging()

# Configure page settings
st.set_page_config(
    page_title=settings.app.name,
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize application session states
state_manager.initialize_state()

# Apply custom glassmorphic styling
apply_custom_theme()

# Sidebar Branding
st.sidebar.markdown(
    f"""
    <div style="text-align: center; padding: 15px 0;">
        <span style="font-size: 2.8rem;">💼</span>
        <h3 style="color: #f1f5f9; margin-top: 10px; margin-bottom: 2px;">{settings.app.name}</h3>
        <code style="background-color: #1e293b; color: #38bdf8; padding: 2px 6px; border-radius: 4px;">v{settings.app.version}</code>
    </div>
    """,
    unsafe_allow_html=True,
)

# Render main dashboard header
render_header(
    title="Career Intelligence Dashboard",
    subtitle="Accelerating target profiles and job placement tracking through structured local AI.",
)

st.write("---")

# Analytics Stat KPI Panels
col1, col2, col3 = st.columns(3)

with col1:
    render_kpi_card(
        title="Resume Alignment Matches",
        value="84.7%",
        delta="+4.2% since last week",
        description="Average match score across uploaded targets",
    )

with col2:
    render_kpi_card(
        title="Mock Interview Practice",
        value="12 Sessions",
        delta="3 completed today",
        description="Behavioral and technical coaching cycles",
    )

with col3:
    render_kpi_card(
        title="Active Applications",
        value="8 Tracked",
        delta=None,
        description="SQLite database tracked job applications",
    )

st.write("### Featured Platforms Insights")

col_left, col_right = st.columns(2)

with col_left:
    render_card(
        title="📝 Resume Optimization Module",
        content="Target keyword maps, structural recommendations, and automated sentence modifications mapped using Jinja2 prompts templates.",
    )
    render_card(
        title="📊 Labor Career Insights Module",
        content="Eagerly checks skill demands, formats step-by-step career path guides, and logs timelines for transition goals.",
    )

with col_right:
    render_card(
        title="🎙️ Interactive Mock Recruiter",
        content="STAR-method training interface evaluating communication structures, technical accuracy, and generating scores on a 1-10 scale.",
    )
    st.info("👈 Open the left sidebar panel to navigate directly into the operational pages.")
