import streamlit as st

from src.app.components.ui import apply_custom_theme, render_header, validate_uploaded_file
from src.core.config import settings

st.set_page_config(page_title="Resume Tailoring", page_icon="📝", layout="wide")

# Apply custom styling theme
apply_custom_theme()

# Render title header
render_header(
    title="📝 Resume Tailoring & Alignment",
    subtitle="Evaluate compatibility metrics and tailor document copy targeting job descriptions.",
)

st.write("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Candidate Resume")
    uploaded_file = st.file_uploader(
        "Upload Resume (PDF, DOCX, TXT)", type=settings.modules.resume.allowed_extensions
    )

    # Pre-populate if already parsed in session state
    resume_text = st.text_area("Or paste raw resume text here", height=200)

with col2:
    st.subheader("Target Job Requirements")
    job_desc = st.text_area("Paste the job description you are targeting", height=270)

if st.button("Analyze Alignment", type="primary"):
    input_text = resume_text

    # Validate file if uploaded
    if uploaded_file:
        if validate_uploaded_file(uploaded_file):
            input_text = uploaded_file.read().decode("utf-8", errors="ignore")
        else:
            input_text = ""

    if not input_text or not job_desc:
        st.warning("Please provide your resume content and a target job description.")
    else:
        with st.spinner("Executing alignment analytics..."):
            try:
                # Lazy loading backend logic to decrease memory footprint
                from src.modules.resume import analyze_resume

                results = analyze_resume(input_text, job_desc)
                st.success("Analysis complete!")
                st.write("### AI Alignment Insights")
                st.markdown(results)
            except Exception as e:
                st.error(f"Failed to process analysis: {e!s}")
