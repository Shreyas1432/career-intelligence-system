from typing import Any

import streamlit as st

from src.core.config import settings


def apply_custom_theme() -> None:
    """
    Applies unified CSS stylesheets targeting premium glassmorphism aesthetics.
    """
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #0b0f19;
            color: #f8fafc;
        }
        .main-title {
            font-family: 'Outfit', 'Inter', sans-serif;
            background: linear-gradient(90deg, #38bdf8 0%, #a855f7 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            font-size: 2.8rem;
            margin-bottom: 0.2rem;
            padding-top: 10px;
        }
        .subtitle {
            color: #94a3b8;
            font-size: 1.1rem;
            margin-bottom: 2rem;
        }
        .glass-card {
            background: rgba(30, 41, 59, 0.45);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 15px;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.25);
        }
        .card-header {
            color: #f1f5f9;
            font-size: 1.3rem;
            font-weight: 600;
            margin-bottom: 8px;
        }
        .card-content {
            color: #94a3b8;
            font-size: 0.95rem;
            line-height: 1.5;
        }
        .kpi-value {
            font-size: 2rem;
            font-weight: 700;
            color: #38bdf8;
            margin: 5px 0;
        }
        .kpi-delta {
            font-size: 0.85rem;
            color: #10b981;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(title: str, subtitle: str) -> None:
    """
    Renders top header sections on dashboard page modules.
    """
    st.markdown(f'<h1 class="main-title">{title}</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="subtitle">{subtitle}</p>', unsafe_allow_html=True)


def render_card(title: str, content: str) -> None:
    """
    Renders stylized static information boxes.
    """
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="card-header">{title}</div>
            <div class="card-content">{content}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(
    title: str, value: str, delta: str | None = None, description: str | None = None
) -> None:
    """
    Renders statistical KPI metrics on dashboard pages.
    """
    delta_html = f'<span class="kpi-delta">▲ {delta}</span>' if delta else ""
    desc_html = (
        f'<div style="color: #64748b; font-size: 0.8rem; margin-top: 5px;">{description}</div>'
        if description
        else ""
    )

    st.markdown(
        f"""
        <div class="glass-card">
            <div class="card-content" style="font-weight: 500;">{title}</div>
            <div class="kpi-value">{value}</div>
            {delta_html}
            {desc_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def validate_uploaded_file(uploaded_file: Any) -> bool:
    """
    Checks uploaded file sizes and extensions dynamically against configured modules constraints.
    """
    if uploaded_file is None:
        return False

    # Check extension
    file_ext = "." + uploaded_file.name.split(".")[-1].lower()
    allowed_exts = settings.modules.resume.allowed_extensions
    if file_ext not in allowed_exts:
        st.error(f"Unsupported file format: {file_ext}. Allowed: {', '.join(allowed_exts)}")
        return False

    # Check size
    max_bytes = settings.modules.resume.max_upload_size_mb * 1024 * 1024
    if uploaded_file.size > max_bytes:
        st.error(
            f"File exceeds maximum size limit of {settings.modules.resume.max_upload_size_mb}MB."
        )
        return False

    return True
