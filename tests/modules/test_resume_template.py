from pathlib import Path
from typing import Any

import pytest

from src.modules.resume.rendering import ResumeTemplateEngine
from src.modules.resume.schemas import RenderResponse, TemplateStyleConfig


@pytest.fixture
def mock_tailored_resume() -> dict[str, Any]:
    """
    Returns a sample structured tailored resume dictionary matching expectations.
    """
    return {
        "full_name": "Jane Doe",
        "email": "jane.doe@example.com",
        "phone": "555-0199",
        "suggested_headline": "Senior Software Architect | Python & AI Expert",
        "professional_summary": "Experienced architect specializing in distributed Python applications and scalable AI cloud setups.",
        "experiences": [
            {
                "title": "Lead Software Architect",
                "company": "ScaleInc",
                "start_date": "2021-01",
                "end_date": "Present",
                "bullets": [
                    "Designed cloud systems saving 15% in AWS infrastructure costs.",
                    "Led a team of 5 engineers to deliver AI integrations.",
                ],
            }
        ],
        "skills": ["Python", "SQL", "Cloud Infrastructure", "Machine Learning"],
    }


def test_html_template_variable_rendering(mock_tailored_resume: dict[str, Any]) -> None:
    """
    Verifies that the template engine successfully renders context variables
    into classic and modern HTML layouts.
    """
    engine = ResumeTemplateEngine()
    config = TemplateStyleConfig(
        font_family="Calibri, sans-serif",
        primary_color="#002244",
        margin_top="1in",
    )

    # 1. Classic layout rendering
    html_classic = engine.render_html(
        tailored_resume=mock_tailored_resume,
        layout_name="classic_enterprise",
        config=config,
    )
    assert "Jane Doe" in html_classic
    assert "Senior Software Architect" in html_classic
    assert "ScaleInc" in html_classic
    assert "Designed cloud systems saving 15%" in html_classic
    assert "Calibri, sans-serif" in html_classic
    assert "#002244" in html_classic
    assert "margin: 1in" in html_classic

    # 2. Modern clean layout rendering
    html_modern = engine.render_html(
        tailored_resume=mock_tailored_resume,
        layout_name="modern_clean",
        config=config,
    )
    assert "Jane Doe" in html_modern
    assert "border-bottom: 2px solid #002244" in html_modern


def test_section_ordering_configuration(mock_tailored_resume: dict[str, Any]) -> None:
    """
    Verifies that changing section_order changes section placement sequence.
    """
    engine = ResumeTemplateEngine()

    # Put skills first
    config_skills_first = TemplateStyleConfig(section_order=["skills", "summary", "experience"])
    html_out = engine.render_html(
        tailored_resume=mock_tailored_resume,
        layout_name="classic_enterprise",
        config=config_skills_first,
    )

    # Verify order: skills index < experience index
    skills_index = html_out.find('id="skills"')
    exp_index = html_out.find('id="experience"')
    assert skills_index != -1
    assert exp_index != -1
    assert skills_index < exp_index


def test_pdf_export_service_and_fallback(
    tmp_path: Path, mock_tailored_resume: dict[str, Any]
) -> None:
    """
    Verifies that PDFExportService successfully writes to disk, either generating
    a PDF or falling back gracefully to HTML format if WeasyPrint fails.
    """
    engine = ResumeTemplateEngine()
    out_pdf_path = str(tmp_path / "jane_doe_resume.pdf")

    # Call export
    response = engine.export_resume(
        tailored_resume=mock_tailored_resume,
        output_path=out_pdf_path,
        layout_name="classic_enterprise",
        export_format="pdf",
    )

    assert isinstance(response, RenderResponse)
    assert Path(response.output_path).exists()
    assert response.format in ("pdf", "html")
    assert "Jane Doe" in response.html_content

    # Clean up
    if Path(response.output_path).exists():
        Path(response.output_path).unlink()


def test_direct_html_export(tmp_path: Path, mock_tailored_resume: dict[str, Any]) -> None:
    """
    Verifies that HTML format exports correctly write HTML files directly to disk.
    """
    engine = ResumeTemplateEngine()
    out_html_path = str(tmp_path / "jane_doe_resume.html")

    response = engine.export_resume(
        tailored_resume=mock_tailored_resume,
        output_path=out_html_path,
        layout_name="classic_enterprise",
        export_format="html",
    )

    assert response.format == "html"
    assert response.output_path.endswith(".html")
    assert Path(response.output_path).exists()

    # Clean up
    if Path(response.output_path).exists():
        Path(response.output_path).unlink()
