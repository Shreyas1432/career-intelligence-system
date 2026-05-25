import logging
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from src.modules.resume.schemas import RenderResponse, TemplateStyleConfig

logger = logging.getLogger("src.modules.resume.rendering")

# Check if WeasyPrint is available
try:
    import weasyprint
    WEASYPRINT_AVAILABLE = True
except (ImportError, OSError):
    WEASYPRINT_AVAILABLE = False
    logger.warning("WeasyPrint not available. PDF export will fall back to HTML.")

class PDFExportService:
    """Service rendering HTML templates to PDF using WeasyPrint.
    If system dependencies are missing, falls back gracefully to HTML export.
    """
    def export_pdf(self, html_content: str, output_path: str) -> str:
        """Compiles HTML string into a PDF file at output_path.
        If WeasyPrint is unavailable, writes HTML instead and returns the HTML path.
        """
        path = Path(output_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        if WEASYPRINT_AVAILABLE:
            try:
                weasyprint.HTML(string=html_content).write_pdf(str(path))
                logger.info(f"Successfully compiled PDF to: {path}")
                return str(path)
            except Exception as e:
                logger.error(f"Failed to generate PDF via WeasyPrint: {e}. Falling back to HTML.")

        # HTML Fallback
        html_path = path.with_suffix(".html")
        with html_path.open("w", encoding="utf-8") as f:
            f.write(html_content)
        logger.warning(f"Exported HTML fallback file to: {html_path}")
        return str(html_path)

class ResumeTemplateEngine:
    """Template engine that compiles tailored resumes into HTML pages
    and exports them to PDF/HTML format using style configurations.
    """
    def __init__(self, templates_dir: str | None = None, pdf_service: PDFExportService | None = None) -> None:
        if templates_dir is None:
            templates_dir = str(Path(__file__).parent / "templates")
        self.templates_dir = Path(templates_dir).resolve()
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            autoescape=True
        )
        self.pdf_service = pdf_service or PDFExportService()

    def render_html(self, tailored_resume: Any, layout_name: str = 'classic_enterprise', config: TemplateStyleConfig | None = None) -> str:
        """Renders the tailored resume structure to HTML with embedded styling."""
        active_config = config or TemplateStyleConfig()

        try:
            css_template = self.jinja_env.get_template('style.css')
            css_content = css_template.render(config=active_config)
        except Exception as e:
            logger.error(f"Failed to load/render style.css template: {e}")
            css_content = ''

        layout_filename = f"{layout_name}.html"
        try:
            html_template = self.jinja_env.get_template(layout_filename)
        except Exception as e:
            logger.error(f"Failed to find layout template '{layout_filename}': {e}")
            raise FileNotFoundError(f"Resume layout template '{layout_filename}' not found.") from e

        context = {
            'resume': tailored_resume,
            'config': active_config,
            'css_content': css_content
        }
        rendered_html = html_template.render(context)
        return rendered_html

    def export_resume(self, tailored_resume: Any, output_path: str, layout_name: str = 'classic_enterprise', config: TemplateStyleConfig | None = None, export_format: str = 'pdf') -> RenderResponse:
        """Renders and saves the tailored resume as PDF or HTML."""
        fmt = export_format.lower().strip()
        if fmt not in ('pdf', 'html'):
            raise ValueError("export_format must be 'pdf' or 'html'")

        html_content = self.render_html(tailored_resume=tailored_resume, layout_name=layout_name, config=config)
        out_path = Path(output_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if fmt == 'html':
            html_path = out_path.with_suffix('.html')
            with html_path.open('w', encoding='utf-8') as f:
                f.write(html_content)
            logger.info(f"Successfully exported HTML resume to: {html_path}")
            return RenderResponse(output_path=str(html_path), format='html', html_content=html_content)
        else:
            compiled_path = self.pdf_service.export_pdf(html_content, str(out_path))
            resulting_format = 'pdf' if compiled_path.endswith('.pdf') else 'html'
            return RenderResponse(output_path=compiled_path, format=resulting_format, html_content=html_content)
