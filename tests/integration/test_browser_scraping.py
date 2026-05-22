from urllib.parse import quote

import pytest

from src.core.browser import BrowserScrapingService
from src.core.browser.exceptions import BrowserStartupError
from src.core.config.browser import BrowserConfig


def _data_url(html: str) -> str:
    return f"data:text/html,{quote(html)}"


@pytest.mark.asyncio
async def test_browser_scraping_extracts_rendered_content() -> None:
    html = """
    <html>
      <body>
        <h1>AI Platform Engineer</h1>
        <script>
          document.body.insertAdjacentHTML("beforeend", "<p>Rendered after JS</p>");
        </script>
      </body>
    </html>
    """

    service = BrowserScrapingService(config=BrowserConfig(retry_attempts=0))
    try:
        snapshot = await service.capture_page(_data_url(html))
    except BrowserStartupError as exc:
        pytest.skip(f"Playwright browser is not installed: {exc}")

    assert "AI Platform Engineer" in snapshot.text
    assert "Rendered after JS" in snapshot.rendered_dom
    assert service._manager.is_started is False


@pytest.mark.asyncio
async def test_browser_scraping_captures_screenshot_bytes() -> None:
    html = "<html><body><h1>Screenshot Job</h1></body></html>"

    service = BrowserScrapingService(config=BrowserConfig(retry_attempts=0))
    try:
        image = await service.screenshot(_data_url(html), full_page=False)
    except BrowserStartupError as exc:
        pytest.skip(f"Playwright browser is not installed: {exc}")

    assert image.startswith(b"\x89PNG")
    assert service._manager.is_started is False
