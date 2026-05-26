# tests/fixtures/scraping_extraction.py

from typing import Any

import pytest
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from pydantic import BaseModel

# ------------------------------------------------------------------------------
# Playwright Fakes and Routing Mock Manager
# ------------------------------------------------------------------------------


class DynamicFakePage:
    def __init__(self, routing_rules: dict[str, Any]):
        self.routing_rules = routing_rules
        self.closed = False
        self.url = "about:blank"
        self.goto_calls: list[str] = []

    async def goto(self, url: str, **kwargs: Any) -> None:
        _ = kwargs
        self.goto_calls.append(url)
        self.url = url
        if url in self.routing_rules:
            rule = self.routing_rules[url]
            if isinstance(rule, Exception):
                raise rule
        if "timeout" in url:
            raise PlaywrightTimeoutError("navigation timed out")

    async def content(self) -> str:
        rule = self.routing_rules.get(self.url, "")
        if isinstance(rule, str):
            return rule
        elif isinstance(rule, dict) and "html" in rule:
            html = rule["html"]
            if isinstance(html, str):
                return html
        return "<html><body>Default Page Content</body></html>"

    async def evaluate(self, expression: str) -> str:
        _ = expression
        # evaluate usually fetches DOM layout
        return await self.content()

    async def wait_for_load_state(self, state: str, timeout: int) -> None:
        _ = (state, timeout)
        if "network_idle_timeout" in self.url:
            raise PlaywrightTimeoutError("network idle timed out")

    def locator(self, selector: str) -> Any:
        _ = selector

        class DynamicLocator:
            async def inner_text(self) -> str:
                return "Mocked locator text"

        return DynamicLocator()

    async def screenshot(self, full_page: bool) -> bytes:
        _ = full_page
        return b"mocked-png-data"

    def set_default_timeout(self, timeout: int) -> None:
        pass

    def set_default_navigation_timeout(self, timeout: int) -> None:
        pass

    def is_closed(self) -> bool:
        return self.closed

    async def close(self) -> None:
        self.closed = True


class DynamicFakeContext:
    def __init__(self, page: DynamicFakePage):
        self.page = page
        self.closed = False

    async def new_page(self) -> DynamicFakePage:
        return self.page

    async def close(self) -> None:
        self.closed = True

    async def route(self, pattern: str, handler: Any) -> None:
        pass


class DynamicFakeBrowser:
    def __init__(self, context: DynamicFakeContext):
        self.context = context
        self.closed = False

    def is_connected(self) -> bool:
        return not self.closed

    async def new_context(self, **kwargs: Any) -> DynamicFakeContext:
        _ = kwargs
        return self.context

    async def close(self) -> None:
        self.closed = True


class DynamicFakeLauncher:
    def __init__(self, browser: DynamicFakeBrowser):
        self.browser = browser

    async def launch(self, **kwargs: Any) -> DynamicFakeBrowser:
        _ = kwargs
        return self.browser


class DynamicFakePlaywright:
    def __init__(self, browser: DynamicFakeBrowser):
        self.chromium = DynamicFakeLauncher(browser)
        self.stopped = False

    async def stop(self) -> None:
        self.stopped = True


class DynamicFakePlaywrightStarter:
    def __init__(self, playwright: DynamicFakePlaywright):
        self.playwright = playwright

    async def start(self) -> DynamicFakePlaywright:
        return self.playwright


@pytest.fixture
def mock_playwright_routes(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """
    Fixture monkeypatching async_playwright to route requests dynamically.
    Keys are request URLs, values are HTML strings or Exceptions to raise.
    """
    routes: dict[str, Any] = {}
    page = DynamicFakePage(routes)
    context = DynamicFakeContext(page)
    browser = DynamicFakeBrowser(context)
    playwright = DynamicFakePlaywright(browser)
    starter = DynamicFakePlaywrightStarter(playwright)

    monkeypatch.setattr(
        "src.core.browser.manager.async_playwright",
        lambda: starter,
    )
    return routes


# ------------------------------------------------------------------------------
# ScrapeGraphAI Extraction Adapter Mock Manager
# ------------------------------------------------------------------------------


class MockScrapeGraphAdapter:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.outputs: list[Any] = []
        self.error_to_raise: Exception | None = None

    def set_output(self, output: Any) -> None:
        self.outputs = [output]

    def set_outputs(self, outputs: list[Any]) -> None:
        self.outputs = outputs

    def set_error(self, exc: Exception) -> None:
        self.error_to_raise = exc

    def run(self, *, prompt: str, source: str, response_model: type[BaseModel]) -> Any:
        self.calls.append({"prompt": prompt, "source": source, "response_model": response_model})
        if self.error_to_raise:
            raise self.error_to_raise
        if self.outputs:
            out = self.outputs.pop(0)
            if isinstance(out, Exception):
                raise out
            return out
        # Default mock fallback schema
        return {
            "company": "Mock Corp",
            "title": "Software Engineer",
            "skills": ["Python"],
            "experience_required": "2+ years",
            "location": "Remote",
            "visa_signal": "unknown",
            "employment_type": "full_time",
            "domain": "software_engineering",
            "confidence_score": 0.85,
        }


@pytest.fixture
def mock_scrapegraph_adapter(monkeypatch: pytest.MonkeyPatch) -> MockScrapeGraphAdapter:
    """
    Fixture intercepting ScrapeGraphAIAdapter execution calls and returning
    registered mock extraction outputs or raising mock errors.
    """
    adapter = MockScrapeGraphAdapter()
    monkeypatch.setattr(
        "src.modules.scraping.extraction.ScrapeGraphAIAdapter.run",
        adapter.run,
    )
    return adapter
