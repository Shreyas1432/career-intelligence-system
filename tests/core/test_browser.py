from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from src.core.browser import BrowserManager, BrowserScrapingService, PageLoadOptions
from src.core.browser.exceptions import BrowserNavigationError, BrowserStartupError
from src.core.config.browser import BrowserConfig


class FakeLocator:
    async def inner_text(self) -> str:
        return "Example job text"


class FakePage:
    def __init__(self, fail_goto_count: int = 0, network_idle_timeout: bool = False):
        self.fail_goto_count = fail_goto_count
        self.network_idle_timeout = network_idle_timeout
        self.goto_calls = 0
        self.closed = False
        self.url = "about:blank"

    async def goto(self, url: str, **_kwargs: object) -> None:
        self.goto_calls += 1
        if self.goto_calls <= self.fail_goto_count:
            raise PlaywrightTimeoutError("navigation timed out")
        self.url = url

    async def wait_for_load_state(self, state: str, timeout: int) -> None:
        _ = (state, timeout)
        if self.network_idle_timeout:
            raise PlaywrightTimeoutError("network idle timed out")

    async def content(self) -> str:
        return "<html><body>Example job text</body></html>"

    async def evaluate(self, _expression: str) -> str:
        return "<html><body>Rendered job text</body></html>"

    def locator(self, _selector: str) -> FakeLocator:
        return FakeLocator()

    async def screenshot(self, full_page: bool) -> bytes:
        _ = full_page
        return b"fake-png"

    def set_default_timeout(self, timeout: int) -> None:
        self.action_timeout = timeout

    def set_default_navigation_timeout(self, timeout: int) -> None:
        self.navigation_timeout = timeout

    def is_closed(self) -> bool:
        return self.closed

    async def close(self) -> None:
        self.closed = True


class FakeContext:
    def __init__(self):
        self.page = FakePage()
        self.closed = False

    async def new_page(self) -> FakePage:
        return self.page

    async def close(self) -> None:
        self.closed = True

    async def route(self, pattern: str, handler: object) -> None:
        self.route_pattern = pattern
        self.route_handler = handler


class FakeBrowser:
    def __init__(self):
        self.context = FakeContext()
        self.closed = False

    def is_connected(self) -> bool:
        return not self.closed

    async def new_context(self, **kwargs: object) -> FakeContext:
        self.context_options = kwargs
        return self.context

    async def close(self) -> None:
        self.closed = True


class FakeLauncher:
    def __init__(self, browser: FakeBrowser):
        self.browser = browser

    async def launch(self, **kwargs: object) -> FakeBrowser:
        self.launch_options = kwargs
        return self.browser


class FakePlaywright:
    def __init__(self, browser: FakeBrowser):
        self.chromium = FakeLauncher(browser)
        self.stopped = False

    async def stop(self) -> None:
        self.stopped = True


class FakePlaywrightStarter:
    def __init__(self, playwright: FakePlaywright):
        self.playwright = playwright

    async def start(self) -> FakePlaywright:
        return self.playwright


class FakeManager:
    def __init__(self, page: FakePage):
        self.page_instance = page
        self.start_calls = 0
        self.close_calls = 0
        self.page_context_exited = False

    async def start(self) -> None:
        self.start_calls += 1

    async def close(self) -> None:
        self.close_calls += 1

    @asynccontextmanager
    async def page(self, _context_options: object = None) -> AsyncIterator[FakePage]:
        try:
            yield self.page_instance
        finally:
            self.page_context_exited = True


@pytest.mark.asyncio
async def test_browser_manager_starts_and_cleans_page_context(monkeypatch) -> None:
    fake_browser = FakeBrowser()
    fake_playwright = FakePlaywright(fake_browser)

    monkeypatch.setattr(
        "src.core.browser.manager.async_playwright",
        lambda: FakePlaywrightStarter(fake_playwright),
    )

    manager = BrowserManager(BrowserConfig())

    async with manager.page() as page:
        assert page is fake_browser.context.page
        assert manager.is_started is True

    assert fake_browser.context.page.closed is True
    assert fake_browser.context.closed is True

    await manager.close()

    assert fake_browser.closed is True
    assert fake_playwright.stopped is True


@pytest.mark.asyncio
async def test_browser_manager_enforces_instance_cap(monkeypatch) -> None:
    fake_browser = FakeBrowser()
    fake_playwright = FakePlaywright(fake_browser)

    monkeypatch.setattr(
        "src.core.browser.manager.async_playwright",
        lambda: FakePlaywrightStarter(fake_playwright),
    )

    first_manager = BrowserManager(BrowserConfig(max_browser_instances=1))
    second_manager = BrowserManager(BrowserConfig(max_browser_instances=1))

    await first_manager.start()
    try:
        with pytest.raises(BrowserStartupError):
            await second_manager.start()
    finally:
        await first_manager.close()
        await second_manager.close()


@pytest.mark.asyncio
async def test_scraping_service_captures_page_snapshot() -> None:
    page = FakePage()
    manager = FakeManager(page)
    service = BrowserScrapingService(manager=manager, config=BrowserConfig(retry_attempts=0))

    snapshot = await service.capture_page(
        "https://example.com/jobs/1",
        include_screenshot=True,
    )

    assert snapshot.final_url == "https://example.com/jobs/1"
    assert "Example job text" in snapshot.html
    assert "Rendered job text" in snapshot.rendered_dom
    assert snapshot.text == "Example job text"
    assert snapshot.screenshot == b"fake-png"
    assert manager.page_context_exited is True


@pytest.mark.asyncio
async def test_scraping_service_retries_navigation() -> None:
    page = FakePage(fail_goto_count=1)
    manager = FakeManager(page)
    service = BrowserScrapingService(
        manager=manager,
        config=BrowserConfig(retry_attempts=1, retry_backoff_seconds=0),
    )

    snapshot = await service.capture_page("https://example.com/jobs/1")

    assert snapshot.final_url == "https://example.com/jobs/1"
    assert page.goto_calls == 2


@pytest.mark.asyncio
async def test_scraping_service_raises_after_retry_exhaustion() -> None:
    page = FakePage(fail_goto_count=2)
    manager = FakeManager(page)
    service = BrowserScrapingService(
        manager=manager,
        config=BrowserConfig(retry_attempts=1, retry_backoff_seconds=0),
    )

    with pytest.raises(BrowserNavigationError):
        await service.capture_page("https://example.com/jobs/1")

    assert manager.page_context_exited is True


@pytest.mark.asyncio
async def test_network_idle_timeout_does_not_fail_snapshot() -> None:
    page = FakePage(network_idle_timeout=True)
    manager = FakeManager(page)
    service = BrowserScrapingService(manager=manager, config=BrowserConfig(retry_attempts=0))

    snapshot = await service.capture_page(
        "https://example.com/jobs/1",
        load_options=PageLoadOptions(network_idle=True),
    )

    assert snapshot.text == "Example job text"
