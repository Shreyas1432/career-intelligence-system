import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from playwright.async_api import Error as PlaywrightError
from playwright.async_api import Page
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from src.core.config import settings
from src.core.config.browser import BrowserConfig

from .exceptions import BrowserExtractionError, BrowserNavigationError
from .manager import BrowserManager
from .types import BrowserContextOptions, PageLoadOptions, PageSnapshot

logger = logging.getLogger("src.core.browser.service")


class BrowserScrapingService:
    """
    High-level browser scraping API built on short-lived isolated contexts.
    """

    def __init__(
        self,
        manager: BrowserManager | None = None,
        config: BrowserConfig | None = None,
    ):
        self.config = config or settings.browser
        self._manager = manager or BrowserManager(self.config)
        self._owns_manager = manager is None
        self._entered = False

    async def __aenter__(self) -> "BrowserScrapingService":
        await self._manager.start()
        self._entered = True
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        await self.close()

    async def close(self) -> None:
        if self._owns_manager:
            await self._manager.close()
        self._entered = False

    async def capture_page(
        self,
        url: str,
        *,
        load_options: PageLoadOptions | None = None,
        context_options: BrowserContextOptions | None = None,
        include_screenshot: bool = False,
        full_page_screenshot: bool = True,
    ) -> PageSnapshot:
        """
        Load a page and return rendered HTML, DOM, text, and optional screenshot bytes.
        """
        async with self._manager_scope():
            async with self._manager.page(context_options) as page:
                await self._load_page(page, url, load_options)

                try:
                    html = await page.content()
                    rendered_dom = await page.evaluate("() => document.documentElement.outerHTML")
                    text = await page.locator("body").inner_text()
                    screenshot = (
                        await page.screenshot(full_page=full_page_screenshot)
                        if include_screenshot
                        else None
                    )
                except (PlaywrightError, PlaywrightTimeoutError) as exc:
                    raise BrowserExtractionError(f"Failed to extract content from {url}") from exc

                return PageSnapshot(
                    requested_url=url,
                    final_url=page.url,
                    html=html,
                    rendered_dom=rendered_dom,
                    text=text,
                    screenshot=screenshot,
                )

    async def extract_html(
        self,
        url: str,
        *,
        load_options: PageLoadOptions | None = None,
        context_options: BrowserContextOptions | None = None,
    ) -> str:
        snapshot = await self.capture_page(
            url,
            load_options=load_options,
            context_options=context_options,
        )
        return snapshot.html

    async def extract_rendered_dom(
        self,
        url: str,
        *,
        load_options: PageLoadOptions | None = None,
        context_options: BrowserContextOptions | None = None,
    ) -> str:
        snapshot = await self.capture_page(
            url,
            load_options=load_options,
            context_options=context_options,
        )
        return snapshot.rendered_dom

    async def extract_text(
        self,
        url: str,
        *,
        load_options: PageLoadOptions | None = None,
        context_options: BrowserContextOptions | None = None,
    ) -> str:
        snapshot = await self.capture_page(
            url,
            load_options=load_options,
            context_options=context_options,
        )
        return snapshot.text

    async def screenshot(
        self,
        url: str,
        *,
        load_options: PageLoadOptions | None = None,
        context_options: BrowserContextOptions | None = None,
        full_page: bool = True,
    ) -> bytes:
        snapshot = await self.capture_page(
            url,
            load_options=load_options,
            context_options=context_options,
            include_screenshot=True,
            full_page_screenshot=full_page,
        )
        if snapshot.screenshot is None:
            raise BrowserExtractionError(f"Failed to capture screenshot for {url}")
        return snapshot.screenshot

    async def _load_page(self, page: Page, url: str, options: PageLoadOptions | None) -> None:
        load_options = options or PageLoadOptions()
        wait_until = load_options.wait_until or self.config.wait_until
        timeout_ms = load_options.timeout_ms or self.config.navigation_timeout_ms
        retries = (
            self.config.retry_attempts if load_options.retries is None else load_options.retries
        )

        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                await page.goto(url, wait_until=wait_until, timeout=timeout_ms)
                if load_options.network_idle and wait_until != "networkidle":
                    await self._wait_for_network_idle(page)
                return
            except (PlaywrightError, PlaywrightTimeoutError) as exc:
                last_error = exc
                if attempt >= retries:
                    break

                backoff = self.config.retry_backoff_seconds * (2**attempt)
                if backoff > 0:
                    await asyncio.sleep(backoff)

        raise BrowserNavigationError(f"Failed to load {url}") from last_error

    async def _wait_for_network_idle(self, page: Page) -> None:
        try:
            await page.wait_for_load_state(
                "networkidle",
                timeout=self.config.network_idle_timeout_ms,
            )
        except PlaywrightTimeoutError:
            logger.debug("Network idle wait timed out; continuing with rendered page snapshot")

    @asynccontextmanager
    async def _manager_scope(self) -> AsyncIterator[None]:
        if self._entered or not self._owns_manager:
            yield
            return

        await self._manager.start()
        try:
            yield
        finally:
            await self._manager.close()


async def capture_page(
    url: str,
    *,
    load_options: PageLoadOptions | None = None,
    context_options: BrowserContextOptions | None = None,
    include_screenshot: bool = False,
) -> PageSnapshot:
    async with BrowserScrapingService() as service:
        return await service.capture_page(
            url,
            load_options=load_options,
            context_options=context_options,
            include_screenshot=include_screenshot,
        )


async def extract_html(url: str) -> str:
    async with BrowserScrapingService() as service:
        return await service.extract_html(url)


async def extract_text(url: str) -> str:
    async with BrowserScrapingService() as service:
        return await service.extract_text(url)


async def screenshot(url: str, *, full_page: bool = True) -> bytes:
    async with BrowserScrapingService() as service:
        return await service.screenshot(url, full_page=full_page)
