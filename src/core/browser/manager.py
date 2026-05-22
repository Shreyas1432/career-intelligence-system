import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from threading import Lock
from typing import Any, ClassVar

from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    Route,
    async_playwright,
)
from playwright.async_api import (
    Error as PlaywrightError,
)

from src.core.config import settings
from src.core.config.browser import BrowserConfig

from .exceptions import BrowserStartupError
from .types import BrowserContextOptions

logger = logging.getLogger("src.core.browser.manager")


class BrowserManager:
    """
    Reusable async Playwright manager with bounded context usage.
    """

    _active_browser_count: ClassVar[int] = 0
    _browser_count_lock: ClassVar[Lock] = Lock()

    def __init__(self, config: BrowserConfig | None = None):
        self.config = config or settings.browser
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._browser_slot_acquired = False
        self._start_lock = asyncio.Lock()
        self._context_slots = asyncio.Semaphore(self.config.max_contexts)

    @property
    def is_started(self) -> bool:
        return self._browser is not None and self._browser.is_connected()

    async def __aenter__(self) -> "BrowserManager":
        await self.start()
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        await self.close()

    async def start(self) -> None:
        """
        Start Playwright and one browser process if needed.
        """
        async with self._start_lock:
            if self.is_started:
                return

            try:
                self._acquire_browser_slot()
                self._playwright = await async_playwright().start()
                launcher = getattr(self._playwright, self.config.browser_type)
                self._browser = await launcher.launch(
                    headless=self.config.headless,
                    args=self._launch_args(),
                )
            except Exception as exc:
                await self.close()
                raise BrowserStartupError(
                    f"Failed to start {self.config.browser_type} browser"
                ) from exc

    async def close(self) -> None:
        """
        Close browser resources and stop Playwright cleanly.
        """
        browser = self._browser
        playwright = self._playwright
        self._browser = None
        self._playwright = None

        if browser is not None:
            try:
                await browser.close()
            except PlaywrightError:
                logger.warning("Browser close failed", exc_info=True)

        if playwright is not None:
            try:
                await playwright.stop()
            except PlaywrightError:
                logger.warning("Playwright stop failed", exc_info=True)

        self._release_browser_slot()

    @asynccontextmanager
    async def page(
        self, context_options: BrowserContextOptions | None = None
    ) -> AsyncIterator[Page]:
        """
        Yield an isolated page and always close its context afterward.
        """
        await self.start()
        await self._context_slots.acquire()

        context: BrowserContext | None = None
        page: Page | None = None

        try:
            context = await self._new_context(context_options)
            if self.config.block_resource_types:
                await self._install_resource_blocking(context)

            page = await context.new_page()
            page.set_default_timeout(self.config.action_timeout_ms)
            page.set_default_navigation_timeout(self.config.navigation_timeout_ms)
            yield page
        finally:
            if page is not None and not page.is_closed():
                await page.close()

            if context is not None:
                await context.close()

            self._context_slots.release()

    async def _new_context(
        self, context_options: BrowserContextOptions | None = None
    ) -> BrowserContext:
        if self._browser is None:
            raise BrowserStartupError("Browser is not started")

        viewport_width = context_options.viewport_width if context_options else None
        viewport_height = context_options.viewport_height if context_options else None
        user_agent = context_options.user_agent if context_options else None
        java_script_enabled = context_options.java_script_enabled if context_options else None

        options: dict[str, Any] = {
            "viewport": {
                "width": viewport_width or self.config.viewport_width,
                "height": viewport_height or self.config.viewport_height,
            },
            "java_script_enabled": (
                self.config.java_script_enabled
                if java_script_enabled is None
                else java_script_enabled
            ),
        }

        effective_user_agent = user_agent or self.config.user_agent
        if effective_user_agent:
            options["user_agent"] = effective_user_agent

        return await self._browser.new_context(**options)

    async def _install_resource_blocking(self, context: BrowserContext) -> None:
        blocked_types = set(self.config.block_resource_types)

        async def handle_route(route: Route) -> None:
            if route.request.resource_type in blocked_types:
                await route.abort()
                return
            await route.continue_()

        await context.route("**/*", handle_route)

    def _launch_args(self) -> list[str]:
        if self.config.browser_type != "chromium":
            return []

        return [
            "--disable-background-networking",
            "--disable-background-timer-throttling",
            "--disable-breakpad",
            "--disable-dev-shm-usage",
            "--disable-extensions",
            "--disable-features=site-per-process",
            "--disable-sync",
            "--no-first-run",
        ]

    def _acquire_browser_slot(self) -> None:
        with self._browser_count_lock:
            if self._browser_slot_acquired:
                return

            if self._active_browser_count >= self.config.max_browser_instances:
                raise BrowserStartupError(
                    f"Browser instance limit reached: {self.config.max_browser_instances}"
                )

            type(self)._active_browser_count += 1
            self._browser_slot_acquired = True

    def _release_browser_slot(self) -> None:
        with self._browser_count_lock:
            if not self._browser_slot_acquired:
                return

            type(self)._active_browser_count = max(0, self._active_browser_count - 1)
            self._browser_slot_acquired = False
