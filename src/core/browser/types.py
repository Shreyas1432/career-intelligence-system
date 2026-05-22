from dataclasses import dataclass
from typing import Literal

WaitUntilState = Literal["domcontentloaded", "load", "networkidle"]


@dataclass(frozen=True, slots=True)
class BrowserContextOptions:
    """
    Per-context browser overrides for isolated scraping sessions.
    """

    viewport_width: int | None = None
    viewport_height: int | None = None
    user_agent: str | None = None
    java_script_enabled: bool | None = None


@dataclass(frozen=True, slots=True)
class PageLoadOptions:
    """
    Controls page navigation and retry behavior for one request.
    """

    wait_until: WaitUntilState | None = None
    timeout_ms: int | None = None
    network_idle: bool = True
    retries: int | None = None


@dataclass(frozen=True, slots=True)
class PageSnapshot:
    """
    Captured browser page state.
    """

    requested_url: str
    final_url: str
    html: str
    rendered_dom: str
    text: str
    screenshot: bytes | None = None
