class BrowserAutomationError(RuntimeError):
    """
    Base exception for Playwright browser automation failures.
    """


class BrowserStartupError(BrowserAutomationError):
    """
    Raised when Playwright or the configured browser cannot start.
    """


class BrowserNavigationError(BrowserAutomationError):
    """
    Raised when a page cannot be loaded after retry handling.
    """


class BrowserExtractionError(BrowserAutomationError):
    """
    Raised when page content extraction fails.
    """
