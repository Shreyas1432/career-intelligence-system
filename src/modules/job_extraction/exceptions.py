class JobExtractionError(RuntimeError):
    """
    Base exception for intelligent job extraction failures.
    """


class ScrapeGraphAIUnavailableError(JobExtractionError):
    """
    Raised when ScrapeGraphAI is not installed or cannot be imported.
    """


class JobExtractionTimeoutError(JobExtractionError):
    """
    Raised when extraction exceeds the configured timeout.
    """


class JobExtractionValidationError(JobExtractionError):
    """
    Raised when extraction output cannot be validated.
    """
