class PipelineError(Exception):
    """
    Base exception for all errors occurring within the scraping orchestration pipeline.
    """

    pass


class StepExecutionError(PipelineError):
    """
    Raised when a specific step in the pipeline fails.
    """

    def __init__(self, step_name: str, message: str, original_error: Exception | None = None):
        super().__init__(f"Step '{step_name}' failed: {message}")
        self.step_name = step_name
        self.original_error = original_error


class PipelineFatalError(PipelineError):
    """
    Raised when a non-recoverable error halts the scraping pipeline execution entirely.
    """

    pass
