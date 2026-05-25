"""
Example script demonstrating usage of the Structlog structured logging module.
To run this:
    uv run python examples/logging_usage.py
"""

from src.core.logging import configure_logging, get_logger, set_correlation_id

# Initialize logging configuration at module level — safe because configure_logging
# has no environment-injection dependencies and is idempotent.
configure_logging()

# Module-aware logger bound to this example's namespace.
logger = get_logger("examples.logging_usage")


def simulate_job_execution(job_id: str) -> None:
    # Bind a correlation ID to trace logs for this execution context
    set_correlation_id(job_id)

    logger.info("Starting processing task", status="initialized", task_count=3)

    try:
        # Contextual metadata binding
        context_logger = logger.bind(step="resume_parsing")
        context_logger.info("Reading document content", format="pdf", size_kb=340)

        # Simulate business error log
        context_logger.warn("Parsing mismatch detected", skipped_sections=["hobbies"])

        # Simulate unhandled crash logging
        raise ValueError("Unsupported resume format schema")

    except Exception as e:
        logger.error(
            "Fatal error during job run",
            error=str(e),
            exc_info=True,
            status="failed",
        )
    finally:
        # Clear correlation ID after task finishes
        set_correlation_id(None)
        logger.info("Job processing finalized", status="cleared")


def main() -> None:
    print("=== SCENARIO 1: Processing Job Alpha ===")
    simulate_job_execution("job-alpha-001")

    print("\n=== SCENARIO 2: Processing Job Beta ===")
    simulate_job_execution("job-beta-099")


if __name__ == "__main__":
    main()
