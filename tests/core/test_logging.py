import asyncio

import pytest
import structlog

from src.core.logging import (
    configure_logging,
    correlation_id_processor,
    get_correlation_id,
    get_logger,
    set_correlation_id,
)


def test_logger_factory():
    """
    Ensure the module-aware factory returns a valid structlog BoundLogger.
    """
    configure_logging()
    logger = get_logger("test.module")
    assert (
        isinstance(logger, structlog.stdlib.BoundLogger)
        or type(logger).__name__ == "BoundLoggerLazyProxy"
    )
    assert hasattr(logger, "bind")
    assert hasattr(logger, "info")


def test_correlation_id_management():
    """
    Verify correlation ID gets set and retrieved correctly in the context.
    """
    set_correlation_id(None)
    assert get_correlation_id() is None

    set_correlation_id("job-1234")
    assert get_correlation_id() == "job-1234"

    # Reset
    set_correlation_id(None)
    assert get_correlation_id() is None


@pytest.mark.asyncio
async def test_correlation_id_async_safety():
    """
    Verify that correlation IDs are isolated across async tasks (async-safe).
    """

    async def task_one():
        set_correlation_id("task-1")
        await asyncio.sleep(0.05)
        return get_correlation_id()

    async def task_two():
        set_correlation_id("task-2")
        await asyncio.sleep(0.01)  # Finish before task_one resumes
        return get_correlation_id()

    res_one, res_two = await asyncio.gather(task_one(), task_two())

    assert res_one == "task-1"
    assert res_two == "task-2"


def test_correlation_id_processor():
    """
    Verify the correlation processor injects values into the event dictionary.
    """
    set_correlation_id("corr-xyz")

    event_dict = {"event": "test message"}
    processed_dict = correlation_id_processor(None, "info", event_dict)

    assert "correlation_id" in processed_dict
    assert processed_dict["correlation_id"] == "corr-xyz"

    # Clean up
    set_correlation_id(None)
