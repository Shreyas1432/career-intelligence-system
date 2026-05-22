import contextvars
import logging
import logging.config
import sys
from typing import Any, cast

import structlog

from src.core.config import settings

# Async-safe ContextVar to manage request/job correlation IDs
_CORRELATION_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "correlation_id", default=None
)


def set_correlation_id(correlation_id: str | None) -> None:
    """
    Sets the current thread/task context correlation ID.
    """
    _CORRELATION_ID.set(correlation_id)


def get_correlation_id() -> str | None:
    """
    Retrieves the current context correlation ID.
    """
    return _CORRELATION_ID.get()


def correlation_id_processor(
    _logger: Any, _method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """
    Structlog processor that injects the correlation ID if set.
    """
    corr_id = _CORRELATION_ID.get()
    if corr_id:
        event_dict["correlation_id"] = corr_id
    return event_dict


def configure_logging() -> None:
    """
    Configures standard library logging and structlog processors.
    Utilizes JSON format for production environments and a colored, readable
    console output for local development.
    """
    # Define structlog processors list
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        correlation_id_processor,
    ]

    # Detect if we should use console-friendly formatting or JSON
    is_development = settings.env == "development"

    renderer: Any
    if is_development:
        # Development console-friendly renderer
        renderer = structlog.dev.ConsoleRenderer(colors=True)
    else:
        # Production JSON renderer
        renderer = structlog.processors.JSONRenderer()

    # Configure structlog
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Standard logging configuration wrapper
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=[
            structlog.stdlib.ExtraAdder(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
        ],
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    # Configure stdout logging handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()

    # Remove existing handlers to avoid duplicate prints
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)

    root_logger.addHandler(handler)

    # Set default system levels from configuration
    log_level_str = settings.logging.level.upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    root_logger.setLevel(log_level)

    # Prevent external libraries from polluting logs with DEBUG messages
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Module-aware logger factory method.
    Returns a configured structlog BoundLogger.
    """
    return cast(structlog.stdlib.BoundLogger, structlog.get_logger(name))
