import logging
import sys
from logging.handlers import RotatingFileHandler
import os
from datetime import datetime
import json
from typing import Any, Dict


class CustomJSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        if hasattr(record, "extra_data"):
            log_data.update(record.extra_data)

        return json.dumps(log_data)


class ContextLogger(logging.Logger):
    """Logger that supports context information."""

    def _log_with_context(
        self,
        level: int,
        msg: str,
        args: tuple,
        exc_info: Any = None,
        extra: Dict[str, Any] = None,
        stack_info: bool = False,
        context: Dict[str, Any] = None,
    ) -> None:
        """Log with additional context information."""
        if context:
            if not extra:
                extra = {}
            extra["extra_data"] = context

        super().log(level, msg, args, exc_info, extra, stack_info)

    def debug_with_context(
        self, msg: str, *args: Any, context: Dict[str, Any] = None, **kwargs: Any
    ) -> None:
        """Log debug message with context."""
        self._log_with_context(logging.DEBUG, msg, args, context=context, **kwargs)

    def info_with_context(
        self, msg: str, *args: Any, context: Dict[str, Any] = None, **kwargs: Any
    ) -> None:
        """Log info message with context."""
        self._log_with_context(logging.INFO, msg, args, context=context, **kwargs)

    def warning_with_context(
        self, msg: str, *args: Any, context: Dict[str, Any] = None, **kwargs: Any
    ) -> None:
        """Log warning message with context."""
        self._log_with_context(logging.WARNING, msg, args, context=context, **kwargs)

    def error_with_context(
        self, msg: str, *args: Any, context: Dict[str, Any] = None, **kwargs: Any
    ) -> None:
        """Log error message with context."""
        self._log_with_context(logging.ERROR, msg, args, context=context, **kwargs)


def setup_logging(
    log_level: str = "INFO",
    log_file: str = "app.log",
    max_bytes: int = 10485760,  # 10MB
    backup_count: int = 5,
    json_format: bool = True,
) -> None:
    """
    Set up logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup files to keep
        json_format: Whether to use JSON formatting
    """
    # Register custom logger class
    logging.setLoggerClass(ContextLogger)

    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create handlers
    handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    handlers.append(console_handler)

    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_file, maxBytes=max_bytes, backupCount=backup_count
    )
    handlers.append(file_handler)

    # Set formatter
    if json_format:
        formatter = CustomJSONFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    # Configure handlers
    for handler in handlers:
        handler.setFormatter(formatter)

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()), handlers=handlers, force=True
    )

    # Log startup message
    logger = logging.getLogger(__name__)
    logger.info(
        "Logging configured",
        extra={"extra_data": {"log_level": log_level, "log_file": log_file}},
    )


def get_logger(name: str) -> ContextLogger:
    """
    Get a logger instance with the specified name.

    Args:
        name: Logger name

    Returns:
        ContextLogger instance
    """
    return logging.getLogger(name)
