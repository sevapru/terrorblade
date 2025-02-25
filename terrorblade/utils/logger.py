import logging
import sys
from pathlib import Path
from typing import Any, Optional


class ColorFormatter(logging.Formatter):
    grey = "\033[37m"
    blue = "\033[36m"  # Cyan instead of bright blue
    yellow = "\033[33m"  # Soft yellow
    red = "\033[91m"  # Light red
    bold_red = "\033[31m"  # Red for critical errors
    green = "\033[32m"  # Green for NICE
    reset = "\033[0m"

    format_str = "[ %(asctime)s ] %(name)s - %(levelname)-8s %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format_str + reset,
        logging.INFO: blue + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset,
        25: green + format_str + reset,  # NICE level between INFO(20) and WARNING(30)
    }

    def format(self, record: logging.LogRecord) -> str:
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


# Register new log level
NICE = 25
logging.addLevelName(NICE, "NICE")


def nice(self: logging.Logger, message: str, *args: Any, **kwargs: Any) -> None:
    """Log 'msg % args' with severity 'NICE'."""
    if self.isEnabledFor(NICE):
        self._log(NICE, message, args, **kwargs)


def Logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_dir: Optional[str] = None,
) -> logging.Logger:
    """
    Setup and configure logger with color formatting

    Args:
        name (str): Logger name that will be displayed in logs
        level (int): Logging level (default: logging.INFO)
        log_file (str, optional): Name of the log file
        log_dir (str, optional): Directory for log files

    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Console handler with color formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(ColorFormatter())
    logger.addHandler(console_handler)

    # File handler if log_file is specified
    if log_file and log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path / log_file, encoding="utf-8")
        file_handler.setLevel(level)
        # Use simple formatter for file logs (without colors)
        file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)-8s - %(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger
