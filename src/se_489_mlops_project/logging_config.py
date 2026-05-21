"""Centralized logging configuration.

Provides two handlers:
  - Console  : ``rich.logging.RichHandler`` for colored, human-readable output.
  - File     : ``RotatingFileHandler`` writing to ``logs/app.log`` (5 MB × 3 backups).

``rich.traceback.install()`` is called once so unhandled exceptions print a
nicely formatted traceback in the terminal.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Literal

from rich.logging import RichHandler
from rich.traceback import install as _install_rich_traceback

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

_LOGS_DIR = Path(__file__).resolve().parents[2] / "logs"
_LOG_FILE = _LOGS_DIR / "app.log"
_MAX_BYTES = 5 * 1024 * 1024  # 5 MB
_BACKUP_COUNT = 3

# Install rich tracebacks once at import time.
_install_rich_traceback(show_locals=False)


def setup_logging(level: LogLevel = "INFO") -> None:
    """Configure the root logger for the application.

    Sets up a Rich console handler (colored, formatted) and a rotating file
    handler.  Idempotent: safe to call multiple times.
    """
    root = logging.getLogger()
    root.setLevel(level)

    # Remove any handlers added by a previous call or by basicConfig.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    # --- Console handler (Rich) ---
    console_handler = RichHandler(
        level=level,
        rich_tracebacks=True,
        show_path=False,
        markup=True,
    )
    root.addHandler(console_handler)

    # --- File handler (rotating) ---
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        filename=_LOG_FILE,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger."""
    return logging.getLogger(name)
