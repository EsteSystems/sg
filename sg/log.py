"""Structured logging for Software Genomics.

Provides JSON and human-readable formatters, correlation IDs for tracing
requests through the evolutionary loop, and a central ``configure_logging``
entry point wired from the CLI.

Usage::

    from sg.log import get_logger, configure_logging, correlation_scope

    configure_logging(level="DEBUG", json_format=True)
    logger = get_logger("orchestrator")

    with correlation_scope() as cid:
        logger.info("executing locus", extra={"locus": "bridge_create"})
"""
from __future__ import annotations

import contextvars
import json
import logging
import time
import uuid
from contextlib import contextmanager
from typing import Generator

_correlation_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "sg_correlation_id", default=None
)


def get_correlation_id() -> str | None:
    """Return the current correlation ID, or *None* if outside a scope."""
    return _correlation_id.get()


@contextmanager
def correlation_scope(cid: str | None = None) -> Generator[str, None, None]:
    """Context manager that sets a correlation ID for the duration of its block.

    If *cid* is ``None`` a new UUID-4 hex string is generated.
    Nested scopes are supported — the outer ID is restored on exit.
    """
    cid = cid or uuid.uuid4().hex[:12]
    token = _correlation_id.set(cid)
    try:
        yield cid
    finally:
        _correlation_id.reset(token)


class JSONFormatter(logging.Formatter):
    """Emit each record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "ts": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        cid = _correlation_id.get()
        if cid:
            payload["correlation_id"] = cid

        # Include any extra keys passed via ``extra={...}``
        for key in ("locus", "sha", "fitness", "allele", "pathway", "event"):
            val = getattr(record, key, None)
            if val is not None:
                payload[key] = val

        if record.exc_info and record.exc_info[1]:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


class HumanFormatter(logging.Formatter):
    """Human-readable format: ``[cid] [logger] level: message``."""

    def format(self, record: logging.LogRecord) -> str:
        cid = _correlation_id.get()
        cid_part = f"[{cid}] " if cid else ""
        base = f"{cid_part}[{record.name}] {record.levelname}: {record.getMessage()}"
        if record.exc_info and record.exc_info[1]:
            base += "\n" + self.formatException(record.exc_info)
        return base


_configured = False


def configure_logging(
    level: str = "INFO",
    json_format: bool = False,
) -> None:
    """Configure the ``sg`` logger hierarchy.

    Safe to call multiple times — only the first call attaches handlers.
    Subsequent calls update the level.
    """
    global _configured
    root = logging.getLogger("sg")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    if not _configured:
        handler = logging.StreamHandler()
        formatter: logging.Formatter
        if json_format:
            formatter = JSONFormatter()
        else:
            formatter = HumanFormatter()
        handler.setFormatter(formatter)
        root.addHandler(handler)
        _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the ``sg.`` namespace."""
    return logging.getLogger(f"sg.{name}")
