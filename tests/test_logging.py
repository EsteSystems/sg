"""Tests for sg.log — structured logging."""
from __future__ import annotations

import json
import logging

import pytest

from sg.log import (
    JSONFormatter,
    HumanFormatter,
    configure_logging,
    correlation_scope,
    get_correlation_id,
    get_logger,
)


# ---------------------------------------------------------------------------
# Correlation ID
# ---------------------------------------------------------------------------

class TestCorrelationId:
    def test_none_outside_scope(self):
        assert get_correlation_id() is None

    def test_scope_sets_id(self):
        with correlation_scope() as cid:
            assert get_correlation_id() == cid
            assert len(cid) == 12  # uuid4 hex[:12]

    def test_explicit_cid(self):
        with correlation_scope("abc123") as cid:
            assert cid == "abc123"
            assert get_correlation_id() == "abc123"

    def test_nested_scopes(self):
        with correlation_scope("outer") as outer:
            assert get_correlation_id() == "outer"
            with correlation_scope("inner") as inner:
                assert get_correlation_id() == "inner"
            assert get_correlation_id() == "outer"
        assert get_correlation_id() is None


# ---------------------------------------------------------------------------
# JSONFormatter
# ---------------------------------------------------------------------------

class TestJSONFormatter:
    def _make_record(self, msg="test message", **extras):
        logger = logging.getLogger("sg.test.json")
        record = logger.makeRecord(
            "sg.test.json", logging.INFO, "test.py", 1, msg, (), None
        )
        for k, v in extras.items():
            setattr(record, k, v)
        return record

    def test_basic_output(self):
        fmt = JSONFormatter()
        record = self._make_record()
        line = fmt.format(record)
        data = json.loads(line)
        assert data["level"] == "INFO"
        assert data["logger"] == "sg.test.json"
        assert data["msg"] == "test message"
        assert "correlation_id" not in data

    def test_with_correlation_id(self):
        fmt = JSONFormatter()
        with correlation_scope("cid42"):
            record = self._make_record()
            data = json.loads(fmt.format(record))
            assert data["correlation_id"] == "cid42"

    def test_extra_fields(self):
        fmt = JSONFormatter()
        record = self._make_record(locus="bridge_create", sha="abc123")
        data = json.loads(fmt.format(record))
        assert data["locus"] == "bridge_create"
        assert data["sha"] == "abc123"

    def test_exception_included(self):
        fmt = JSONFormatter()
        try:
            raise ValueError("boom")
        except ValueError:
            import sys
            exc_info = sys.exc_info()

        logger = logging.getLogger("sg.test.json")
        record = logger.makeRecord(
            "sg.test.json", logging.ERROR, "test.py", 1, "fail", (), exc_info
        )
        data = json.loads(fmt.format(record))
        assert "exception" in data
        assert "boom" in data["exception"]


# ---------------------------------------------------------------------------
# HumanFormatter
# ---------------------------------------------------------------------------

class TestHumanFormatter:
    def _make_record(self, msg="hello"):
        logger = logging.getLogger("sg.test.human")
        return logger.makeRecord(
            "sg.test.human", logging.WARNING, "test.py", 1, msg, (), None
        )

    def test_basic_format(self):
        fmt = HumanFormatter()
        record = self._make_record()
        line = fmt.format(record)
        assert "[sg.test.human]" in line
        assert "WARNING" in line
        assert "hello" in line

    def test_with_correlation_id(self):
        fmt = HumanFormatter()
        with correlation_scope("req99"):
            record = self._make_record()
            line = fmt.format(record)
            assert line.startswith("[req99]")


# ---------------------------------------------------------------------------
# configure_logging / get_logger
# ---------------------------------------------------------------------------

class TestConfigureLogging:
    def test_get_logger_namespace(self):
        logger = get_logger("mymod")
        assert logger.name == "sg.mymod"

    def test_configure_sets_level(self):
        # Reset _configured so we can test
        import sg.log
        sg.log._configured = False
        # Remove existing handlers to avoid accumulation
        root = logging.getLogger("sg")
        root.handlers.clear()

        configure_logging(level="DEBUG", json_format=False)
        assert logging.getLogger("sg").level == logging.DEBUG

        # Calling again only updates level
        configure_logging(level="WARNING")
        assert logging.getLogger("sg").level == logging.WARNING
        # Should still have exactly one handler
        assert len(logging.getLogger("sg").handlers) == 1

        # Cleanup
        root.handlers.clear()
        sg.log._configured = False
