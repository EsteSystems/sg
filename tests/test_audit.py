"""Tests for sg.audit — append-only audit trail."""
from __future__ import annotations

import json

import pytest

from sg.audit import AuditLog, AuditEntry, MAX_AUDIT_ENTRIES, MAX_AUDIT_SIZE


class TestAuditLog:
    def test_record_creates_file(self, tmp_path):
        log = AuditLog(tmp_path / "sub" / "audit.jsonl")
        entry = log.record("promotion", locus="bridge_create", sha="abc123")
        assert entry.event == "promotion"
        assert entry.locus == "bridge_create"
        assert entry.sha == "abc123"
        assert (tmp_path / "sub" / "audit.jsonl").exists()

    def test_record_appends(self, tmp_path):
        log = AuditLog(tmp_path / "audit.jsonl")
        log.record("promotion", locus="a")
        log.record("demotion", locus="b")
        lines = (tmp_path / "audit.jsonl").read_text().strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["event"] == "promotion"
        assert json.loads(lines[1])["event"] == "demotion"

    def test_read_all(self, tmp_path):
        log = AuditLog(tmp_path / "audit.jsonl")
        log.record("a")
        log.record("b")
        log.record("c")
        entries = log.read_all()
        assert len(entries) == 3
        assert entries[0].event == "a"
        assert entries[2].event == "c"

    def test_read_all_empty(self, tmp_path):
        log = AuditLog(tmp_path / "audit.jsonl")
        assert log.read_all() == []

    def test_read_recent(self, tmp_path):
        log = AuditLog(tmp_path / "audit.jsonl")
        for i in range(10):
            log.record(f"event-{i}")
        recent = log.read_recent(3)
        assert len(recent) == 3
        assert recent[0].event == "event-7"
        assert recent[2].event == "event-9"

    def test_details_stored(self, tmp_path):
        log = AuditLog(tmp_path / "audit.jsonl")
        log.record("mutation", locus="x", sha="y", attempt=3, fitness=0.85)
        entries = log.read_all()
        assert entries[0].details["attempt"] == 3
        assert entries[0].details["fitness"] == 0.85

    def test_timestamp_present(self, tmp_path):
        log = AuditLog(tmp_path / "audit.jsonl")
        entry = log.record("test")
        assert entry.timestamp > 0


    def test_rotation_caps_entries(self, tmp_path, monkeypatch):
        """When file exceeds size limit, rotation trims to MAX_AUDIT_ENTRIES."""
        monkeypatch.setattr("sg.audit.MAX_AUDIT_SIZE", 500)
        monkeypatch.setattr("sg.audit.MAX_AUDIT_ENTRIES", 10)
        log = AuditLog(tmp_path / "audit.jsonl")
        for i in range(50):
            log.record(f"event-{i}", locus="test")
        lines = (tmp_path / "audit.jsonl").read_text().strip().splitlines()
        assert len(lines) <= 10

    def test_rotation_preserves_recent(self, tmp_path, monkeypatch):
        """After rotation, the most recent entries are preserved."""
        monkeypatch.setattr("sg.audit.MAX_AUDIT_SIZE", 500)
        monkeypatch.setattr("sg.audit.MAX_AUDIT_ENTRIES", 10)
        log = AuditLog(tmp_path / "audit.jsonl")
        for i in range(50):
            log.record(f"event-{i}", locus="test")
        recent = log.read_recent(5)
        # The last 5 entries should be the most recent
        assert recent[-1].event == "event-49"
        assert recent[-2].event == "event-48"


class TestAuditEntry:
    def test_round_trip(self):
        entry = AuditEntry(
            timestamp=1000.0, event="promotion",
            locus="bridge_create", sha="abc",
            details={"fitness": 0.9},
        )
        d = entry.to_dict()
        restored = AuditEntry.from_dict(d)
        assert restored.event == "promotion"
        assert restored.details["fitness"] == 0.9
