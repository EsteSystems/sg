"""Append-only audit trail — JSONL at .sg/audit.jsonl.

Records significant evolutionary events: promotions, demotions, mutations,
regressions, and pathway completions. Each entry is a single JSON line
with a timestamp, event type, locus/sha context, and free-form details.

Usage::

    from sg.audit import AuditLog

    audit = AuditLog(Path(".sg/audit.jsonl"))
    audit.record("promotion", locus="bridge_create", sha="abc123")
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class AuditEntry:
    timestamp: float
    event: str
    locus: str = ""
    sha: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> AuditEntry:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class AuditLog:
    """Append-only JSONL audit log."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def record(self, event: str, locus: str = "", sha: str = "",
               **details: Any) -> AuditEntry:
        """Append an audit entry. Creates parent dirs if needed."""
        entry = AuditEntry(
            timestamp=time.time(),
            event=event,
            locus=locus,
            sha=sha,
            details=details,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(entry.to_dict(), default=str) + "\n")
        return entry

    def read_all(self) -> list[AuditEntry]:
        """Read all entries from the log."""
        if not self.path.exists():
            return []
        entries = []
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if line:
                entries.append(AuditEntry.from_dict(json.loads(line)))
        return entries

    def read_recent(self, count: int = 100) -> list[AuditEntry]:
        """Read the most recent *count* entries (tail of file)."""
        all_entries = self.read_all()
        return all_entries[-count:]
