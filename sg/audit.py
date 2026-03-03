"""Append-only audit trail — JSONL at .sg/audit.jsonl.

Records significant evolutionary events: promotions, demotions, mutations,
regressions, and pathway completions. Each entry is a single JSON line
with a timestamp, event type, locus/sha context, and free-form details.

Internal rotation: when the log exceeds MAX_AUDIT_SIZE bytes, it is
truncated to the most recent entries. read_recent() uses tail-seeking
for O(count) read cost.

Usage::

    from sg.audit import AuditLog

    audit = AuditLog(Path(".sg/audit.jsonl"))
    audit.record("promotion", locus="bridge_create", sha="abc123")
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from sg.filelock import file_lock, file_lock_shared

MAX_AUDIT_SIZE = 1_000_000  # 1 MB — triggers rotation
MAX_AUDIT_ENTRIES = 10_000  # keep this many entries after rotation


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
        """Append an audit entry.  Rotates when file exceeds MAX_AUDIT_SIZE."""
        entry = AuditEntry(
            timestamp=time.time(),
            event=event,
            locus=locus,
            sha=sha,
            details=details,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with file_lock(self.path):
            with open(self.path, "a") as f:
                f.write(json.dumps(entry.to_dict(), default=str) + "\n")
            self._maybe_rotate()
        return entry

    def _maybe_rotate(self) -> None:
        """Truncate to MAX_AUDIT_ENTRIES if file exceeds MAX_AUDIT_SIZE.

        Must be called while holding the file lock.
        """
        try:
            size = os.path.getsize(self.path)
        except OSError:
            return
        if size <= MAX_AUDIT_SIZE:
            return
        try:
            lines = self.path.read_text().splitlines()
        except OSError:
            return
        if len(lines) <= MAX_AUDIT_ENTRIES:
            return
        kept = lines[-MAX_AUDIT_ENTRIES:]
        self.path.write_text("\n".join(kept) + "\n")

    def read_all(self) -> list[AuditEntry]:
        """Read all entries from the log."""
        if not self.path.exists():
            return []
        with file_lock_shared(self.path):
            lines = self.path.read_text().splitlines()
        entries = []
        for line in lines:
            line = line.strip()
            if line:
                entries.append(AuditEntry.from_dict(json.loads(line)))
        return entries

    def read_recent(self, count: int = 100) -> list[AuditEntry]:
        """Read the most recent *count* entries via tail-seeking."""
        if not self.path.exists():
            return []
        with file_lock_shared(self.path):
            with open(self.path, "rb") as f:
                f.seek(0, 2)
                file_size = f.tell()
                if file_size == 0:
                    return []
                lines: list[str] = []
                chunk_size = 8192
                remaining = file_size
                leftover = b""
                while remaining > 0 and len(lines) < count:
                    read_size = min(chunk_size, remaining)
                    remaining -= read_size
                    f.seek(remaining)
                    chunk = f.read(read_size) + leftover
                    chunk_lines = chunk.split(b"\n")
                    leftover = chunk_lines[0]
                    for raw in reversed(chunk_lines[1:]):
                        line = raw.decode().strip()
                        if line:
                            lines.append(line)
                        if len(lines) >= count:
                            break
                if leftover and len(lines) < count:
                    line = leftover.decode().strip()
                    if line:
                        lines.append(line)
        entries = []
        for line in reversed(lines[:count]):
            entries.append(AuditEntry.from_dict(json.loads(line)))
        return entries
