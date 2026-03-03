"""Failure mode discovery — proposes new 'fails when' contract entries.

Watches error patterns flowing through execution, compares against
existing ``fails when`` entries in the gene contract, and proposes
new entries when novel patterns appear with enough frequency.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from sg.decomposition import _normalize_error
from sg.filelock import atomic_write_text, file_lock
from sg.log import get_logger

logger = get_logger("failure_discovery")

PROPOSAL_THRESHOLD = 5
MAX_PROPOSALS = 50
MAX_TRACKED_PATTERNS = 100

# Words to ignore when checking coverage overlap
_STOP_WORDS = frozenset({
    "<n>", "<hex>", "<hash>", "<path>", "<str>",
    "the", "a", "an", "is", "not", "in", "at", "for",
    "of", "to", "and", "or", "was", "with", "no", "on",
})


@dataclass
class NovelErrorPattern:
    """A normalized error pattern not covered by existing fails_when."""
    pattern: str = ""
    count: int = 0
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    representative_messages: list[str] = field(default_factory=list)
    allele_shas: list[str] = field(default_factory=list)

    MAX_REPRESENTATIVES = 3
    MAX_SHAS = 5

    def to_dict(self) -> dict:
        return {
            "pattern": self.pattern,
            "count": self.count,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "representative_messages": self.representative_messages,
            "allele_shas": self.allele_shas,
        }

    @classmethod
    def from_dict(cls, d: dict) -> NovelErrorPattern:
        return cls(
            pattern=d.get("pattern", ""),
            count=d.get("count", 0),
            first_seen=d.get("first_seen", 0.0),
            last_seen=d.get("last_seen", 0.0),
            representative_messages=d.get("representative_messages", []),
            allele_shas=d.get("allele_shas", []),
        )


@dataclass
class FailureProposal:
    """A proposed 'fails when' entry for a gene contract."""
    locus: str = ""
    proposed_text: str = ""
    pattern: str = ""
    occurrence_count: int = 0
    representative_messages: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    status: str = "pending"

    def to_dict(self) -> dict:
        return {
            "locus": self.locus,
            "proposed_text": self.proposed_text,
            "pattern": self.pattern,
            "occurrence_count": self.occurrence_count,
            "representative_messages": self.representative_messages,
            "created_at": self.created_at,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, d: dict) -> FailureProposal:
        return cls(
            locus=d.get("locus", ""),
            proposed_text=d.get("proposed_text", ""),
            pattern=d.get("pattern", ""),
            occurrence_count=d.get("occurrence_count", 0),
            representative_messages=d.get("representative_messages", []),
            created_at=d.get("created_at", 0.0),
            status=d.get("status", "pending"),
        )


class FailureDiscovery:
    """Tracks novel error patterns and proposes 'fails when' entries."""

    def __init__(self) -> None:
        self.patterns: dict[str, dict[str, NovelErrorPattern]] = {}
        self.proposals: dict[str, list[FailureProposal]] = {}

    def record_error(
        self,
        locus: str,
        sha: str,
        error_message: str,
        known_fails_when: list[str],
    ) -> FailureProposal | None:
        """Record an error and return a proposal if threshold is met."""
        normalized = _normalize_error(error_message)

        if self._is_covered(normalized, known_fails_when):
            return None

        if locus not in self.patterns:
            self.patterns[locus] = {}
        patterns = self.patterns[locus]

        if normalized not in patterns:
            patterns[normalized] = NovelErrorPattern(pattern=normalized)
            if len(patterns) > MAX_TRACKED_PATTERNS:
                oldest = min(patterns.values(), key=lambda p: p.last_seen)
                del patterns[oldest.pattern]

        entry = patterns[normalized]
        entry.count += 1
        entry.last_seen = time.time()
        if len(entry.representative_messages) < NovelErrorPattern.MAX_REPRESENTATIVES:
            entry.representative_messages.append(error_message)
        if sha not in entry.allele_shas:
            entry.allele_shas.append(sha)
            if len(entry.allele_shas) > NovelErrorPattern.MAX_SHAS:
                entry.allele_shas = entry.allele_shas[-NovelErrorPattern.MAX_SHAS:]

        if entry.count >= PROPOSAL_THRESHOLD:
            if not self._already_proposed(locus, normalized):
                proposal = self._create_proposal(locus, entry)
                if locus not in self.proposals:
                    self.proposals[locus] = []
                self.proposals[locus].append(proposal)
                if len(self.proposals[locus]) > MAX_PROPOSALS:
                    self.proposals[locus] = self.proposals[locus][-MAX_PROPOSALS:]
                logger.info(
                    "failure mode discovered for '%s': %s (%d occurrences)",
                    locus, normalized, entry.count,
                )
                return proposal
        return None

    @staticmethod
    def _is_covered(normalized: str, known_fails_when: list[str]) -> bool:
        """Check if a normalized error pattern is semantically covered."""
        pattern_words = set(normalized.lower().split()) - _STOP_WORDS
        if not pattern_words:
            return False
        for fw in known_fails_when:
            fw_words = set(fw.lower().split())
            overlap = len(pattern_words & fw_words)
            if overlap / len(pattern_words) > 0.5:
                return True
        return False

    def _already_proposed(self, locus: str, pattern: str) -> bool:
        for p in self.proposals.get(locus, []):
            if p.pattern == pattern:
                return True
        return False

    @staticmethod
    def _create_proposal(locus: str, entry: NovelErrorPattern) -> FailureProposal:
        """Create a human-readable 'fails when' proposal from a pattern."""
        text = entry.pattern
        text = text.replace("<N>", "N")
        text = text.replace("<HEX>", "address")
        text = text.replace("<HASH>", "hash")
        text = text.replace("<PATH>", "path")
        text = text.replace("'<STR>'", "value")
        text = text.replace('"<STR>"', "value")
        return FailureProposal(
            locus=locus,
            proposed_text=text,
            pattern=entry.pattern,
            occurrence_count=entry.count,
            representative_messages=list(entry.representative_messages),
        )

    def get_proposals(
        self, locus: str, status: str = "pending",
    ) -> list[FailureProposal]:
        return [p for p in self.proposals.get(locus, []) if p.status == status]

    def get_all_proposals(self) -> dict[str, list[FailureProposal]]:
        return {
            locus: [p for p in proposals if p.status == "pending"]
            for locus, proposals in self.proposals.items()
            if any(p.status == "pending" for p in proposals)
        }

    def accept_proposal(self, locus: str, pattern: str) -> bool:
        for p in self.proposals.get(locus, []):
            if p.pattern == pattern and p.status == "pending":
                p.status = "accepted"
                return True
        return False

    def reject_proposal(self, locus: str, pattern: str) -> bool:
        for p in self.proposals.get(locus, []):
            if p.pattern == pattern and p.status == "pending":
                p.status = "rejected"
                return True
        return False

    # --- Persistence ---

    def save(self, path: Path) -> None:
        data = {
            "patterns": {
                locus: {p: e.to_dict() for p, e in entries.items()}
                for locus, entries in self.patterns.items()
            },
            "proposals": {
                locus: [p.to_dict() for p in proposals]
                for locus, proposals in self.proposals.items()
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with file_lock(path):
            atomic_write_text(path, json.dumps(data, indent=2))

    def load(self, path: Path) -> None:
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            logger.warning("failure discovery data corrupted at %s, starting fresh", path)
            return
        self.patterns = {
            locus: {
                p: NovelErrorPattern.from_dict(e)
                for p, e in entries.items()
            }
            for locus, entries in data.get("patterns", {}).items()
        }
        self.proposals = {
            locus: [FailureProposal.from_dict(p) for p in proposals]
            for locus, proposals in data.get("proposals", {}).items()
        }

    @classmethod
    def open(cls, path: Path) -> FailureDiscovery:
        fd = cls()
        fd.load(path)
        return fd
