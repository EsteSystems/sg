"""Cross-locus failure analysis — discover error patterns spanning multiple loci.

Records per-locus errors, normalizes them, and detects when the same error
pattern appears across 3+ loci with 10+ total occurrences.  Generates
LocusProposal items suggesting new genes to handle the cross-cutting concern.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from sg.decomposition import _normalize_error
from sg.filelock import atomic_write_text, file_lock_shared
from sg.log import get_logger

logger = get_logger("locus_discovery")

MIN_LOCI_FOR_PROPOSAL = 3
MIN_OCCURRENCES_FOR_PROPOSAL = 10
MAX_CROSS_LOCUS_PATTERNS = 200
MAX_PROPOSALS = 50
MAX_REPRESENTATIVE = 5


@dataclass
class CrossLocusPattern:
    """A normalized error pattern seen across multiple loci."""
    pattern: str = ""
    loci: dict[str, int] = field(default_factory=dict)
    total_count: int = 0
    representative_messages: list[str] = field(default_factory=list)
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "pattern": self.pattern,
            "loci": self.loci,
            "total_count": self.total_count,
            "representative_messages": self.representative_messages[:MAX_REPRESENTATIVE],
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
        }

    @classmethod
    def from_dict(cls, d: dict) -> CrossLocusPattern:
        return cls(
            pattern=d.get("pattern", ""),
            loci=d.get("loci", {}),
            total_count=d.get("total_count", 0),
            representative_messages=d.get("representative_messages", []),
            first_seen=d.get("first_seen", 0.0),
            last_seen=d.get("last_seen", 0.0),
        )


@dataclass
class LocusProposal:
    """A proposed new gene to handle a cross-locus concern."""
    proposed_name: str = ""
    family: str = "diagnostic"
    description: str = ""
    evidence_pattern: str = ""
    affected_loci: list[str] = field(default_factory=list)
    occurrence_count: int = 0
    created_at: float = field(default_factory=time.time)
    status: str = "pending"

    def to_dict(self) -> dict:
        return {
            "proposed_name": self.proposed_name,
            "family": self.family,
            "description": self.description,
            "evidence_pattern": self.evidence_pattern,
            "affected_loci": self.affected_loci,
            "occurrence_count": self.occurrence_count,
            "created_at": self.created_at,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, d: dict) -> LocusProposal:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class CrossLocusFailureAnalyzer:
    """Tracks error patterns across loci and proposes new genes."""

    def __init__(self) -> None:
        self.patterns: dict[str, CrossLocusPattern] = {}
        self.proposals: list[LocusProposal] = []
        self._path: Path | None = None

    def record_error(
        self, locus: str, error_message: str,
    ) -> LocusProposal | None:
        """Record an error from a locus.  Returns proposal if threshold met."""
        normalized = _normalize_error(error_message)
        if not normalized:
            return None

        pat = self.patterns.get(normalized)
        if pat is None:
            if len(self.patterns) >= MAX_CROSS_LOCUS_PATTERNS:
                self._evict_oldest()
            pat = CrossLocusPattern(pattern=normalized)
            self.patterns[normalized] = pat

        pat.loci[locus] = pat.loci.get(locus, 0) + 1
        pat.total_count += 1
        pat.last_seen = time.time()

        if (len(pat.representative_messages) < MAX_REPRESENTATIVE
                and error_message not in pat.representative_messages):
            pat.representative_messages.append(error_message)

        if (len(pat.loci) >= MIN_LOCI_FOR_PROPOSAL
                and pat.total_count >= MIN_OCCURRENCES_FOR_PROPOSAL):
            existing = {p.evidence_pattern for p in self.proposals}
            if normalized not in existing:
                proposal = self._generate_proposal(pat)
                self.proposals.append(proposal)
                if len(self.proposals) > MAX_PROPOSALS:
                    self.proposals = self.proposals[-MAX_PROPOSALS:]
                return proposal

        return None

    def _generate_proposal(self, pattern: CrossLocusPattern) -> LocusProposal:
        """Generate a locus proposal from a cross-locus error pattern."""
        text = pattern.pattern.lower()

        if "not found" in text or "does not exist" in text:
            family = "diagnostic"
            desc = f"Diagnostic gene to check for resource existence before use"
        elif "timeout" in text or "timed out" in text:
            family = "diagnostic"
            desc = f"Diagnostic gene to monitor response times"
        elif "permission" in text or "denied" in text:
            family = "diagnostic"
            desc = f"Diagnostic gene to verify permissions"
        elif "setup" in text or "init" in text or "config" in text:
            family = "configuration"
            desc = f"Configuration gene for prerequisite setup"
        else:
            family = "diagnostic"
            desc = f"Diagnostic gene for cross-cutting concern: {pattern.pattern[:60]}"

        # Derive name from pattern keywords
        words = [w for w in text.split() if len(w) > 2 and w.isalpha()]
        name_parts = words[:3] if words else ["cross_locus"]
        proposed_name = "check_" + "_".join(name_parts)

        return LocusProposal(
            proposed_name=proposed_name,
            family=family,
            description=desc,
            evidence_pattern=pattern.pattern,
            affected_loci=list(pattern.loci.keys()),
            occurrence_count=pattern.total_count,
        )

    def get_proposals(self, status: str = "pending") -> list[LocusProposal]:
        return [p for p in self.proposals if p.status == status]

    def accept_proposal(self, index: int) -> bool:
        pending = [p for p in self.proposals if p.status == "pending"]
        if 0 <= index < len(pending):
            pending[index].status = "accepted"
            return True
        return False

    def reject_proposal(self, index: int) -> bool:
        pending = [p for p in self.proposals if p.status == "pending"]
        if 0 <= index < len(pending):
            pending[index].status = "rejected"
            return True
        return False

    def _evict_oldest(self) -> None:
        """Remove the oldest pattern to stay under the cap."""
        if not self.patterns:
            return
        oldest_key = min(self.patterns, key=lambda k: self.patterns[k].last_seen)
        del self.patterns[oldest_key]

    # --- Persistence ---

    def save(self, path: Path | None = None) -> None:
        p = path or self._path
        if p is None:
            return
        data = {
            "patterns": {k: v.to_dict() for k, v in self.patterns.items()},
            "proposals": [p.to_dict() for p in self.proposals],
        }
        p.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(p, json.dumps(data, indent=2))

    @classmethod
    def open(cls, path: Path) -> CrossLocusFailureAnalyzer:
        analyzer = cls()
        analyzer._path = path
        if not path.exists():
            return analyzer
        try:
            with file_lock_shared(path):
                data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            logger.warning("corrupted locus discovery state, starting fresh")
            return analyzer
        for k, v in data.get("patterns", {}).items():
            analyzer.patterns[k] = CrossLocusPattern.from_dict(v)
        analyzer.proposals = [
            LocusProposal.from_dict(p)
            for p in data.get("proposals", [])
        ]
        return analyzer
