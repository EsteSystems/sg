"""Automatic decomposition — splitting unstable genes into pathways.

Detects when a gene fails for qualitatively different reasons (error
diversity) and signals that it should be decomposed into multiple
focused sub-genes composed by a pathway. Completes the bidirectional
fusion/decomposition cycle.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from sg.filelock import file_lock
from sg.log import get_logger

logger = get_logger("decomposition")

ERROR_WINDOW = 100
MIN_ERRORS_FOR_SIGNAL = 10
MIN_CLUSTERS_FOR_SIGNAL = 3
MAX_SPLIT_COUNT = 5
MAX_REPRESENTATIVE = 5


def _normalize_error(message: str) -> str:
    """Normalize an error message for clustering.

    Takes the first line, strips numbers, hex strings,
    file paths, and quoted strings.
    """
    first_line = message.split("\n")[0].strip()
    # Replace hex strings and hashes
    first_line = re.sub(r"0x[0-9a-fA-F]+", "<HEX>", first_line)
    first_line = re.sub(r"\b[0-9a-fA-F]{8,}\b", "<HASH>", first_line)
    # Replace numbers
    first_line = re.sub(r"\b\d+\b", "<N>", first_line)
    # Replace file paths
    first_line = re.sub(r"/[\w/.-]+", "<PATH>", first_line)
    # Replace quoted strings
    first_line = re.sub(r"'[^']*'", "'<STR>'", first_line)
    first_line = re.sub(r'"[^"]*"', '"<STR>"', first_line)
    return first_line


@dataclass
class ErrorCluster:
    """A group of similar error messages at a locus."""
    pattern: str
    count: int = 0
    representative_messages: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "pattern": self.pattern,
            "count": self.count,
            "representative_messages": self.representative_messages,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ErrorCluster:
        return cls(
            pattern=d.get("pattern", ""),
            count=d.get("count", 0),
            representative_messages=d.get("representative_messages", []),
        )


@dataclass
class DecompositionSignal:
    """Signal that a locus should be decomposed into a pathway."""
    locus: str
    error_clusters: list[ErrorCluster]
    recommended_split_count: int
    total_errors: int


@dataclass
class DecompositionResult:
    """Result of an LLM-assisted gene decomposition."""
    pathway_contract_source: str
    sub_gene_contract_sources: list[str]
    sub_gene_seed_sources: list[str]
    original_locus: str


@dataclass
class LocusErrorHistory:
    """Windowed error history for a single locus."""
    errors: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"errors": self.errors}

    @classmethod
    def from_dict(cls, d: dict) -> LocusErrorHistory:
        return cls(errors=d.get("errors", []))


class DecompositionDetector:
    """Detects when a locus should be decomposed based on error diversity.

    JSON-persisted following the same pattern as RegressionDetector.
    """

    def __init__(self) -> None:
        self.histories: dict[str, LocusErrorHistory] = {}
        self._decomposition_state: dict[str, dict] = {}

    def record_error(self, locus: str, sha: str, error_message: str) -> None:
        """Accumulate an error for a locus. Windowed to last ERROR_WINDOW."""
        if locus not in self.histories:
            self.histories[locus] = LocusErrorHistory()
        history = self.histories[locus]
        history.errors.append({
            "message": error_message,
            "sha": sha,
            "timestamp": time.time(),
        })
        if len(history.errors) > ERROR_WINDOW:
            history.errors = history.errors[-ERROR_WINDOW:]

    def analyze(self, locus: str) -> DecompositionSignal | None:
        """Cluster errors and return a signal if decomposition is warranted.

        Returns DecompositionSignal if 3+ distinct error clusters exist
        across 10+ total errors. Returns None otherwise.
        """
        history = self.histories.get(locus)
        if history is None or len(history.errors) < MIN_ERRORS_FOR_SIGNAL:
            return None

        clusters_map: dict[str, ErrorCluster] = {}
        for entry in history.errors:
            pattern = _normalize_error(entry["message"])
            if pattern not in clusters_map:
                clusters_map[pattern] = ErrorCluster(pattern=pattern)
            cluster = clusters_map[pattern]
            cluster.count += 1
            if len(cluster.representative_messages) < MAX_REPRESENTATIVE:
                cluster.representative_messages.append(entry["message"])

        clusters = sorted(
            clusters_map.values(), key=lambda c: c.count, reverse=True
        )

        if len(clusters) < MIN_CLUSTERS_FOR_SIGNAL:
            return None

        return DecompositionSignal(
            locus=locus,
            error_clusters=clusters[:MAX_SPLIT_COUNT],
            recommended_split_count=min(len(clusters), MAX_SPLIT_COUNT),
            total_errors=len(history.errors),
        )

    def is_decomposed(self, locus: str) -> bool:
        """Check if a locus currently has an active decomposition."""
        return locus in self._decomposition_state

    def record_decomposition(
        self, locus: str, pathway_name: str, sub_loci: list[str],
    ) -> None:
        """Record that a locus was decomposed into a pathway."""
        self._decomposition_state[locus] = {
            "pathway_name": pathway_name,
            "sub_loci": sub_loci,
            "timestamp": time.time(),
            "original_locus": locus,
        }

    def get_decomposition(self, locus: str) -> dict | None:
        """Get active decomposition state for a locus."""
        return self._decomposition_state.get(locus)

    def clear_decomposition(self, locus: str) -> None:
        """Clear decomposition state (revert to monolithic gene)."""
        self._decomposition_state.pop(locus, None)

    def record_fusion_of_decomposition(
        self, original_locus: str, fused_sha: str,
    ) -> None:
        """Record that a decomposed pathway has re-fused."""
        state = self._decomposition_state.get(original_locus)
        if state is not None:
            state["fused_sha"] = fused_sha
            state["fused_at"] = time.time()
            state["status"] = "refined"

    # --- Persistence ---

    def save(self, path: Path) -> None:
        data = {
            "histories": {
                locus: h.to_dict() for locus, h in self.histories.items()
            },
            "decomposition_state": self._decomposition_state,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with file_lock(path):
            path.write_text(json.dumps(data, indent=2))

    def load(self, path: Path) -> None:
        if path.exists():
            data = json.loads(path.read_text())
            self.histories = {
                locus: LocusErrorHistory.from_dict(h)
                for locus, h in data.get("histories", {}).items()
            }
            self._decomposition_state = data.get("decomposition_state", {})

    @classmethod
    def open(cls, path: Path) -> DecompositionDetector:
        det = cls()
        det.load(path)
        return det
