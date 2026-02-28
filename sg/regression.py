"""Fitness regression detection — proactive mutation on fitness decline.

Tracks per-allele peak fitness. When current fitness drops > threshold
below peak, triggers proactive mutation (generates a competing allele).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

from sg.registry import AlleleMetadata
from sg import arena


REGRESSION_THRESHOLD = 0.2   # fitness drop from peak → proactive mutation
SEVERE_REGRESSION = 0.4      # fitness drop from peak → auto-demote
MIN_INVOCATIONS = 10         # need data before detecting regression


@dataclass
class FitnessHistory:
    """Tracks peak fitness for regression detection."""
    peak_fitness: float = 0.0
    last_fitness: float = 0.0
    samples: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> FitnessHistory:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class RegressionDetector:
    """Monitors allele fitness for regressions.

    Returns severity when current fitness drops significantly below
    the allele's historical peak. JSON-persisted.
    """

    def __init__(self, threshold: float = REGRESSION_THRESHOLD) -> None:
        self.threshold = threshold
        self.history: dict[str, FitnessHistory] = {}

    def record(self, allele: AlleleMetadata) -> str | None:
        """Record current fitness, return regression severity if detected.

        Returns: None (no regression), "mild", or "severe".
        """
        sha = allele.sha256
        fitness = arena.compute_fitness(allele)

        if sha not in self.history:
            self.history[sha] = FitnessHistory()

        h = self.history[sha]
        h.last_fitness = fitness
        h.samples += 1

        if fitness > h.peak_fitness:
            h.peak_fitness = fitness
            return None

        if allele.total_invocations < MIN_INVOCATIONS:
            return None

        drop = h.peak_fitness - fitness
        if drop >= SEVERE_REGRESSION:
            return "severe"
        elif drop >= self.threshold:
            return "mild"
        return None

    def get_history(self, sha: str) -> FitnessHistory | None:
        return self.history.get(sha)

    def save(self, path: Path) -> None:
        """Persist regression history to JSON."""
        data = {sha: h.to_dict() for sha, h in self.history.items()}
        path.write_text(json.dumps(data, indent=2))

    def load(self, path: Path) -> None:
        """Load regression history from JSON."""
        if path.exists():
            data = json.loads(path.read_text())
            self.history = {
                sha: FitnessHistory.from_dict(h)
                for sha, h in data.items()
            }

    @classmethod
    def open(cls, path: Path, threshold: float = REGRESSION_THRESHOLD) -> RegressionDetector:
        """Create a RegressionDetector and load state from disk."""
        det = cls(threshold=threshold)
        det.load(path)
        return det
