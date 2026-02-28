"""Fitness regression detection — proactive mutation on fitness decline.

Tracks per-allele peak fitness. When current fitness drops > threshold
below peak, triggers proactive mutation (generates a competing allele).
"""
from __future__ import annotations

from dataclasses import dataclass, field
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


class RegressionDetector:
    """Monitors allele fitness for regressions.

    Returns severity when current fitness drops significantly below
    the allele's historical peak.
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
