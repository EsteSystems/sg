"""Temporal fitness — three-timescale scoring with retroactive decay.

Immediate (30%): Did the gene succeed right now?
Convergence (50%): Does the system converge to stable state after?
Resilience (20%): Does the change hold up over time?

Diagnostic genes feed results back to configuration genes via the
`feeds` declarations in their contracts.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from sg.registry import AlleleMetadata


# Timescale weights
IMMEDIATE_WEIGHT = 0.30
CONVERGENCE_WEIGHT = 0.50
RESILIENCE_WEIGHT = 0.20

# Retroactive decay: each convergence failure reduces immediate score
CONVERGENCE_DECAY_FACTOR = 0.2

# Maximum fitness records kept per allele (sliding window)
MAX_FITNESS_RECORDS = 200


@dataclass
class FitnessRecord:
    """A single fitness observation at a specific timescale."""
    timescale: str  # "immediate", "convergence", "resilience"
    success: bool
    source_locus: str  # which locus produced this observation
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "timescale": self.timescale,
            "success": self.success,
            "source": self.source_locus,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> FitnessRecord:
        return cls(
            timescale=d["timescale"],
            success=d["success"],
            source_locus=d.get("source", ""),
            timestamp=d.get("timestamp", 0.0),
        )


def _score_for_timescale(records: list[FitnessRecord], timescale: str) -> float | None:
    """Compute score for a single timescale. Returns None if no records."""
    relevant = [r for r in records if r.timescale == timescale]
    if not relevant:
        return None
    successes = sum(1 for r in relevant if r.success)
    return successes / len(relevant)


def compute_temporal_fitness(allele: AlleleMetadata) -> float:
    """Compute weighted temporal fitness across three timescales.

    Falls back to simple fitness (immediate only) when no diagnostic
    feedback has been recorded.
    """
    records = [FitnessRecord.from_dict(r) for r in allele.fitness_records]

    # Immediate score from invocation counts (existing mechanism)
    total = allele.total_invocations
    if total == 0:
        return 0.0
    immediate = allele.successful_invocations / max(total, 10)

    # Convergence and resilience from diagnostic feedback
    convergence = _score_for_timescale(records, "convergence")
    resilience = _score_for_timescale(records, "resilience")

    # If no diagnostic feedback, return simple fitness (backward compat)
    if convergence is None and resilience is None:
        return immediate

    # Retroactive decay: convergence failures reduce immediate
    convergence_failures = sum(
        1 for r in records
        if r.timescale == "convergence" and not r.success
    )
    if convergence_failures > 0:
        decay = max(0.0, 1.0 - CONVERGENCE_DECAY_FACTOR * convergence_failures)
        immediate *= decay

    # Default to 1.0 (assume good) for timescales with no data
    conv_score = convergence if convergence is not None else 1.0
    res_score = resilience if resilience is not None else 1.0

    return (
        immediate * IMMEDIATE_WEIGHT
        + conv_score * CONVERGENCE_WEIGHT
        + res_score * RESILIENCE_WEIGHT
    )


def record_feedback(
    allele: AlleleMetadata,
    timescale: str,
    success: bool,
    source_locus: str,
) -> None:
    """Record a diagnostic feedback observation against a config allele."""
    record = FitnessRecord(
        timescale=timescale,
        success=success,
        source_locus=source_locus,
    )
    allele.fitness_records.append(record.to_dict())
    # Sliding window: keep only the most recent records
    if len(allele.fitness_records) > MAX_FITNESS_RECORDS:
        allele.fitness_records = allele.fitness_records[-MAX_FITNESS_RECORDS:]
