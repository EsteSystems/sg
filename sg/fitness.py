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

# Discount base for fitness records recorded under a different pathway structure.
# Records N structures ago get weight OLD_STRUCTURE_BASE ** N.
# Kept as OLD_STRUCTURE_WEIGHT for backward compatibility when no history provided.
OLD_STRUCTURE_WEIGHT = 0.5
OLD_STRUCTURE_BASE = 0.7


@dataclass
class FitnessRecord:
    """A single fitness observation at a specific timescale."""
    timescale: str  # "immediate", "convergence", "resilience"
    success: bool
    source_locus: str  # which locus produced this observation
    timestamp: float = field(default_factory=time.time)
    structure_hash: str = ""  # pathway structure context when recorded

    def to_dict(self) -> dict:
        d = {
            "timescale": self.timescale,
            "success": self.success,
            "source": self.source_locus,
            "timestamp": self.timestamp,
        }
        if self.structure_hash:
            d["structure_hash"] = self.structure_hash
        return d

    @classmethod
    def from_dict(cls, d: dict) -> FitnessRecord:
        return cls(
            timescale=d["timescale"],
            success=d["success"],
            source_locus=d.get("source", ""),
            timestamp=d.get("timestamp", 0.0),
            structure_hash=d.get("structure_hash", ""),
        )


def _structure_discount(
    record_hash: str,
    current_hash: str,
    structure_history: tuple[str, ...] | list[str] = (),
) -> float:
    """Compute progressive discount for a record from an older structure.

    With *structure_history* (newest-first list of prior structure hashes),
    records from N generations ago get weight ``OLD_STRUCTURE_BASE ** N``.
    Without history, falls back to flat ``OLD_STRUCTURE_WEIGHT``.
    """
    if not record_hash or not current_hash:
        return 1.0
    if record_hash == current_hash:
        return 1.0
    if not structure_history:
        return OLD_STRUCTURE_WEIGHT
    for i, h in enumerate(structure_history):
        if h == record_hash:
            return OLD_STRUCTURE_BASE ** (i + 1)
    # Unknown old structure: maximum decay
    return OLD_STRUCTURE_BASE ** max(len(structure_history), 1)


def _score_for_timescale(
    records: list[FitnessRecord],
    timescale: str,
    current_structure_hash: str = "",
    structure_history: tuple[str, ...] | list[str] = (),
) -> float | None:
    """Compute score for a single timescale. Returns None if no records.

    When *current_structure_hash* is provided, records from a different
    structure are discounted progressively using *structure_history*.
    """
    relevant = [r for r in records if r.timescale == timescale]
    if not relevant:
        return None
    weighted_successes = 0.0
    total_weight = 0.0
    for r in relevant:
        w = _structure_discount(
            r.structure_hash, current_structure_hash, structure_history,
        )
        total_weight += w
        if r.success:
            weighted_successes += w
    return weighted_successes / total_weight if total_weight > 0 else 0.0


def compute_temporal_fitness(
    allele: AlleleMetadata,
    current_structure_hash: str = "",
    structure_history: tuple[str, ...] | list[str] = (),
) -> float:
    """Compute weighted temporal fitness across three timescales.

    Falls back to simple fitness (immediate only) when no diagnostic
    feedback has been recorded.  When *current_structure_hash* is given,
    records from older structures are discounted progressively using
    *structure_history* (newest-first list of prior structure hashes).
    """
    records = [FitnessRecord.from_dict(r) for r in allele.fitness_records]

    # Immediate score from invocation counts (existing mechanism)
    total = allele.total_invocations
    if total == 0:
        return 0.0
    immediate = allele.successful_invocations / max(total, 10)

    # Convergence and resilience from diagnostic feedback
    convergence = _score_for_timescale(
        records, "convergence", current_structure_hash, structure_history,
    )
    resilience = _score_for_timescale(
        records, "resilience", current_structure_hash, structure_history,
    )

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
    structure_hash: str = "",
) -> None:
    """Record a diagnostic feedback observation against a config allele."""
    record = FitnessRecord(
        timescale=timescale,
        success=success,
        source_locus=source_locus,
        structure_hash=structure_hash,
    )
    allele.fitness_records.append(record.to_dict())
    # Sliding window: keep only the most recent records
    if len(allele.fitness_records) > MAX_FITNESS_RECORDS:
        allele.fitness_records = allele.fitness_records[-MAX_FITNESS_RECORDS:]
