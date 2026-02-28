"""Fitness scoring, promotion, and demotion.

Simple fitness = successful_invocations / max(total_invocations, 10).
Temporal fitness = weighted (immediate 30%, convergence 50%, resilience 20%).
Uses temporal fitness when diagnostic feedback records exist.

Promotion: candidate fitness >= dominant + 0.1, with >= 50 invocations.
Demotion: 3 consecutive failures.
"""
from __future__ import annotations

from sg.registry import AlleleMetadata, AlleleState


MIN_INVOCATIONS_FOR_SCORE = 10
PROMOTION_ADVANTAGE = 0.1
PROMOTION_MIN_INVOCATIONS = 50
DEMOTION_CONSECUTIVE_FAILURES = 3


def compute_fitness(allele: AlleleMetadata) -> float:
    """Compute fitness, using temporal scoring when feedback records exist."""
    if allele.fitness_records:
        from sg.fitness import compute_temporal_fitness
        return compute_temporal_fitness(allele)
    total = allele.total_invocations
    if total == 0:
        return 0.0
    denominator = max(total, MIN_INVOCATIONS_FOR_SCORE)
    return allele.successful_invocations / denominator


def compute_distributed_fitness(allele: AlleleMetadata) -> float:
    """Fitness incorporating peer observations (70% local, 30% peers).

    Falls back to local fitness if no peer data is available or
    insufficient peer invocations.
    """
    local = compute_fitness(allele)
    if not allele.peer_observations:
        return local
    peer_total_s = sum(o.get("successes", 0) for o in allele.peer_observations)
    peer_total_f = sum(o.get("failures", 0) for o in allele.peer_observations)
    peer_total = peer_total_s + peer_total_f
    if peer_total < MIN_INVOCATIONS_FOR_SCORE:
        return local
    peer_fitness = peer_total_s / max(peer_total, MIN_INVOCATIONS_FOR_SCORE)
    return 0.7 * local + 0.3 * peer_fitness


def record_success(allele: AlleleMetadata) -> None:
    allele.successful_invocations += 1
    allele.consecutive_failures = 0


def record_failure(allele: AlleleMetadata) -> None:
    allele.failed_invocations += 1
    allele.consecutive_failures += 1


def should_promote(candidate: AlleleMetadata, dominant: AlleleMetadata | None) -> bool:
    if candidate.total_invocations < PROMOTION_MIN_INVOCATIONS:
        return False
    candidate_fitness = compute_fitness(candidate)
    if dominant is None:
        return candidate_fitness > 0.0
    dominant_fitness = compute_fitness(dominant)
    return candidate_fitness >= dominant_fitness + PROMOTION_ADVANTAGE


def should_demote(allele: AlleleMetadata) -> bool:
    return allele.consecutive_failures >= DEMOTION_CONSECUTIVE_FAILURES


def set_dominant(allele: AlleleMetadata) -> None:
    allele.state = AlleleState.DOMINANT.value


def set_recessive(allele: AlleleMetadata) -> None:
    allele.state = AlleleState.RECESSIVE.value


def set_deprecated(allele: AlleleMetadata) -> None:
    allele.state = AlleleState.DEPRECATED.value
