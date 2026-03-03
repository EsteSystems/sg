"""Fitness scoring, promotion, and demotion.

Simple fitness = successful_invocations / max(total_invocations, 10).
Temporal fitness = weighted (immediate 30%, convergence 50%, resilience 20%).
Uses temporal fitness when diagnostic feedback records exist.

Promotion: candidate fitness >= dominant + 0.1, with >= 50 invocations.
Demotion: 3 consecutive failures.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from sg.registry import AlleleMetadata, AlleleState

if TYPE_CHECKING:
    from sg.meta_params import EvolutionaryParams


MIN_INVOCATIONS_FOR_SCORE = 10
PROMOTION_ADVANTAGE = 0.1
PROMOTION_MIN_INVOCATIONS = 50
DEMOTION_CONSECUTIVE_FAILURES = 3
CANARY_MIN_SUCCESSES = 10
CANARY_TRAFFIC_FRACTION = 0.2


def compute_fitness(
    allele: AlleleMetadata,
    current_structure_hash: str = "",
    structure_history: tuple[str, ...] | list[str] = (),
    params: EvolutionaryParams | None = None,
) -> float:
    """Compute fitness, using temporal scoring when feedback records exist."""
    if allele.fitness_records:
        from sg.fitness import compute_temporal_fitness
        return compute_temporal_fitness(
            allele, current_structure_hash, structure_history, params=params,
        )
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


def should_promote(
    candidate: AlleleMetadata,
    dominant: AlleleMetadata | None,
    params: EvolutionaryParams | None = None,
) -> bool:
    min_invocations = params.promotion_min_invocations if params else PROMOTION_MIN_INVOCATIONS
    advantage = params.promotion_advantage if params else PROMOTION_ADVANTAGE
    if candidate.total_invocations < min_invocations:
        return False
    candidate_fitness = compute_fitness(candidate, params=params)
    if dominant is None:
        return candidate_fitness > 0.0
    dominant_fitness = compute_fitness(dominant, params=params)
    return candidate_fitness >= dominant_fitness + advantage


def should_demote(
    allele: AlleleMetadata,
    params: EvolutionaryParams | None = None,
) -> bool:
    threshold = params.demotion_consecutive_failures if params else DEMOTION_CONSECUTIVE_FAILURES
    return allele.consecutive_failures >= threshold


def set_dominant(allele: AlleleMetadata) -> None:
    allele.state = AlleleState.DOMINANT.value


def set_recessive(allele: AlleleMetadata) -> None:
    allele.state = AlleleState.RECESSIVE.value


def set_canary(allele: AlleleMetadata) -> None:
    allele.state = AlleleState.CANARY.value


def set_deprecated(allele: AlleleMetadata) -> None:
    allele.state = AlleleState.DEPRECATED.value


def should_graduate_canary(
    allele: AlleleMetadata,
    params: EvolutionaryParams | None = None,
) -> bool:
    """Check if a canary allele has accumulated enough successes to graduate."""
    min_successes = CANARY_MIN_SUCCESSES
    if allele.state != AlleleState.CANARY.value:
        return False
    return allele.canary_successes >= min_successes


def should_fail_canary(allele: AlleleMetadata) -> bool:
    """Check if a canary allele has failed enough times to be reverted."""
    if allele.state != AlleleState.CANARY.value:
        return False
    total = allele.canary_successes + allele.canary_failures
    if total < 5:
        return False
    failure_rate = allele.canary_failures / total
    return failure_rate > 0.5
