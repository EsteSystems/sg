"""Topology evolutionary pressure — fitness, promotion, demotion.

Mirrors sg/pathway_arena.py but for topology alleles. Topology thresholds
are the most conservative because topology changes affect entire
multi-resource deployments.
"""
from __future__ import annotations

from sg.topology_registry import TopologyAllele

TOPOLOGY_PROMOTION_ADVANTAGE = 0.20
TOPOLOGY_PROMOTION_MIN_EXECUTIONS = 500
TOPOLOGY_DEMOTION_CONSECUTIVE_FAILURES = 8


def compute_topology_fitness(allele: TopologyAllele) -> float:
    """Simple fitness for topology alleles: success_rate with min denominator."""
    if allele.total_executions == 0:
        return 0.0
    denominator = max(allele.total_executions, 30)
    return allele.successful_executions / denominator


def record_topology_success(allele: TopologyAllele) -> None:
    allele.total_executions += 1
    allele.successful_executions += 1
    allele.consecutive_failures = 0
    allele.fitness = compute_topology_fitness(allele)


def record_topology_failure(allele: TopologyAllele) -> None:
    allele.total_executions += 1
    allele.consecutive_failures += 1
    allele.fitness = compute_topology_fitness(allele)


def should_promote_topology(
    candidate: TopologyAllele, dominant: TopologyAllele | None,
) -> bool:
    if candidate.total_executions < TOPOLOGY_PROMOTION_MIN_EXECUTIONS:
        return False
    candidate_fitness = compute_topology_fitness(candidate)
    if dominant is None:
        return candidate_fitness > 0.0
    dominant_fitness = compute_topology_fitness(dominant)
    return candidate_fitness >= dominant_fitness + TOPOLOGY_PROMOTION_ADVANTAGE


def should_demote_topology(allele: TopologyAllele) -> bool:
    return allele.consecutive_failures >= TOPOLOGY_DEMOTION_CONSECUTIVE_FAILURES


def set_topology_dominant(allele: TopologyAllele) -> None:
    allele.state = "dominant"


def set_topology_recessive(allele: TopologyAllele) -> None:
    allele.state = "recessive"


def set_topology_deprecated(allele: TopologyAllele) -> None:
    allele.state = "deprecated"
