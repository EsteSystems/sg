"""Pathway evolutionary pressure — fitness, promotion, demotion.

Mirrors sg/arena.py but for pathway alleles (structural evolution).
Pathway thresholds are more conservative than gene thresholds because
structural changes have broader impact.
"""
from __future__ import annotations

from sg.pathway_registry import PathwayAllele

PATHWAY_PROMOTION_ADVANTAGE = 0.15
PATHWAY_PROMOTION_MIN_EXECUTIONS = 200
PATHWAY_DEMOTION_CONSECUTIVE_FAILURES = 5


def compute_pathway_fitness(allele: PathwayAllele) -> float:
    """Simple fitness for pathway alleles: success_rate with min denominator."""
    if allele.total_executions == 0:
        return 0.0
    denominator = max(allele.total_executions, 20)
    return allele.successful_executions / denominator


def record_pathway_success(allele: PathwayAllele) -> None:
    allele.total_executions += 1
    allele.successful_executions += 1
    allele.consecutive_failures = 0
    allele.fitness = compute_pathway_fitness(allele)


def record_pathway_failure(allele: PathwayAllele) -> None:
    allele.total_executions += 1
    allele.consecutive_failures += 1
    allele.fitness = compute_pathway_fitness(allele)


def should_promote_pathway(
    candidate: PathwayAllele, dominant: PathwayAllele | None,
) -> bool:
    if candidate.total_executions < PATHWAY_PROMOTION_MIN_EXECUTIONS:
        return False
    candidate_fitness = compute_pathway_fitness(candidate)
    if dominant is None:
        return candidate_fitness > 0.0
    dominant_fitness = compute_pathway_fitness(dominant)
    return candidate_fitness >= dominant_fitness + PATHWAY_PROMOTION_ADVANTAGE


def should_demote_pathway(allele: PathwayAllele) -> bool:
    return allele.consecutive_failures >= PATHWAY_DEMOTION_CONSECUTIVE_FAILURES


def set_pathway_dominant(allele: PathwayAllele) -> None:
    allele.state = "dominant"


def set_pathway_recessive(allele: PathwayAllele) -> None:
    allele.state = "recessive"


def set_pathway_deprecated(allele: PathwayAllele) -> None:
    allele.state = "deprecated"
