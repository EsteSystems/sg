"""Tests for pathway evolutionary pressure (Phase 4)."""
import pytest

from sg.pathway_registry import PathwayAllele, StepSpec
from sg.pathway_arena import (
    compute_pathway_fitness,
    record_pathway_success,
    record_pathway_failure,
    should_promote_pathway,
    should_demote_pathway,
    set_pathway_dominant,
    set_pathway_recessive,
    set_pathway_deprecated,
    PATHWAY_PROMOTION_ADVANTAGE,
    PATHWAY_PROMOTION_MIN_EXECUTIONS,
    PATHWAY_DEMOTION_CONSECUTIVE_FAILURES,
)


def _make_allele(**kwargs) -> PathwayAllele:
    defaults = dict(
        structure_sha="abc", pathway_name="test_pw", steps=[],
    )
    defaults.update(kwargs)
    return PathwayAllele(**defaults)


class TestComputeFitness:
    def test_zero_executions(self):
        allele = _make_allele()
        assert compute_pathway_fitness(allele) == 0.0

    def test_with_executions(self):
        allele = _make_allele(total_executions=100, successful_executions=80)
        assert compute_pathway_fitness(allele) == pytest.approx(0.8)

    def test_min_denominator(self):
        """Below 20 total, denominator is clamped to 20."""
        allele = _make_allele(total_executions=5, successful_executions=5)
        assert compute_pathway_fitness(allele) == pytest.approx(5 / 20)


class TestRecordSuccess:
    def test_increments_and_resets(self):
        allele = _make_allele(consecutive_failures=3)
        record_pathway_success(allele)
        assert allele.total_executions == 1
        assert allele.successful_executions == 1
        assert allele.consecutive_failures == 0
        assert allele.fitness > 0.0


class TestRecordFailure:
    def test_increments_failures(self):
        allele = _make_allele()
        record_pathway_failure(allele)
        assert allele.total_executions == 1
        assert allele.successful_executions == 0
        assert allele.consecutive_failures == 1


class TestPromotion:
    def test_insufficient_executions(self):
        candidate = _make_allele(
            total_executions=100, successful_executions=100,
        )
        assert not should_promote_pathway(candidate, None)

    def test_no_dominant(self):
        candidate = _make_allele(
            total_executions=200, successful_executions=200,
        )
        assert should_promote_pathway(candidate, None)

    def test_over_dominant(self):
        candidate = _make_allele(
            total_executions=200, successful_executions=190,
        )
        dominant = _make_allele(
            total_executions=200, successful_executions=150,
        )
        # candidate fitness = 190/200 = 0.95, dominant = 150/200 = 0.75
        # advantage = 0.20 >= 0.15
        assert should_promote_pathway(candidate, dominant)

    def test_insufficient_advantage(self):
        candidate = _make_allele(
            total_executions=200, successful_executions=180,
        )
        dominant = _make_allele(
            total_executions=200, successful_executions=170,
        )
        # candidate = 0.90, dominant = 0.85, advantage = 0.05 < 0.15
        assert not should_promote_pathway(candidate, dominant)


class TestDemotion:
    def test_five_failures(self):
        allele = _make_allele(consecutive_failures=5)
        assert should_demote_pathway(allele)

    def test_four_failures(self):
        allele = _make_allele(consecutive_failures=4)
        assert not should_demote_pathway(allele)


class TestPromotionBoundary:
    def test_exactly_200_executions_qualifies(self):
        """Exactly 200 executions should qualify for promotion."""
        candidate = _make_allele(
            total_executions=200, successful_executions=200,
        )
        assert should_promote_pathway(candidate, None)

    def test_exactly_199_executions_rejects(self):
        candidate = _make_allele(
            total_executions=199, successful_executions=199,
        )
        assert not should_promote_pathway(candidate, None)

    def test_exactly_0_15_advantage_promotes(self):
        """Exactly +0.15 advantage uses >= so should promote."""
        # candidate = 180/200 = 0.90, dominant = 150/200 = 0.75, advantage = 0.15
        candidate = _make_allele(
            total_executions=200, successful_executions=180,
        )
        dominant = _make_allele(
            total_executions=200, successful_executions=150,
        )
        assert should_promote_pathway(candidate, dominant)


class TestSetStates:
    def test_set_dominant(self):
        allele = _make_allele()
        set_pathway_dominant(allele)
        assert allele.state == "dominant"

    def test_set_recessive(self):
        allele = _make_allele(state="dominant")
        set_pathway_recessive(allele)
        assert allele.state == "recessive"

    def test_set_deprecated(self):
        allele = _make_allele()
        set_pathway_deprecated(allele)
        assert allele.state == "deprecated"
