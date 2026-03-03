"""Tests for fitness scoring, promotion, and demotion."""
import pytest
from sg.arena import (
    compute_fitness, record_success, record_failure,
    should_promote, should_demote,
)
from sg.registry import AlleleMetadata


def make_allele(**kwargs) -> AlleleMetadata:
    defaults = {"sha256": "abc123", "locus": "bridge_create"}
    defaults.update(kwargs)
    return AlleleMetadata(**defaults)


def test_fitness_zero_invocations():
    a = make_allele()
    assert compute_fitness(a) == 0.0


def test_fitness_below_min_invocations():
    a = make_allele(successful_invocations=3, failed_invocations=2)
    assert compute_fitness(a) == pytest.approx(0.3)


def test_fitness_above_min_invocations():
    a = make_allele(successful_invocations=8, failed_invocations=2)
    assert compute_fitness(a) == pytest.approx(0.8)


def test_fitness_high_invocations():
    a = make_allele(successful_invocations=90, failed_invocations=10)
    assert compute_fitness(a) == pytest.approx(0.9)


def test_record_success():
    a = make_allele(consecutive_failures=2)
    record_success(a)
    assert a.successful_invocations == 1
    assert a.consecutive_failures == 0


def test_record_failure():
    a = make_allele()
    record_failure(a)
    assert a.failed_invocations == 1
    assert a.consecutive_failures == 1


def test_should_promote_insufficient_invocations():
    candidate = make_allele(successful_invocations=10, failed_invocations=0)
    assert not should_promote(candidate, None)


def test_should_promote_no_dominant():
    candidate = make_allele(successful_invocations=50, failed_invocations=0)
    assert should_promote(candidate, None)


def test_should_promote_over_dominant():
    dominant = make_allele(successful_invocations=40, failed_invocations=10)
    candidate = make_allele(
        sha256="def456",
        successful_invocations=50,
        failed_invocations=0,
    )
    assert should_promote(candidate, dominant)


def test_should_not_promote_insufficient_advantage():
    dominant = make_allele(successful_invocations=45, failed_invocations=5)
    candidate = make_allele(
        sha256="def456",
        successful_invocations=48,
        failed_invocations=2,
    )
    assert not should_promote(candidate, dominant)


def test_should_demote_three_failures():
    a = make_allele(consecutive_failures=3)
    assert should_demote(a)


def test_should_not_demote_two_failures():
    a = make_allele(consecutive_failures=2)
    assert not should_demote(a)


# --- Tests for custom params (Item 6) ---


def test_should_promote_with_custom_params():
    """Custom params lower the promotion threshold."""
    from sg.meta_params import EvolutionaryParams
    params = EvolutionaryParams(promotion_min_invocations=5, promotion_advantage=0.05)
    candidate = make_allele(successful_invocations=10, failed_invocations=0)
    dominant = make_allele(sha256="dom", successful_invocations=8, failed_invocations=2)
    # With defaults (50 invocations min), wouldn't promote
    assert not should_promote(candidate, dominant)
    # With custom params (5 min), should promote
    assert should_promote(candidate, dominant, params=params)


def test_should_demote_with_custom_params():
    """Custom params change the demotion threshold."""
    from sg.meta_params import EvolutionaryParams
    params = EvolutionaryParams(demotion_consecutive_failures=5)
    a = make_allele(consecutive_failures=3)
    # Default (3) would demote
    assert should_demote(a)
    # Custom (5) should not
    assert not should_demote(a, params=params)


def test_params_none_uses_defaults():
    """Passing params=None preserves default behavior."""
    candidate = make_allele(successful_invocations=50, failed_invocations=0)
    assert should_promote(candidate, None, params=None)
    a = make_allele(consecutive_failures=3)
    assert should_demote(a, params=None)


def test_should_promote_uses_custom_fitness_weights():
    """should_promote passes params through to compute_fitness for scoring."""
    from sg.meta_params import EvolutionaryParams
    from sg.fitness import record_feedback

    # Candidate with convergence failure — fitness depends on weights
    candidate = make_allele(successful_invocations=60, failed_invocations=0)
    record_feedback(candidate, "convergence", False, "check1")

    # Dominant with moderate fitness (simple: 55/80 = 0.69)
    dominant = make_allele(sha256="dom", successful_invocations=55, failed_invocations=25)

    # Default weights: convergence=0.5 penalizes candidate heavily
    # Candidate temporal: 0.3*0.8 + 0.5*0.0 + 0.2*1.0 = 0.44 < 0.69 + 0.1
    assert not should_promote(candidate, dominant)

    # Custom weights: zero convergence weight + zero decay → candidate = 1.0
    # 1.0 >= 0.69 + 0.05 → promoted
    params = EvolutionaryParams(
        promotion_min_invocations=5,
        promotion_advantage=0.05,
        immediate_weight=0.9,
        convergence_weight=0.0,
        resilience_weight=0.1,
        convergence_decay_factor=0.0,
    )
    assert should_promote(candidate, dominant, params=params)
