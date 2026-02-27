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
