"""Tests for adaptive parameter tuning and safety policies."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import pytest

from sg.adaptation import (
    AdaptiveParamTuner, AdaptiveSafety,
    TuningRecommendation, SafetyAdjustment,
    MIN_SNAPSHOTS_FOR_TUNING, MAX_TUNING_STEP,
    TUNING_INTERVAL_SNAPSHOTS, MAX_RECOMMENDATIONS,
)


# --- Fakes ---

@dataclass
class FakeParams:
    immediate_weight: float = 0.30
    convergence_weight: float = 0.50
    resilience_weight: float = 0.20
    promotion_advantage: float = 0.1
    promotion_min_invocations: int = 50
    demotion_consecutive_failures: int = 3
    pathway_promotion_advantage: float = 0.15
    pathway_promotion_min_executions: int = 200
    pathway_demotion_consecutive_failures: int = 5


@dataclass
class FakeSnapshot:
    entity_name: str = ""
    entity_type: str = "gene"
    params: dict = field(default_factory=dict)
    outcome_fitness: float = 0.5
    allele_sha: str = ""
    allele_survived: bool = True
    timestamp: float = field(default_factory=time.time)


class FakeTracker:
    def __init__(self):
        self.defaults = FakeParams()
        self.overrides: dict[str, dict[str, float | int]] = {}
        self.snapshots: dict[str, list[FakeSnapshot]] = {}

    def get_params(self, entity_name):
        return self.defaults

    def get_snapshots(self, entity_name, entity_type=None):
        snaps = self.snapshots.get(entity_name, [])
        if entity_type:
            snaps = [s for s in snaps if s.entity_type == entity_type]
        return snaps

    def survival_rate(self, entity_name):
        snaps = self.snapshots.get(entity_name, [])
        if not snaps:
            return None
        survived = sum(1 for s in snaps if s.allele_survived)
        return survived / len(snaps)

    def record_snapshot(self, **kw):
        name = kw["entity_name"]
        snap = FakeSnapshot(**kw)
        self.snapshots.setdefault(name, []).append(snap)


@dataclass
class FakeAuditEntry:
    timestamp: float = 0.0
    event: str = ""
    locus: str = ""
    sha: str = ""
    details: dict = field(default_factory=dict)


class FakeAuditLog:
    def __init__(self, entries=None):
        self._entries = entries or []

    def read_recent(self, count=100):
        return self._entries[-count:]


class FakeGeneContract:
    def __init__(self, risk_value="low"):
        self.risk = type('R', (), {'value': risk_value})()


class FakeContractStore:
    def __init__(self, loci=None):
        self._loci = loci or {}

    def known_loci(self):
        return list(self._loci.keys())

    def get_gene(self, name):
        return self._loci.get(name)


# --- TuningRecommendation ---

class TestTuningRecommendation:
    def test_roundtrip(self):
        r = TuningRecommendation(
            entity_name="bridge_create",
            param_name="promotion_min_invocations",
            current_value=50.0,
            recommended_value=52.0,
            reason="test",
            confidence=0.7,
        )
        d = r.to_dict()
        restored = TuningRecommendation.from_dict(d)
        assert restored.entity_name == "bridge_create"
        assert restored.recommended_value == 52.0


# --- AdaptiveParamTuner ---

class TestAdaptiveParamTuner:
    def test_insufficient_snapshots(self):
        tracker = FakeTracker()
        for i in range(MIN_SNAPSHOTS_FOR_TUNING - 1):
            tracker.record_snapshot(
                entity_name="l", entity_type="gene",
                outcome_fitness=0.5, allele_sha=f"sha{i}", allele_survived=True,
            )
        tuner = AdaptiveParamTuner(tracker)
        assert tuner.analyze("l", "gene") == []

    def test_low_survival_recommends_relaxation(self):
        tracker = FakeTracker()
        for i in range(MIN_SNAPSHOTS_FOR_TUNING):
            tracker.record_snapshot(
                entity_name="l", entity_type="gene",
                outcome_fitness=0.3, allele_sha=f"sha{i}",
                allele_survived=(i < 3),  # 3/20 = 15% survival
            )
        tuner = AdaptiveParamTuner(tracker)
        recs = tuner.analyze("l", "gene")
        assert len(recs) >= 1
        names = {r.param_name for r in recs}
        assert "promotion_min_invocations" in names
        # Should recommend increase
        for r in recs:
            if r.param_name == "promotion_min_invocations":
                assert r.recommended_value > r.current_value

    def test_high_survival_recommends_tightening(self):
        tracker = FakeTracker()
        for i in range(MIN_SNAPSHOTS_FOR_TUNING):
            tracker.record_snapshot(
                entity_name="l", entity_type="gene",
                outcome_fitness=0.9, allele_sha=f"sha{i}",
                allele_survived=True,  # 100% survival
            )
        tuner = AdaptiveParamTuner(tracker)
        recs = tuner.analyze("l", "gene")
        assert len(recs) >= 1
        for r in recs:
            if r.param_name == "promotion_min_invocations":
                assert r.recommended_value < r.current_value

    def test_max_step_enforced(self):
        tracker = FakeTracker()
        for i in range(MIN_SNAPSHOTS_FOR_TUNING):
            tracker.record_snapshot(
                entity_name="l", entity_type="gene",
                outcome_fitness=0.1, allele_sha=f"sha{i}",
                allele_survived=False,  # 0% survival
            )
        tuner = AdaptiveParamTuner(tracker)
        recs = tuner.analyze("l", "gene")
        for r in recs:
            if r.param_name == "promotion_min_invocations":
                step = abs(r.recommended_value - r.current_value)
                assert step <= r.current_value * MAX_TUNING_STEP + 1

    def test_apply_recommendations_filters_by_confidence(self):
        tracker = FakeTracker()
        tuner = AdaptiveParamTuner(tracker)
        recs = [
            TuningRecommendation("l", "p1", 1.0, 2.0, "test", confidence=0.3),
            TuningRecommendation("l", "p2", 1.0, 2.0, "test", confidence=0.8),
        ]
        applied = tuner.apply_recommendations(recs, min_confidence=0.6)
        assert len(applied) == 1
        assert applied[0].param_name == "p2"

    def test_apply_sets_override(self):
        tracker = FakeTracker()
        tuner = AdaptiveParamTuner(tracker)
        recs = [TuningRecommendation("l", "promotion_min_invocations",
                                     50.0, 55.0, "test", confidence=0.9)]
        tuner.apply_recommendations(recs)
        assert tracker.overrides["l"]["promotion_min_invocations"] == 55.0

    def test_fitness_weight_reversal_detection(self):
        tracker = FakeTracker()
        # Promotions followed by demotions of same SHA
        for i in range(MIN_SNAPSHOTS_FOR_TUNING):
            sha = f"sha{i % 5}"
            tracker.record_snapshot(
                entity_name="l", entity_type="gene",
                outcome_fitness=0.5, allele_sha=sha,
                allele_survived=(i % 2 == 0),  # alternate promote/demote
            )
        tuner = AdaptiveParamTuner(tracker)
        recs = tuner.analyze_fitness_weights("l")
        # With alternating, half are reversed
        if recs:
            assert any(r.param_name == "immediate_weight" for r in recs)

    def test_auto_tune_respects_interval(self):
        tracker = FakeTracker()
        # Less than TUNING_INTERVAL_SNAPSHOTS
        for i in range(MIN_SNAPSHOTS_FOR_TUNING):
            tracker.record_snapshot(
                entity_name="l", entity_type="gene",
                outcome_fitness=0.5, allele_sha=f"sha{i}",
                allele_survived=True,
            )
        tuner = AdaptiveParamTuner(tracker)
        recs = tuner.auto_tune()
        # Should work first time (last_tune_count starts at 0)
        # But we need TUNING_INTERVAL_SNAPSHOTS to trigger
        # 20 snapshots < 50 interval, so no tuning
        assert len(recs) == 0

    def test_auto_tune_fires_at_interval(self):
        tracker = FakeTracker()
        for i in range(TUNING_INTERVAL_SNAPSHOTS):
            tracker.record_snapshot(
                entity_name="l", entity_type="gene",
                outcome_fitness=0.5, allele_sha=f"sha{i}",
                allele_survived=False,  # 0% survival to trigger recs
            )
        tuner = AdaptiveParamTuner(tracker)
        recs = tuner.auto_tune()
        assert len(recs) >= 1

    def test_pathway_low_survival(self):
        tracker = FakeTracker()
        for i in range(MIN_SNAPSHOTS_FOR_TUNING):
            tracker.record_snapshot(
                entity_name="pw", entity_type="pathway",
                outcome_fitness=0.2, allele_sha=f"sha{i}",
                allele_survived=False,
            )
        tuner = AdaptiveParamTuner(tracker)
        recs = tuner.analyze("pw", "pathway")
        assert len(recs) >= 1
        assert recs[0].param_name == "pathway_promotion_min_executions"


# --- Persistence ---

class TestAdaptiveParamTunerPersistence:
    def test_save_and_open(self, tmp_path):
        tracker = FakeTracker()
        tuner = AdaptiveParamTuner(tracker)
        tuner.history.append(TuningRecommendation(
            "l", "p", 1.0, 2.0, "test", confidence=0.9,
        ))
        tuner._last_tune_count["l"] = 100
        path = tmp_path / "adaptation.json"
        tuner.save(path)

        restored = AdaptiveParamTuner.open(path, tracker)
        assert len(restored.history) == 1
        assert restored._last_tune_count["l"] == 100

    def test_open_missing_file(self, tmp_path):
        tracker = FakeTracker()
        tuner = AdaptiveParamTuner.open(tmp_path / "missing.json", tracker)
        assert len(tuner.history) == 0


# --- AdaptiveSafety ---

class TestAdaptiveSafety:
    def test_low_risk_with_rollbacks(self):
        entries = [
            FakeAuditEntry(event="rollback", locus="bridge_create"),
            FakeAuditEntry(event="rollback", locus="bridge_create"),
            FakeAuditEntry(event="rollback", locus="bridge_create"),
        ]
        audit = FakeAuditLog(entries)
        store = FakeContractStore({"bridge_create": FakeGeneContract("low")})
        safety = AdaptiveSafety(audit)
        adjustments = safety.analyze(store)
        assert len(adjustments) == 1
        assert adjustments[0].recommended_risk == "medium"

    def test_high_risk_with_shadows(self):
        entries = [
            FakeAuditEntry(event="shadow_success", locus="bridge_create")
            for _ in range(100)
        ]
        audit = FakeAuditLog(entries)
        store = FakeContractStore({"bridge_create": FakeGeneContract("high")})
        safety = AdaptiveSafety(audit)
        adjustments = safety.analyze(store)
        assert len(adjustments) == 1
        assert adjustments[0].recommended_risk == "medium"

    def test_no_anomalies(self):
        entries = [FakeAuditEntry(event="promotion", locus="bridge_create")]
        audit = FakeAuditLog(entries)
        store = FakeContractStore({"bridge_create": FakeGeneContract("low")})
        safety = AdaptiveSafety(audit)
        assert safety.analyze(store) == []

    def test_empty_audit(self):
        audit = FakeAuditLog([])
        store = FakeContractStore({"bridge_create": FakeGeneContract("low")})
        safety = AdaptiveSafety(audit)
        assert safety.analyze(store) == []
