"""Adaptive parameter tuning and safety policies.

Analyzes MetaParamTracker snapshots to recommend evolutionary parameter
adjustments.  Safety analysis reviews audit logs for risk-level anomalies.

All recommendations are advisory — human approval required for safety changes.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sg.filelock import atomic_write_text, file_lock, file_lock_shared
from sg.log import get_logger

logger = get_logger("adaptation")

MIN_SNAPSHOTS_FOR_TUNING = 20
MAX_TUNING_STEP = 0.05
TUNING_INTERVAL_SNAPSHOTS = 50
MAX_RECOMMENDATIONS = 200


@dataclass
class TuningRecommendation:
    """A recommended change to an evolutionary parameter."""
    entity_name: str
    param_name: str
    current_value: float
    recommended_value: float
    reason: str
    confidence: float = 0.0
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "entity_name": self.entity_name,
            "param_name": self.param_name,
            "current_value": self.current_value,
            "recommended_value": self.recommended_value,
            "reason": self.reason,
            "confidence": self.confidence,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TuningRecommendation:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class SafetyAdjustment:
    """A recommended risk-level change for a locus."""
    locus: str
    current_risk: str
    recommended_risk: str
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "locus": self.locus,
            "current_risk": self.current_risk,
            "recommended_risk": self.recommended_risk,
            "reason": self.reason,
            "evidence": self.evidence,
        }

    @classmethod
    def from_dict(cls, d: dict) -> SafetyAdjustment:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class AdaptiveParamTuner:
    """Analyzes MetaParamTracker snapshots and recommends parameter changes."""

    def __init__(self, tracker) -> None:
        self._tracker = tracker
        self.history: list[TuningRecommendation] = []
        self._last_tune_count: dict[str, int] = {}
        self._path: Path | None = None

    def analyze(
        self, entity_name: str, entity_type: str,
    ) -> list[TuningRecommendation]:
        """Analyze snapshots for an entity and recommend param changes."""
        snapshots = self._tracker.get_snapshots(entity_name, entity_type)
        if len(snapshots) < MIN_SNAPSHOTS_FOR_TUNING:
            return []

        survival = self._tracker.survival_rate(entity_name)
        if survival is None:
            return []

        params = self._tracker.get_params(entity_name)
        results: list[TuningRecommendation] = []

        if entity_type == "gene":
            results.extend(self._analyze_gene(entity_name, params, survival))
        elif entity_type == "pathway":
            results.extend(self._analyze_pathway(entity_name, params, survival))

        return results

    def _analyze_gene(
        self, entity_name: str, params, survival: float,
    ) -> list[TuningRecommendation]:
        results: list[TuningRecommendation] = []

        if survival < 0.3:
            # Too many demotions — relax thresholds
            current = params.promotion_min_invocations
            step = max(1, int(current * MAX_TUNING_STEP))
            results.append(TuningRecommendation(
                entity_name=entity_name,
                param_name="promotion_min_invocations",
                current_value=float(current),
                recommended_value=float(current + step),
                reason=f"low survival rate ({survival:.2f}): increase observation window",
                confidence=min(0.9, (0.3 - survival) * 3),
            ))

            current_adv = params.promotion_advantage
            new_adv = max(0.02, current_adv - MAX_TUNING_STEP)
            if new_adv != current_adv:
                results.append(TuningRecommendation(
                    entity_name=entity_name,
                    param_name="promotion_advantage",
                    current_value=current_adv,
                    recommended_value=new_adv,
                    reason=f"low survival rate ({survival:.2f}): reduce advantage requirement",
                    confidence=min(0.8, (0.3 - survival) * 2),
                ))

        elif survival > 0.8:
            # Very stable — tighten thresholds to allow more competition
            current = params.promotion_min_invocations
            step = max(1, int(current * MAX_TUNING_STEP))
            new_val = max(10, current - step)
            if new_val != current:
                results.append(TuningRecommendation(
                    entity_name=entity_name,
                    param_name="promotion_min_invocations",
                    current_value=float(current),
                    recommended_value=float(new_val),
                    reason=f"high survival rate ({survival:.2f}): reduce observation window",
                    confidence=min(0.7, (survival - 0.8) * 3),
                ))

        return results

    def _analyze_pathway(
        self, entity_name: str, params, survival: float,
    ) -> list[TuningRecommendation]:
        results: list[TuningRecommendation] = []

        if survival < 0.3:
            current = params.pathway_promotion_min_executions
            step = max(1, int(current * MAX_TUNING_STEP))
            results.append(TuningRecommendation(
                entity_name=entity_name,
                param_name="pathway_promotion_min_executions",
                current_value=float(current),
                recommended_value=float(current + step),
                reason=f"low pathway survival ({survival:.2f}): increase observation window",
                confidence=min(0.9, (0.3 - survival) * 3),
            ))

        return results

    def analyze_fitness_weights(
        self, entity_name: str,
    ) -> list[TuningRecommendation]:
        """Analyze whether fitness weight distribution should change."""
        snapshots = self._tracker.get_snapshots(entity_name)
        if len(snapshots) < MIN_SNAPSHOTS_FOR_TUNING:
            return []

        params = self._tracker.get_params(entity_name)
        results: list[TuningRecommendation] = []

        # Count demotions that followed promotions (reversed promotions)
        reversed_count = 0
        promoted_shas: set[str] = set()
        for snap in snapshots:
            if snap.allele_survived:
                promoted_shas.add(snap.allele_sha)
            elif snap.allele_sha in promoted_shas:
                reversed_count += 1

        reversal_rate = reversed_count / max(len(promoted_shas), 1)
        if reversal_rate > 0.3 and params.immediate_weight > 0.15:
            results.append(TuningRecommendation(
                entity_name=entity_name,
                param_name="immediate_weight",
                current_value=params.immediate_weight,
                recommended_value=max(0.15, params.immediate_weight - MAX_TUNING_STEP),
                reason=f"high reversal rate ({reversal_rate:.2f}): reduce immediate weight",
                confidence=min(0.8, reversal_rate),
            ))

        return results

    def apply_recommendations(
        self, recs: list[TuningRecommendation], min_confidence: float = 0.6,
    ) -> list[TuningRecommendation]:
        """Apply recommendations that meet confidence threshold."""
        applied: list[TuningRecommendation] = []
        for rec in recs:
            if rec.confidence < min_confidence:
                continue
            overrides = self._tracker.overrides.setdefault(rec.entity_name, {})
            overrides[rec.param_name] = rec.recommended_value
            applied.append(rec)
            logger.info("applied tuning: %s.%s = %.4f (was %.4f, confidence %.2f)",
                        rec.entity_name, rec.param_name,
                        rec.recommended_value, rec.current_value,
                        rec.confidence)
        self.history.extend(applied)
        if len(self.history) > MAX_RECOMMENDATIONS:
            self.history = self.history[-MAX_RECOMMENDATIONS:]
        return applied

    def auto_tune(self) -> list[TuningRecommendation]:
        """Run a full auto-tune cycle across all tracked entities."""
        all_recs: list[TuningRecommendation] = []
        for entity_name, snapshots in self._tracker.snapshots.items():
            if not snapshots:
                continue
            snapshot_count = len(snapshots)
            last_count = self._last_tune_count.get(entity_name, 0)
            if snapshot_count - last_count < TUNING_INTERVAL_SNAPSHOTS:
                continue

            entity_type = snapshots[-1].entity_type
            recs = self.analyze(entity_name, entity_type)
            recs.extend(self.analyze_fitness_weights(entity_name))
            applied = self.apply_recommendations(recs)
            all_recs.extend(applied)
            self._last_tune_count[entity_name] = snapshot_count

        return all_recs

    # --- Persistence ---

    def save(self, path: Path | None = None) -> None:
        p = path or self._path
        if p is None:
            return
        data = {
            "history": [r.to_dict() for r in self.history[-MAX_RECOMMENDATIONS:]],
            "last_tune_count": self._last_tune_count,
        }
        p.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(p, json.dumps(data, indent=2))

    @classmethod
    def open(cls, path: Path, tracker) -> AdaptiveParamTuner:
        tuner = cls(tracker)
        tuner._path = path
        if not path.exists():
            return tuner
        try:
            with file_lock_shared(path):
                data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            logger.warning("corrupted adaptive tuner state, starting fresh")
            return tuner
        tuner.history = [
            TuningRecommendation.from_dict(r)
            for r in data.get("history", [])
        ]
        tuner._last_tune_count = data.get("last_tune_count", {})
        return tuner


class AdaptiveSafety:
    """Analyzes audit logs to recommend risk-level adjustments.

    All recommendations are advisory — requires human approval.
    """

    def __init__(self, audit_log) -> None:
        self._audit_log = audit_log

    def analyze(self, contract_store) -> list[SafetyAdjustment]:
        """Analyze recent audit history for safety-level anomalies."""
        entries = self._audit_log.read_recent(count=500)
        if not entries:
            return []

        results: list[SafetyAdjustment] = []

        # Track rollbacks and shadow successes per locus
        rollback_counts: dict[str, int] = {}
        shadow_counts: dict[str, int] = {}
        for entry in entries:
            if entry.event == "rollback" or "rolled back" in entry.event:
                locus = entry.locus
                if locus:
                    rollback_counts[locus] = rollback_counts.get(locus, 0) + 1
            if entry.event == "shadow_success":
                locus = entry.locus
                if locus:
                    shadow_counts[locus] = shadow_counts.get(locus, 0) + 1

        for locus_name in contract_store.known_loci():
            gc = contract_store.get_gene(locus_name)
            if gc is None:
                continue
            current_risk = gc.risk.value if hasattr(gc.risk, 'value') else str(gc.risk)

            rollbacks = rollback_counts.get(locus_name, 0)
            shadows = shadow_counts.get(locus_name, 0)

            # Low risk with many rollbacks → recommend medium
            if current_risk == "low" and rollbacks >= 3:
                results.append(SafetyAdjustment(
                    locus=locus_name,
                    current_risk="low",
                    recommended_risk="medium",
                    reason=f"{rollbacks} rollbacks detected — consider medium risk",
                    evidence={"rollback_count": rollbacks},
                ))

            # High risk with many shadow successes, zero rollbacks → recommend medium
            if current_risk == "high" and shadows >= 100 and rollbacks == 0:
                results.append(SafetyAdjustment(
                    locus=locus_name,
                    current_risk="high",
                    recommended_risk="medium",
                    reason=f"{shadows} shadow successes with 0 rollbacks — consider reducing to medium",
                    evidence={"shadow_success_count": shadows, "rollback_count": 0},
                ))

        return results
