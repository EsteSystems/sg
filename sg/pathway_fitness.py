"""Pathway-level fitness tracking.

Tracks per-pathway execution metrics: success rates, per-step timing,
failure distribution, timing anomaly detection, and input clustering.
JSON-persisted following the same pattern as FusionTracker.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from sg.log import get_logger

logger = get_logger("pathway_fitness")

MAX_STEP_TIMINGS = 50
MAX_EXECUTION_WINDOW = 100
ANOMALY_THRESHOLD = 2.0
INPUT_TRUNCATE_LEN = 200


@dataclass
class InputCluster:
    """Group of inputs that share a failure outcome."""
    failure_step: str | None = None
    count: int = 0
    recent_inputs: list[str] = field(default_factory=list)

    MAX_RECENT = 5

    def to_dict(self) -> dict:
        return {
            "failure_step": self.failure_step,
            "count": self.count,
            "recent_inputs": self.recent_inputs,
        }

    @classmethod
    def from_dict(cls, d: dict) -> InputCluster:
        return cls(
            failure_step=d.get("failure_step"),
            count=d.get("count", 0),
            recent_inputs=d.get("recent_inputs", []),
        )


@dataclass
class TimingAnomaly:
    """A step whose timing deviates significantly from baseline."""
    step_name: str
    latest_ms: float
    avg_ms: float
    ratio: float

    def to_dict(self) -> dict:
        return {
            "step_name": self.step_name,
            "latest_ms": round(self.latest_ms, 2),
            "avg_ms": round(self.avg_ms, 2),
            "ratio": round(self.ratio, 2),
        }


@dataclass
class PathwayFitnessRecord:
    """Per-pathway fitness state."""
    pathway_name: str = ""
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    failure_step_distribution: dict[str, int] = field(default_factory=dict)
    avg_execution_time_ms: float = 0.0
    step_timings: dict[str, list[float]] = field(default_factory=dict)
    input_failure_clusters: list[InputCluster] = field(default_factory=list)
    consecutive_failures: int = 0
    last_structure_hash: str = ""
    _recent_outcomes: list[bool] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "pathway_name": self.pathway_name,
            "total_executions": self.total_executions,
            "successful_executions": self.successful_executions,
            "failed_executions": self.failed_executions,
            "failure_step_distribution": self.failure_step_distribution,
            "avg_execution_time_ms": round(self.avg_execution_time_ms, 2),
            "step_timings": self.step_timings,
            "input_failure_clusters": [c.to_dict() for c in self.input_failure_clusters],
            "consecutive_failures": self.consecutive_failures,
            "last_structure_hash": self.last_structure_hash,
            "recent_outcomes": self._recent_outcomes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> PathwayFitnessRecord:
        clusters = [InputCluster.from_dict(c) for c in d.get("input_failure_clusters", [])]
        return cls(
            pathway_name=d.get("pathway_name", ""),
            total_executions=d.get("total_executions", 0),
            successful_executions=d.get("successful_executions", 0),
            failed_executions=d.get("failed_executions", 0),
            failure_step_distribution=d.get("failure_step_distribution", {}),
            avg_execution_time_ms=d.get("avg_execution_time_ms", 0.0),
            step_timings=d.get("step_timings", {}),
            input_failure_clusters=clusters,
            consecutive_failures=d.get("consecutive_failures", 0),
            last_structure_hash=d.get("last_structure_hash", ""),
            _recent_outcomes=d.get("recent_outcomes", []),
        )


class PathwayFitnessTracker:
    """Tracks pathway-level fitness metrics. JSON-persisted."""

    def __init__(self) -> None:
        self.records: dict[str, PathwayFitnessRecord] = {}

    def _ensure_record(self, pathway_name: str) -> PathwayFitnessRecord:
        if pathway_name not in self.records:
            self.records[pathway_name] = PathwayFitnessRecord(
                pathway_name=pathway_name,
            )
        return self.records[pathway_name]

    def record_execution(
        self,
        pathway_name: str,
        steps_executed: list[str],
        step_timings: dict[str, float],
        success: bool,
        failure_step: str | None,
        input_json: str,
        structure_hash: str = "",
    ) -> None:
        """Record a pathway execution result with per-step timing data."""
        rec = self._ensure_record(pathway_name)
        rec.total_executions += 1

        if success:
            rec.successful_executions += 1
            rec.consecutive_failures = 0
        else:
            rec.failed_executions += 1
            rec.consecutive_failures += 1
            if failure_step:
                rec.failure_step_distribution[failure_step] = (
                    rec.failure_step_distribution.get(failure_step, 0) + 1
                )

        # Sliding window of outcomes
        rec._recent_outcomes.append(success)
        if len(rec._recent_outcomes) > MAX_EXECUTION_WINDOW:
            rec._recent_outcomes = rec._recent_outcomes[-MAX_EXECUTION_WINDOW:]

        # Per-step timings (windowed)
        total_time_ms = 0.0
        for step_name, ms in step_timings.items():
            if step_name not in rec.step_timings:
                rec.step_timings[step_name] = []
            rec.step_timings[step_name].append(ms)
            if len(rec.step_timings[step_name]) > MAX_STEP_TIMINGS:
                rec.step_timings[step_name] = rec.step_timings[step_name][-MAX_STEP_TIMINGS:]
            total_time_ms += ms

        # Rolling average execution time (EMA, alpha=0.1)
        if rec.total_executions == 1:
            rec.avg_execution_time_ms = total_time_ms
        else:
            alpha = 0.1
            rec.avg_execution_time_ms = (
                (1 - alpha) * rec.avg_execution_time_ms + alpha * total_time_ms
            )

        # Structure hash
        if structure_hash and structure_hash != rec.last_structure_hash:
            rec.last_structure_hash = structure_hash

        # Input clusters
        self._update_input_clusters(rec, failure_step, input_json)

    def _update_input_clusters(
        self,
        rec: PathwayFitnessRecord,
        failure_step: str | None,
        input_json: str,
    ) -> None:
        truncated = input_json[:INPUT_TRUNCATE_LEN]
        for cluster in rec.input_failure_clusters:
            if cluster.failure_step == failure_step:
                cluster.count += 1
                cluster.recent_inputs.append(truncated)
                if len(cluster.recent_inputs) > InputCluster.MAX_RECENT:
                    cluster.recent_inputs = cluster.recent_inputs[-InputCluster.MAX_RECENT:]
                return
        rec.input_failure_clusters.append(InputCluster(
            failure_step=failure_step,
            count=1,
            recent_inputs=[truncated],
        ))

    def compute_fitness(self, pathway_name: str) -> float:
        """Windowed success rate with exponential recency weighting."""
        rec = self.records.get(pathway_name)
        if rec is None or rec.total_executions == 0:
            return 0.0

        outcomes = rec._recent_outcomes
        if not outcomes:
            return rec.successful_executions / max(rec.total_executions, 1)

        decay = 0.95
        n = len(outcomes)
        weighted_success = 0.0
        total_weight = 0.0
        for i, outcome in enumerate(outcomes):
            weight = decay ** (n - 1 - i)
            total_weight += weight
            if outcome:
                weighted_success += weight

        return weighted_success / total_weight if total_weight > 0 else 0.0

    def get_failure_distribution(self, pathway_name: str) -> dict[str, float]:
        """Per-step failure probability: failures_at_step / total_failures."""
        rec = self.records.get(pathway_name)
        if rec is None or rec.failed_executions == 0:
            return {}
        return {
            step: count / rec.failed_executions
            for step, count in rec.failure_step_distribution.items()
        }

    def get_timing_anomalies(self, pathway_name: str) -> list[TimingAnomaly]:
        """Steps where latest timing > ANOMALY_THRESHOLD x rolling average."""
        rec = self.records.get(pathway_name)
        if rec is None:
            return []
        anomalies = []
        for step_name, timings in rec.step_timings.items():
            if len(timings) < 2:
                continue
            avg = sum(timings[:-1]) / len(timings[:-1])
            latest = timings[-1]
            if avg > 0 and latest / avg > ANOMALY_THRESHOLD:
                anomalies.append(TimingAnomaly(
                    step_name=step_name,
                    latest_ms=latest,
                    avg_ms=avg,
                    ratio=latest / avg,
                ))
        return anomalies

    def get_input_clusters(self, pathway_name: str) -> list[InputCluster]:
        """Return input clusters for a pathway."""
        rec = self.records.get(pathway_name)
        if rec is None:
            return []
        return list(rec.input_failure_clusters)

    def get_record(self, pathway_name: str) -> PathwayFitnessRecord | None:
        return self.records.get(pathway_name)

    # --- Persistence ---

    def save(self, path: Path) -> None:
        data = {name: rec.to_dict() for name, rec in self.records.items()}
        path.write_text(json.dumps(data, indent=2))

    def load(self, path: Path) -> None:
        if path.exists():
            data = json.loads(path.read_text())
            self.records = {
                name: PathwayFitnessRecord.from_dict(rec)
                for name, rec in data.items()
            }

    @classmethod
    def open(cls, path: Path) -> PathwayFitnessTracker:
        tracker = cls()
        tracker.load(path)
        return tracker
