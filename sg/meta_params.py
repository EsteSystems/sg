"""Meta-evolution — adaptive parameter tracking infrastructure.

Centralizes all evolutionary parameters that are currently hardcoded
constants across fitness.py, arena.py, pathway_arena.py, topology_arena.py,
fusion.py, and stabilization.py. Provides per-entity override support and
history tracking for future adaptive tuning.

Phase E.4 tracking only — actual adaptation is deferred until enough
operational data validates the approach.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

from sg.filelock import atomic_write_text, file_lock
from sg.log import get_logger

logger = get_logger("meta_params")

MAX_SNAPSHOTS_PER_ENTITY = 200
MAX_ENTITIES = 500


@dataclass
class EvolutionaryParams:
    """All tunable evolutionary parameters with current defaults."""

    # Fitness weights (fitness.py)
    immediate_weight: float = 0.30
    convergence_weight: float = 0.50
    resilience_weight: float = 0.20
    convergence_decay_factor: float = 0.2
    old_structure_weight: float = 0.5
    max_fitness_records: int = 200

    # Gene promotion/demotion (arena.py)
    promotion_advantage: float = 0.1
    promotion_min_invocations: int = 50
    demotion_consecutive_failures: int = 3

    # Pathway promotion/demotion (pathway_arena.py)
    pathway_promotion_advantage: float = 0.15
    pathway_promotion_min_executions: int = 200
    pathway_demotion_consecutive_failures: int = 5

    # Topology promotion/demotion (topology_arena.py)
    topology_promotion_advantage: float = 0.20
    topology_promotion_min_executions: int = 500
    topology_demotion_consecutive_failures: int = 8

    # Fusion (fusion.py)
    fusion_threshold: int = 10

    # Stabilization (stabilization.py)
    cv_threshold: float = 0.05
    stabilization_timeout_hours: float = 24.0
    min_stabilization_observations: int = 10

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> EvolutionaryParams:
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class ParamSnapshot:
    """A point-in-time observation of parameters and associated outcome."""
    entity_name: str = ""
    entity_type: str = ""         # "gene", "pathway", "topology"
    params: dict = field(default_factory=dict)
    outcome_fitness: float = 0.0
    allele_sha: str = ""
    allele_survived: bool = True
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "entity_name": self.entity_name,
            "entity_type": self.entity_type,
            "params": self.params,
            "outcome_fitness": self.outcome_fitness,
            "allele_sha": self.allele_sha,
            "allele_survived": self.allele_survived,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ParamSnapshot:
        return cls(
            entity_name=d.get("entity_name", ""),
            entity_type=d.get("entity_type", ""),
            params=d.get("params", {}),
            outcome_fitness=d.get("outcome_fitness", 0.0),
            allele_sha=d.get("allele_sha", ""),
            allele_survived=d.get("allele_survived", True),
            timestamp=d.get("timestamp", 0.0),
        )


# Param subsets relevant to each entity type
_GENE_PARAMS = {
    "immediate_weight", "convergence_weight", "resilience_weight",
    "promotion_advantage", "promotion_min_invocations",
    "demotion_consecutive_failures",
}

_PATHWAY_PARAMS = {
    "pathway_promotion_advantage", "pathway_promotion_min_executions",
    "pathway_demotion_consecutive_failures", "fusion_threshold",
    "cv_threshold",
}

_TOPOLOGY_PARAMS = {
    "topology_promotion_advantage", "topology_promotion_min_executions",
    "topology_demotion_consecutive_failures",
}


class MetaParamTracker:
    """Tracks evolutionary parameter values and outcome snapshots.

    JSON-persisted. Records snapshots of which parameters were active
    when alleles were promoted/demoted, building the dataset needed
    for future adaptive tuning.
    """

    def __init__(self) -> None:
        self.defaults: EvolutionaryParams = EvolutionaryParams()
        self.overrides: dict[str, dict[str, float | int]] = {}
        self.snapshots: dict[str, list[ParamSnapshot]] = {}

    def get_params(self, entity_name: str) -> EvolutionaryParams:
        """Return effective params for an entity (defaults + overrides)."""
        if entity_name not in self.overrides:
            return self.defaults
        params_dict = self.defaults.to_dict()
        params_dict.update(self.overrides[entity_name])
        return EvolutionaryParams.from_dict(params_dict)

    def record_snapshot(
        self,
        entity_name: str,
        entity_type: str,
        outcome_fitness: float,
        allele_sha: str,
        allele_survived: bool,
    ) -> None:
        """Record a parameter snapshot with outcome."""
        params = self.get_params(entity_name)
        all_params = params.to_dict()

        if entity_type == "gene":
            relevant = {k: all_params[k] for k in _GENE_PARAMS}
        elif entity_type == "pathway":
            relevant = {k: all_params[k] for k in _PATHWAY_PARAMS}
        elif entity_type == "topology":
            relevant = {k: all_params[k] for k in _TOPOLOGY_PARAMS}
        else:
            relevant = all_params

        snapshot = ParamSnapshot(
            entity_name=entity_name,
            entity_type=entity_type,
            params=relevant,
            outcome_fitness=outcome_fitness,
            allele_sha=allele_sha,
            allele_survived=allele_survived,
        )
        if entity_name not in self.snapshots:
            self.snapshots[entity_name] = []
        self.snapshots[entity_name].append(snapshot)
        if len(self.snapshots[entity_name]) > MAX_SNAPSHOTS_PER_ENTITY:
            self.snapshots[entity_name] = self.snapshots[entity_name][-MAX_SNAPSHOTS_PER_ENTITY:]

        if len(self.snapshots) > MAX_ENTITIES:
            oldest = min(
                self.snapshots,
                key=lambda k: self.snapshots[k][-1].timestamp,
            )
            del self.snapshots[oldest]

    def get_snapshots(
        self, entity_name: str, entity_type: str | None = None,
    ) -> list[ParamSnapshot]:
        """Return snapshots for an entity, optionally filtered by type."""
        snaps = self.snapshots.get(entity_name, [])
        if entity_type is not None:
            snaps = [s for s in snaps if s.entity_type == entity_type]
        return snaps

    def survival_rate(self, entity_name: str) -> float | None:
        """Fraction of snapshots where the allele survived. None if no data."""
        snaps = self.snapshots.get(entity_name, [])
        if not snaps:
            return None
        survived = sum(1 for s in snaps if s.allele_survived)
        return survived / len(snaps)

    # --- Persistence ---

    def save(self, path: Path) -> None:
        data = {
            "defaults": self.defaults.to_dict(),
            "overrides": self.overrides,
            "snapshots": {
                name: [s.to_dict() for s in snaps]
                for name, snaps in self.snapshots.items()
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with file_lock(path):
            atomic_write_text(path, json.dumps(data, indent=2))

    def load(self, path: Path) -> None:
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            logger.warning("meta params data corrupted at %s, starting fresh", path)
            return
        if "defaults" in data:
            self.defaults = EvolutionaryParams.from_dict(data["defaults"])
        self.overrides = data.get("overrides", {})
        self.snapshots = {
            name: [ParamSnapshot.from_dict(s) for s in snaps]
            for name, snaps in data.get("snapshots", {}).items()
        }

    @classmethod
    def open(cls, path: Path) -> MetaParamTracker:
        tracker = cls()
        tracker.load(path)
        return tracker
