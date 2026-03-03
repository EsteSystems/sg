"""Bottom-up stabilization — blocks pathway mutation after promotion.

After a pathway allele is promoted, gene fitness at constituent loci must
stabilize (CV < 0.05) before further pathway mutations are allowed. This
prevents structural oscillation during the transient period when gene
fitness records are adapting to the new pathway structure.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

from sg.filelock import atomic_write_text, file_lock, file_lock_shared
from sg.log import get_logger

logger = get_logger("stabilization")

CV_THRESHOLD = 0.05
STABILIZATION_TIMEOUT_HOURS = 24
MIN_OBSERVATIONS = 10
MAX_STABILIZATION_STATES = 50
MAX_FITNESS_SNAPSHOTS = 200


def coefficient_of_variation(values: list[float]) -> float:
    """CV = std_dev / mean.  Returns inf if fewer than 2 values or mean is 0."""
    if len(values) < 2:
        return math.inf
    mean = sum(values) / len(values)
    if mean == 0:
        return math.inf
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance) / mean


@dataclass
class StabilizationState:
    """Tracks stabilization progress for a single pathway."""
    pathway_name: str = ""
    promoted_structure_sha: str = ""
    started_at: float = field(default_factory=time.time)
    gene_fitness_snapshots: dict[str, list[float]] = field(default_factory=dict)
    stabilized: bool = False
    timed_out: bool = False

    def to_dict(self) -> dict:
        return {
            "pathway_name": self.pathway_name,
            "promoted_structure_sha": self.promoted_structure_sha,
            "started_at": self.started_at,
            "gene_fitness_snapshots": self.gene_fitness_snapshots,
            "stabilized": self.stabilized,
            "timed_out": self.timed_out,
        }

    @classmethod
    def from_dict(cls, d: dict) -> StabilizationState:
        return cls(
            pathway_name=d.get("pathway_name", ""),
            promoted_structure_sha=d.get("promoted_structure_sha", ""),
            started_at=d.get("started_at", 0.0),
            gene_fitness_snapshots=d.get("gene_fitness_snapshots", {}),
            stabilized=d.get("stabilized", False),
            timed_out=d.get("timed_out", False),
        )


class StabilizationTracker:
    """Tracks pathway stabilization state.  JSON-persisted."""

    def __init__(self) -> None:
        self.states: dict[str, StabilizationState] = {}

    def start_stabilization(
        self, pathway_name: str, structure_sha: str, loci: list[str],
    ) -> None:
        """Begin tracking stabilization after a pathway promotion."""
        if len(self.states) >= MAX_STABILIZATION_STATES:
            # Prune completed/timed_out entries first
            pruneable = [k for k, v in self.states.items()
                         if v.stabilized or v.timed_out]
            if pruneable:
                del self.states[pruneable[0]]
            else:
                oldest = min(self.states,
                             key=lambda k: self.states[k].started_at)
                del self.states[oldest]
        self.states[pathway_name] = StabilizationState(
            pathway_name=pathway_name,
            promoted_structure_sha=structure_sha,
            gene_fitness_snapshots={locus: [] for locus in loci},
        )
        logger.info("stabilization started for '%s'", pathway_name)

    def is_stabilizing(self, pathway_name: str) -> bool:
        """True if the pathway is currently in stabilization period."""
        state = self.states.get(pathway_name)
        if state is None:
            return False
        return not state.stabilized and not state.timed_out

    def record_gene_fitness(
        self, pathway_name: str, locus: str, fitness: float,
    ) -> None:
        """Record a gene fitness observation during stabilization."""
        state = self.states.get(pathway_name)
        if state is None:
            return
        if locus in state.gene_fitness_snapshots:
            state.gene_fitness_snapshots[locus].append(fitness)
            if len(state.gene_fitness_snapshots[locus]) > MAX_FITNESS_SNAPSHOTS:
                state.gene_fitness_snapshots[locus] = (
                    state.gene_fitness_snapshots[locus][-MAX_FITNESS_SNAPSHOTS:]
                )

    def check_stabilization(self, pathway_name: str) -> str:
        """Check stabilization progress.

        Returns ``"stabilizing"``, ``"stabilized"``, or ``"timed_out"``.
        """
        state = self.states.get(pathway_name)
        if state is None:
            return "stabilized"

        if state.stabilized:
            return "stabilized"
        if state.timed_out:
            return "timed_out"

        # Check timeout
        elapsed = time.time() - state.started_at
        if elapsed >= STABILIZATION_TIMEOUT_HOURS * 3600:
            state.timed_out = True
            logger.warning("stabilization timed out for '%s'", pathway_name)
            return "timed_out"

        # Check CV for all loci
        for locus, snapshots in state.gene_fitness_snapshots.items():
            if len(snapshots) < MIN_OBSERVATIONS:
                return "stabilizing"
            cv = coefficient_of_variation(snapshots)
            if cv >= CV_THRESHOLD:
                return "stabilizing"

        state.stabilized = True
        logger.info("stabilization complete for '%s'", pathway_name)
        return "stabilized"

    # --- Persistence ---

    def save(self, path: Path) -> None:
        data = {name: s.to_dict() for name, s in self.states.items()}
        path.parent.mkdir(parents=True, exist_ok=True)
        with file_lock(path):
            atomic_write_text(path, json.dumps(data, indent=2))

    def load(self, path: Path) -> None:
        if path.exists():
            try:
                with file_lock_shared(path):
                    data = json.loads(path.read_text())
            except json.JSONDecodeError:
                logger.warning("stabilization data corrupted at %s, starting fresh", path)
                return
            self.states = {
                name: StabilizationState.from_dict(s)
                for name, s in data.items()
            }

    @classmethod
    def open(cls, path: Path) -> StabilizationTracker:
        tracker = cls()
        tracker.load(path)
        return tracker
