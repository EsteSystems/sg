"""Topology allele registry — structural identity for topology evolution.

Topologies are evolvable entities. A topology allele represents a specific
decomposition strategy — which resources in what order, mapped to which
pathways/genes. The structural SHA-256 is computed from the normalized
step specs (resource_name, action, target), excluding resolved input_json.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from sg.filelock import atomic_write_text, file_lock, file_lock_shared
from sg.log import get_logger

logger = get_logger("topology_registry")


@dataclass
class TopologyStepSpec:
    """Normalized topology step specification for structural identity."""
    resource_name: str  # from topology has block (e.g. "management")
    action: str         # "pathway" | "gene" | "loop_gene"
    target: str         # pathway/gene name to execute
    loop_target_count: int = 0  # for loop_gene: item count (structural info)

    def to_dict(self) -> dict:
        d: dict = {
            "resource_name": self.resource_name,
            "action": self.action,
            "target": self.target,
        }
        if self.loop_target_count:
            d["loop_target_count"] = self.loop_target_count
        return d

    @classmethod
    def from_dict(cls, d: dict) -> TopologyStepSpec:
        return cls(
            resource_name=d.get("resource_name", ""),
            action=d.get("action", ""),
            target=d.get("target", ""),
            loop_target_count=d.get("loop_target_count", 0),
        )


def compute_topology_structure_sha(steps: list[TopologyStepSpec]) -> str:
    """SHA-256 of the structural identity only.

    Excludes loop_target_count (input-dependent, not structural).
    The identity is: which resources map to which actions/targets in what order.
    """
    structural = [
        {
            "resource_name": s.resource_name,
            "action": s.action,
            "target": s.target,
        }
        for s in steps
    ]
    normalized = json.dumps(structural, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode()).hexdigest()


def steps_from_decomposition(topo_steps: list) -> list[TopologyStepSpec]:
    """Convert runtime TopologyStep objects into TopologyStepSpec list."""
    specs: list[TopologyStepSpec] = []
    for step in topo_steps:
        specs.append(TopologyStepSpec(
            resource_name=step.resource_name,
            action=step.action,
            target=step.target,
            loop_target_count=len(step.loop_items),
        ))
    return specs


@dataclass
class TopologyAllele:
    """A specific decomposition strategy for a topology."""
    structure_sha: str
    topology_name: str
    steps: list[TopologyStepSpec]
    fitness: float = 0.0
    total_executions: int = 0
    successful_executions: int = 0
    consecutive_failures: int = 0
    state: str = "recessive"
    parent_sha: str | None = None
    mutation_operator: str | None = None
    created_at: float = field(default_factory=time.time)

    @property
    def failed_executions(self) -> int:
        return self.total_executions - self.successful_executions

    def to_dict(self) -> dict:
        return {
            "structure_sha": self.structure_sha,
            "topology_name": self.topology_name,
            "steps": [s.to_dict() for s in self.steps],
            "fitness": self.fitness,
            "total_executions": self.total_executions,
            "successful_executions": self.successful_executions,
            "consecutive_failures": self.consecutive_failures,
            "state": self.state,
            "parent_sha": self.parent_sha,
            "mutation_operator": self.mutation_operator,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TopologyAllele:
        steps = [TopologyStepSpec.from_dict(s) for s in d.get("steps", [])]
        return cls(
            structure_sha=d.get("structure_sha", ""),
            topology_name=d.get("topology_name", ""),
            steps=steps,
            fitness=d.get("fitness", 0.0),
            total_executions=d.get("total_executions", 0),
            successful_executions=d.get("successful_executions", 0),
            consecutive_failures=d.get("consecutive_failures", 0),
            state=d.get("state", "recessive"),
            parent_sha=d.get("parent_sha"),
            mutation_operator=d.get("mutation_operator"),
            created_at=d.get("created_at", 0.0),
        )


class TopologyRegistry:
    """Registry of topology structural alleles. JSON-persisted."""

    def __init__(self, root: Path):
        self.root = root
        self.index_path = root / "topology_registry.json"
        self.alleles: dict[str, TopologyAllele] = {}
        self._topology_index: dict[str, list[str]] = {}

    def ensure_dirs(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def register(
        self,
        topology_name: str,
        steps: list[TopologyStepSpec],
        parent_sha: str | None = None,
        mutation_operator: str | None = None,
    ) -> str:
        """Register a topology allele. Returns its structure SHA-256."""
        sha = compute_topology_structure_sha(steps)
        if sha not in self.alleles:
            self.alleles[sha] = TopologyAllele(
                structure_sha=sha,
                topology_name=topology_name,
                steps=steps,
                parent_sha=parent_sha,
                mutation_operator=mutation_operator,
            )
            self._topology_index.setdefault(topology_name, [])
            if sha not in self._topology_index[topology_name]:
                self._topology_index[topology_name].append(sha)
        return sha

    def get(self, sha: str) -> TopologyAllele | None:
        return self.alleles.get(sha)

    def get_for_topology(self, name: str) -> list[TopologyAllele]:
        """Return all alleles for a topology, sorted by fitness descending."""
        shas = self._topology_index.get(name, [])
        alleles = [self.alleles[s] for s in shas if s in self.alleles]
        alleles.sort(key=lambda a: a.fitness, reverse=True)
        return alleles

    def _rebuild_topology_index(self) -> None:
        self._topology_index.clear()
        for sha, allele in self.alleles.items():
            self._topology_index.setdefault(allele.topology_name, [])
            if sha not in self._topology_index[allele.topology_name]:
                self._topology_index[allele.topology_name].append(sha)

    def save_index(self) -> None:
        self.ensure_dirs()
        data = {sha: a.to_dict() for sha, a in self.alleles.items()}
        with file_lock(self.index_path):
            atomic_write_text(self.index_path, json.dumps(data, indent=2))

    def load_index(self) -> None:
        if self.index_path.exists():
            try:
                with file_lock_shared(self.index_path):
                    data = json.loads(self.index_path.read_text())
                self.alleles = {
                    sha: TopologyAllele.from_dict(a)
                    for sha, a in data.items()
                }
            except json.JSONDecodeError:
                logger.warning("topology registry index corrupted, starting fresh")
                self.alleles = {}
        self._rebuild_topology_index()

    @classmethod
    def open(cls, root: Path) -> TopologyRegistry:
        reg = cls(root)
        reg.ensure_dirs()
        reg.load_index()
        return reg
