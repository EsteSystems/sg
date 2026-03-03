"""Pathway allele registry — structural identity for pathway evolution.

Pathways are evolvable entities with their own alleles (structural
arrangements of steps). A pathway allele is identified by the SHA-256
of its normalized step sequence — which loci in what order, with what
conditional/loop structure, but NOT which gene alleles are bound.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from sg.filelock import atomic_write_text, file_lock, file_lock_shared
from sg.log import get_logger

logger = get_logger("pathway_registry")

MAX_ALLELES_PER_PATHWAY = 20


@dataclass
class StepSpec:
    """Normalized step specification for structural identity."""
    step_type: str  # "locus", "composed", "loop", "conditional"
    target: str     # locus name, pathway name, or loop body locus
    params: dict[str, str] = field(default_factory=dict)
    loop_variable: str | None = None
    loop_iterable: str | None = None
    condition_step_index: int | None = None
    condition_field: str | None = None
    branches: dict[str, dict] | None = None

    def to_dict(self) -> dict:
        d: dict = {"step_type": self.step_type, "target": self.target}
        if self.params:
            d["params"] = self.params
        if self.loop_variable is not None:
            d["loop_variable"] = self.loop_variable
        if self.loop_iterable is not None:
            d["loop_iterable"] = self.loop_iterable
        if self.condition_step_index is not None:
            d["condition_step_index"] = self.condition_step_index
        if self.condition_field is not None:
            d["condition_field"] = self.condition_field
        if self.branches is not None:
            d["branches"] = self.branches
        return d

    @classmethod
    def from_dict(cls, d: dict) -> StepSpec:
        return cls(
            step_type=d.get("step_type", ""),
            target=d.get("target", ""),
            params=d.get("params", {}),
            loop_variable=d.get("loop_variable"),
            loop_iterable=d.get("loop_iterable"),
            condition_step_index=d.get("condition_step_index"),
            condition_field=d.get("condition_field"),
            branches=d.get("branches"),
        )


@dataclass
class PathwayAllele:
    """A specific structural arrangement of a pathway."""
    structure_sha: str
    pathway_name: str
    steps: list[StepSpec]
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
            "pathway_name": self.pathway_name,
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
    def from_dict(cls, d: dict) -> PathwayAllele:
        steps = [StepSpec.from_dict(s) for s in d.get("steps", [])]
        return cls(
            structure_sha=d.get("structure_sha", ""),
            pathway_name=d.get("pathway_name", ""),
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


def compute_structure_sha(steps: list[StepSpec]) -> str:
    """SHA-256 of the structural identity only.

    Excludes params (input bindings) — two pathways with the same steps
    but different param naming are structurally identical.
    """
    structural = []
    for s in steps:
        entry: dict = {"step_type": s.step_type, "target": s.target}
        if s.loop_variable is not None:
            entry["loop_variable"] = s.loop_variable
            entry["loop_iterable"] = s.loop_iterable
        if s.condition_step_index is not None:
            entry["condition_step_index"] = s.condition_step_index
            entry["condition_field"] = s.condition_field
        if s.branches is not None:
            entry["branches"] = s.branches
        structural.append(entry)
    normalized = json.dumps(structural, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode()).hexdigest()


def steps_from_pathway(pathway) -> list[StepSpec]:
    """Convert a runtime Pathway's steps to StepSpec list for registration."""
    from sg.pathway import (
        PathwayStep, ComposedStep, LoopStep, ConditionalExecStep,
    )
    specs: list[StepSpec] = []
    for step in pathway.steps:
        if isinstance(step, PathwayStep):
            specs.append(StepSpec(step_type="locus", target=step.locus))
        elif isinstance(step, ComposedStep):
            specs.append(StepSpec(step_type="composed", target=step.pathway_name))
        elif isinstance(step, LoopStep):
            specs.append(StepSpec(
                step_type="loop",
                target=step.body_locus,
                loop_variable=step.variable,
                loop_iterable=step.iterable_field,
            ))
        elif isinstance(step, ConditionalExecStep):
            branch_dict = {}
            for value, branch in step.branches.items():
                if isinstance(branch, PathwayStep):
                    branch_dict[value] = {"step_type": "locus", "target": branch.locus}
                elif isinstance(branch, ComposedStep):
                    branch_dict[value] = {"step_type": "composed", "target": branch.pathway_name}
            specs.append(StepSpec(
                step_type="conditional",
                target="",
                condition_step_index=step.condition_step_index,
                condition_field=step.condition_field,
                branches=branch_dict,
            ))
    return specs


class PathwayRegistry:
    """Registry of pathway structural alleles. JSON-persisted."""

    def __init__(self, root: Path):
        self.root = root
        self.index_path = root / "pathway_registry.json"
        self.alleles: dict[str, PathwayAllele] = {}
        self._pathway_index: dict[str, list[str]] = {}

    def ensure_dirs(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def register(
        self,
        pathway_name: str,
        steps: list[StepSpec],
        parent_sha: str | None = None,
        mutation_operator: str | None = None,
    ) -> str:
        """Register a pathway allele. Returns its structure SHA-256."""
        sha = compute_structure_sha(steps)
        if sha not in self.alleles:
            self.alleles[sha] = PathwayAllele(
                structure_sha=sha,
                pathway_name=pathway_name,
                steps=steps,
                parent_sha=parent_sha,
                mutation_operator=mutation_operator,
            )
            self._pathway_index.setdefault(pathway_name, [])
            if sha not in self._pathway_index[pathway_name]:
                self._pathway_index[pathway_name].append(sha)
            self._evict_if_over_cap(pathway_name)
        return sha

    def _evict_if_over_cap(self, pathway_name: str) -> None:
        """Evict oldest deprecated allele if pathway exceeds cap."""
        shas = self._pathway_index.get(pathway_name, [])
        if len(shas) <= MAX_ALLELES_PER_PATHWAY:
            return
        deprecated = [s for s in shas if s in self.alleles
                      and self.alleles[s].state == "deprecated"]
        if deprecated:
            victim = min(deprecated,
                         key=lambda s: self.alleles[s].created_at)
        else:
            victim = min(shas, key=lambda s: self.alleles[s].created_at
                         if s in self.alleles else 0)
        shas.remove(victim)
        self.alleles.pop(victim, None)

    def get(self, sha: str) -> PathwayAllele | None:
        return self.alleles.get(sha)

    def get_for_pathway(self, name: str) -> list[PathwayAllele]:
        """Return all alleles for a pathway, sorted by fitness descending."""
        shas = self._pathway_index.get(name, [])
        alleles = [self.alleles[s] for s in shas if s in self.alleles]
        alleles.sort(key=lambda a: a.fitness, reverse=True)
        return alleles

    def _rebuild_pathway_index(self) -> None:
        self._pathway_index.clear()
        for sha, allele in self.alleles.items():
            self._pathway_index.setdefault(allele.pathway_name, [])
            if sha not in self._pathway_index[allele.pathway_name]:
                self._pathway_index[allele.pathway_name].append(sha)

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
                    sha: PathwayAllele.from_dict(a)
                    for sha, a in data.items()
                }
            except json.JSONDecodeError:
                logger.warning("pathway registry index corrupted, starting fresh")
                self.alleles = {}
        self._rebuild_pathway_index()

    @classmethod
    def open(cls, root: Path) -> PathwayRegistry:
        reg = cls(root)
        reg.ensure_dirs()
        reg.load_index()
        return reg
