"""Speciation tracking — divergence analysis across organisms.

Compares dominant allele compositions between organisms (identified by
pool membership or organism_id) to detect evolutionary divergence.
When two organisms share enough loci but have highly divergent alleles,
speciation has occurred.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from sg.filelock import atomic_write_text, file_lock_shared
from sg.log import get_logger

logger = get_logger("speciation")

DIVERGENCE_THRESHOLD = 0.6
MAX_SPECIES_HISTORY = 100


@dataclass
class OrganismSnapshot:
    """Point-in-time snapshot of an organism's dominant alleles."""
    organism_id: str = ""
    timestamp: float = field(default_factory=time.time)
    dominant_alleles: dict[str, str] = field(default_factory=dict)
    fitness_summary: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "organism_id": self.organism_id,
            "timestamp": self.timestamp,
            "dominant_alleles": self.dominant_alleles,
            "fitness_summary": self.fitness_summary,
        }

    @classmethod
    def from_dict(cls, d: dict) -> OrganismSnapshot:
        return cls(
            organism_id=d.get("organism_id", ""),
            timestamp=d.get("timestamp", 0.0),
            dominant_alleles=d.get("dominant_alleles", {}),
            fitness_summary=d.get("fitness_summary", {}),
        )


@dataclass
class SpeciationMetric:
    """Divergence measurement between two organisms."""
    organism_a: str
    organism_b: str
    shared_loci: int
    divergent_loci: int
    divergence_ratio: float

    def to_dict(self) -> dict:
        return {
            "organism_a": self.organism_a,
            "organism_b": self.organism_b,
            "shared_loci": self.shared_loci,
            "divergent_loci": self.divergent_loci,
            "divergence_ratio": self.divergence_ratio,
        }


class SpeciationTracker:
    """Tracks organism snapshots and detects speciation events."""

    def __init__(self) -> None:
        self.snapshots: dict[str, list[OrganismSnapshot]] = {}
        self._path: Path | None = None

    def record_snapshot(
        self, organism_id: str, phenotype, registry,
        meta_param_tracker=None,
    ) -> None:
        """Record a snapshot of an organism's current dominant alleles."""
        from sg import arena

        dominant_alleles: dict[str, str] = {}
        fitness_summary: dict[str, float] = {}
        for locus_name, config in phenotype.loci.items():
            if config.dominant:
                dominant_alleles[locus_name] = config.dominant
                allele = registry.get(config.dominant)
                if allele:
                    params = meta_param_tracker.get_params(locus_name) if meta_param_tracker else None
                    fitness_summary[locus_name] = arena.compute_fitness(allele, params=params)

        snap = OrganismSnapshot(
            organism_id=organism_id,
            dominant_alleles=dominant_alleles,
            fitness_summary=fitness_summary,
        )

        snapshots = self.snapshots.setdefault(organism_id, [])
        snapshots.append(snap)
        if len(snapshots) > MAX_SPECIES_HISTORY:
            self.snapshots[organism_id] = snapshots[-MAX_SPECIES_HISTORY:]

    def compute_divergence(
        self, organism_a: str, organism_b: str,
    ) -> SpeciationMetric | None:
        """Compute divergence between two organisms' latest snapshots."""
        snaps_a = self.snapshots.get(organism_a)
        snaps_b = self.snapshots.get(organism_b)
        if not snaps_a or not snaps_b:
            return None

        latest_a = snaps_a[-1]
        latest_b = snaps_b[-1]

        loci_a = set(latest_a.dominant_alleles.keys())
        loci_b = set(latest_b.dominant_alleles.keys())
        shared = loci_a & loci_b

        if not shared:
            return None

        divergent = sum(
            1 for locus in shared
            if latest_a.dominant_alleles[locus] != latest_b.dominant_alleles[locus]
        )

        return SpeciationMetric(
            organism_a=organism_a,
            organism_b=organism_b,
            shared_loci=len(shared),
            divergent_loci=divergent,
            divergence_ratio=divergent / len(shared),
        )

    def detect_speciation(self) -> list[tuple[str, str]]:
        """Find organism pairs that have diverged past the threshold."""
        organisms = list(self.snapshots.keys())
        speciated: list[tuple[str, str]] = []

        for i in range(len(organisms)):
            for j in range(i + 1, len(organisms)):
                metric = self.compute_divergence(organisms[i], organisms[j])
                if metric is not None and metric.divergence_ratio > DIVERGENCE_THRESHOLD:
                    speciated.append((organisms[i], organisms[j]))

        return speciated

    # --- Persistence ---

    def save(self, path: Path | None = None) -> None:
        p = path or self._path
        if p is None:
            return
        data = {
            org_id: [s.to_dict() for s in snaps[-MAX_SPECIES_HISTORY:]]
            for org_id, snaps in self.snapshots.items()
        }
        p.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(p, json.dumps(data, indent=2))

    @classmethod
    def open(cls, path: Path) -> SpeciationTracker:
        tracker = cls()
        tracker._path = path
        if not path.exists():
            return tracker
        try:
            with file_lock_shared(path):
                data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            logger.warning("corrupted speciation state, starting fresh")
            return tracker
        for org_id, snaps in data.items():
            tracker.snapshots[org_id] = [
                OrganismSnapshot.from_dict(s) for s in snaps
            ]
        return tracker
