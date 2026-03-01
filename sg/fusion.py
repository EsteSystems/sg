"""Pathway fusion — reinforcement tracking, fuse/decompose cycle.

When a pathway executes successfully 10 consecutive times with the same
allele composition, the pathway is "fused" into a single optimized gene.
If the fused gene fails, it decomposes back to individual steps.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

from sg.kernel.base import Kernel
from sg.loader import load_gene, call_gene
from sg.mutation import MutationEngine
from sg.registry import Registry
from sg.phenotype import PhenotypeMap


FUSION_THRESHOLD = 10


@dataclass
class PathwayTrack:
    composition_fingerprint: str | None = None
    constituent_alleles: list[str] = field(default_factory=list)
    reinforcement_count: int = 0
    total_successes: int = 0
    total_failures: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> PathwayTrack:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class FusionTracker:
    """Tracks reinforcement state for pathway fusion. JSON-persisted."""

    def __init__(self) -> None:
        self.tracks: dict[str, PathwayTrack] = {}

    def record_success(self, pathway: str, allele_shas: list[str]) -> str | None:
        """Record a successful pathway execution.

        Returns the composition fingerprint if fusion threshold is met.
        """
        fingerprint = composition_fingerprint(allele_shas)
        track = self.tracks.get(pathway)

        if track is None:
            track = PathwayTrack()
            self.tracks[pathway] = track

        if track.composition_fingerprint != fingerprint:
            track.composition_fingerprint = fingerprint
            track.constituent_alleles = list(allele_shas)
            track.reinforcement_count = 0

        track.reinforcement_count += 1
        track.total_successes += 1

        if track.reinforcement_count >= FUSION_THRESHOLD:
            return fingerprint
        return None

    def record_failure(self, pathway: str) -> None:
        track = self.tracks.get(pathway)
        if track is not None:
            track.reinforcement_count = 0
            track.total_failures += 1

    def get_track(self, pathway: str) -> PathwayTrack | None:
        return self.tracks.get(pathway)

    def save(self, path: Path) -> None:
        data = {name: track.to_dict() for name, track in self.tracks.items()}
        path.write_text(json.dumps(data, indent=2))

    def load(self, path: Path) -> None:
        if path.exists():
            data = json.loads(path.read_text())
            self.tracks = {
                name: PathwayTrack.from_dict(track)
                for name, track in data.items()
            }

    @classmethod
    def open(cls, path: Path) -> FusionTracker:
        ft = cls()
        ft.load(path)
        return ft


def composition_fingerprint(allele_shas: list[str]) -> str:
    """SHA-256 of the ordered allele list, used as fusion identity."""
    combined = ":".join(allele_shas)
    return hashlib.sha256(combined.encode()).hexdigest()


def fuse_pathway(
    pathway_name: str,
    allele_shas: list[str],
    loci: list[str],
    registry: Registry,
    phenotype: PhenotypeMap,
    mutation_engine: MutationEngine,
) -> str | None:
    """Generate and register a fused gene for a pathway.

    Returns the SHA of the fused gene, or None on failure.
    """
    sources = []
    for sha in allele_shas:
        source = registry.load_source(sha)
        if source is None:
            print(f"  [fusion] cannot load source for {sha[:12]}")
            return None
        sources.append(source)

    try:
        fused_source = mutation_engine.generate_fused(pathway_name, sources, loci)
    except Exception as e:
        print(f"  [fusion] generation failed: {e}")
        return None

    fused_sha = registry.register(fused_source, loci[0])
    fingerprint = composition_fingerprint(allele_shas)
    phenotype.set_fused(pathway_name, fused_sha, fingerprint)

    print(f"  [fusion] pathway '{pathway_name}' fused → {fused_sha[:12]}")
    return fused_sha


def try_fused_execution(
    pathway_name: str,
    input_json: str,
    registry: Registry,
    phenotype: PhenotypeMap,
    fusion_tracker: FusionTracker,
    kernel: Kernel,
) -> str | None:
    """Try executing the fused gene for a pathway.

    Returns the output JSON on success, or None if fused execution fails
    (which triggers decomposition back to individual steps).
    """
    fusion_config = phenotype.get_fused(pathway_name)
    if fusion_config is None or fusion_config.fused_sha is None:
        return None

    fused_sha = fusion_config.fused_sha
    source = registry.load_source(fused_sha)
    if source is None:
        print(f"  [fusion] fused source not found: {fused_sha[:12]}")
        phenotype.clear_fused(pathway_name)
        return None

    try:
        execute_fn = load_gene(source, kernel)
        result = call_gene(execute_fn, input_json)
        print(f"  [fusion] fused execution succeeded for '{pathway_name}'")
        return result
    except Exception as e:
        print(f"  [fusion] fused execution failed: {e}")
        print(f"  [fusion] decomposing pathway '{pathway_name}' back to individual steps")
        fusion_tracker.record_failure(pathway_name)
        phenotype.clear_fused(pathway_name)
        return None
