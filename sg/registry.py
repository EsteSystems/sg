"""Content-addressed allele registry.

Alleles are identified by SHA-256 of their source code. The registry stores
source files and a JSON index of metadata (fitness, invocations, lineage, state).
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path


class AlleleState(str, Enum):
    DOMINANT = "dominant"
    RECESSIVE = "recessive"
    DEPRECATED = "deprecated"


@dataclass
class AlleleMetadata:
    sha256: str
    locus: str
    generation: int = 0
    parent_sha: str | None = None
    state: str = "recessive"
    successful_invocations: int = 0
    failed_invocations: int = 0
    consecutive_failures: int = 0
    shadow_successes: int = 0
    created_at: float = field(default_factory=time.time)
    fitness_records: list[dict] = field(default_factory=list)
    peer_observations: list[dict] = field(default_factory=list)

    @property
    def total_invocations(self) -> int:
        return self.successful_invocations + self.failed_invocations

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> AlleleMetadata:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class Registry:
    """SHA-256 content-addressed store for gene source code + metadata index."""

    def __init__(self, root: Path):
        self.root = root
        self.sources_dir = root / "sources"
        self.index_path = root / "registry.json"
        self.alleles: dict[str, AlleleMetadata] = {}

    def ensure_dirs(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.sources_dir.mkdir(exist_ok=True)

    def register(self, source: str, locus: str,
                 generation: int = 0, parent_sha: str | None = None) -> str:
        """Register a gene source. Returns its SHA-256 hash."""
        sha = hashlib.sha256(source.encode()).hexdigest()
        source_path = self.sources_dir / f"{sha}.py"
        source_path.write_text(source)
        if sha not in self.alleles:
            self.alleles[sha] = AlleleMetadata(
                sha256=sha,
                locus=locus,
                generation=generation,
                parent_sha=parent_sha,
            )
        return sha

    def get(self, sha: str) -> AlleleMetadata | None:
        return self.alleles.get(sha)

    def source_path(self, sha: str) -> Path:
        return self.sources_dir / f"{sha}.py"

    def load_source(self, sha: str) -> str | None:
        path = self.source_path(sha)
        if path.exists():
            return path.read_text()
        return None

    def alleles_for_locus(self, locus: str) -> list[AlleleMetadata]:
        """Return alleles for a locus, sorted by fitness descending."""
        from sg.arena import compute_fitness
        matching = [a for a in self.alleles.values() if a.locus == locus]
        matching.sort(key=lambda a: compute_fitness(a), reverse=True)
        return matching

    def save_index(self) -> None:
        data = {sha: meta.to_dict() for sha, meta in self.alleles.items()}
        self.index_path.write_text(json.dumps(data, indent=2))

    def load_index(self) -> None:
        if self.index_path.exists():
            data = json.loads(self.index_path.read_text())
            self.alleles = {
                sha: AlleleMetadata.from_dict(meta)
                for sha, meta in data.items()
            }

    @classmethod
    def open(cls, root: Path) -> Registry:
        reg = cls(root)
        reg.ensure_dirs()
        reg.load_index()
        return reg
