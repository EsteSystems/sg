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

from sg.filelock import atomic_write_text, file_lock, file_lock_shared
from sg.log import get_logger

logger = get_logger("registry")


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
        self._locus_index: dict[str, list[str]] = {}  # locus -> [sha, ...]

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
            # Update locus index
            self._locus_index.setdefault(locus, [])
            if sha not in self._locus_index[locus]:
                self._locus_index[locus].append(sha)
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
        """Return alleles for a locus, sorted by fitness descending.

        Uses the locus index for O(k) lookup instead of O(n) scan.
        """
        from sg.arena import compute_fitness
        shas = self._locus_index.get(locus, [])
        matching = [self.alleles[s] for s in shas if s in self.alleles]
        matching.sort(key=lambda a: compute_fitness(a), reverse=True)
        return matching

    def save_index(self) -> None:
        data = {sha: meta.to_dict() for sha, meta in self.alleles.items()}
        with file_lock(self.index_path):
            atomic_write_text(self.index_path, json.dumps(data, indent=2))

    def _rebuild_locus_index(self) -> None:
        """Rebuild the locus index from current alleles dict."""
        self._locus_index.clear()
        for sha, meta in self.alleles.items():
            self._locus_index.setdefault(meta.locus, [])
            if sha not in self._locus_index[meta.locus]:
                self._locus_index[meta.locus].append(sha)

    def load_index(self) -> None:
        if self.index_path.exists():
            try:
                with file_lock_shared(self.index_path):
                    data = json.loads(self.index_path.read_text())
                self.alleles = {
                    sha: AlleleMetadata.from_dict(meta)
                    for sha, meta in data.items()
                }
            except json.JSONDecodeError:
                logger.warning("registry index corrupted, attempting recovery")
                self.rebuild_index()
        self._rebuild_locus_index()

    def rebuild_index(self) -> int:
        """Rebuild the allele index by scanning source files.

        For each .py file in sources/, verify the SHA matches the filename
        and create a minimal AlleleMetadata entry. Returns the number of
        recovered alleles.
        """
        recovered = 0
        if not self.sources_dir.exists():
            return recovered
        for path in sorted(self.sources_dir.glob("*.py")):
            sha_from_name = path.stem
            source = path.read_text()
            actual_sha = hashlib.sha256(source.encode()).hexdigest()
            if actual_sha != sha_from_name:
                logger.warning("SHA mismatch for %s (expected %s), skipping",
                               path.name, actual_sha[:12])
                continue
            if sha_from_name not in self.alleles:
                self.alleles[sha_from_name] = AlleleMetadata(
                    sha256=sha_from_name,
                    locus="unknown",
                )
                recovered += 1
        self._rebuild_locus_index()
        logger.info("recovered %d allele(s) from source files", recovered)
        return recovered

    @classmethod
    def open(cls, root: Path) -> Registry:
        reg = cls(root)
        reg.ensure_dirs()
        reg.load_index()
        return reg
