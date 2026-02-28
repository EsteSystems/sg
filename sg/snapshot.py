"""Genome snapshots — save/restore complete genome state.

Captures registry, phenotype, fusion tracker, and regression state
into a named snapshot directory under .sg/snapshots/.
"""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class SnapshotMeta:
    name: str
    timestamp: float
    description: str = ""
    allele_count: int = 0
    loci_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> SnapshotMeta:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class SnapshotManager:
    """Manage genome state snapshots."""

    def __init__(self, project_root: Path):
        self.root = project_root
        self.snapshots_dir = project_root / ".sg" / "snapshots"

    def _snapshot_dir(self, name: str) -> Path:
        return self.snapshots_dir / name

    def create(self, name: str | None = None, description: str = "") -> SnapshotMeta:
        """Create a snapshot of the current genome state.

        Copies: .sg/registry/, phenotype.toml, fusion_tracker.json, .sg/regression.json
        """
        if name is None:
            name = f"snapshot-{int(time.time())}"

        snap_dir = self._snapshot_dir(name)
        if snap_dir.exists():
            raise ValueError(f"snapshot '{name}' already exists")
        snap_dir.mkdir(parents=True)

        # Copy registry directory
        registry_src = self.root / ".sg" / "registry"
        if registry_src.exists():
            shutil.copytree(registry_src, snap_dir / "registry")

        # Copy individual state files
        for filename in ["phenotype.toml", "fusion_tracker.json"]:
            src = self.root / filename
            if src.exists():
                shutil.copy2(src, snap_dir / filename)

        regression_src = self.root / ".sg" / "regression.json"
        if regression_src.exists():
            shutil.copy2(regression_src, snap_dir / "regression.json")

        # Count alleles and loci for metadata
        allele_count = 0
        loci_count = 0
        registry_index = snap_dir / "registry" / "registry.json"
        if registry_index.exists():
            data = json.loads(registry_index.read_text())
            allele_count = len(data)
            loci_count = len({v.get("locus", "") for v in data.values()})

        meta = SnapshotMeta(
            name=name,
            timestamp=time.time(),
            description=description,
            allele_count=allele_count,
            loci_count=loci_count,
        )
        (snap_dir / "meta.json").write_text(json.dumps(meta.to_dict(), indent=2))
        return meta

    def restore(self, name: str) -> None:
        """Restore genome state from a named snapshot."""
        snap_dir = self._snapshot_dir(name)
        if not snap_dir.exists():
            raise ValueError(f"snapshot '{name}' does not exist")

        # Restore registry directory
        registry_snap = snap_dir / "registry"
        registry_dest = self.root / ".sg" / "registry"
        if registry_snap.exists():
            if registry_dest.exists():
                shutil.rmtree(registry_dest)
            shutil.copytree(registry_snap, registry_dest)

        # Restore individual state files
        for filename in ["phenotype.toml", "fusion_tracker.json"]:
            snap_file = snap_dir / filename
            if snap_file.exists():
                shutil.copy2(snap_file, self.root / filename)

        regression_snap = snap_dir / "regression.json"
        if regression_snap.exists():
            shutil.copy2(regression_snap, self.root / ".sg" / "regression.json")

    def list_snapshots(self) -> list[SnapshotMeta]:
        """List all snapshots, sorted by timestamp (newest first)."""
        if not self.snapshots_dir.exists():
            return []

        snapshots = []
        for snap_dir in sorted(self.snapshots_dir.iterdir()):
            meta_path = snap_dir / "meta.json"
            if meta_path.exists():
                data = json.loads(meta_path.read_text())
                snapshots.append(SnapshotMeta.from_dict(data))

        snapshots.sort(key=lambda s: s.timestamp, reverse=True)
        return snapshots

    def delete(self, name: str) -> None:
        """Delete a named snapshot."""
        snap_dir = self._snapshot_dir(name)
        if not snap_dir.exists():
            raise ValueError(f"snapshot '{name}' does not exist")
        shutil.rmtree(snap_dir)

    def get(self, name: str) -> SnapshotMeta | None:
        """Get metadata for a named snapshot."""
        snap_dir = self._snapshot_dir(name)
        meta_path = snap_dir / "meta.json"
        if not meta_path.exists():
            return None
        data = json.loads(meta_path.read_text())
        return SnapshotMeta.from_dict(data)
