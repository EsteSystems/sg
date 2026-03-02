"""Tests for genome snapshots and rollback."""
import json
import shutil
import time
import pytest
from pathlib import Path

from sg.contracts import ContractStore
from sg.fusion import FusionTracker
from sg.phenotype import PhenotypeMap
from sg.registry import Registry
from sg.snapshot import SnapshotManager, SnapshotMeta
import sg_network

CONTRACTS_DIR = sg_network.contracts_path()
GENES_DIR = sg_network.genes_path()


@pytest.fixture
def project(tmp_path):
    """Set up a minimal project."""
    shutil.copytree(CONTRACTS_DIR, tmp_path / "contracts")

    contract_store = ContractStore.open(tmp_path / "contracts")
    registry = Registry.open(tmp_path / ".sg" / "registry")
    phenotype = PhenotypeMap()

    for locus in contract_store.known_loci():
        candidates = sorted(GENES_DIR.glob(f"{locus}_*.py"))
        if candidates:
            source = candidates[0].read_text()
            sha = registry.register(source, locus)
            phenotype.promote(locus, sha)
            allele = registry.get(sha)
            allele.state = "dominant"

    registry.save_index()
    phenotype.save(tmp_path / "phenotype.toml")
    FusionTracker.open(tmp_path / "fusion_tracker.json").save(tmp_path / "fusion_tracker.json")
    return tmp_path


class TestSnapshotCreate:
    def test_create_snapshot(self, project):
        """Creating a snapshot saves all state files."""
        mgr = SnapshotManager(project)
        meta = mgr.create(name="test-snap", description="test snapshot")

        assert meta.name == "test-snap"
        assert meta.description == "test snapshot"
        assert meta.allele_count > 0
        assert meta.loci_count > 0
        assert meta.timestamp > 0

        snap_dir = project / ".sg" / "snapshots" / "test-snap"
        assert (snap_dir / "registry" / "registry.json").exists()
        assert (snap_dir / "phenotype.toml").exists()
        assert (snap_dir / "fusion_tracker.json").exists()
        assert (snap_dir / "meta.json").exists()

    def test_auto_name(self, project):
        """Snapshot without name gets timestamp-based name."""
        mgr = SnapshotManager(project)
        meta = mgr.create()
        assert meta.name.startswith("snapshot-")

    def test_create_duplicate_fails(self, project):
        """Cannot create two snapshots with the same name."""
        mgr = SnapshotManager(project)
        mgr.create(name="dup")
        with pytest.raises(ValueError, match="already exists"):
            mgr.create(name="dup")


class TestSnapshotRestore:
    def test_restore_snapshot(self, project):
        """Modifying state then restoring brings back original."""
        mgr = SnapshotManager(project)
        mgr.create(name="before")

        # Modify state
        registry = Registry.open(project / ".sg" / "registry")
        original_count = len(registry.alleles)
        new_sha = registry.register("def execute(i): return '{}'", "bridge_create", generation=99)
        registry.save_index()

        # Verify modification
        registry2 = Registry.open(project / ".sg" / "registry")
        assert len(registry2.alleles) == original_count + 1

        # Restore
        mgr.restore("before")

        # Verify original state
        registry3 = Registry.open(project / ".sg" / "registry")
        assert len(registry3.alleles) == original_count

    def test_restore_nonexistent_fails(self, project):
        """Restoring a nonexistent snapshot raises."""
        mgr = SnapshotManager(project)
        with pytest.raises(ValueError, match="does not exist"):
            mgr.restore("nonexistent")

    def test_restore_phenotype(self, project):
        """Restore brings back original phenotype."""
        mgr = SnapshotManager(project)
        phenotype = PhenotypeMap.load(project / "phenotype.toml")
        original_loci = set(phenotype.loci.keys())
        mgr.create(name="before")

        # Add a new locus to phenotype
        phenotype.promote("new_locus", "fake_sha")
        phenotype.save(project / "phenotype.toml")

        mgr.restore("before")

        restored = PhenotypeMap.load(project / "phenotype.toml")
        assert set(restored.loci.keys()) == original_loci


class TestSnapshotList:
    def test_list_snapshots(self, project):
        """Multiple snapshots listed in order."""
        mgr = SnapshotManager(project)
        mgr.create(name="first")
        mgr.create(name="second")
        mgr.create(name="third")

        snapshots = mgr.list_snapshots()
        assert len(snapshots) == 3
        # Newest first
        assert snapshots[0].name == "third"
        assert snapshots[2].name == "first"

    def test_list_empty(self, project):
        """No snapshots returns empty list."""
        mgr = SnapshotManager(project)
        assert mgr.list_snapshots() == []


class TestSnapshotDelete:
    def test_delete_snapshot(self, project):
        """Deleting a snapshot removes the directory."""
        mgr = SnapshotManager(project)
        mgr.create(name="to-delete")
        assert mgr.get("to-delete") is not None

        mgr.delete("to-delete")
        assert mgr.get("to-delete") is None
        assert not (project / ".sg" / "snapshots" / "to-delete").exists()

    def test_delete_nonexistent_fails(self, project):
        """Deleting a nonexistent snapshot raises."""
        mgr = SnapshotManager(project)
        with pytest.raises(ValueError, match="does not exist"):
            mgr.delete("nonexistent")


class TestSnapshotMeta:
    def test_meta_roundtrip(self):
        """SnapshotMeta serializes/deserializes correctly."""
        meta = SnapshotMeta(
            name="test", timestamp=1234567890.0,
            description="desc", allele_count=5, loci_count=3,
        )
        d = meta.to_dict()
        restored = SnapshotMeta.from_dict(d)
        assert restored.name == "test"
        assert restored.timestamp == 1234567890.0
        assert restored.description == "desc"
        assert restored.allele_count == 5
        assert restored.loci_count == 3
