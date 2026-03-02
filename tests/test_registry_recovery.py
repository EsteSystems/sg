"""Tests for registry locus index and recovery (Phase 11)."""
from __future__ import annotations

import hashlib
import json

import pytest

from sg.registry import Registry, AlleleMetadata


class TestLocusIndex:
    def test_register_populates_index(self, tmp_path):
        reg = Registry.open(tmp_path / "registry")
        reg.register("src_a", "locus_a")
        reg.register("src_b", "locus_a")
        reg.register("src_c", "locus_b")

        assert len(reg._locus_index["locus_a"]) == 2
        assert len(reg._locus_index["locus_b"]) == 1

    def test_alleles_for_locus_uses_index(self, tmp_path):
        reg = Registry.open(tmp_path / "registry")
        sha_a = reg.register("source_a", "mylock")
        sha_b = reg.register("source_b", "mylock")
        reg.register("source_c", "other")

        result = reg.alleles_for_locus("mylock")
        shas = [a.sha256 for a in result]
        assert sha_a in shas
        assert sha_b in shas
        assert len(result) == 2

    def test_index_rebuilt_on_load(self, tmp_path):
        reg = Registry.open(tmp_path / "registry")
        reg.register("src1", "loc1")
        reg.register("src2", "loc2")
        reg.save_index()

        reg2 = Registry.open(tmp_path / "registry")
        assert len(reg2._locus_index["loc1"]) == 1
        assert len(reg2._locus_index["loc2"]) == 1


class TestRegistryRecovery:
    def test_rebuild_from_sources(self, tmp_path):
        """rebuild_index() recovers alleles from source files."""
        reg_dir = tmp_path / "registry"
        reg_dir.mkdir()
        sources_dir = reg_dir / "sources"
        sources_dir.mkdir()

        # Create source files directly (simulating orphaned sources)
        for i in range(3):
            source = f"def execute(x): return '{i}'"
            sha = hashlib.sha256(source.encode()).hexdigest()
            (sources_dir / f"{sha}.py").write_text(source)

        reg = Registry(reg_dir)
        reg.ensure_dirs()
        recovered = reg.rebuild_index()
        assert recovered == 3
        assert len(reg.alleles) == 3

        # All recovered alleles have locus="unknown"
        for meta in reg.alleles.values():
            assert meta.locus == "unknown"

    def test_rebuild_skips_bad_sha(self, tmp_path):
        """Files with mismatched SHA are skipped during recovery."""
        reg_dir = tmp_path / "registry"
        reg_dir.mkdir()
        sources_dir = reg_dir / "sources"
        sources_dir.mkdir()

        # Create a file with wrong name
        (sources_dir / "badsha.py").write_text("def execute(x): pass")

        reg = Registry(reg_dir)
        reg.ensure_dirs()
        recovered = reg.rebuild_index()
        assert recovered == 0

    def test_corrupted_index_triggers_recovery(self, tmp_path):
        """JSONDecodeError in load_index triggers rebuild."""
        reg_dir = tmp_path / "registry"
        reg_dir.mkdir()
        sources_dir = reg_dir / "sources"
        sources_dir.mkdir()

        # Create a valid source file
        source = "def execute(x): return 'ok'"
        sha = hashlib.sha256(source.encode()).hexdigest()
        (sources_dir / f"{sha}.py").write_text(source)

        # Create corrupted index
        (reg_dir / "registry.json").write_text("{corrupted json!!")

        reg = Registry.open(reg_dir)
        assert sha in reg.alleles

    def test_rebuild_preserves_existing(self, tmp_path):
        """rebuild_index doesn't overwrite existing allele metadata."""
        reg = Registry.open(tmp_path / "registry")
        sha = reg.register("def execute(x): return 'known'", "my_locus", generation=5)
        reg.save_index()

        # Add an orphan source
        orphan_src = "def execute(x): return 'orphan'"
        orphan_sha = hashlib.sha256(orphan_src.encode()).hexdigest()
        (reg.sources_dir / f"{orphan_sha}.py").write_text(orphan_src)

        recovered = reg.rebuild_index()
        assert recovered == 1
        # Original allele preserved
        assert reg.alleles[sha].locus == "my_locus"
        assert reg.alleles[sha].generation == 5
        # Orphan recovered
        assert reg.alleles[orphan_sha].locus == "unknown"
