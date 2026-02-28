"""Tests for phenotype diffing."""
import pytest
from pathlib import Path

from sg.diff import diff_phenotypes, format_diff, PhenotypeDiff, LocusDiff
from sg.phenotype import PhenotypeMap, PathwayFusionConfig
from sg.registry import Registry


class TestDiffPhenotypes:
    def test_no_changes(self):
        """Identical phenotypes produce empty diff."""
        pm = PhenotypeMap()
        pm.promote("bridge_create", "sha_abc123")
        diff = diff_phenotypes(pm, pm)
        assert not diff.has_changes
        assert diff.loci_changes == []
        assert diff.fusions_added == []
        assert diff.fusions_removed == []

    def test_dominant_changed(self):
        """Detects when dominant SHA changes."""
        old = PhenotypeMap()
        old.promote("bridge_create", "sha_old_allele")
        new = PhenotypeMap()
        new.promote("bridge_create", "sha_new_allele")

        diff = diff_phenotypes(old, new)
        assert diff.has_changes
        assert len(diff.loci_changes) == 1
        change = diff.loci_changes[0]
        assert change.locus == "bridge_create"
        assert change.change == "dominant_changed"
        assert change.old_dominant == "sha_old_alle"  # truncated to 12
        assert change.new_dominant == "sha_new_alle"

    def test_locus_added(self):
        """Detects new locus in new phenotype."""
        old = PhenotypeMap()
        old.promote("bridge_create", "sha1")
        new = PhenotypeMap()
        new.promote("bridge_create", "sha1")
        new.promote("bond_create", "sha2")

        diff = diff_phenotypes(old, new)
        assert diff.has_changes
        assert len(diff.loci_changes) == 1
        assert diff.loci_changes[0].change == "added"
        assert diff.loci_changes[0].locus == "bond_create"

    def test_locus_removed(self):
        """Detects locus removed from new phenotype."""
        old = PhenotypeMap()
        old.promote("bridge_create", "sha1")
        old.promote("bond_create", "sha2")
        new = PhenotypeMap()
        new.promote("bridge_create", "sha1")

        diff = diff_phenotypes(old, new)
        assert diff.has_changes
        assert len(diff.loci_changes) == 1
        assert diff.loci_changes[0].change == "removed"
        assert diff.loci_changes[0].locus == "bond_create"

    def test_fallback_changed(self):
        """Detects when fallback stack changes but dominant stays the same."""
        old = PhenotypeMap()
        old.promote("bridge_create", "sha1")

        new = PhenotypeMap()
        new.promote("bridge_create", "sha1")
        new.add_to_fallback("bridge_create", "sha_fallback")

        diff = diff_phenotypes(old, new)
        assert diff.has_changes
        assert len(diff.loci_changes) == 1
        assert diff.loci_changes[0].change == "fallback_changed"

    def test_fusion_added(self):
        """Detects new pathway fusion."""
        old = PhenotypeMap()
        new = PhenotypeMap()
        new.set_fused("health_check", "fused_sha", "fingerprint")

        diff = diff_phenotypes(old, new)
        assert diff.fusions_added == ["health_check"]
        assert diff.fusions_removed == []

    def test_fusion_removed(self):
        """Detects removed pathway fusion."""
        old = PhenotypeMap()
        old.set_fused("health_check", "fused_sha", "fingerprint")
        new = PhenotypeMap()

        diff = diff_phenotypes(old, new)
        assert diff.fusions_added == []
        assert diff.fusions_removed == ["health_check"]


class TestDiffWithFitness:
    def test_fitness_included(self, tmp_path):
        """Fitness is included when registries are provided."""
        reg = Registry.open(tmp_path / "reg")
        sha = reg.register("def execute(i): return '{}'", "bridge_create")
        allele = reg.get(sha)
        allele.successful_invocations = 50
        allele.failed_invocations = 0

        old = PhenotypeMap()
        new = PhenotypeMap()
        new.promote("bridge_create", sha)

        diff = diff_phenotypes(old, new, new_reg=reg)
        assert diff.loci_changes[0].new_fitness > 0


class TestFormatDiff:
    def test_no_changes(self):
        """No changes produces 'No changes.' message."""
        diff = PhenotypeDiff()
        assert format_diff(diff) == "No changes."

    def test_format_added(self):
        """Added locus formatted with + prefix."""
        diff = PhenotypeDiff(loci_changes=[
            LocusDiff(locus="bond_create", change="added",
                      new_dominant="abc123", new_fitness=0.85),
        ])
        output = format_diff(diff)
        assert output.startswith("+ bond_create")
        assert "abc123" in output

    def test_format_removed(self):
        """Removed locus formatted with - prefix."""
        diff = PhenotypeDiff(loci_changes=[
            LocusDiff(locus="bond_create", change="removed",
                      old_dominant="abc123", old_fitness=0.5),
        ])
        output = format_diff(diff)
        assert output.startswith("- bond_create")

    def test_format_changed(self):
        """Changed dominant formatted with ~ prefix and arrow."""
        diff = PhenotypeDiff(loci_changes=[
            LocusDiff(locus="bridge_create", change="dominant_changed",
                      old_dominant="old_sha", new_dominant="new_sha",
                      old_fitness=0.5, new_fitness=0.9),
        ])
        output = format_diff(diff)
        assert output.startswith("~ bridge_create")
        assert "->" in output

    def test_format_fusion(self):
        """Fusion changes formatted correctly."""
        diff = PhenotypeDiff(
            fusions_added=["health_check"],
            fusions_removed=["old_pathway"],
        )
        output = format_diff(diff)
        assert "+ fusion: health_check" in output
        assert "- fusion: old_pathway" in output
