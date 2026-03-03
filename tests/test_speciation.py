"""Tests for speciation tracking."""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from sg.speciation import (
    SpeciationTracker, OrganismSnapshot, SpeciationMetric,
    DIVERGENCE_THRESHOLD, MAX_SPECIES_HISTORY,
)


# --- Fakes ---

@dataclass
class FakeLocusConfig:
    dominant: str | None = None
    fallback: list = field(default_factory=list)


@dataclass
class FakeAllele:
    sha256: str = ""
    locus: str = ""
    successful_invocations: int = 80
    failed_invocations: int = 20
    consecutive_failures: int = 0
    fitness_records: list = field(default_factory=list)
    generation: int = 0
    state: str = "dominant"

    @property
    def total_invocations(self):
        return self.successful_invocations + self.failed_invocations


class FakePhenotype:
    def __init__(self, loci=None):
        self.loci = loci or {}


class FakeRegistry:
    def __init__(self, alleles=None):
        self._alleles = alleles or {}

    def get(self, sha):
        return self._alleles.get(sha)


# --- OrganismSnapshot ---

class TestOrganismSnapshot:
    def test_roundtrip(self):
        snap = OrganismSnapshot(
            organism_id="org_a",
            dominant_alleles={"bridge_create": "sha1", "check_conn": "sha2"},
            fitness_summary={"bridge_create": 0.8, "check_conn": 0.9},
        )
        d = snap.to_dict()
        restored = OrganismSnapshot.from_dict(d)
        assert restored.organism_id == "org_a"
        assert restored.dominant_alleles == snap.dominant_alleles


# --- SpeciationTracker ---

class TestSpeciationTracker:
    def test_record_snapshot(self):
        tracker = SpeciationTracker()
        phenotype = FakePhenotype({
            "bridge_create": FakeLocusConfig(dominant="sha1"),
            "check_conn": FakeLocusConfig(dominant="sha2"),
        })
        registry = FakeRegistry({
            "sha1": FakeAllele(sha256="sha1", locus="bridge_create"),
            "sha2": FakeAllele(sha256="sha2", locus="check_conn"),
        })
        tracker.record_snapshot("org_a", phenotype, registry)
        assert len(tracker.snapshots["org_a"]) == 1
        snap = tracker.snapshots["org_a"][0]
        assert snap.dominant_alleles["bridge_create"] == "sha1"

    def test_identical_organisms_no_divergence(self):
        tracker = SpeciationTracker()
        phenotype = FakePhenotype({
            "l1": FakeLocusConfig(dominant="sha1"),
            "l2": FakeLocusConfig(dominant="sha2"),
        })
        registry = FakeRegistry({
            "sha1": FakeAllele(sha256="sha1"),
            "sha2": FakeAllele(sha256="sha2"),
        })
        tracker.record_snapshot("org_a", phenotype, registry)
        tracker.record_snapshot("org_b", phenotype, registry)

        metric = tracker.compute_divergence("org_a", "org_b")
        assert metric is not None
        assert metric.divergence_ratio == 0.0
        assert metric.shared_loci == 2
        assert metric.divergent_loci == 0

    def test_fully_divergent_organisms(self):
        tracker = SpeciationTracker()
        pheno_a = FakePhenotype({
            "l1": FakeLocusConfig(dominant="sha_a1"),
            "l2": FakeLocusConfig(dominant="sha_a2"),
        })
        pheno_b = FakePhenotype({
            "l1": FakeLocusConfig(dominant="sha_b1"),
            "l2": FakeLocusConfig(dominant="sha_b2"),
        })
        registry = FakeRegistry({
            "sha_a1": FakeAllele(sha256="sha_a1"),
            "sha_a2": FakeAllele(sha256="sha_a2"),
            "sha_b1": FakeAllele(sha256="sha_b1"),
            "sha_b2": FakeAllele(sha256="sha_b2"),
        })
        tracker.record_snapshot("org_a", pheno_a, registry)
        tracker.record_snapshot("org_b", pheno_b, registry)

        metric = tracker.compute_divergence("org_a", "org_b")
        assert metric is not None
        assert metric.divergence_ratio == 1.0

    def test_partial_divergence(self):
        tracker = SpeciationTracker()
        pheno_a = FakePhenotype({
            "l1": FakeLocusConfig(dominant="shared"),
            "l2": FakeLocusConfig(dominant="sha_a"),
            "l3": FakeLocusConfig(dominant="shared3"),
        })
        pheno_b = FakePhenotype({
            "l1": FakeLocusConfig(dominant="shared"),
            "l2": FakeLocusConfig(dominant="sha_b"),
            "l3": FakeLocusConfig(dominant="shared3"),
        })
        registry = FakeRegistry({
            "shared": FakeAllele(sha256="shared"),
            "sha_a": FakeAllele(sha256="sha_a"),
            "sha_b": FakeAllele(sha256="sha_b"),
            "shared3": FakeAllele(sha256="shared3"),
        })
        tracker.record_snapshot("org_a", pheno_a, registry)
        tracker.record_snapshot("org_b", pheno_b, registry)

        metric = tracker.compute_divergence("org_a", "org_b")
        assert metric is not None
        # 1/3 divergent
        assert abs(metric.divergence_ratio - 1.0 / 3.0) < 0.01

    def test_detect_speciation(self):
        tracker = SpeciationTracker()
        # Create highly divergent organisms
        pheno_a = FakePhenotype({
            f"l{i}": FakeLocusConfig(dominant=f"sha_a_{i}")
            for i in range(5)
        })
        pheno_b = FakePhenotype({
            f"l{i}": FakeLocusConfig(dominant=f"sha_b_{i}")
            for i in range(5)
        })
        alleles = {}
        for i in range(5):
            alleles[f"sha_a_{i}"] = FakeAllele(sha256=f"sha_a_{i}")
            alleles[f"sha_b_{i}"] = FakeAllele(sha256=f"sha_b_{i}")
        registry = FakeRegistry(alleles)
        tracker.record_snapshot("org_a", pheno_a, registry)
        tracker.record_snapshot("org_b", pheno_b, registry)

        speciated = tracker.detect_speciation()
        assert ("org_a", "org_b") in speciated

    def test_no_speciation_for_similar_organisms(self):
        tracker = SpeciationTracker()
        phenotype = FakePhenotype({
            f"l{i}": FakeLocusConfig(dominant=f"sha_{i}")
            for i in range(5)
        })
        alleles = {f"sha_{i}": FakeAllele(sha256=f"sha_{i}") for i in range(5)}
        registry = FakeRegistry(alleles)
        tracker.record_snapshot("org_a", phenotype, registry)
        tracker.record_snapshot("org_b", phenotype, registry)

        assert tracker.detect_speciation() == []

    def test_no_shared_loci_returns_none(self):
        tracker = SpeciationTracker()
        pheno_a = FakePhenotype({"l1": FakeLocusConfig(dominant="sha1")})
        pheno_b = FakePhenotype({"l2": FakeLocusConfig(dominant="sha2")})
        registry = FakeRegistry({
            "sha1": FakeAllele(sha256="sha1"),
            "sha2": FakeAllele(sha256="sha2"),
        })
        tracker.record_snapshot("org_a", pheno_a, registry)
        tracker.record_snapshot("org_b", pheno_b, registry)

        assert tracker.compute_divergence("org_a", "org_b") is None

    def test_unknown_organism_returns_none(self):
        tracker = SpeciationTracker()
        assert tracker.compute_divergence("org_a", "org_b") is None

    def test_bounded_history(self):
        tracker = SpeciationTracker()
        phenotype = FakePhenotype({"l1": FakeLocusConfig(dominant="sha1")})
        registry = FakeRegistry({"sha1": FakeAllele(sha256="sha1")})
        for _ in range(MAX_SPECIES_HISTORY + 20):
            tracker.record_snapshot("org_a", phenotype, registry)
        assert len(tracker.snapshots["org_a"]) == MAX_SPECIES_HISTORY


class TestSpeciationPersistence:
    def test_save_and_open(self, tmp_path):
        tracker = SpeciationTracker()
        phenotype = FakePhenotype({"l1": FakeLocusConfig(dominant="sha1")})
        registry = FakeRegistry({"sha1": FakeAllele(sha256="sha1")})
        tracker.record_snapshot("org_a", phenotype, registry)

        path = tmp_path / "speciation.json"
        tracker.save(path)

        restored = SpeciationTracker.open(path)
        assert len(restored.snapshots["org_a"]) == 1
        assert restored.snapshots["org_a"][0].organism_id == "org_a"

    def test_open_missing(self, tmp_path):
        tracker = SpeciationTracker.open(tmp_path / "missing.json")
        assert len(tracker.snapshots) == 0

    def test_open_corrupted(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("invalid{")
        tracker = SpeciationTracker.open(path)
        assert len(tracker.snapshots) == 0
