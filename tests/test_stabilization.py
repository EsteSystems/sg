"""Phase E.1b: Bottom-up stabilization check tests.

Verifies CV computation, stabilization lifecycle, pathway mutation
blocking during stabilization, and persistence round-trips.
"""
from __future__ import annotations

import json
import math
import shutil
import time
from unittest.mock import patch
import pytest

import sg_network
from sg.contracts import ContractStore
from sg.fusion import FusionTracker
from sg_network import MockNetworkKernel
from sg.mutation import MockMutationEngine
from sg.orchestrator import Orchestrator
from sg.pathway_fitness import PathwayFitnessTracker
from sg.pathway_registry import PathwayRegistry
from sg.phenotype import PhenotypeMap
from sg.registry import Registry
from sg.stabilization import (
    coefficient_of_variation,
    StabilizationTracker,
    StabilizationState,
    CV_THRESHOLD,
    STABILIZATION_TIMEOUT_HOURS,
    MIN_OBSERVATIONS,
)


CONTRACTS_DIR = sg_network.contracts_path()
GENES_DIR = sg_network.genes_path()
FIXTURES_DIR = sg_network.fixtures_path()


class TestCoefficientOfVariation:
    """CV computation correctness."""

    def test_stable_values(self):
        values = [0.8] * 20
        cv = coefficient_of_variation(values)
        assert cv == 0.0

    def test_volatile_values(self):
        values = [0.1, 0.9, 0.2, 0.8, 0.15, 0.85]
        cv = coefficient_of_variation(values)
        assert cv > CV_THRESHOLD

    def test_single_value(self):
        assert math.isinf(coefficient_of_variation([0.5]))

    def test_empty(self):
        assert math.isinf(coefficient_of_variation([]))

    def test_zero_mean(self):
        assert math.isinf(coefficient_of_variation([0.0, 0.0, 0.0]))


class TestStabilizationTracker:
    """Tracker lifecycle: start, record, check, persist."""

    def test_start_and_is_stabilizing(self):
        tracker = StabilizationTracker()
        tracker.start_stabilization("pw", "sha_new", ["bridge_create", "stp_set"])
        assert tracker.is_stabilizing("pw")
        assert not tracker.is_stabilizing("other")

    def test_not_stabilizing_without_state(self):
        tracker = StabilizationTracker()
        assert not tracker.is_stabilizing("pw")

    def test_stabilized_when_cv_below_threshold(self):
        tracker = StabilizationTracker()
        tracker.start_stabilization("pw", "sha_new", ["locus_a"])

        # Record stable fitness observations
        for _ in range(MIN_OBSERVATIONS):
            tracker.record_gene_fitness("pw", "locus_a", 0.85)

        status = tracker.check_stabilization("pw")
        assert status == "stabilized"
        assert not tracker.is_stabilizing("pw")

    def test_still_stabilizing_with_few_observations(self):
        tracker = StabilizationTracker()
        tracker.start_stabilization("pw", "sha_new", ["locus_a"])

        for _ in range(MIN_OBSERVATIONS - 1):
            tracker.record_gene_fitness("pw", "locus_a", 0.85)

        status = tracker.check_stabilization("pw")
        assert status == "stabilizing"

    def test_still_stabilizing_with_volatile_fitness(self):
        tracker = StabilizationTracker()
        tracker.start_stabilization("pw", "sha_new", ["locus_a"])

        for i in range(MIN_OBSERVATIONS):
            # Alternate high/low to produce high CV
            tracker.record_gene_fitness("pw", "locus_a", 0.9 if i % 2 else 0.1)

        status = tracker.check_stabilization("pw")
        assert status == "stabilizing"

    def test_timed_out(self):
        tracker = StabilizationTracker()
        tracker.start_stabilization("pw", "sha_new", ["locus_a"])

        # Set started_at to 25 hours ago
        tracker.states["pw"].started_at = time.time() - (25 * 3600)

        status = tracker.check_stabilization("pw")
        assert status == "timed_out"
        assert not tracker.is_stabilizing("pw")

    def test_persistence_roundtrip(self, tmp_path):
        tracker = StabilizationTracker()
        tracker.start_stabilization("pw", "sha_new", ["locus_a", "locus_b"])
        tracker.record_gene_fitness("pw", "locus_a", 0.85)
        tracker.record_gene_fitness("pw", "locus_b", 0.90)

        path = tmp_path / "stabilization.json"
        tracker.save(path)

        tracker2 = StabilizationTracker.open(path)
        assert tracker2.is_stabilizing("pw")
        state = tracker2.states["pw"]
        assert state.promoted_structure_sha == "sha_new"
        assert len(state.gene_fitness_snapshots["locus_a"]) == 1
        assert len(state.gene_fitness_snapshots["locus_b"]) == 1


class TestStabilizationIntegration:
    """Stabilization blocks pathway mutation through orchestrator."""

    @pytest.fixture
    def full_project(self, tmp_path):
        shutil.copytree(FIXTURES_DIR, tmp_path / "fixtures")
        shutil.copytree(CONTRACTS_DIR, tmp_path / "contracts")
        (tmp_path / ".sg").mkdir(exist_ok=True)
        contract_store = ContractStore.open(tmp_path / "contracts")
        registry = Registry.open(tmp_path / ".sg" / "registry")
        phenotype = PhenotypeMap()
        for locus in contract_store.known_loci():
            candidates = sorted(GENES_DIR.glob(f"{locus}_*.py"))
            if candidates:
                source = candidates[0].read_text()
                sha = registry.register(source, locus)
                phenotype.promote(locus, sha)
                registry.get(sha).state = "dominant"
        registry.save_index()
        phenotype.save(tmp_path / "phenotype.toml")
        return tmp_path

    def _make_orchestrator(self, project_root):
        from sg.parser.types import GeneFamily
        contract_store = ContractStore.open(project_root / "contracts")
        for locus in contract_store.known_loci():
            gc = contract_store.get_gene(locus)
            if gc and gc.family == GeneFamily.CONFIGURATION and gc.verify_within:
                gc.verify_within = "0.01s"
        registry = Registry.open(project_root / ".sg" / "registry")
        phenotype = PhenotypeMap.load(project_root / "phenotype.toml")
        fusion_tracker = FusionTracker.open(project_root / "fusion_tracker.json")
        pft = PathwayFitnessTracker.open(project_root / "pathway_fitness.json")
        pr = PathwayRegistry.open(project_root / ".sg" / "pathway_registry")
        mutation_engine = MockMutationEngine(project_root / "fixtures")
        kernel = MockNetworkKernel()
        return Orchestrator(
            registry=registry, phenotype=phenotype,
            mutation_engine=mutation_engine, fusion_tracker=fusion_tracker,
            kernel=kernel, contract_store=contract_store,
            project_root=project_root,
            pathway_fitness_tracker=pft, pathway_registry=pr,
        )

    def test_blocks_pathway_mutation_during_stabilization(self, full_project):
        """_try_pathway_mutation returns None when stabilization is active."""
        orch = self._make_orchestrator(full_project)
        pathway_name = "configure_bridge_with_stp"

        # Start stabilization manually
        orch._stabilization_tracker.start_stabilization(
            pathway_name, "sha_test", ["bridge_create", "stp_set"],
        )

        from unittest.mock import MagicMock
        result = orch._try_pathway_mutation(pathway_name, MagicMock())
        assert result is None
