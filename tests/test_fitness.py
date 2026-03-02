"""Tests for temporal fitness scoring and two-family feedback."""
import json
import shutil
import pytest
from pathlib import Path

import sg_network
from sg import arena
from sg.contracts import ContractStore
from sg.fitness import (
    compute_temporal_fitness, record_feedback,
    IMMEDIATE_WEIGHT, CONVERGENCE_WEIGHT, RESILIENCE_WEIGHT,
    CONVERGENCE_DECAY_FACTOR,
)
from sg.fusion import FusionTracker
from sg_network import MockNetworkKernel
from sg.mutation import MockMutationEngine
from sg.orchestrator import Orchestrator
from sg.phenotype import PhenotypeMap
from sg.registry import Registry, AlleleMetadata


CONTRACTS_DIR = sg_network.contracts_path()
GENES_DIR = sg_network.genes_path()
FIXTURES_DIR = sg_network.fixtures_path()


def _make_allele(successful=0, failed=0, records=None):
    allele = AlleleMetadata(sha256="abc123", locus="bridge_create")
    allele.successful_invocations = successful
    allele.failed_invocations = failed
    allele.fitness_records = records or []
    return allele


# --- Temporal fitness computation ---

class TestTemporalFitness:
    def test_zero_invocations(self):
        allele = _make_allele()
        assert compute_temporal_fitness(allele) == 0.0

    def test_no_records_falls_back_to_simple(self):
        allele = _make_allele(successful=8, failed=2)
        # simple: 8/10 = 0.8
        assert compute_temporal_fitness(allele) == 0.8

    def test_immediate_only_with_convergence_success(self):
        allele = _make_allele(successful=10, failed=0)
        record_feedback(allele, "convergence", True, "check_connectivity")
        fitness = compute_temporal_fitness(allele)
        # immediate=1.0, convergence=1.0, resilience=1.0 (default)
        expected = 1.0 * IMMEDIATE_WEIGHT + 1.0 * CONVERGENCE_WEIGHT + 1.0 * RESILIENCE_WEIGHT
        assert abs(fitness - expected) < 0.001

    def test_convergence_failure_reduces_fitness(self):
        allele = _make_allele(successful=10, failed=0)
        record_feedback(allele, "convergence", False, "check_connectivity")
        fitness = compute_temporal_fitness(allele)
        # immediate=1.0 * decay(0.8) = 0.8, convergence=0.0, resilience=1.0
        expected = (0.8 * IMMEDIATE_WEIGHT
                    + 0.0 * CONVERGENCE_WEIGHT
                    + 1.0 * RESILIENCE_WEIGHT)
        assert abs(fitness - expected) < 0.001
        assert fitness < 0.5

    def test_convergence_failure_makes_fitness_below_half(self):
        """Config gene passing immediately but failing convergence has fitness < 0.5."""
        allele = _make_allele(successful=10, failed=0)
        record_feedback(allele, "convergence", False, "check_connectivity")
        assert compute_temporal_fitness(allele) < 0.5

    def test_multiple_convergence_failures_compound_decay(self):
        allele = _make_allele(successful=10, failed=0)
        record_feedback(allele, "convergence", False, "check_connectivity")
        record_feedback(allele, "convergence", False, "check_mac_stability")
        fitness = compute_temporal_fitness(allele)
        # decay = 1.0 - 0.2 * 2 = 0.6
        # immediate = 1.0 * 0.6 = 0.6
        expected = (0.6 * IMMEDIATE_WEIGHT
                    + 0.0 * CONVERGENCE_WEIGHT
                    + 1.0 * RESILIENCE_WEIGHT)
        assert abs(fitness - expected) < 0.001

    def test_mixed_convergence_results(self):
        allele = _make_allele(successful=10, failed=0)
        record_feedback(allele, "convergence", True, "check_connectivity")
        record_feedback(allele, "convergence", False, "check_mac_stability")
        fitness = compute_temporal_fitness(allele)
        # convergence = 1/2 = 0.5, decay from 1 failure = 0.8
        # immediate = 1.0 * 0.8 = 0.8
        expected = (0.8 * IMMEDIATE_WEIGHT
                    + 0.5 * CONVERGENCE_WEIGHT
                    + 1.0 * RESILIENCE_WEIGHT)
        assert abs(fitness - expected) < 0.001

    def test_resilience_feedback(self):
        allele = _make_allele(successful=10, failed=0)
        record_feedback(allele, "convergence", True, "check_connectivity")
        record_feedback(allele, "resilience", False, "check_connectivity")
        fitness = compute_temporal_fitness(allele)
        # immediate=1.0, convergence=1.0, resilience=0.0
        expected = (1.0 * IMMEDIATE_WEIGHT
                    + 1.0 * CONVERGENCE_WEIGHT
                    + 0.0 * RESILIENCE_WEIGHT)
        assert abs(fitness - expected) < 0.001

    def test_decay_clamps_at_zero(self):
        allele = _make_allele(successful=10, failed=0)
        for _ in range(10):
            record_feedback(allele, "convergence", False, "check_connectivity")
        fitness = compute_temporal_fitness(allele)
        assert fitness >= 0.0

    def test_weights_sum_to_one(self):
        assert abs(IMMEDIATE_WEIGHT + CONVERGENCE_WEIGHT + RESILIENCE_WEIGHT - 1.0) < 0.001


# --- Arena integration ---

class TestArenaIntegration:
    def test_compute_fitness_uses_temporal_when_records_exist(self):
        allele = _make_allele(successful=10, failed=0)
        # Without records: simple fitness
        simple = arena.compute_fitness(allele)
        assert simple == 1.0

        # With convergence failure: temporal fitness
        record_feedback(allele, "convergence", False, "check_connectivity")
        temporal = arena.compute_fitness(allele)
        assert temporal < simple

    def test_compute_fitness_backward_compatible(self):
        allele = _make_allele(successful=5, failed=5)
        # No records: should use simple fitness = 5/10 = 0.5
        assert arena.compute_fitness(allele) == 0.5


# --- Record serialization ---

class TestRecordSerialization:
    def test_fitness_records_persist(self, tmp_path):
        registry = Registry.open(tmp_path / ".sg" / "registry")
        sha = registry.register("source", "bridge_create")
        allele = registry.get(sha)
        record_feedback(allele, "convergence", False, "check_connectivity")

        registry.save_index()

        registry2 = Registry.open(tmp_path / ".sg" / "registry")
        allele2 = registry2.get(sha)
        assert len(allele2.fitness_records) == 1
        assert allele2.fitness_records[0]["timescale"] == "convergence"
        assert allele2.fitness_records[0]["success"] is False


# --- Two-family feedback via orchestrator ---

class TestTwoFamilyFeedback:
    @pytest.fixture
    def project(self, tmp_path):
        fixtures_dst = tmp_path / "fixtures"
        shutil.copytree(FIXTURES_DIR, fixtures_dst)
        contracts_dst = tmp_path / "contracts"
        shutil.copytree(CONTRACTS_DIR, contracts_dst)

        contract_store = ContractStore.open(contracts_dst)
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
        return tmp_path

    def _make_orch(self, project_root):
        contract_store = ContractStore.open(project_root / "contracts")
        registry = Registry.open(project_root / ".sg" / "registry")
        phenotype = PhenotypeMap.load(project_root / "phenotype.toml")
        fusion_tracker = FusionTracker.open(project_root / "fusion_tracker.json")
        mutation_engine = MockMutationEngine(project_root / "fixtures")
        kernel = MockNetworkKernel()
        return Orchestrator(
            registry=registry,
            phenotype=phenotype,
            mutation_engine=mutation_engine,
            fusion_tracker=fusion_tracker,
            kernel=kernel,
            contract_store=contract_store,
            project_root=project_root,
        )

    def test_healthy_diagnostic_feeds_positive(self, project):
        orch = self._make_orch(project)

        # Provision bridge
        orch.run_pathway("provision_management_bridge", json.dumps({
            "bridge_name": "br0",
            "interfaces": ["eth0"],
            "uplink": "eth1",
            "stp_enabled": True,
            "forward_delay": 15,
        }))

        # Run health check — should feed positive results to config genes
        orch.run_pathway("health_check_bridge", json.dumps({
            "bridge_name": "br0",
        }))
        orch.save_state()

        # Check that bridge_create got positive convergence feedback
        bridge_create_sha = orch.phenotype.get_dominant("bridge_create")
        allele = orch.registry.get(bridge_create_sha)
        convergence_records = [
            r for r in allele.fitness_records
            if r["timescale"] == "convergence"
        ]
        assert len(convergence_records) > 0

    def test_unhealthy_diagnostic_reduces_config_fitness(self, project):
        orch = self._make_orch(project)

        # Provision bridge
        orch.run_pathway("provision_management_bridge", json.dumps({
            "bridge_name": "br0",
            "interfaces": ["eth0"],
            "uplink": "eth1",
            "stp_enabled": True,
            "forward_delay": 15,
        }))

        # Run healthy health check first
        orch.run_pathway("health_check_bridge", json.dumps({
            "bridge_name": "br0",
        }))

        bridge_create_sha = orch.phenotype.get_dominant("bridge_create")
        healthy_fitness = arena.compute_fitness(orch.registry.get(bridge_create_sha))

        # Inject link failure to make diagnostic report unhealthy
        orch.kernel.inject_link_failure("eth0")

        # Run unhealthy health check
        orch.run_pathway("health_check_bridge", json.dumps({
            "bridge_name": "br0",
        }))
        orch.save_state()

        # Config gene fitness should decrease after unhealthy feedback
        allele = orch.registry.get(bridge_create_sha)
        unhealthy_fitness = arena.compute_fitness(allele)
        assert unhealthy_fitness < healthy_fitness

    def test_mac_flapping_reduces_config_fitness(self, project):
        orch = self._make_orch(project)

        # Provision bridge
        orch.run_pathway("provision_management_bridge", json.dumps({
            "bridge_name": "br0",
            "interfaces": ["eth0"],
            "uplink": "eth1",
            "stp_enabled": True,
            "forward_delay": 15,
        }))

        # Inject MAC flapping
        orch.kernel.inject_mac_flapping("br0", "de:ad:be:ef:00:01", ["eth0", "eth1"])

        # Run health check
        orch.run_pathway("health_check_bridge", json.dumps({
            "bridge_name": "br0",
        }))
        orch.save_state()

        # check_mac_stability feeds bridge_create and mac_preserve
        bridge_create_sha = orch.phenotype.get_dominant("bridge_create")
        allele = orch.registry.get(bridge_create_sha)
        unhealthy_records = [
            r for r in allele.fitness_records if not r["success"]
        ]
        assert len(unhealthy_records) > 0
