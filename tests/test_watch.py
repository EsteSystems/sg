"""Tests for sg watch daemon mode (resilience fitness)."""
import json
import shutil
import pytest
from pathlib import Path

from sg import arena
from sg.contracts import ContractStore
from sg.fusion import FusionTracker
from sg_network import MockNetworkKernel
from sg.mutation import MockMutationEngine
from sg.orchestrator import Orchestrator
from sg.phenotype import PhenotypeMap
from sg.registry import Registry


import sg_network

CONTRACTS_DIR = sg_network.contracts_path()
GENES_DIR = sg_network.genes_path()
FIXTURES_DIR = sg_network.fixtures_path()


class TestFeedbackTimescaleOverride:
    @pytest.fixture
    def project(self, tmp_path):
        shutil.copytree(FIXTURES_DIR, tmp_path / "fixtures")
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

    def test_default_timescale_is_convergence(self, project):
        """Without override, feedback uses contract's declared timescale."""
        orch = self._make_orch(project)
        assert orch.feedback_timescale is None

        # Provision bridge, then run health check
        orch.run_pathway("provision_management_bridge", json.dumps({
            "bridge_name": "br0", "interfaces": ["eth0"],
            "uplink": "eth1", "stp_enabled": True, "forward_delay": 15,
        }))
        orch.verify_scheduler.wait()
        orch.run_pathway("health_check_bridge", json.dumps({
            "bridge_name": "br0",
        }))

        config_sha = orch.phenotype.get_dominant("bridge_create")
        allele = orch.registry.get(config_sha)
        convergence_records = [
            r for r in allele.fitness_records if r["timescale"] == "convergence"
        ]
        assert len(convergence_records) > 0

    def test_resilience_override(self, project):
        """With feedback_timescale='resilience', all feedback goes to resilience."""
        orch = self._make_orch(project)
        orch.feedback_timescale = "resilience"

        # Provision bridge, then run health check
        orch.run_pathway("provision_management_bridge", json.dumps({
            "bridge_name": "br0", "interfaces": ["eth0"],
            "uplink": "eth1", "stp_enabled": True, "forward_delay": 15,
        }))
        orch.verify_scheduler.wait()
        orch.run_pathway("health_check_bridge", json.dumps({
            "bridge_name": "br0",
        }))

        config_sha = orch.phenotype.get_dominant("bridge_create")
        allele = orch.registry.get(config_sha)
        resilience_records = [
            r for r in allele.fitness_records if r["timescale"] == "resilience"
        ]
        assert len(resilience_records) > 0

    def test_resilience_failure_reduces_fitness(self, project):
        """Resilience failure lowers overall fitness."""
        orch = self._make_orch(project)
        orch.feedback_timescale = "resilience"

        # Provision bridge
        orch.run_pathway("provision_management_bridge", json.dumps({
            "bridge_name": "br0", "interfaces": ["eth0"],
            "uplink": "eth1", "stp_enabled": True, "forward_delay": 15,
        }))
        orch.verify_scheduler.wait()

        config_sha = orch.phenotype.get_dominant("bridge_create")
        healthy_fitness = arena.compute_fitness(orch.registry.get(config_sha))

        # Inject link failure, run health check with resilience timescale
        orch.kernel.inject_link_failure("eth0")
        orch.run_pathway("health_check_bridge", json.dumps({
            "bridge_name": "br0",
        }))

        unhealthy_fitness = arena.compute_fitness(orch.registry.get(config_sha))
        assert unhealthy_fitness < healthy_fitness

    def test_multiple_watch_iterations(self, project):
        """Simulates what sg watch does: multiple diagnostic runs accumulate resilience records."""
        orch = self._make_orch(project)
        orch.feedback_timescale = "resilience"

        # Provision
        orch.run_pathway("provision_management_bridge", json.dumps({
            "bridge_name": "br0", "interfaces": ["eth0"],
            "uplink": "eth1", "stp_enabled": True, "forward_delay": 15,
        }))
        orch.verify_scheduler.wait()

        # Simulate 3 watch iterations
        for _ in range(3):
            orch.run_pathway("health_check_bridge", json.dumps({
                "bridge_name": "br0",
            }))

        config_sha = orch.phenotype.get_dominant("bridge_create")
        allele = orch.registry.get(config_sha)
        resilience_records = [
            r for r in allele.fitness_records if r["timescale"] == "resilience"
        ]
        assert len(resilience_records) >= 3
