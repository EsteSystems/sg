"""End-to-end test: MAC flapping detection, fitness decay, and remediation.

Tests the full evolutionary loop:
  provision → inject flapping → diagnostic detection → fitness decay
  → remediation pathway → verify recovery

This is the canonical demonstration that Software Genomics works:
diagnostic genes observe the environment, feed results back to
configuration genes, and the system autonomously responds.
"""
import json
import shutil
import pytest
from pathlib import Path

from sg import arena
from sg.contracts import ContractStore
from sg.fitness import compute_temporal_fitness
from sg.fusion import FusionTracker
from sg.kernel.mock import MockNetworkKernel
from sg.mutation import MockMutationEngine
from sg.orchestrator import Orchestrator
from sg.phenotype import PhenotypeMap
from sg.registry import Registry


CONTRACTS_DIR = Path(__file__).parent.parent / "contracts"
GENES_DIR = Path(__file__).parent.parent / "genes"
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def project(tmp_path):
    """Full project setup — all genes, contracts, fixtures."""
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


def make_orch(project_root: Path) -> Orchestrator:
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


PROVISION_INPUT = json.dumps({
    "bridge_name": "br0",
    "interfaces": ["eth0"],
    "uplink": "eth1",
    "stp_enabled": True,
    "forward_delay": 15,
})


class TestMACFlappingE2E:
    """Full MAC flapping scenario — the canonical Software Genomics demo."""

    def test_provision_then_healthy_check(self, project):
        """Phase 1: Provision bridge, verify health check passes."""
        orch = make_orch(project)

        # Provision
        outputs = orch.run_pathway("provision_management_bridge", PROVISION_INPUT)
        assert len(outputs) == 4  # bridge_create, bridge_stp, bridge_uplink, mac_preserve
        assert all(json.loads(o)["success"] for o in outputs)

        # Health check — should be healthy
        health_outputs = orch.run_pathway("health_check_bridge", json.dumps({
            "bridge_name": "br0",
        }))
        assert len(health_outputs) == 3

        # check_connectivity should report healthy
        connectivity = json.loads(health_outputs[0])
        assert connectivity["healthy"] is True

        orch.save_state()

    def test_mac_flapping_detection(self, project):
        """Phase 2: Inject MAC flapping, verify diagnostic detects it."""
        orch = make_orch(project)

        # Provision
        orch.run_pathway("provision_management_bridge", PROVISION_INPUT)

        # Inject MAC flapping
        orch.kernel.inject_mac_flapping("br0", "de:ad:be:ef:00:01", ["eth0", "eth1"])

        # Health check — should detect flapping
        health_outputs = orch.run_pathway("health_check_bridge", json.dumps({
            "bridge_name": "br0",
        }))

        # check_mac_stability should report unhealthy with flapping MACs
        mac_stability = json.loads(health_outputs[1])
        assert mac_stability["healthy"] is False
        assert len(mac_stability.get("flapping_macs", [])) > 0

        orch.save_state()

    def test_fitness_decay_after_unhealthy_diagnostic(self, project):
        """Phase 3: Unhealthy diagnostics reduce config gene fitness."""
        orch = make_orch(project)

        # Provision
        orch.run_pathway("provision_management_bridge", PROVISION_INPUT)

        # Healthy check first
        orch.run_pathway("health_check_bridge", json.dumps({
            "bridge_name": "br0",
        }))
        orch.save_state()

        # Get bridge_create fitness after healthy check
        bridge_sha = orch.phenotype.get_dominant("bridge_create")
        allele = orch.registry.get(bridge_sha)
        healthy_fitness = arena.compute_fitness(allele)

        # Inject flapping and run unhealthy check
        orch.kernel.inject_mac_flapping("br0", "de:ad:be:ef:00:01", ["eth0", "eth1"])
        orch.run_pathway("health_check_bridge", json.dumps({
            "bridge_name": "br0",
        }))
        orch.save_state()

        # Fitness should have decreased
        orch2 = make_orch(project)
        allele2 = orch2.registry.get(bridge_sha)
        unhealthy_fitness = arena.compute_fitness(allele2)
        assert unhealthy_fitness < healthy_fitness

    def test_convergence_records_persist_across_sessions(self, project):
        """Phase 4: Fitness records survive save/load cycle."""
        orch = make_orch(project)

        # Provision + healthy check
        orch.run_pathway("provision_management_bridge", PROVISION_INPUT)
        orch.run_pathway("health_check_bridge", json.dumps({"bridge_name": "br0"}))
        orch.save_state()

        # Inject flapping + unhealthy check
        orch.kernel.inject_mac_flapping("br0", "de:ad:be:ef:00:01", ["eth0", "eth1"])
        orch.run_pathway("health_check_bridge", json.dumps({"bridge_name": "br0"}))
        orch.save_state()

        # Reload from disk — records should persist
        orch2 = make_orch(project)
        bridge_sha = orch2.phenotype.get_dominant("bridge_create")
        allele = orch2.registry.get(bridge_sha)
        assert len(allele.fitness_records) > 0

        # Should have both healthy and unhealthy records
        healthy_count = sum(1 for r in allele.fitness_records if r["success"])
        unhealthy_count = sum(1 for r in allele.fitness_records if not r["success"])
        assert healthy_count > 0
        assert unhealthy_count > 0

    def test_remediation_pathway_triggers_on_flapping(self, project):
        """Phase 5: remediate_mac_flapping detects and remediates."""
        orch = make_orch(project)

        # Provision
        orch.run_pathway("provision_management_bridge", PROVISION_INPUT)

        # Inject flapping
        orch.kernel.inject_mac_flapping("br0", "de:ad:be:ef:00:01", ["eth0", "eth1"])

        # Run remediation pathway
        outputs = orch.run_pathway("remediate_mac_flapping", json.dumps({
            "bridge_name": "br0",
        }))

        # Should have diagnostic output + remediation output
        assert len(outputs) == 2

        # First output: check_mac_stability reports unhealthy
        diagnostic = json.loads(outputs[0])
        assert diagnostic["healthy"] is False

        # Second output: mac_preserve ran as remediation
        remediation = json.loads(outputs[1])
        assert remediation["success"] is True

    def test_link_failure_detection(self, project):
        """Link failure detected by check_connectivity diagnostic."""
        orch = make_orch(project)

        # Provision
        orch.run_pathway("provision_management_bridge", PROVISION_INPUT)

        # Inject link failure
        orch.kernel.inject_link_failure("eth0")

        # Health check
        health_outputs = orch.run_pathway("health_check_bridge", json.dumps({
            "bridge_name": "br0",
        }))

        # check_connectivity should detect the link failure
        connectivity = json.loads(health_outputs[0])
        assert connectivity["healthy"] is False


class TestDeployServerNetwork:
    """Tests for the composed deploy_server_network pathway."""

    def test_full_server_deployment(self, project):
        """Deploy bridge + bond + VLANs in a single pathway."""
        orch = make_orch(project)

        outputs = orch.run_pathway("deploy_server_network", json.dumps({
            "bridge_name": "br0",
            "interfaces": ["eth0"],
            "uplink": "eth1",
            "stp_enabled": True,
            "forward_delay": 15,
            "bond_name": "bond0",
            "bond_mode": "802.3ad",
            "bond_members": ["eth2", "eth3"],
            "vlans": [100, 200],
        }))

        # Verify all resources created
        assert orch.kernel.get_bridge("br0") is not None
        assert orch.kernel.get_bond("bond0") is not None
        assert orch.kernel.get_vlan("bond0", 100) is not None
        assert orch.kernel.get_vlan("bond0", 200) is not None

    def test_deployment_with_health_check(self, project):
        """Deploy then health check — end-to-end."""
        orch = make_orch(project)

        # Deploy
        orch.run_pathway("deploy_server_network", json.dumps({
            "bridge_name": "br0",
            "interfaces": ["eth0"],
            "uplink": "eth1",
            "stp_enabled": True,
            "forward_delay": 15,
            "bond_name": "bond0",
            "bond_mode": "802.3ad",
            "bond_members": ["eth2", "eth3"],
            "vlans": [100, 200],
        }))

        # Health check on deployed bridge
        health = orch.run_pathway("health_check_bridge", json.dumps({
            "bridge_name": "br0",
        }))
        assert len(health) == 3
        assert json.loads(health[0])["healthy"] is True


class TestEvolutionaryLoop:
    """Tests demonstrating the full evolutionary loop."""

    def test_mutation_on_failure_then_recovery(self, project):
        """Gene failure → mutation → fix → successful execution."""
        orch = make_orch(project)

        # Register a gene that always raises (doesn't catch errors)
        bad_gene = (
            'import json\n'
            'def execute(input_json):\n'
            '    raise RuntimeError("always fails")\n'
        )
        sha = orch.registry.register(bad_gene, "bridge_create")
        orch.phenotype.promote("bridge_create", sha)
        allele = orch.registry.get(sha)
        allele.state = "dominant"

        # The dominant allele raises, mutation fix should succeed
        result = orch.execute_locus("bridge_create", json.dumps({
            "bridge_name": "br0",
            "interfaces": ["eth0"],
        }))

        # Should succeed via mutation fix
        assert result is not None
        output, used_sha = result
        data = json.loads(output)
        assert data["success"] is True
        # The mutation fix sha should differ from the bad gene
        assert used_sha != sha

    def test_fitness_tracks_across_invocations(self, project):
        """Multiple invocations build up fitness tracking."""
        orch = make_orch(project)

        for i in range(5):
            orch.execute_locus("bridge_create", json.dumps({
                "bridge_name": f"br{i}",
                "interfaces": ["eth0"],
            }))

        bridge_sha = orch.phenotype.get_dominant("bridge_create")
        allele = orch.registry.get(bridge_sha)
        assert allele.successful_invocations >= 5
        assert arena.compute_fitness(allele) > 0.0

    def test_demotion_on_consecutive_failures(self, project):
        """Three consecutive failures demote an allele."""
        orch = make_orch(project)

        # Register a gene that always fails
        bad_gene = (
            'import json\n'
            'def execute(input_json):\n'
            '    raise RuntimeError("always fails")\n'
        )
        sha = orch.registry.register(bad_gene, "bridge_create")
        orch.phenotype.promote("bridge_create", sha)
        allele = orch.registry.get(sha)
        allele.state = "dominant"

        # Execute multiple times — should fail 3 times then demote
        for _ in range(4):
            orch.execute_locus("bridge_create", json.dumps({
                "bridge_name": "br0",
                "interfaces": ["eth0"],
            }))

        allele = orch.registry.get(sha)
        assert allele.state == "deprecated"

    def test_two_family_feedback_loop(self, project):
        """Diagnostic genes feed fitness back to config genes."""
        orch = make_orch(project)

        # Provision
        orch.run_pathway("provision_management_bridge", PROVISION_INPUT)

        # Get initial fitness
        bridge_sha = orch.phenotype.get_dominant("bridge_create")
        allele = orch.registry.get(bridge_sha)
        initial_records = len(allele.fitness_records)

        # Healthy health check — feeds positive convergence
        orch.run_pathway("health_check_bridge", json.dumps({"bridge_name": "br0"}))
        assert len(allele.fitness_records) > initial_records

        healthy_fitness = arena.compute_fitness(allele)

        # Inject failure + unhealthy check
        orch.kernel.inject_link_failure("eth0")
        orch.run_pathway("health_check_bridge", json.dumps({"bridge_name": "br0"}))

        # Fitness should decrease after unhealthy feedback
        unhealthy_fitness = arena.compute_fitness(allele)
        assert unhealthy_fitness < healthy_fitness

    def test_safety_rollback_on_gene_failure(self, project):
        """Gene failure rolls back partial kernel state."""
        orch = make_orch(project)

        # Register gene that creates state then raises
        bad_gene = (
            'import json\n'
            'def execute(input_json):\n'
            '    gene_sdk.create_bridge("leaked_br", ["eth0"])\n'
            '    raise RuntimeError("partial failure")\n'
        )
        sha = orch.registry.register(bad_gene, "bridge_create")
        orch.phenotype.promote("bridge_create", sha)
        allele = orch.registry.get(sha)
        allele.state = "dominant"

        orch.execute_locus("bridge_create", json.dumps({
            "bridge_name": "br0",
            "interfaces": ["eth0"],
        }))

        # "leaked_br" should have been rolled back
        assert orch.kernel.get_bridge("leaked_br") is None
