"""Tests for network gene contracts, seed genes, and pathways.

M5: Full network gene set running against MockNetworkKernel.
"""
import json
import shutil
import pytest
from pathlib import Path

from sg.contracts import ContractStore
from sg.fusion import FusionTracker
from sg.kernel.mock import MockNetworkKernel
from sg.loader import load_gene, call_gene
from sg.mutation import MockMutationEngine
from sg.orchestrator import Orchestrator
from sg.phenotype import PhenotypeMap
from sg.registry import Registry


CONTRACTS_DIR = Path(__file__).parent.parent / "contracts"
GENES_DIR = Path(__file__).parent.parent / "genes"
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def kernel():
    k = MockNetworkKernel()
    k.reset()
    return k


@pytest.fixture
def store():
    return ContractStore.open(CONTRACTS_DIR)


# --- Contract parsing ---

class TestContracts:
    def test_all_config_genes_parsed(self, store):
        config_genes = ["bridge_create", "bridge_stp", "bridge_uplink",
                        "mac_preserve", "bond_create", "vlan_create"]
        for name in config_genes:
            gene = store.get_gene(name)
            assert gene is not None, f"missing gene contract: {name}"
            assert gene.family.value == "configuration"

    def test_all_diagnostic_genes_parsed(self, store):
        diag_genes = ["check_connectivity", "check_mac_stability",
                      "check_fdb_stability", "check_link_state",
                      "check_bond_state"]
        for name in diag_genes:
            gene = store.get_gene(name)
            assert gene is not None, f"missing gene contract: {name}"
            assert gene.family.value == "diagnostic"

    def test_all_pathways_parsed(self, store):
        pathways = ["configure_bridge_with_stp", "provision_management_bridge",
                    "health_check_bridge"]
        for name in pathways:
            pw = store.get_pathway(name)
            assert pw is not None, f"missing pathway contract: {name}"

    def test_provision_management_bridge_has_four_steps(self, store):
        pw = store.get_pathway("provision_management_bridge")
        assert len(pw.steps) == 4

    def test_health_check_bridge_has_three_steps(self, store):
        pw = store.get_pathway("health_check_bridge")
        assert len(pw.steps) == 3

    def test_diagnostic_genes_have_feeds(self, store):
        gene = store.get_gene("check_connectivity")
        assert len(gene.feeds) > 0
        targets = [f.target_locus for f in gene.feeds]
        assert "bridge_create" in targets


# --- Seed gene execution ---

class TestConfigGenes:
    def test_bridge_uplink(self, kernel):
        kernel.create_bridge("br0", ["eth0"])
        source = (GENES_DIR / "bridge_uplink_v1.py").read_text()
        fn = load_gene(source, kernel)
        result = json.loads(call_gene(fn, json.dumps({
            "bridge_name": "br0", "uplink": "eth1"
        })))
        assert result["success"] is True
        bridge = kernel.get_bridge("br0")
        assert "eth1" in bridge["interfaces"]

    def test_mac_preserve(self, kernel):
        kernel.create_bridge("br0", ["eth0"])
        source = (GENES_DIR / "mac_preserve_v1.py").read_text()
        fn = load_gene(source, kernel)
        result = json.loads(call_gene(fn, json.dumps({
            "device": "br0", "source_mac": "aa:bb:cc:dd:ee:ff"
        })))
        assert result["success"] is True
        assert result["new_mac"] == "aa:bb:cc:dd:ee:ff"
        assert kernel.get_device_mac("br0") == "aa:bb:cc:dd:ee:ff"

    def test_mac_preserve_default_reads_existing(self, kernel):
        kernel.create_bridge("br0", ["eth0"])
        original_mac = kernel.get_device_mac("br0")
        source = (GENES_DIR / "mac_preserve_v1.py").read_text()
        fn = load_gene(source, kernel)
        result = json.loads(call_gene(fn, json.dumps({"device": "br0"})))
        assert result["success"] is True
        assert result["original_mac"] == original_mac
        assert result["new_mac"] == original_mac

    def test_bond_create(self, kernel):
        source = (GENES_DIR / "bond_create_v1.py").read_text()
        fn = load_gene(source, kernel)
        result = json.loads(call_gene(fn, json.dumps({
            "bond_name": "bond0", "mode": "802.3ad", "members": ["eth0", "eth1"]
        })))
        assert result["success"] is True
        assert result["bond"] == "bond0"
        bond = kernel.get_bond("bond0")
        assert bond is not None

    def test_vlan_create(self, kernel):
        source = (GENES_DIR / "vlan_create_v1.py").read_text()
        fn = load_gene(source, kernel)
        result = json.loads(call_gene(fn, json.dumps({
            "parent": "eth0", "vlan_id": 100
        })))
        assert result["success"] is True
        assert result["vlan_name"] == "eth0.100"
        vlan = kernel.get_vlan("eth0", 100)
        assert vlan is not None


class TestDiagnosticGenes:
    def test_check_connectivity_healthy(self, kernel):
        kernel.create_bridge("br0", ["eth0", "eth1"])
        source = (GENES_DIR / "check_connectivity_v1.py").read_text()
        fn = load_gene(source, kernel)
        result = json.loads(call_gene(fn, json.dumps({"bridge_name": "br0"})))
        assert result["success"] is True
        assert result["healthy"] is True
        assert result["bridge_up"] is True
        assert result["ports_down"] == []

    def test_check_connectivity_link_down(self, kernel):
        kernel.create_bridge("br0", ["eth0", "eth1"])
        kernel.inject_link_failure("eth0")
        source = (GENES_DIR / "check_connectivity_v1.py").read_text()
        fn = load_gene(source, kernel)
        result = json.loads(call_gene(fn, json.dumps({"bridge_name": "br0"})))
        assert result["success"] is True
        assert result["healthy"] is False
        assert "eth0" in result["ports_down"]

    def test_check_mac_stability_healthy(self, kernel):
        kernel.create_bridge("br0", ["eth0"])
        kernel.add_fdb_entry("br0", "aa:bb:cc:dd:ee:ff", "eth0")
        source = (GENES_DIR / "check_mac_stability_v1.py").read_text()
        fn = load_gene(source, kernel)
        result = json.loads(call_gene(fn, json.dumps({"bridge_name": "br0"})))
        assert result["success"] is True
        assert result["healthy"] is True
        assert result["flapping_macs"] == []

    def test_check_mac_stability_detects_flapping(self, kernel):
        kernel.create_bridge("br0", ["eth0", "eth1"])
        kernel.inject_mac_flapping("br0", "aa:bb:cc:dd:ee:ff", ["eth0", "eth1"])
        source = (GENES_DIR / "check_mac_stability_v1.py").read_text()
        fn = load_gene(source, kernel)
        result = json.loads(call_gene(fn, json.dumps({"bridge_name": "br0"})))
        assert result["success"] is True
        assert result["healthy"] is False
        assert "aa:bb:cc:dd:ee:ff" in result["flapping_macs"]

    def test_check_fdb_stability_healthy(self, kernel):
        kernel.create_bridge("br0", ["eth0"])
        source = (GENES_DIR / "check_fdb_stability_v1.py").read_text()
        fn = load_gene(source, kernel)
        result = json.loads(call_gene(fn, json.dumps({"bridge_name": "br0"})))
        assert result["success"] is True
        assert result["healthy"] is True
        assert result["anomalies"] == []

    def test_check_fdb_stability_detects_duplicates(self, kernel):
        kernel.create_bridge("br0", ["eth0", "eth1"])
        kernel.inject_mac_flapping("br0", "de:ad:be:ef:00:01", ["eth0", "eth1"])
        source = (GENES_DIR / "check_fdb_stability_v1.py").read_text()
        fn = load_gene(source, kernel)
        result = json.loads(call_gene(fn, json.dumps({"bridge_name": "br0"})))
        assert result["success"] is True
        assert result["healthy"] is False
        assert len(result["anomalies"]) > 0

    def test_check_link_state_up(self, kernel):
        kernel.create_bridge("br0", ["eth0"])
        source = (GENES_DIR / "check_link_state_v1.py").read_text()
        fn = load_gene(source, kernel)
        result = json.loads(call_gene(fn, json.dumps({"interface": "eth0"})))
        assert result["success"] is True
        assert result["healthy"] is True
        assert result["carrier"] is True

    def test_check_link_state_down(self, kernel):
        kernel.create_bridge("br0", ["eth0"])
        kernel.inject_link_failure("eth0")
        source = (GENES_DIR / "check_link_state_v1.py").read_text()
        fn = load_gene(source, kernel)
        result = json.loads(call_gene(fn, json.dumps({"interface": "eth0"})))
        assert result["success"] is True
        assert result["healthy"] is False
        assert result["carrier"] is False

    def test_check_bond_state_healthy(self, kernel):
        kernel.create_bond("bond0", "802.3ad", ["eth0", "eth1"])
        source = (GENES_DIR / "check_bond_state_v1.py").read_text()
        fn = load_gene(source, kernel)
        result = json.loads(call_gene(fn, json.dumps({"bond_name": "bond0"})))
        assert result["success"] is True
        assert result["healthy"] is True
        assert result["members_down"] == []

    def test_check_bond_state_member_down(self, kernel):
        kernel.create_bond("bond0", "802.3ad", ["eth0", "eth1"])
        kernel.inject_link_failure("eth0")
        source = (GENES_DIR / "check_bond_state_v1.py").read_text()
        fn = load_gene(source, kernel)
        result = json.loads(call_gene(fn, json.dumps({"bond_name": "bond0"})))
        assert result["success"] is True
        assert result["healthy"] is False
        assert "eth0" in result["members_down"]


# --- Pathway execution ---

def _make_project(tmp_path):
    """Set up a full project with all genes and contracts."""
    fixtures_dst = tmp_path / "fixtures"
    shutil.copytree(FIXTURES_DIR, fixtures_dst)
    contracts_dst = tmp_path / "contracts"
    shutil.copytree(CONTRACTS_DIR, contracts_dst)

    contract_store = ContractStore.open(contracts_dst)
    registry = Registry.open(tmp_path / ".sg" / "registry")
    phenotype = PhenotypeMap()

    # Register all seed genes
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


def _make_orchestrator(project_root):
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


class TestProvisionManagementBridge:
    @pytest.fixture
    def project(self, tmp_path):
        return _make_project(tmp_path)

    def test_full_pathway(self, project):
        orch = _make_orchestrator(project)
        input_json = json.dumps({
            "bridge_name": "br-mgmt",
            "interfaces": ["eth0"],
            "uplink": "eth1",
            "stp_enabled": True,
            "forward_delay": 15,
        })
        outputs = orch.run_pathway("provision_management_bridge", input_json)
        orch.save_state()

        assert len(outputs) == 4
        for output in outputs:
            data = json.loads(output)
            assert data["success"] is True

    def test_failure_triggers_mutation(self, project):
        orch = _make_orchestrator(project)
        orch.kernel.inject_failure("create_bridge", "simulated failure")
        input_json = json.dumps({
            "bridge_name": "br-mgmt",
            "interfaces": ["eth0"],
            "uplink": "eth1",
            "stp_enabled": True,
            "forward_delay": 15,
        })
        outputs = orch.run_pathway("provision_management_bridge", input_json)
        orch.save_state()
        # Should succeed via mutation
        assert len(outputs) == 4


class TestHealthCheckBridge:
    @pytest.fixture
    def project(self, tmp_path):
        return _make_project(tmp_path)

    def test_healthy_bridge(self, project):
        orch = _make_orchestrator(project)
        # First provision the bridge
        orch.run_pathway("provision_management_bridge", json.dumps({
            "bridge_name": "br0",
            "interfaces": ["eth0"],
            "uplink": "eth1",
            "stp_enabled": True,
            "forward_delay": 15,
        }))
        # Then check health
        outputs = orch.run_pathway("health_check_bridge", json.dumps({
            "bridge_name": "br0",
        }))
        orch.save_state()

        assert len(outputs) == 3
        for output in outputs:
            data = json.loads(output)
            assert data["success"] is True
            assert data["healthy"] is True

    def test_unhealthy_bridge_link_down(self, project):
        orch = _make_orchestrator(project)
        orch.run_pathway("provision_management_bridge", json.dumps({
            "bridge_name": "br0",
            "interfaces": ["eth0"],
            "uplink": "eth1",
            "stp_enabled": True,
            "forward_delay": 15,
        }))
        # Inject link failure
        orch.kernel.inject_link_failure("eth0")
        outputs = orch.run_pathway("health_check_bridge", json.dumps({
            "bridge_name": "br0",
        }))
        orch.save_state()

        connectivity = json.loads(outputs[0])
        assert connectivity["success"] is True
        assert connectivity["healthy"] is False

    def test_unhealthy_bridge_mac_flapping(self, project):
        orch = _make_orchestrator(project)
        orch.run_pathway("provision_management_bridge", json.dumps({
            "bridge_name": "br0",
            "interfaces": ["eth0"],
            "uplink": "eth1",
            "stp_enabled": True,
            "forward_delay": 15,
        }))
        # Inject MAC flapping
        orch.kernel.inject_mac_flapping("br0", "de:ad:be:ef:00:01", ["eth0", "eth1"])
        outputs = orch.run_pathway("health_check_bridge", json.dumps({
            "bridge_name": "br0",
        }))
        orch.save_state()

        mac_check = json.loads(outputs[1])
        assert mac_check["success"] is True
        assert mac_check["healthy"] is False
        assert "de:ad:be:ef:00:01" in mac_check["flapping_macs"]
