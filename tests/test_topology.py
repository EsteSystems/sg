"""Tests for topology decomposition and execution engine."""
import json
import shutil
import pytest
from pathlib import Path

from sg.contracts import ContractStore
from sg.fusion import FusionTracker
from sg.kernel.mock import MockNetworkKernel
from sg.mutation import MockMutationEngine
from sg.orchestrator import Orchestrator
from sg.parser.parser import parse_sg
from sg.parser.types import TopologyContract, TopologyResource
from sg.phenotype import PhenotypeMap
from sg.registry import Registry
from sg.kernel.network_mappers import NETWORK_RESOURCE_MAPPERS
from sg.topology import (
    TopologyStep, decompose, execute_topology,
    _build_dependency_graph, _topological_sort,
)


CONTRACTS_DIR = Path(__file__).parent.parent / "contracts"
GENES_DIR = Path(__file__).parent.parent / "genes"
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


# --- Decomposition unit tests ---

class TestDecompose:
    def test_bridge_with_uplink(self):
        """Bridge with uplink property → provision_management_bridge pathway."""
        topo = TopologyContract(
            name="test",
            does="test",
            has=[TopologyResource(
                name="mgmt",
                resource_type="bridge",
                properties={"uplink": "{uplink}", "stp": "enabled"},
            )],
        )
        steps = decompose(topo, json.dumps({
            "bridge_name": "br0",
            "bridge_ifaces": ["eth0"],
            "uplink": "eth1",
        }), NETWORK_RESOURCE_MAPPERS)
        assert len(steps) == 1
        assert steps[0].action == "pathway"
        assert steps[0].target == "provision_management_bridge"
        assert steps[0].resource_name == "mgmt"
        data = json.loads(steps[0].input_json)
        assert data["bridge_name"] == "br0"
        assert data["uplink"] == "eth1"
        assert data["stp_enabled"] is True

    def test_bridge_with_stp_no_uplink(self):
        """Bridge with stp but no uplink → configure_bridge_with_stp pathway."""
        topo = TopologyContract(
            name="test",
            does="test",
            has=[TopologyResource(
                name="internal",
                resource_type="bridge",
                properties={"stp": "enabled"},
            )],
        )
        steps = decompose(topo, json.dumps({
            "bridge_name": "br0",
            "bridge_ifaces": ["eth0"],
        }), NETWORK_RESOURCE_MAPPERS)
        assert len(steps) == 1
        assert steps[0].action == "pathway"
        assert steps[0].target == "configure_bridge_with_stp"

    def test_bare_bridge(self):
        """Bridge with no special properties → bridge_create gene."""
        topo = TopologyContract(
            name="test",
            does="test",
            has=[TopologyResource(
                name="simple",
                resource_type="bridge",
                properties={},
            )],
        )
        steps = decompose(topo, json.dumps({
            "bridge_name": "br0",
            "bridge_ifaces": ["eth0"],
        }), NETWORK_RESOURCE_MAPPERS)
        assert len(steps) == 1
        assert steps[0].action == "gene"
        assert steps[0].target == "bridge_create"

    def test_bond(self):
        """Bond resource → bond_create gene."""
        topo = TopologyContract(
            name="test",
            does="test",
            has=[TopologyResource(
                name="storage",
                resource_type="bond",
                properties={"mode": "{bond_mode}", "members": "{bond_members}"},
            )],
        )
        steps = decompose(topo, json.dumps({
            "bond_name": "bond0",
            "bond_mode": "active-backup",
            "bond_members": ["eth2", "eth3"],
        }), NETWORK_RESOURCE_MAPPERS)
        assert len(steps) == 1
        assert steps[0].action == "gene"
        assert steps[0].target == "bond_create"
        data = json.loads(steps[0].input_json)
        assert data["bond_name"] == "bond0"
        assert data["mode"] == "active-backup"
        assert data["members"] == ["eth2", "eth3"]

    def test_vlan_bridges(self):
        """vlan_bridges resource → loop vlan_create gene over vlans."""
        topo = TopologyContract(
            name="test",
            does="test",
            has=[TopologyResource(
                name="vlans",
                resource_type="vlan_bridges",
                properties={"vlans": "{vlans}"},
            )],
        )
        steps = decompose(topo, json.dumps({
            "bond_name": "bond0",
            "vlans": [100, 200, 300],
        }), NETWORK_RESOURCE_MAPPERS)
        assert len(steps) == 1
        assert steps[0].action == "loop_gene"
        assert steps[0].target == "vlan_create"
        assert len(steps[0].loop_items) == 3
        # Check first loop item
        item0 = json.loads(steps[0].loop_items[0])
        assert item0["parent"] == "bond0"
        assert item0["vlan_id"] == 100

    def test_unknown_resource_type(self):
        """Unknown resource type raises ValueError."""
        topo = TopologyContract(
            name="test",
            does="test",
            has=[TopologyResource(
                name="unknown",
                resource_type="quantum_link",
                properties={},
            )],
        )
        with pytest.raises(ValueError, match="unknown resource type"):
            decompose(topo, "{}", NETWORK_RESOURCE_MAPPERS)


# --- Dependency ordering ---

class TestDependencyOrdering:
    def test_trunk_creates_dependency(self):
        """'trunk storage' means vm_traffic depends on storage."""
        resources = [
            TopologyResource(name="vm_traffic", resource_type="vlan_bridges",
                             properties={"trunk": "storage", "vlans": "{vlans}"}),
            TopologyResource(name="storage", resource_type="bond",
                             properties={"mode": "active-backup"}),
        ]
        deps = _build_dependency_graph(resources)
        assert "vm_traffic" in deps
        assert "storage" in deps["vm_traffic"]

    def test_topological_sort_respects_deps(self):
        """Resources with dependencies come after their dependencies."""
        resources = [
            TopologyResource(name="vm_traffic", resource_type="vlan_bridges",
                             properties={"trunk": "storage", "vlans": "{vlans}"}),
            TopologyResource(name="management", resource_type="bridge",
                             properties={"uplink": "eth0"}),
            TopologyResource(name="storage", resource_type="bond",
                             properties={"mode": "active-backup"}),
        ]
        deps = _build_dependency_graph(resources)
        ordered = _topological_sort(resources, deps)
        names = [r.name for r in ordered]

        # storage must come before vm_traffic
        assert names.index("storage") < names.index("vm_traffic")
        # management has no deps — can be anywhere before vm_traffic
        assert "management" in names

    def test_circular_dependency_raises(self):
        """Circular dependencies raise ValueError."""
        resources = [
            TopologyResource(name="a", resource_type="bridge",
                             properties={"depends_on": "b"}),
            TopologyResource(name="b", resource_type="bridge",
                             properties={"depends_on": "a"}),
        ]
        deps = _build_dependency_graph(resources)
        with pytest.raises(ValueError, match="circular dependency"):
            _topological_sort(resources, deps)

    def test_full_production_server_order(self):
        """production_server.sg decomposes with correct dependency order."""
        cs = ContractStore.open(CONTRACTS_DIR)
        topo = cs.get_topology("production_server")
        assert topo is not None

        steps = decompose(topo, json.dumps({
            "bridge_name": "br0",
            "bridge_ifaces": ["eth0"],
            "uplink": "eth1",
            "bond_name": "bond0",
            "bond_mode": "active-backup",
            "bond_members": ["eth2", "eth3"],
            "vlans": [100, 200],
        }), NETWORK_RESOURCE_MAPPERS)

        names = [s.resource_name for s in steps]
        # storage before vm_traffic (trunk dependency)
        assert names.index("storage") < names.index("vm_traffic")
        # management has no dependency on storage — it can be first or second
        assert "management" in names
        assert len(steps) == 3


# --- Integration: full topology execution ---

class TestExecuteTopology:
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

    def test_execute_production_server(self, project):
        """Full production_server topology executes all resources."""
        orch = self._make_orch(project)
        outputs = orch.run_topology("production_server", json.dumps({
            "bridge_name": "br0",
            "bridge_ifaces": ["eth0"],
            "uplink": "eth1",
            "bond_name": "bond0",
            "bond_mode": "active-backup",
            "bond_members": ["eth2", "eth3"],
            "vlans": [100, 200],
        }))
        # Should have outputs from: pathway (multiple steps) + bond_create + 2x vlan_create
        assert len(outputs) >= 3

        # Bridge should exist
        assert orch.kernel.get_bridge("br0") is not None
        # Bond should exist
        assert orch.kernel.get_bond("bond0") is not None
        # VLANs should exist
        assert orch.kernel.get_vlan("bond0", 100) is not None
        assert orch.kernel.get_vlan("bond0", 200) is not None

    def test_unknown_topology_raises(self, project):
        """Unknown topology name raises ValueError."""
        orch = self._make_orch(project)
        with pytest.raises(ValueError, match="unknown topology"):
            orch.run_topology("nonexistent", "{}")

    def test_preserve_what_works(self, project):
        """Failure in one resource doesn't abort others."""
        orch = self._make_orch(project)

        # Remove bond_create allele AND its mutation fixture so the step
        # truly fails (no allele, no mutation rescue)
        orch.phenotype.loci.pop("bond_create", None)
        fixture = project / "fixtures" / "bond_create_fix.py"
        if fixture.exists():
            fixture.unlink()

        # Use a two-resource topology: bridge + bond
        # Bridge should succeed, bond should fail
        topo = TopologyContract(
            name="test_partial",
            does="test partial failure",
            has=[
                TopologyResource(name="mgmt", resource_type="bridge",
                                 properties={}),
                TopologyResource(name="storage", resource_type="bond",
                                 properties={"mode": "active-backup",
                                             "members": "{bond_members}"}),
            ],
            on_failure="preserve what works",
        )
        from sg.topology import execute_topology

        with pytest.raises(RuntimeError, match="partially failed"):
            execute_topology(topo, json.dumps({
                "bridge_name": "br0",
                "bridge_ifaces": ["eth0"],
                "bond_name": "bond0",
                "bond_members": ["eth2", "eth3"],
            }), orch, NETWORK_RESOURCE_MAPPERS)

        # Bridge should still have been created (preserve what works)
        assert orch.kernel.get_bridge("br0") is not None


# --- Contract parsing ---

class TestTopologyContractParsing:
    def test_production_server_loads(self):
        """production_server.sg loads and parses correctly."""
        cs = ContractStore.open(CONTRACTS_DIR)
        assert "production_server" in cs.known_topologies()
        topo = cs.get_topology("production_server")
        assert topo is not None
        assert len(topo.has) == 3
        assert topo.on_failure == "preserve what works"

    def test_resource_types(self):
        """Resources have correct types."""
        cs = ContractStore.open(CONTRACTS_DIR)
        topo = cs.get_topology("production_server")
        types = {r.name: r.resource_type for r in topo.has}
        assert types["management"] == "bridge"
        assert types["storage"] == "bond"
        assert types["vm_traffic"] == "vlan_bridges"

    def test_verify_steps_parsed(self):
        """Verify steps are parsed from topology contract."""
        cs = ContractStore.open(CONTRACTS_DIR)
        topo = cs.get_topology("production_server")
        assert len(topo.verify) == 2
        loci = [v.locus for v in topo.verify]
        assert "check_connectivity" in loci
        assert "check_bond_state" in loci
