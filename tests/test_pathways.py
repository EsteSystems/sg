"""Tests for composed pathways — -> refs, for loops, requires, when conditionals."""
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
from sg.parser.types import (
    PathwayStep as ASTPathwayStep, ForStep as ASTForStep,
    ConditionalStep as ASTConditionalStep, Dependency,
)
from sg.pathway import (
    pathway_from_contract, Pathway, PathwayStep,
    ComposedStep, LoopStep, ConditionalExecStep,
    _validate_dependencies,
)
from sg.phenotype import PhenotypeMap
from sg.registry import Registry


CONTRACTS_DIR = Path(__file__).parent.parent / "contracts"
GENES_DIR = Path(__file__).parent.parent / "genes"
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


# --- Parser tests for new step types ---

class TestParserComposedSteps:
    def test_arrow_ref_parsed(self):
        source = CONTRACTS_DIR / "pathways" / "deploy_server_network.sg"
        contract = parse_sg(source.read_text())
        assert contract.steps[0].is_pathway_ref is True
        assert contract.steps[0].locus == "provision_management_bridge"

    def test_for_step_parsed(self):
        source = CONTRACTS_DIR / "pathways" / "deploy_server_network.sg"
        contract = parse_sg(source.read_text())
        for_step = contract.steps[2]
        assert isinstance(for_step, ASTForStep)
        assert for_step.variable == "vlan"
        assert for_step.iterable == "vlans"
        assert for_step.body is not None
        assert for_step.body.locus == "vlan_create"

    def test_for_step_body_params(self):
        source = CONTRACTS_DIR / "pathways" / "deploy_server_network.sg"
        contract = parse_sg(source.read_text())
        for_step = contract.steps[2]
        assert for_step.body.params == {
            "parent": "{bond_name}",
            "vlan_id": "{vlan}",
        }

    def test_when_step_parsed(self):
        source = CONTRACTS_DIR / "pathways" / "remediate_mac_flapping.sg"
        contract = parse_sg(source.read_text())
        when_step = contract.steps[1]
        assert isinstance(when_step, ASTConditionalStep)
        assert when_step.condition_step == 1
        assert when_step.condition_field == "healthy"
        assert "false" in when_step.branches

    def test_when_branch_not_pathway_ref(self):
        """Arrow in when branch is a visual connector, not a pathway ref."""
        source = CONTRACTS_DIR / "pathways" / "remediate_mac_flapping.sg"
        contract = parse_sg(source.read_text())
        when_step = contract.steps[1]
        branch = when_step.branches["false"]
        assert branch.is_pathway_ref is False
        assert branch.locus == "mac_preserve"

    def test_requires_parsed(self):
        source = CONTRACTS_DIR / "pathways" / "deploy_server_network.sg"
        contract = parse_sg(source.read_text())
        assert len(contract.requires) == 2
        assert Dependency(step=2, needs=1) in contract.requires
        assert Dependency(step=3, needs=2) in contract.requires

    def test_on_failure_report_partial(self):
        source = CONTRACTS_DIR / "pathways" / "remediate_mac_flapping.sg"
        contract = parse_sg(source.read_text())
        assert contract.on_failure == "report partial"


# --- pathway_from_contract conversion ---

class TestPathwayConversion:
    def test_composed_step_conversion(self):
        source = CONTRACTS_DIR / "pathways" / "deploy_server_network.sg"
        contract = parse_sg(source.read_text())
        pathway = pathway_from_contract(contract)
        assert isinstance(pathway.steps[0], ComposedStep)
        assert pathway.steps[0].pathway_name == "provision_management_bridge"

    def test_loop_step_conversion(self):
        source = CONTRACTS_DIR / "pathways" / "deploy_server_network.sg"
        contract = parse_sg(source.read_text())
        pathway = pathway_from_contract(contract)
        loop = pathway.steps[2]
        assert isinstance(loop, LoopStep)
        assert loop.variable == "vlan"
        assert loop.iterable_field == "vlans"
        assert loop.body_locus == "vlan_create"

    def test_conditional_step_conversion(self):
        source = CONTRACTS_DIR / "pathways" / "remediate_mac_flapping.sg"
        contract = parse_sg(source.read_text())
        pathway = pathway_from_contract(contract)
        cond = pathway.steps[1]
        assert isinstance(cond, ConditionalExecStep)
        assert cond.condition_step_index == 0  # 1-based → 0-based
        assert cond.condition_field == "healthy"
        assert "false" in cond.branches

    def test_mixed_step_types(self):
        source = CONTRACTS_DIR / "pathways" / "deploy_server_network.sg"
        contract = parse_sg(source.read_text())
        pathway = pathway_from_contract(contract)
        assert len(pathway.steps) == 3
        assert isinstance(pathway.steps[0], ComposedStep)
        assert isinstance(pathway.steps[1], PathwayStep)
        assert isinstance(pathway.steps[2], LoopStep)


# --- Dependency validation ---

class TestDependencyValidation:
    def test_valid_dependencies(self):
        steps = [
            ASTPathwayStep(index=1, locus="a"),
            ASTPathwayStep(index=2, locus="b"),
            ASTPathwayStep(index=3, locus="c"),
        ]
        deps = [Dependency(step=2, needs=1), Dependency(step=3, needs=2)]
        _validate_dependencies(steps, deps)  # no error

    def test_backward_dependency_raises(self):
        steps = [
            ASTPathwayStep(index=1, locus="a"),
            ASTPathwayStep(index=2, locus="b"),
        ]
        deps = [Dependency(step=1, needs=2)]
        with pytest.raises(ValueError, match="appears after"):
            _validate_dependencies(steps, deps)

    def test_unknown_step_raises(self):
        steps = [ASTPathwayStep(index=1, locus="a")]
        deps = [Dependency(step=5, needs=1)]
        with pytest.raises(ValueError, match="unknown step"):
            _validate_dependencies(steps, deps)

    def test_empty_dependencies_ok(self):
        steps = [ASTPathwayStep(index=1, locus="a")]
        _validate_dependencies(steps, [])  # no error


# --- Execution integration ---

class TestPathwayExecution:
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

    def test_composed_pathway_executes_sub_pathway(self, project):
        """deploy_server_network composes provision_management_bridge."""
        orch = self._make_orch(project)
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
        # Should have outputs from: sub-pathway(1), bond_create(1), vlan(2)
        assert len(outputs) >= 3
        # Bridge from composed sub-pathway
        assert orch.kernel.get_bridge("br0") is not None
        # Bond from step 2
        assert orch.kernel.get_bond("bond0") is not None

    def test_for_loop_creates_multiple_vlans(self, project):
        orch = self._make_orch(project)
        orch.run_pathway("deploy_server_network", json.dumps({
            "bridge_name": "br0",
            "interfaces": ["eth0"],
            "uplink": "eth1",
            "stp_enabled": True,
            "forward_delay": 15,
            "bond_name": "bond0",
            "bond_mode": "802.3ad",
            "bond_members": ["eth2", "eth3"],
            "vlans": [100, 200, 300],
        }))
        # All three VLANs should be created on bond0
        assert orch.kernel.get_vlan("bond0", 100) is not None
        assert orch.kernel.get_vlan("bond0", 200) is not None
        assert orch.kernel.get_vlan("bond0", 300) is not None

    def test_conditional_step_healthy_no_remediation(self, project):
        """When diagnostic reports healthy, conditional step doesn't execute."""
        orch = self._make_orch(project)

        # Provision bridge first
        orch.run_pathway("provision_management_bridge", json.dumps({
            "bridge_name": "br0",
            "interfaces": ["eth0"],
            "uplink": "eth1",
            "stp_enabled": True,
            "forward_delay": 15,
        }))

        # Run remediation — check_mac_stability should report healthy
        outputs = orch.run_pathway("remediate_mac_flapping", json.dumps({
            "bridge_name": "br0",
        }))
        # Step 1 output + no step 2 (condition not met)
        assert len(outputs) == 1

    def test_conditional_step_unhealthy_triggers_remediation(self, project):
        """When diagnostic reports unhealthy, conditional step executes."""
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

        # Run remediation — check_mac_stability should detect flapping
        outputs = orch.run_pathway("remediate_mac_flapping", json.dumps({
            "bridge_name": "br0",
        }))
        # Step 1 output + step 2 output (remediation triggered)
        assert len(outputs) == 2
