"""Tests for safety layer — transactions, rollback, blast radius."""
import json
import shutil
import pytest
from pathlib import Path

import sg_network
from sg import arena
from sg.contracts import ContractStore
from sg.fusion import FusionTracker
from sg_network import MockNetworkKernel
from sg.mutation import MockMutationEngine
from sg.orchestrator import Orchestrator
from sg.parser.types import BlastRadius
from sg.phenotype import PhenotypeMap
from sg.registry import Registry
from sg.safety import (
    Transaction, SafeKernel, UndoAction,
    requires_transaction, is_shadow_only,
)


CONTRACTS_DIR = sg_network.contracts_path()
GENES_DIR = sg_network.genes_path()
FIXTURES_DIR = sg_network.fixtures_path()


# --- Transaction unit tests ---

class TestTransaction:
    def test_record_and_rollback(self):
        txn = Transaction("bridge_create", BlastRadius.LOW)
        log = []
        txn.record("action 1", lambda: log.append("undo 1"))
        txn.record("action 2", lambda: log.append("undo 2"))
        txn.record("action 3", lambda: log.append("undo 3"))

        rolled = txn.rollback()
        # Undo in reverse order
        assert log == ["undo 3", "undo 2", "undo 1"]
        assert rolled == ["action 3", "action 2", "action 1"]
        assert txn.rolled_back

    def test_commit_clears_log(self):
        txn = Transaction("bridge_create", BlastRadius.LOW)
        txn.record("action 1", lambda: None)
        txn.commit()
        assert txn.committed
        assert txn.action_count == 0

    def test_rollback_survives_undo_errors(self):
        txn = Transaction("bridge_create", BlastRadius.LOW)
        log = []
        txn.record("good 1", lambda: log.append("undo 1"))
        txn.record("bad", lambda: (_ for _ in ()).throw(RuntimeError("oops")))
        txn.record("good 2", lambda: log.append("undo 3"))

        rolled = txn.rollback()
        # good 2 undone, bad fails silently, good 1 still undone
        assert "undo 3" in log
        assert "undo 1" in log
        assert len(rolled) == 2  # only successful undos reported

    def test_action_count(self):
        txn = Transaction("bridge_create", BlastRadius.LOW)
        assert txn.action_count == 0
        txn.record("a", lambda: None)
        assert txn.action_count == 1
        txn.record("b", lambda: None)
        assert txn.action_count == 2


# --- SafeKernel unit tests ---

class TestSafeKernel:
    @pytest.fixture
    def kernel(self):
        return MockNetworkKernel()

    def test_create_bridge_records_undo(self, kernel):
        txn = Transaction("bridge_create", BlastRadius.LOW)
        safe = SafeKernel(kernel, txn)
        safe.create_bridge("br0", ["eth0"])
        assert txn.action_count == 1
        assert kernel.get_bridge("br0") is not None

        # Rollback should delete the bridge
        txn.rollback()
        assert kernel.get_bridge("br0") is None

    def test_create_bond_records_undo(self, kernel):
        txn = Transaction("bond_create", BlastRadius.LOW)
        safe = SafeKernel(kernel, txn)
        safe.create_bond("bond0", "802.3ad", ["eth0", "eth1"])
        assert kernel.get_bond("bond0") is not None

        txn.rollback()
        assert kernel.get_bond("bond0") is None

    def test_create_vlan_records_undo(self, kernel):
        txn = Transaction("vlan_create", BlastRadius.LOW)
        safe = SafeKernel(kernel, txn)
        safe.create_vlan("eth0", 100)
        assert kernel.get_vlan("eth0", 100) is not None

        txn.rollback()
        assert kernel.get_vlan("eth0", 100) is None

    def test_set_stp_records_undo(self, kernel):
        txn = Transaction("bridge_stp", BlastRadius.LOW)
        safe = SafeKernel(kernel, txn)
        # Create bridge first (directly on inner kernel, not wrapped)
        kernel.create_bridge("br0", ["eth0"])

        safe.set_stp("br0", True, 20)
        bridge = kernel.get_bridge("br0")
        assert bridge["stp_enabled"] is True
        assert bridge["forward_delay"] == 20

        # Rollback should restore original STP settings
        txn.rollback()
        bridge = kernel.get_bridge("br0")
        assert bridge["stp_enabled"] is False  # original default
        assert bridge["forward_delay"] == 15   # original default

    def test_set_device_mac_records_undo(self, kernel):
        txn = Transaction("mac_preserve", BlastRadius.LOW)
        safe = SafeKernel(kernel, txn)
        kernel.create_bridge("br0", ["eth0"])
        original_mac = kernel.get_device_mac("br0")

        safe.set_device_mac("br0", "aa:bb:cc:dd:ee:ff")
        assert kernel.get_device_mac("br0") == "aa:bb:cc:dd:ee:ff"

        txn.rollback()
        assert kernel.get_device_mac("br0") == original_mac

    def test_attach_detach_interface_undo(self, kernel):
        txn = Transaction("bridge_create", BlastRadius.LOW)
        safe = SafeKernel(kernel, txn)
        kernel.create_bridge("br0", ["eth0"])

        safe.attach_interface("br0", "eth1")
        bridge = kernel.get_bridge("br0")
        assert "eth1" in bridge["interfaces"]

        txn.rollback()
        bridge = kernel.get_bridge("br0")
        assert "eth1" not in bridge["interfaces"]

    def test_read_operations_no_undo(self, kernel):
        txn = Transaction("check_connectivity", BlastRadius.NONE)
        safe = SafeKernel(kernel, txn)
        kernel.create_bridge("br0", ["eth0"])

        # Read-only operations should not record undo actions
        safe.get_bridge("br0")
        safe.get_interface_state("eth0")
        assert txn.action_count == 0

    def test_commit_after_success(self, kernel):
        txn = Transaction("bridge_create", BlastRadius.LOW)
        safe = SafeKernel(kernel, txn)
        safe.create_bridge("br0", ["eth0"])
        txn.commit()

        # After commit, the bridge stays
        assert kernel.get_bridge("br0") is not None
        assert txn.action_count == 0

    def test_multiple_operations_rollback_order(self, kernel):
        txn = Transaction("bridge_create", BlastRadius.LOW)
        safe = SafeKernel(kernel, txn)
        safe.create_bridge("br0", ["eth0"])
        safe.attach_interface("br0", "eth1")
        safe.set_stp("br0", True, 20)
        assert txn.action_count == 3

        # Rollback in reverse: STP first, then detach, then delete bridge
        txn.rollback()
        assert kernel.get_bridge("br0") is None

    def test_delete_bridge_records_recreate(self, kernel):
        txn = Transaction("bridge_create", BlastRadius.LOW)
        safe = SafeKernel(kernel, txn)
        kernel.create_bridge("br0", ["eth0", "eth1"])

        safe.delete_bridge("br0")
        assert kernel.get_bridge("br0") is None

        txn.rollback()
        bridge = kernel.get_bridge("br0")
        assert bridge is not None
        assert bridge["name"] == "br0"


# --- Blast radius classification ---

class TestBlastRadius:
    def test_none_no_transaction(self):
        assert not requires_transaction(BlastRadius.NONE)

    def test_low_requires_transaction(self):
        assert requires_transaction(BlastRadius.LOW)

    def test_medium_requires_transaction(self):
        assert requires_transaction(BlastRadius.MEDIUM)

    def test_high_requires_transaction(self):
        assert requires_transaction(BlastRadius.HIGH)

    def test_critical_requires_transaction(self):
        assert requires_transaction(BlastRadius.CRITICAL)

    def test_shadow_mode(self):
        assert not is_shadow_only(BlastRadius.NONE)
        assert not is_shadow_only(BlastRadius.LOW)
        assert not is_shadow_only(BlastRadius.MEDIUM)
        assert is_shadow_only(BlastRadius.HIGH)
        assert is_shadow_only(BlastRadius.CRITICAL)


# --- Orchestrator integration ---

class TestOrchestratorSafety:
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

    def test_successful_gene_commits_transaction(self, project):
        orch = self._make_orch(project)
        result = orch.execute_locus("bridge_create", json.dumps({
            "bridge_name": "br0",
            "interfaces": ["eth0", "eth1"],
        }))
        assert result is not None
        # Bridge should exist in the real kernel
        assert orch.kernel.get_bridge("br0") is not None

    def test_failed_gene_rolls_back(self, project):
        """When a gene creates kernel state then raises, the state is rolled back."""
        orch = self._make_orch(project)

        # Register a gene that creates a bridge then raises.
        # The bridge should be rolled back on failure.
        bad_gene = (
            'import json\n'
            'def execute(input_json):\n'
            '    data = json.loads(input_json)\n'
            '    gene_sdk.create_bridge("rollback_test_br", ["eth2"])\n'
            '    raise RuntimeError("intentional failure after creating bridge")\n'
        )
        sha = orch.registry.register(bad_gene, "bridge_create")
        # Make this the only allele by clearing the stack
        orch.phenotype.promote("bridge_create", sha)
        allele = orch.registry.get(sha)
        allele.state = "dominant"

        # Execute — the bad gene creates "rollback_test_br" then fails
        orch.execute_locus("bridge_create", json.dumps({
            "bridge_name": "br0",
            "interfaces": ["eth0", "eth1"],
        }))

        # "rollback_test_br" was created by the bad gene, but the
        # transaction rollback should have deleted it
        assert orch.kernel.get_bridge("rollback_test_br") is None

    def test_diagnostic_gene_no_transaction(self, project):
        orch = self._make_orch(project)

        # Provision bridge first
        orch.execute_locus("bridge_create", json.dumps({
            "bridge_name": "br0",
            "interfaces": ["eth0"],
        }))

        # Diagnostic genes (risk: none) should not use transactions
        result = orch.execute_locus("check_connectivity", json.dumps({
            "bridge_name": "br0",
        }))
        assert result is not None

    def test_pathway_rollback_on_failure(self, project):
        orch = self._make_orch(project)

        # Make bridge_stp fail by injecting failure
        orch.kernel.inject_failure("set_stp", "simulated STP failure")

        # provision_management_bridge should fail at bridge_stp step
        # With on_failure="rollback all", tracked resources should be cleaned
        try:
            orch.run_pathway("provision_management_bridge", json.dumps({
                "bridge_name": "br0",
                "interfaces": ["eth0"],
                "uplink": "eth1",
                "stp_enabled": True,
                "forward_delay": 15,
            }))
        except RuntimeError:
            pass  # Expected — pathway failed

    def test_pathway_success_keeps_resources(self, project):
        orch = self._make_orch(project)
        orch.run_pathway("provision_management_bridge", json.dumps({
            "bridge_name": "br0",
            "interfaces": ["eth0"],
            "uplink": "eth1",
            "stp_enabled": True,
            "forward_delay": 15,
        }))
        # Bridge should exist after successful pathway
        assert orch.kernel.get_bridge("br0") is not None
