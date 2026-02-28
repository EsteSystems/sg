"""Tests for shadow mode execution."""
import json
import shutil
import pytest
from pathlib import Path

from sg import arena
from sg.contracts import ContractStore
from sg.fusion import FusionTracker
from sg.kernel.mock import MockNetworkKernel
from sg.mutation import MockMutationEngine
from sg.orchestrator import Orchestrator
from sg.parser.types import BlastRadius
from sg.phenotype import PhenotypeMap
from sg.registry import Registry
from sg.safety import is_shadow_only, SHADOW_PROMOTION_THRESHOLD


CONTRACTS_DIR = Path(__file__).parent.parent / "contracts"
GENES_DIR = Path(__file__).parent.parent / "genes"
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


class TestIsShawdowOnly:
    def test_none_not_shadow(self):
        assert is_shadow_only(BlastRadius.NONE) is False

    def test_low_not_shadow(self):
        assert is_shadow_only(BlastRadius.LOW) is False

    def test_medium_not_shadow(self):
        assert is_shadow_only(BlastRadius.MEDIUM) is False

    def test_high_is_shadow(self):
        assert is_shadow_only(BlastRadius.HIGH) is True

    def test_critical_is_shadow(self):
        assert is_shadow_only(BlastRadius.CRITICAL) is True


class TestShadowExecution:
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

    def test_shadow_increments_counter(self, project):
        """HIGH risk gene runs in shadow and increments shadow_successes."""
        orch = self._make_orch(project)

        # mac_preserve is risk=medium, change contract to HIGH for this test
        gene_contract = orch.contract_store.get_gene("mac_preserve")
        original_risk = gene_contract.risk
        gene_contract.risk = BlastRadius.HIGH

        try:
            # Create a device so mac_preserve has something to work with
            orch.kernel.create_bridge("br0", [])

            sha = orch.phenotype.get_dominant("mac_preserve")
            allele = orch.registry.get(sha)
            assert allele.shadow_successes == 0

            result = orch.execute_locus("mac_preserve", json.dumps({
                "device": "br0",
                "source_mac": "02:aa:bb:cc:dd:ee",
            }))
            assert result is not None

            # Shadow success — counter incremented
            assert allele.shadow_successes == 1

            # Real kernel should NOT have been modified
            real_mac = orch.kernel.get_device_mac("br0")
            assert real_mac != "02:aa:bb:cc:dd:ee"
        finally:
            gene_contract.risk = original_risk

    def test_shadow_threshold_then_live(self, project):
        """After SHADOW_PROMOTION_THRESHOLD shadow successes, gene runs live."""
        orch = self._make_orch(project)

        gene_contract = orch.contract_store.get_gene("mac_preserve")
        original_risk = gene_contract.risk
        gene_contract.risk = BlastRadius.HIGH

        try:
            orch.kernel.create_bridge("br0", [])

            sha = orch.phenotype.get_dominant("mac_preserve")
            allele = orch.registry.get(sha)

            # Accumulate shadow successes up to threshold
            for i in range(SHADOW_PROMOTION_THRESHOLD):
                result = orch.execute_locus("mac_preserve", json.dumps({
                    "device": "br0",
                    "source_mac": "02:aa:bb:cc:dd:ee",
                }))
                assert result is not None

            assert allele.shadow_successes >= SHADOW_PROMOTION_THRESHOLD

            # Next execution should be live — real kernel gets modified
            result = orch.execute_locus("mac_preserve", json.dumps({
                "device": "br0",
                "source_mac": "02:aa:bb:cc:dd:ee",
            }))
            assert result is not None

            # Real kernel should now have the new MAC
            real_mac = orch.kernel.get_device_mac("br0")
            assert real_mac == "02:aa:bb:cc:dd:ee"
        finally:
            gene_contract.risk = original_risk

    def test_shadow_failure_resets_counter(self, project):
        """Shadow failure resets shadow_successes to 0."""
        orch = self._make_orch(project)

        # Register a gene that always raises
        failing_source = '''
import json
def execute(input_json):
    raise RuntimeError("always fails")
'''
        sha = orch.registry.register(failing_source, "mac_preserve", generation=1)
        allele = orch.registry.get(sha)
        allele.state = "recessive"
        allele.shadow_successes = 2  # almost there
        orch.phenotype.add_to_fallback("mac_preserve", sha)

        gene_contract = orch.contract_store.get_gene("mac_preserve")
        original_risk = gene_contract.risk
        gene_contract.risk = BlastRadius.HIGH

        try:
            orch.kernel.create_bridge("br0", [])

            # Make the failing gene first in the stack by promoting it
            orch.phenotype.promote("mac_preserve", sha)

            orch.execute_locus("mac_preserve", json.dumps({
                "device": "br0",
                "source_mac": "02:aa:bb:cc:dd:ee",
            }))

            # Shadow failure should reset counter
            assert allele.shadow_successes == 0
        finally:
            gene_contract.risk = original_risk

    def test_low_risk_skips_shadow(self, project):
        """LOW risk genes go straight to live execution, no shadow."""
        orch = self._make_orch(project)

        result = orch.execute_locus("bridge_create", json.dumps({
            "bridge_name": "br0",
            "interfaces": ["eth0"],
        }))
        assert result is not None

        sha = orch.phenotype.get_dominant("bridge_create")
        allele = orch.registry.get(sha)
        # shadow_successes stays 0 — shadow was never used
        assert allele.shadow_successes == 0
        # But the real kernel has the bridge
        assert orch.kernel.get_bridge("br0") is not None

    def test_shadow_successes_persist(self, project):
        """shadow_successes survives registry save/load cycle."""
        registry = Registry.open(project / ".sg" / "registry")
        sha = registry.alleles_for_locus("bridge_create")[0].sha256
        allele = registry.get(sha)
        allele.shadow_successes = 2

        registry.save_index()

        registry2 = Registry.open(project / ".sg" / "registry")
        allele2 = registry2.get(sha)
        assert allele2.shadow_successes == 2
