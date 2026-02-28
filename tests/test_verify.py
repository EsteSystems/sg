"""Tests for verify block auto-scheduling."""
import json
import shutil
import time
import pytest
from pathlib import Path

from sg import arena
from sg.contracts import ContractStore
from sg.fusion import FusionTracker
from sg.kernel.mock import MockNetworkKernel
from sg.mutation import MockMutationEngine
from sg.orchestrator import Orchestrator
from sg.phenotype import PhenotypeMap
from sg.registry import Registry
from sg.verify import parse_duration, resolve_verify_params, VerifyScheduler


CONTRACTS_DIR = Path(__file__).parent.parent / "contracts"
GENES_DIR = Path(__file__).parent.parent / "genes"
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


# --- Unit tests ---

class TestParseDuration:
    def test_seconds(self):
        assert parse_duration("30s") == 30.0

    def test_minutes(self):
        assert parse_duration("5m") == 300.0

    def test_hours(self):
        assert parse_duration("1h") == 3600.0

    def test_fractional(self):
        assert parse_duration("0.5s") == 0.5

    def test_whitespace(self):
        assert parse_duration("  30s  ") == 30.0

    def test_bad_input_raises(self):
        with pytest.raises(ValueError, match="unrecognized"):
            parse_duration("badvalue")


class TestResolveVerifyParams:
    def test_reference_resolution(self):
        result = resolve_verify_params(
            {"interface": "{bridge_name}"},
            '{"bridge_name": "br0"}',
        )
        assert json.loads(result) == {"interface": "br0"}

    def test_literal_passthrough(self):
        result = resolve_verify_params(
            {"flag": "true"},
            '{}',
        )
        assert json.loads(result) == {"flag": "true"}

    def test_missing_reference_drops_param(self):
        result = resolve_verify_params(
            {"interface": "{nonexistent}"},
            '{"bridge_name": "br0"}',
        )
        assert json.loads(result) == {}

    def test_multiple_params(self):
        result = resolve_verify_params(
            {"interface": "{bridge_name}", "mode": "strict"},
            '{"bridge_name": "br0"}',
        )
        parsed = json.loads(result)
        assert parsed == {"interface": "br0", "mode": "strict"}


# --- Integration tests ---

class TestVerifyScheduling:
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

    def test_verify_fires_after_config_gene(self, project):
        """Config gene success schedules and fires verify diagnostic."""
        orch = self._make_orch(project)

        # Run bridge_create — its contract has verify: check_connectivity
        orch.execute_locus("bridge_create", json.dumps({
            "bridge_name": "br0",
            "interfaces": ["eth0"],
        }))

        # Diagnostic should be scheduled
        assert orch.verify_scheduler.pending_count > 0

        # Wait for verify to fire
        orch.verify_scheduler.wait()

        # check_connectivity should have been invoked
        diag_sha = orch.phenotype.get_dominant("check_connectivity")
        diag_allele = orch.registry.get(diag_sha)
        assert diag_allele.total_invocations >= 1

    def test_verify_feeds_convergence_fitness(self, project):
        """Full chain: config success → verify fires → feeds → convergence fitness record."""
        orch = self._make_orch(project)

        # Run bridge_create
        orch.execute_locus("bridge_create", json.dumps({
            "bridge_name": "br0",
            "interfaces": ["eth0"],
        }))
        orch.verify_scheduler.wait()

        # bridge_create's dominant allele should have convergence feedback
        # because check_connectivity feeds bridge_create
        config_sha = orch.phenotype.get_dominant("bridge_create")
        config_allele = orch.registry.get(config_sha)
        convergence_records = [
            r for r in config_allele.fitness_records
            if r["timescale"] == "convergence"
        ]
        assert len(convergence_records) > 0

    def test_verify_skipped_for_diagnostic_genes(self, project):
        """Diagnostic gene success does NOT schedule verify diagnostics."""
        orch = self._make_orch(project)

        # Create a bridge so check_connectivity has something to check
        orch.kernel.create_bridge("br0", ["eth0"])

        initial_count = orch.verify_scheduler.pending_count

        # Run a diagnostic gene directly
        orch.execute_locus("check_connectivity", json.dumps({
            "bridge_name": "br0",
        }))

        # No new verify should be scheduled (diagnostics don't trigger verify)
        assert orch.verify_scheduler.pending_count == initial_count

    def test_verify_with_delay(self, project):
        """Verify diagnostic respects the within delay."""
        orch = self._make_orch(project)

        # Patch bridge_stp's verify_within to a small delay
        gene_contract = orch.contract_store.get_gene("bridge_stp")
        original_within = gene_contract.verify_within
        gene_contract.verify_within = "0.2s"

        try:
            # Create bridge first so bridge_stp can configure it
            orch.execute_locus("bridge_create", json.dumps({
                "bridge_name": "br0",
                "interfaces": ["eth0"],
            }))
            # Wait for bridge_create's own verify to complete
            orch.verify_scheduler.wait()

            # Record check_link_state invocations before bridge_stp
            diag_sha = orch.phenotype.get_dominant("check_link_state")
            diag_allele = orch.registry.get(diag_sha)
            invocations_before = diag_allele.total_invocations

            # Run bridge_stp — its verify has a 0.2s delay
            orch.execute_locus("bridge_stp", json.dumps({
                "bridge_name": "br0",
                "stp_enabled": True,
                "forward_delay": 15,
            }))

            # Diagnostic should not have fired yet (delay not elapsed)
            diag_allele = orch.registry.get(diag_sha)
            assert diag_allele.total_invocations == invocations_before

            # Wait for timer to fire
            orch.verify_scheduler.wait()

            # Now it should have been called
            diag_allele = orch.registry.get(diag_sha)
            assert diag_allele.total_invocations > invocations_before
        finally:
            gene_contract.verify_within = original_within

    def test_bridge_stp_verify_param_name(self, project):
        """bridge_stp.sg verify uses 'interface' (matching check_link_state takes)."""
        orch = self._make_orch(project)
        gene_contract = orch.contract_store.get_gene("bridge_stp")
        assert len(gene_contract.verify) > 0
        step = gene_contract.verify[0]
        assert step.locus == "check_link_state"
        assert "interface" in step.params
