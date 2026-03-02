"""Integration tests for pathway allele infrastructure (Phase 4)."""
import json
import shutil
import pytest
from pathlib import Path

from sg.contracts import ContractStore
from sg.fusion import FusionTracker
from sg.mutation import MockMutationEngine
from sg.orchestrator import Orchestrator
from sg.pathway_registry import PathwayRegistry, StepSpec
from sg.phenotype import PhenotypeMap, PathwayAlleleConfig
from sg.registry import Registry

import sg_network
from sg_network import MockNetworkKernel

CONTRACTS_DIR = sg_network.contracts_path()
GENES_DIR = sg_network.genes_path()
FIXTURES_DIR = sg_network.fixtures_path()


class TestPhenotypePathwayAlleles:
    def test_promote_pathway(self):
        pm = PhenotypeMap()
        pm.promote_pathway("pw", "sha1")
        assert pm.get_pathway_dominant("pw") == "sha1"

    def test_promote_shifts_old_dominant(self):
        pm = PhenotypeMap()
        pm.promote_pathway("pw", "sha1")
        pm.promote_pathway("pw", "sha2")
        assert pm.get_pathway_dominant("pw") == "sha2"
        stack = pm.get_pathway_stack("pw")
        assert stack == ["sha2", "sha1"]

    def test_add_pathway_fallback(self):
        pm = PhenotypeMap()
        pm.promote_pathway("pw", "sha1")
        pm.add_pathway_fallback("pw", "sha2")
        stack = pm.get_pathway_stack("pw")
        assert stack == ["sha1", "sha2"]

    def test_get_pathway_stack_empty(self):
        pm = PhenotypeMap()
        assert pm.get_pathway_stack("nonexistent") == []

    def test_get_pathway_dominant_empty(self):
        pm = PhenotypeMap()
        assert pm.get_pathway_dominant("nonexistent") is None

    def test_save_load_roundtrip(self, tmp_path):
        pm = PhenotypeMap()
        pm.promote_pathway("pw_a", "sha1")
        pm.add_pathway_fallback("pw_a", "sha2")
        pm.promote_pathway("pw_b", "sha3")

        path = tmp_path / "phenotype.toml"
        pm.save(path)

        pm2 = PhenotypeMap.load(path)
        assert pm2.get_pathway_dominant("pw_a") == "sha1"
        assert pm2.get_pathway_stack("pw_a") == ["sha1", "sha2"]
        assert pm2.get_pathway_dominant("pw_b") == "sha3"

    def test_backward_compat(self, tmp_path):
        """Old TOML without pathway_allele section loads fine."""
        path = tmp_path / "phenotype.toml"
        # Write a minimal TOML without pathway_allele section
        path.write_text('[locus]\n[pathway_fusion]\n')
        pm = PhenotypeMap.load(path)
        assert pm.pathway_alleles == {}
        assert pm.get_pathway_stack("any") == []


class TestOrchestratorIntegration:
    @pytest.fixture
    def project(self, tmp_path):
        (tmp_path / ".sg").mkdir()
        shutil.copytree(CONTRACTS_DIR, tmp_path / "contracts")
        shutil.copytree(FIXTURES_DIR, tmp_path / "fixtures")

        cs = ContractStore.open(tmp_path / "contracts")
        registry = Registry.open(tmp_path / ".sg" / "registry")
        phenotype = PhenotypeMap()

        for locus in cs.known_loci():
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

    def _make_orchestrator(self, project, with_registry=True):
        cs = ContractStore.open(project / "contracts")
        registry = Registry.open(project / ".sg" / "registry")
        phenotype = PhenotypeMap.load(project / "phenotype.toml")
        ft = FusionTracker.open(project / "fusion_tracker.json")
        kernel = MockNetworkKernel()
        mutation_engine = MockMutationEngine(project / "fixtures")
        pathway_registry = (
            PathwayRegistry.open(project / ".sg" / "pathway_registry")
            if with_registry else None
        )
        return Orchestrator(
            registry=registry, phenotype=phenotype,
            mutation_engine=mutation_engine, fusion_tracker=ft,
            kernel=kernel, contract_store=cs, project_root=project,
            pathway_registry=pathway_registry,
        )

    def test_orchestrator_has_pathway_registry(self, project):
        orch = self._make_orchestrator(project)
        assert isinstance(orch.pathway_registry, PathwayRegistry)

    def test_backward_compat_no_registry(self, project):
        """Orchestrator works without pathway_registry."""
        orch = self._make_orchestrator(project, with_registry=False)
        assert orch.pathway_registry is None
        # Should still be able to run pathways
        result = orch.run_pathway("configure_bridge_with_stp", json.dumps({
            "bridge_name": "br0", "interfaces": ["eth0"],
        }))
        assert len(result) > 0

    def test_allele_registered_on_run(self, project):
        orch = self._make_orchestrator(project)
        orch.run_pathway("configure_bridge_with_stp", json.dumps({
            "bridge_name": "br0", "interfaces": ["eth0"],
        }))
        alleles = orch.pathway_registry.get_for_pathway("configure_bridge_with_stp")
        assert len(alleles) == 1
        assert alleles[0].state == "dominant"

    def test_allele_success_recorded(self, project):
        orch = self._make_orchestrator(project)
        orch.run_pathway("configure_bridge_with_stp", json.dumps({
            "bridge_name": "br0", "interfaces": ["eth0"],
        }))
        alleles = orch.pathway_registry.get_for_pathway("configure_bridge_with_stp")
        assert alleles[0].total_executions == 1
        assert alleles[0].successful_executions == 1
        assert alleles[0].consecutive_failures == 0

    def test_save_state_persists_registry(self, project):
        orch = self._make_orchestrator(project)
        orch.run_pathway("configure_bridge_with_stp", json.dumps({
            "bridge_name": "br0", "interfaces": ["eth0"],
        }))
        orch.save_state()

        registry_path = project / ".sg" / "pathway_registry" / "pathway_registry.json"
        assert registry_path.exists()

        reg2 = PathwayRegistry.open(project / ".sg" / "pathway_registry")
        alleles = reg2.get_for_pathway("configure_bridge_with_stp")
        assert len(alleles) == 1
        assert alleles[0].total_executions == 1
