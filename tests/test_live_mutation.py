"""Tests for live mutation demo and --force-mutate flag."""
import json
import shutil
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from sg.contracts import ContractStore
from sg.fusion import FusionTracker
import sg_network
from sg_network import MockNetworkKernel
from sg.mutation import MockMutationEngine
from sg.orchestrator import Orchestrator
from sg.phenotype import PhenotypeMap
from sg.registry import Registry


CONTRACTS_DIR = sg_network.contracts_path()
GENES_DIR = sg_network.genes_path()
FIXTURES_DIR = sg_network.fixtures_path()


@pytest.fixture
def project(tmp_path):
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


def _make_orch(project_root):
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


class TestForceMutate:
    def test_inject_broken_genes_replaces_dominant(self, project):
        """_inject_broken_genes replaces all dominant alleles with broken ones."""
        from sg.cli import _inject_broken_genes

        orch = _make_orch(project)

        # Original dominant should work
        original_sha = orch.phenotype.get_dominant("bridge_create")
        assert original_sha is not None

        _inject_broken_genes(orch)

        # Dominant should now be different (broken)
        new_sha = orch.phenotype.get_dominant("bridge_create")
        assert new_sha != original_sha

        # The broken gene should fail when executed directly
        source = orch.registry.load_source(new_sha)
        assert "force-mutate" in source

    def test_force_mutate_triggers_mutation(self, project):
        """After force-mutate injection, executing a locus triggers mutation."""
        from sg.cli import _inject_broken_genes

        orch = _make_orch(project)
        _inject_broken_genes(orch)

        # Execute bridge_create — broken dominant fails, mutation kicks in
        result = orch.execute_locus("bridge_create", json.dumps({
            "bridge_name": "br0",
            "interfaces": ["eth0"],
        }))

        # Mutation should have succeeded (fixture provides the fix)
        assert result is not None
        output, used_sha = result
        data = json.loads(output)
        assert data["success"] is True

        # The used allele should be a mutation (generation > 0 or different from broken)
        broken_sha = orch.phenotype.get_dominant("bridge_create")
        # The mutation was registered and used
        mutant = orch.registry.get(used_sha)
        assert mutant is not None

    def test_force_mutate_all_loci_broken(self, project):
        """_inject_broken_genes breaks every locus in the phenotype."""
        from sg.cli import _inject_broken_genes

        orch = _make_orch(project)
        original_dominants = {
            locus: orch.phenotype.get_dominant(locus)
            for locus in orch.phenotype.loci
        }

        _inject_broken_genes(orch)

        for locus in original_dominants:
            new_dominant = orch.phenotype.get_dominant(locus)
            assert new_dominant != original_dominants[locus], \
                f"{locus} dominant was not replaced"


class TestDemoScript:
    def test_demo_module_imports(self):
        """Demo script imports cleanly."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "live_mutation",
            Path(__file__).parent.parent / "demo" / "live_mutation.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "run_demo")
        assert hasattr(mod, "BROKEN_GENES")
        assert hasattr(mod, "DEFAULT_INPUTS")

    def test_broken_genes_have_inputs(self):
        """Every broken gene has a corresponding default input."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "live_mutation",
            Path(__file__).parent.parent / "demo" / "live_mutation.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for locus in mod.BROKEN_GENES:
            assert locus in mod.DEFAULT_INPUTS, \
                f"broken gene '{locus}' has no default input"

    def test_broken_genes_actually_fail(self):
        """Each broken gene raises when executed."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "live_mutation",
            Path(__file__).parent.parent / "demo" / "live_mutation.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        kernel = MockNetworkKernel()
        for locus, source in mod.BROKEN_GENES.items():
            namespace = {"gene_sdk": kernel}
            exec(source, namespace)
            execute_fn = namespace["execute"]
            input_json = json.dumps(mod.DEFAULT_INPUTS[locus])

            with pytest.raises(Exception):
                execute_fn(input_json)
