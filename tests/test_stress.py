"""Phase B: Multi-locus stress tests.

Verifies bounded growth, fitness convergence, and persist/reload
consistency across 100+ invocations.
"""
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
from sg.pathway_fitness import PathwayFitnessTracker, MAX_STEP_TIMINGS
from sg.pathway_registry import PathwayRegistry
from sg.phenotype import PhenotypeMap
from sg.registry import Registry


CONTRACTS_DIR = sg_network.contracts_path()
GENES_DIR = sg_network.genes_path()
FIXTURES_DIR = sg_network.fixtures_path()


@pytest.fixture
def full_project(tmp_path):
    """Full project with real contracts and gene fixtures."""
    shutil.copytree(FIXTURES_DIR, tmp_path / "fixtures")
    shutil.copytree(CONTRACTS_DIR, tmp_path / "contracts")
    (tmp_path / ".sg").mkdir(exist_ok=True)

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


def _make_orchestrator(project_root):
    from sg.parser.types import GeneFamily
    contract_store = ContractStore.open(project_root / "contracts")
    for locus in contract_store.known_loci():
        gc = contract_store.get_gene(locus)
        if gc and gc.family == GeneFamily.CONFIGURATION and gc.verify_within:
            gc.verify_within = "0.01s"

    registry = Registry.open(project_root / ".sg" / "registry")
    phenotype = PhenotypeMap.load(project_root / "phenotype.toml")
    fusion_tracker = FusionTracker.open(project_root / "fusion_tracker.json")
    pft = PathwayFitnessTracker.open(project_root / "pathway_fitness.json")
    pr = PathwayRegistry.open(project_root / ".sg" / "pathway_registry")
    mutation_engine = MockMutationEngine(project_root / "fixtures")
    kernel = MockNetworkKernel()

    return Orchestrator(
        registry=registry, phenotype=phenotype,
        mutation_engine=mutation_engine, fusion_tracker=fusion_tracker,
        kernel=kernel, contract_store=contract_store,
        project_root=project_root,
        pathway_fitness_tracker=pft, pathway_registry=pr,
    )


def _file_size(path: Path) -> int:
    """Return file size in bytes, 0 if not found."""
    return path.stat().st_size if path.exists() else 0


class TestMultiLocusStress:
    """Stress tests: 100+ invocations, bounded growth, convergence."""

    def test_100_invocations_no_unbounded_growth(self, full_project):
        """100 invocations across 2 pathways: all data files stay bounded."""
        orch = _make_orchestrator(full_project)

        for batch in range(4):
            for i in range(25):
                idx = batch * 25 + i
                input_data = {
                    "bridge_name": f"br{idx}",
                    "interfaces": ["eth0"],
                    "stp_enabled": True,
                    "forward_delay": 15,
                }
                orch.execute_locus("bridge_create", json.dumps({
                    "bridge_name": f"br{idx}",
                    "interfaces": ["eth0"],
                }))

            orch.save_state()

            # Check file sizes after each batch
            registry_index = full_project / ".sg" / "registry" / "index.json"
            phenotype_file = full_project / "phenotype.toml"
            fitness_file = full_project / "pathway_fitness.json"
            decomp_file = full_project / ".sg" / "decomposition.json"

            assert _file_size(registry_index) < 500_000, \
                f"Registry index too large at batch {batch}: {_file_size(registry_index)}"
            assert _file_size(phenotype_file) < 50_000, \
                f"Phenotype file too large at batch {batch}: {_file_size(phenotype_file)}"
            if fitness_file.exists():
                assert _file_size(fitness_file) < 100_000, \
                    f"Fitness file too large at batch {batch}"
            if decomp_file.exists():
                assert _file_size(decomp_file) < 100_000, \
                    f"Decomposition file too large at batch {batch}"

        # Also verify step_timings are capped
        if orch.pathway_fitness_tracker:
            for pw_name, rec in orch.pathway_fitness_tracker.records.items():
                for step, timings in rec.step_timings.items():
                    assert len(timings) <= MAX_STEP_TIMINGS, \
                        f"Unbounded step_timings for {pw_name}/{step}: {len(timings)}"

    def test_fitness_convergence(self, full_project):
        """Fitness increases with successes, drops with failures, recovers."""
        orch = _make_orchestrator(full_project)
        dom_sha = orch.phenotype.get_dominant("bridge_create")

        # Phase 1: 50 successes → fitness should be high
        for i in range(50):
            orch.execute_locus("bridge_create", json.dumps({
                "bridge_name": f"br_s{i}", "interfaces": ["eth0"],
            }))

        allele = orch.registry.get(dom_sha)
        fitness_after_success = arena.compute_fitness(allele)
        assert fitness_after_success > 0.8, \
            f"Expected fitness > 0.8 after 50 successes, got {fitness_after_success}"

        # Phase 2: Replace dominant with a gene that raises (not catches).
        # The seed gene catches exceptions, so kernel failures don't count
        # as allele failures. Use a gene that actually raises.
        failing_gene = (
            'import json\n'
            'def execute(input_json):\n'
            '    raise RuntimeError("stress failure")\n'
        )
        fail_sha = orch.registry.register(failing_gene, "bridge_create")
        orch.phenotype.promote("bridge_create", fail_sha)
        fail_allele = orch.registry.get(fail_sha)
        fail_allele.state = "dominant"
        # Keep original as fallback
        orch.phenotype.add_to_fallback("bridge_create", dom_sha)

        for i in range(10):
            orch.execute_locus("bridge_create", json.dumps({
                "bridge_name": f"br_f{i}", "interfaces": ["eth0"],
            }))

        # The original allele should have recorded failures via fallback
        # or the failing allele should show degraded fitness
        fitness_after_failures = arena.compute_fitness(fail_allele)
        assert fitness_after_failures < fitness_after_success, \
            f"Expected fitness to drop: {fitness_after_failures} >= {fitness_after_success}"

        # Phase 3: Restore original dominant, 40 more successes → fitness recovers
        orch.phenotype.promote("bridge_create", dom_sha)
        allele_restored = orch.registry.get(dom_sha)
        allele_restored.state = "dominant"

        for i in range(40):
            orch.execute_locus("bridge_create", json.dumps({
                "bridge_name": f"br_r{i}", "interfaces": ["eth0"],
            }))

        fitness_after_recovery = arena.compute_fitness(allele_restored)
        assert fitness_after_recovery > 0.7, \
            f"Expected fitness > 0.7 after recovery, got {fitness_after_recovery}"

    def test_persist_reload_consistency(self, full_project):
        """Counters accumulate correctly across save/load cycles."""
        # Phase 1: 20 invocations, save
        orch = _make_orchestrator(full_project)
        dom_sha = orch.phenotype.get_dominant("bridge_create")

        for i in range(20):
            orch.execute_locus("bridge_create", json.dumps({
                "bridge_name": f"br_p1_{i}", "interfaces": ["eth0"],
            }))
        orch.save_state()

        allele_p1 = orch.registry.get(dom_sha)
        success_p1 = allele_p1.successful_invocations

        # Phase 2: Reload, 20 more invocations, save
        orch2 = _make_orchestrator(full_project)
        dom_sha2 = orch2.phenotype.get_dominant("bridge_create")
        assert dom_sha2 == dom_sha

        for i in range(20):
            orch2.execute_locus("bridge_create", json.dumps({
                "bridge_name": f"br_p2_{i}", "interfaces": ["eth0"],
            }))
        orch2.save_state()

        allele_p2 = orch2.registry.get(dom_sha)
        success_p2 = allele_p2.successful_invocations

        # All 40 invocations should be accumulated
        assert success_p2 >= success_p1 + 20, \
            f"Expected {success_p1}+20={success_p1+20} successes, got {success_p2}"

        # Phase 3: Reload again and verify
        orch3 = _make_orchestrator(full_project)
        allele_p3 = orch3.registry.get(dom_sha)
        assert allele_p3.successful_invocations == success_p2, \
            "Invocations lost across save/load cycle"
