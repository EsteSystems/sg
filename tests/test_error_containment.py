"""Phase D: Error containment and bounded growth tests.

Verifies that subsystem failures are contained (don't crash the orchestrator)
and that data structures with size caps actually enforce them.
"""
import json
import shutil
from unittest.mock import patch, MagicMock
import pytest

import sg_network
from sg import arena
from sg.contracts import ContractStore
from sg.federation import merge_peer_observation, MAX_PEER_OBSERVATIONS
from sg_network import MockNetworkKernel
from sg.mutation import MockMutationEngine
from sg.orchestrator import Orchestrator
from sg.pathway_fitness import (
    PathwayFitnessTracker, MAX_INPUT_CLUSTERS,
)
from sg.pathway_registry import PathwayRegistry
from sg.phenotype import PhenotypeMap
from sg.fusion import FusionTracker
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


def _make_orchestrator(project_root, **overrides):
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

    kwargs = dict(
        registry=registry, phenotype=phenotype,
        mutation_engine=mutation_engine, fusion_tracker=fusion_tracker,
        kernel=kernel, contract_store=contract_store,
        project_root=project_root,
        pathway_fitness_tracker=pft, pathway_registry=pr,
    )
    kwargs.update(overrides)
    return Orchestrator(**kwargs)


# --- Error Containment Tests ---

class TestErrorContainment:
    """Subsystem failures must not crash the orchestrator."""

    def test_mutation_engine_failure_contained(self, full_project):
        """Patch mutation_engine.mutate to raise; execute_locus returns None."""
        orch = _make_orchestrator(full_project)
        locus = "bridge_create"

        # Register a gene that raises so all alleles fail and mutation is attempted
        failing_gene = (
            'import json\n'
            'def execute(input_json):\n'
            '    raise RuntimeError("always fails")\n'
        )
        fail_sha = orch.registry.register(failing_gene, locus)
        orch.phenotype.promote(locus, fail_sha)
        orch.registry.get(fail_sha).state = "dominant"
        # Clear fallback so the old seed gene doesn't catch the failure
        orch.phenotype.loci[locus].fallback.clear()

        with patch.object(orch.mutation_engine, 'mutate', side_effect=RuntimeError("LLM down")):
            result = orch.execute_locus(locus, json.dumps({
                "bridge_name": "br0", "interfaces": ["eth0"],
            }))

        # Should return None (all exhausted, mutation failed) but NOT crash
        assert result is None

    def test_pathway_mutation_operator_failure_contained(self, full_project):
        """Patch select_operator to raise; _try_pathway_mutation returns None."""
        orch = _make_orchestrator(full_project)

        with patch("sg.orchestrator.Orchestrator._is_structural_problem", return_value=True), \
             patch("sg.orchestrator.Orchestrator._build_pathway_mutation_context",
                   return_value=MagicMock()):
            with patch("sg.pathway_mutation.select_operator",
                       side_effect=RuntimeError("operator crash")):
                result = orch._try_pathway_mutation("configure_bridge_with_stp", MagicMock())

        assert result is None

    def test_interaction_check_failure_contained(self, full_project):
        """Patch check_interactions to raise; _check_promotion proceeds (fail-open)."""
        orch = _make_orchestrator(full_project)
        locus = "bridge_create"
        dom_sha = orch.phenotype.get_dominant(locus)
        allele = orch.registry.get(dom_sha)

        # Give allele enough stats to trigger promotion logic
        allele.successful_invocations = 100
        allele.failed_invocations = 0

        with patch("sg.interactions.check_interactions",
                   side_effect=RuntimeError("interaction crash")):
            # Should not raise
            orch._check_promotion(locus, dom_sha)

    def test_decomposition_failure_contained(self, full_project):
        """Patch mutation_engine.decompose to raise; execute_locus falls through."""
        orch = _make_orchestrator(full_project)
        locus = "bridge_create"

        # Register failing gene
        failing_gene = (
            'import json\n'
            'def execute(input_json):\n'
            '    raise RuntimeError("always fails")\n'
        )
        fail_sha = orch.registry.register(failing_gene, locus)
        orch.phenotype.promote(locus, fail_sha)
        orch.registry.get(fail_sha).state = "dominant"
        # Clear fallback so the old seed gene doesn't catch the failure
        orch.phenotype.loci[locus].fallback.clear()

        # Inject enough diverse errors to trigger decomposition signal
        for msg in [
            "TypeError: x", "TypeError: x", "TypeError: x", "TypeError: x",
            "ValueError: y", "ValueError: y", "ValueError: y",
            "KeyError: z", "KeyError: z",
            "RuntimeError: w", "RuntimeError: w",
        ]:
            orch.decomposition_detector.record_error(locus, fail_sha, msg)

        with patch.object(orch.mutation_engine, 'decompose',
                          side_effect=RuntimeError("LLM down")), \
             patch.object(orch.mutation_engine, 'mutate',
                          side_effect=RuntimeError("LLM down")):
            result = orch.execute_locus(locus, json.dumps({
                "bridge_name": "br0", "interfaces": ["eth0"],
            }))

        # Should return None but NOT crash
        assert result is None

    def test_pathway_fitness_tracker_none_safe(self, full_project):
        """Orchestrator with pathway_fitness_tracker=None runs pathway OK."""
        orch = _make_orchestrator(full_project, pathway_fitness_tracker=None)

        input_data = {
            "bridge_name": "br0",
            "interfaces": ["eth0"],
            "stp_enabled": True,
            "forward_delay": 15,
        }
        outputs = orch.run_pathway(
            "configure_bridge_with_stp", json.dumps(input_data),
        )
        assert len(outputs) >= 1


# --- Bounded Growth Tests ---

class TestBoundedGrowth:
    """Data structures with size caps enforce them."""

    def test_peer_observations_capped(self, full_project):
        """Adding 100 peer observations caps at MAX_PEER_OBSERVATIONS."""
        registry = Registry.open(full_project / ".sg" / "registry")
        # Get any registered allele
        phenotype = PhenotypeMap.load(full_project / "phenotype.toml")
        dom_sha = phenotype.get_dominant("bridge_create")
        allele = registry.get(dom_sha)

        for i in range(100):
            merge_peer_observation(allele, f"peer_{i}", {
                "successful_invocations": 10,
                "total_invocations": 12,
            })

        assert len(allele.peer_observations) == MAX_PEER_OBSERVATIONS
        # Most recent observation should be from peer_99
        assert allele.peer_observations[-1]["peer"] == "peer_99"

    def test_input_failure_clusters_capped(self):
        """Recording failures at 30+ distinct steps caps at MAX_INPUT_CLUSTERS."""
        pft = PathwayFitnessTracker()
        pathway_name = "test_pathway"

        for i in range(30):
            pft.record_execution(
                pathway_name,
                steps_executed=["step_a"],
                step_timings={"step_a": 10.0},
                success=False,
                failure_step=f"step_{i}",
                input_json=f'{{"idx": {i}}}',
            )

        rec = pft.get_record(pathway_name)
        assert len(rec.input_failure_clusters) <= MAX_INPUT_CLUSTERS


# --- File Locking Tests ---

class TestFileLocking:
    """State files use file_lock for concurrent write protection."""

    def test_state_files_create_lock_files(self, full_project):
        """Saving state creates .lock sidecar files for protected files."""
        orch = _make_orchestrator(full_project)

        # Execute something to have state
        orch.execute_locus("bridge_create", json.dumps({
            "bridge_name": "br0", "interfaces": ["eth0"],
        }))
        orch.save_state()

        # Check that lock files were created for locked state files
        locked_files = [
            full_project / "phenotype.toml",
            full_project / "fusion_tracker.json",
            full_project / "pathway_fitness.json",
            full_project / ".sg" / "pathway_mutation_throttle.json",
        ]
        for f in locked_files:
            lock_path = f.with_suffix(f.suffix + ".lock")
            assert lock_path.exists(), f"Missing lock file for {f.name}"
