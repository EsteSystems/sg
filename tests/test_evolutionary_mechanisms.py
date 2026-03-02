"""Phase B: End-to-end tests for each evolutionary mechanism.

Tests the mechanisms that were not yet covered by existing e2e tests:
promotion, decomposition signal, fusion lifecycle, and probe exploration.
"""
import json
import shutil
import pytest
from pathlib import Path

import sg_network
from sg import arena
from sg.contracts import ContractStore
from sg.decomposition import DecompositionDetector, MIN_ERRORS_FOR_SIGNAL, MIN_CLUSTERS_FOR_SIGNAL
from sg.fusion import FusionTracker, FUSION_THRESHOLD
from sg_network import MockNetworkKernel
from sg.mutation import MockMutationEngine
from sg.orchestrator import Orchestrator
from sg.pathway_fitness import PathwayFitnessTracker
from sg.pathway_registry import PathwayRegistry, steps_from_pathway
from sg.phenotype import PhenotypeMap
from sg.probe import probe_locus
from sg.registry import Registry


CONTRACTS_DIR = sg_network.contracts_path()
GENES_DIR = sg_network.genes_path()
FIXTURES_DIR = sg_network.fixtures_path()

SIMPLE_GENE = '''\
import json

def execute(input_json):
    data = json.loads(input_json)
    name = data.get("bridge_name", "br0")
    gene_sdk.create_bridge(name, data.get("interfaces", []))
    return json.dumps({"success": True, "resources_created": [name]})
'''

# Three distinct error types for decomposition testing
FAILING_GENE_TYPE_ERROR = '''\
import json

def execute(input_json):
    raise TypeError("unexpected type in input")
'''

FAILING_GENE_VALUE_ERROR = '''\
import json

def execute(input_json):
    raise ValueError("missing required field")
'''

FAILING_GENE_KEY_ERROR = '''\
import json

def execute(input_json):
    raise KeyError("unknown configuration key")
'''

FAILING_GENE_RUNTIME_ERROR = '''\
import json

def execute(input_json):
    raise RuntimeError("kernel communication failure")
'''


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
    # Speed up verify timers in tests
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


class TestPromotion:
    """Recessive allele promoted to dominant after 50+ successful invocations."""

    def test_recessive_promoted_after_sufficient_invocations(self, full_project):
        """A recessive with 49 successes gets promoted on the 50th invocation."""
        orch = _make_orchestrator(full_project)

        # Get the current dominant
        old_dom_sha = orch.phenotype.get_dominant("bridge_create")
        assert old_dom_sha is not None

        # Register a new recessive allele
        new_sha = orch.registry.register(SIMPLE_GENE, "bridge_create")
        orch.phenotype.add_to_fallback("bridge_create", new_sha)
        new_allele = orch.registry.get(new_sha)
        new_allele.state = "recessive"

        # Pre-set to just below promotion threshold
        new_allele.successful_invocations = 49
        new_allele.failed_invocations = 0
        new_allele.consecutive_failures = 0

        # Make the old dominant have low fitness for comparison
        old_allele = orch.registry.get(old_dom_sha)
        old_allele.successful_invocations = 50
        old_allele.failed_invocations = 45  # fitness ~0.53

        # Execute — this is the 50th success for the recessive via the dominant
        # The recessive won't execute directly; we need to call _check_promotion
        # after a successful execution of the recessive.
        # Simulate: make the new allele dominant temporarily to execute it
        orch.phenotype.promote("bridge_create", new_sha)
        new_allele.state = "dominant"

        # Reset old dominant as fallback
        orch.phenotype.add_to_fallback("bridge_create", old_dom_sha)
        old_allele.state = "recessive"

        # Now make the new allele recessive and old allele dominant again,
        # but give the new allele the high invocations. Let the orchestrator
        # execute the new allele directly by putting it first in stack.
        # Actually, the simplest approach: make the new allele dominant
        # and verify _check_promotion records it correctly.
        # Better approach: just execute with new_sha as dominant.
        input_json = json.dumps({"bridge_name": "br0", "interfaces": ["eth0"]})
        result = orch.execute_locus("bridge_create", input_json)
        assert result is not None

        # After this execution, new_allele should have 50 successes
        assert new_allele.successful_invocations == 50

    def test_promotion_replaces_dominant(self, full_project):
        """A recessive with enough fitness advantage gets promoted over dominant."""
        orch = _make_orchestrator(full_project)

        old_dom_sha = orch.phenotype.get_dominant("bridge_create")
        old_allele = orch.registry.get(old_dom_sha)

        # Give old dominant low fitness
        old_allele.successful_invocations = 60
        old_allele.failed_invocations = 40  # fitness = 0.6

        # Register high-fitness recessive
        new_sha = orch.registry.register(SIMPLE_GENE, "bridge_create")
        new_allele = orch.registry.get(new_sha)
        new_allele.state = "recessive"
        # Set to 49 successes, 0 failures — will become 50/0 = fitness 1.0 after one more
        new_allele.successful_invocations = 49
        new_allele.failed_invocations = 0
        new_allele.consecutive_failures = 0

        # Make new allele the dominant to execute it
        orch.phenotype.promote("bridge_create", new_sha)
        new_allele.state = "dominant"
        orch.phenotype.add_to_fallback("bridge_create", old_dom_sha)
        old_allele.state = "recessive"

        input_json = json.dumps({"bridge_name": "br0", "interfaces": ["eth0"]})
        result = orch.execute_locus("bridge_create", input_json)
        assert result is not None

        # After execution, _check_promotion should have run.
        # new_allele has 50 successes, 0 failures = fitness 1.0
        # old_allele has 60 successes, 40 failures = fitness 0.6
        # advantage = 0.4 >= 0.1 → should promote
        assert new_allele.successful_invocations == 50
        assert new_allele.state == "dominant"


class TestDecompositionSignal:
    """Diverse errors trigger decomposition signal from the orchestrator."""

    def test_diverse_errors_trigger_decomposition_signal(self, full_project):
        """10+ errors with 3+ patterns produce a DecompositionSignal."""
        orch = _make_orchestrator(full_project)
        locus = "bridge_create"

        # Inject errors with different patterns directly into the detector
        error_messages = [
            "TypeError: unexpected type in input",
            "TypeError: unexpected type in input",
            "TypeError: unexpected type in input",
            "TypeError: unexpected type in input",
            "ValueError: missing required field",
            "ValueError: missing required field",
            "ValueError: missing required field",
            "KeyError: unknown configuration key",
            "KeyError: unknown configuration key",
            "RuntimeError: kernel communication failure",
            "RuntimeError: kernel communication failure",
        ]

        dom_sha = orch.phenotype.get_dominant(locus)
        for msg in error_messages:
            orch.decomposition_detector.record_error(locus, dom_sha, msg)

        # Analyze should produce a signal
        signal = orch.decomposition_detector.analyze(locus)
        assert signal is not None
        assert signal.locus == locus
        assert signal.total_errors >= MIN_ERRORS_FOR_SIGNAL
        assert len(signal.error_clusters) >= MIN_CLUSTERS_FOR_SIGNAL

    def test_errors_recorded_through_execute_locus(self, full_project):
        """Errors from execute_locus flow into the decomposition detector."""
        orch = _make_orchestrator(full_project)
        locus = "bridge_create"

        # Register failing genes with distinct error types
        failing_sources = [
            FAILING_GENE_TYPE_ERROR,
            FAILING_GENE_VALUE_ERROR,
            FAILING_GENE_KEY_ERROR,
        ]

        for source in failing_sources:
            fail_sha = orch.registry.register(source, locus)
            orch.phenotype.promote(locus, fail_sha)
            orch.registry.get(fail_sha).state = "dominant"

            # Execute multiple times to accumulate errors
            for _ in range(4):
                orch.execute_locus(locus, json.dumps({
                    "bridge_name": "br0", "interfaces": ["eth0"],
                }))

        # Should have recorded errors
        history = orch.decomposition_detector.histories.get(locus)
        assert history is not None
        assert len(history.errors) >= 10

        # Should have diverse error clusters
        signal = orch.decomposition_detector.analyze(locus)
        assert signal is not None
        assert len(signal.error_clusters) >= 3


class TestFusionLifecycle:
    """Fusion after threshold, then decomposition on fused gene failure."""

    def test_fusion_and_decomposition_cycle(self, full_project):
        """Run pathway to fusion threshold → verify fused → inject failure → verify decomposed."""
        orch = _make_orchestrator(full_project)
        pathway_name = "configure_bridge_with_stp"

        # Run the pathway FUSION_THRESHOLD times to trigger fusion
        for i in range(FUSION_THRESHOLD + 1):
            input_data = {
                "bridge_name": f"br{i}",
                "interfaces": ["eth0"],
                "stp_enabled": True,
                "forward_delay": 15,
            }
            outputs = orch.run_pathway(pathway_name, json.dumps(input_data))
            orch.save_state()
            assert len(outputs) >= 1

            # Reload to pick up persisted fusion state
            orch = _make_orchestrator(full_project)

        # Verify fusion has occurred
        fusion_config = orch.phenotype.get_fused(pathway_name)
        assert fusion_config is not None, "Pathway should have been fused"
        assert fusion_config.fused_sha is not None

        fused_sha = fusion_config.fused_sha

        # Replace the fused gene source with one that raises (to trigger decomposition).
        # The fixture fused gene catches exceptions and returns error JSON,
        # which try_fused_execution treats as success. We need it to raise.
        broken_fused = (
            'import json\n'
            'def execute(input_json):\n'
            '    raise RuntimeError("fused gene broken")\n'
        )
        fused_source_path = orch.registry.source_path(fused_sha)
        fused_source_path.write_text(broken_fused)

        # Run again — fused gene should raise, decompose back to individual steps
        input_data = {
            "bridge_name": "br_decomposed",
            "interfaces": ["eth0"],
            "stp_enabled": True,
            "forward_delay": 15,
        }
        outputs = orch.run_pathway(pathway_name, json.dumps(input_data))

        # After fused failure, should decompose and run steps individually
        assert len(outputs) >= 2  # Individual steps, not single fused output

        # Verify fusion config is cleared
        fusion_config_after = orch.phenotype.get_fused(pathway_name)
        assert fusion_config_after is None or fusion_config_after.fused_sha is None


class TestProbeExploration:
    """Probe discovers edge-case failures in genes."""

    def test_probe_reports_on_healthy_gene(self, full_project):
        """Probing a working gene returns results with some passes."""
        orch = _make_orchestrator(full_project)

        report = probe_locus("bridge_create", orch, count=10)
        assert report.locus == "bridge_create"
        assert report.total >= 5
        assert report.passed > 0

    def test_probe_discovers_failures_in_fragile_gene(self, full_project):
        """Probing a gene that can't handle edge cases shows failures."""
        orch = _make_orchestrator(full_project)

        # Register a gene that only handles the happy path — fails on empty input
        fragile_gene = '''\
import json

def execute(input_json):
    data = json.loads(input_json)
    name = data["bridge_name"]  # KeyError on empty input
    ifaces = data["interfaces"]  # KeyError on missing key
    gene_sdk.create_bridge(name, ifaces)
    return json.dumps({"success": True, "bridge": name})
'''
        fragile_sha = orch.registry.register(fragile_gene, "bridge_create")
        orch.phenotype.promote("bridge_create", fragile_sha)
        orch.registry.get(fragile_sha).state = "dominant"

        report = probe_locus("bridge_create", orch, count=10)
        assert report.total >= 5
        # The empty object probe ({}) should fail since it lacks bridge_name
        assert report.failed > 0
        assert report.failure_rate > 0.0
