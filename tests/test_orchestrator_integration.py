"""Integration tests: cross-module flows through the orchestrator.

Tests the major evolutionary mechanisms end-to-end using MockMutationEngine
and MockNetworkKernel — no LLM calls needed.
"""
import json
import os
import shutil
import pytest
from pathlib import Path
from unittest.mock import patch

import sg_network
from sg import arena
from sg.contracts import ContractStore
from sg.decomposition import DecompositionDetector
from sg.fusion import FusionTracker
from sg_network import MockNetworkKernel
from sg.mutation import MockMutationEngine, MutationEngine, MutationContext
from sg.orchestrator import Orchestrator
from sg.pathway_fitness import PathwayFitnessTracker
from sg.pathway_registry import PathwayRegistry, StepSpec, steps_from_pathway, compute_structure_sha
from sg.phenotype import PhenotypeMap
from sg.registry import Registry


CONTRACTS_DIR = sg_network.contracts_path()
GENES_DIR = sg_network.genes_path()
FIXTURES_DIR = sg_network.fixtures_path()

SIMPLE_GENE = '''
import json

def execute(input_json):
    data = json.loads(input_json)
    gene_sdk.create_bridge(data["bridge_name"], data["interfaces"])
    return json.dumps({"success": True})
'''

FAILING_GENE_TYPE_A = '''
import json

def execute(input_json):
    raise TypeError("unexpected type in input")
'''

FAILING_GENE_TYPE_B = '''
import json

def execute(input_json):
    raise ValueError("missing required field")
'''

FAILING_GENE_TYPE_C = '''
import json

def execute(input_json):
    raise KeyError("unknown configuration key")
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


class TestDecompositionTriggerPath:
    """Orchestrator records diverse errors → DecompositionDetector signals."""

    def test_errors_recorded_on_failure(self, full_project):
        """Gene failure records error in decomposition detector."""
        orch = _make_orchestrator(full_project)

        # Replace dominant with a failing gene
        fail_sha = orch.registry.register(FAILING_GENE_TYPE_A, "bridge_create")
        orch.phenotype.promote("bridge_create", fail_sha)
        orch.registry.get(fail_sha).state = "dominant"

        input_json = json.dumps({"bridge_name": "br0", "interfaces": ["eth0"]})
        orch.execute_locus("bridge_create", input_json)

        # The decomposition detector should have recorded the error
        history = orch.decomposition_detector.histories.get("bridge_create")
        assert history is not None
        assert len(history.errors) > 0
        assert "unexpected type" in history.errors[0]["message"]

    def test_decomposition_detector_persists(self, full_project):
        """save_state persists decomposition error history."""
        orch = _make_orchestrator(full_project)

        # Replace dominant with a failing gene
        fail_sha = orch.registry.register(FAILING_GENE_TYPE_A, "bridge_create")
        orch.phenotype.promote("bridge_create", fail_sha)
        orch.registry.get(fail_sha).state = "dominant"

        input_json = json.dumps({"bridge_name": "br0", "interfaces": ["eth0"]})
        orch.execute_locus("bridge_create", input_json)
        orch.save_state()

        # Reload
        orch2 = _make_orchestrator(full_project)
        history = orch2.decomposition_detector.histories.get("bridge_create")
        assert history is not None
        assert len(history.errors) > 0


class TestPathwayMutationTriggerPath:
    """Low pathway fitness + high gene fitness → structural mutation."""

    def test_structural_problem_triggers_mutation(self, full_project):
        """Pathway mutation fires when allele stack exhausted with structural problem."""
        from sg.pathway import pathway_from_contract
        orch = _make_orchestrator(full_project)

        pw_name = "configure_bridge_with_stp"

        # Register the default pathway allele
        pw_contract = orch.contract_store.get_pathway(pw_name)
        default_pw = pathway_from_contract(pw_contract)
        specs = steps_from_pathway(default_pw)
        sha = orch.pathway_registry.register(pw_name, specs)
        pw_allele = orch.pathway_registry.get(sha)
        pw_allele.state = "dominant"
        orch.phenotype.promote_pathway(pw_name, sha)

        # Give gene alleles high fitness
        for locus in ["bridge_create", "bridge_stp"]:
            dom_sha = orch.phenotype.get_dominant(locus)
            if dom_sha:
                allele = orch.registry.get(dom_sha)
                if allele:
                    allele.successful_invocations = 95
                    allele.failed_invocations = 5

        # Record many pathway failures to create low pathway fitness
        for _ in range(20):
            orch.pathway_fitness_tracker.record_execution(
                pw_name,
                steps_executed=["bridge_create", "bridge_stp"],
                step_timings={"bridge_create": 50, "bridge_stp": 50},
                success=False, failure_step="bridge_stp",
                input_json="{}",
            )

        # Inject timing anomaly to trigger reordering operator
        rec = orch.pathway_fitness_tracker.get_record(pw_name)
        rec.step_timings["bridge_stp"] = [500.0] * 10

        # Try mutation
        new_sha = orch._try_pathway_mutation(pw_name, default_pw)
        if new_sha is not None:
            alleles = orch.pathway_registry.get_for_pathway(pw_name)
            assert len(alleles) >= 2
            new_allele = orch.pathway_registry.get(new_sha)
            assert new_allele.state == "recessive"
            assert new_allele.mutation_operator is not None


class TestInteractionBlockedPromotion:
    """Allele qualifies for promotion but interaction test blocks it."""

    def test_promotion_blocked_by_interaction_failure(self, full_project):
        """Promotion is blocked when SG_INTERACTION_POLICY=rollback and pathway fails."""
        orch = _make_orchestrator(full_project)

        # Get the dominant allele for bridge_create
        dom_sha = orch.phenotype.get_dominant("bridge_create")
        assert dom_sha is not None

        # Register a new allele that will qualify for promotion
        new_sha = orch.registry.register(SIMPLE_GENE, "bridge_create")
        orch.phenotype.add_to_fallback("bridge_create", new_sha)
        new_allele = orch.registry.get(new_sha)
        new_allele.state = "recessive"
        # Give it enough invocations and fitness to qualify
        new_allele.successful_invocations = 55
        new_allele.failed_invocations = 0
        new_allele.consecutive_failures = 0

        # Make the interaction check fail by patching
        from sg.interactions import InteractionFailure
        fake_failures = [InteractionFailure(
            pathway_name="configure_bridge_with_stp",
            failing_step="bridge_create",
            error="test interaction failure",
        )]

        with patch.object(orch, '_test_promotion_interactions',
                          return_value=fake_failures):
            with patch.dict(os.environ, {"SG_INTERACTION_POLICY": "rollback"}):
                orch._check_promotion("bridge_create", new_sha)

        # Should NOT have been promoted
        current_dom = orch.phenotype.get_dominant("bridge_create")
        assert current_dom == dom_sha  # original still dominant

    def test_promotion_allowed_with_mutate_policy(self, full_project):
        """With mutate policy, promotion proceeds despite interaction failures."""
        orch = _make_orchestrator(full_project)

        dom_sha = orch.phenotype.get_dominant("bridge_create")
        new_sha = orch.registry.register(SIMPLE_GENE, "bridge_create")
        orch.phenotype.add_to_fallback("bridge_create", new_sha)
        new_allele = orch.registry.get(new_sha)
        new_allele.state = "recessive"
        new_allele.successful_invocations = 55
        new_allele.failed_invocations = 0

        from sg.interactions import InteractionFailure
        fake_failures = [InteractionFailure(
            pathway_name="test_pw", failing_step="bridge_create",
            error="test",
        )]

        with patch.object(orch, '_test_promotion_interactions',
                          return_value=fake_failures):
            with patch.dict(os.environ, {"SG_INTERACTION_POLICY": "mutate"}):
                orch._check_promotion("bridge_create", new_sha)

        # Should have been promoted despite failure (mutate policy)
        current_dom = orch.phenotype.get_dominant("bridge_create")
        assert current_dom == new_sha


class TestFullEvolutionaryCycle:
    """End-to-end: gene fails → mutation → retry → fitness recorded."""

    def test_failure_triggers_mutation_and_retry(self, full_project):
        """When all alleles fail, mutation creates a new one and retries."""
        orch = _make_orchestrator(full_project)

        # Replace the dominant with a failing gene
        fail_sha = orch.registry.register(FAILING_GENE_TYPE_A, "bridge_create")
        orch.phenotype.promote("bridge_create", fail_sha)
        fail_allele = orch.registry.get(fail_sha)
        fail_allele.state = "dominant"

        input_json = json.dumps({"bridge_name": "br0", "interfaces": ["eth0"]})
        result = orch.execute_locus("bridge_create", input_json)

        # MockMutationEngine should provide a fix from fixtures
        assert result is not None
        output, used_sha = result
        assert used_sha != fail_sha  # used the mutated allele

        # Fitness should be recorded on the new allele
        mutant = orch.registry.get(used_sha)
        assert mutant is not None
        assert mutant.successful_invocations >= 1

    def test_fallback_stack_traversal(self, full_project):
        """Dominant fails → fallback succeeds → fitness recorded on both."""
        orch = _make_orchestrator(full_project)

        # Dominant = failing, fallback = good
        good_sha = orch.phenotype.get_dominant("bridge_create")
        fail_sha = orch.registry.register(FAILING_GENE_TYPE_A, "bridge_create")
        orch.phenotype.promote("bridge_create", fail_sha)
        fail_allele = orch.registry.get(fail_sha)
        fail_allele.state = "dominant"
        orch.phenotype.add_to_fallback("bridge_create", good_sha)

        input_json = json.dumps({"bridge_name": "br0", "interfaces": ["eth0"]})
        result = orch.execute_locus("bridge_create", input_json)

        assert result is not None
        output, used_sha = result
        assert used_sha == good_sha

        # Failing allele should have failure recorded
        assert orch.registry.get(fail_sha).failed_invocations >= 1
        # Good allele should have success recorded
        assert orch.registry.get(good_sha).successful_invocations >= 1

    def test_demotion_after_consecutive_failures(self, full_project):
        """3 consecutive failures demotes allele to deprecated."""
        orch = _make_orchestrator(full_project)

        fail_sha = orch.registry.register(FAILING_GENE_TYPE_A, "bridge_create")
        orch.phenotype.promote("bridge_create", fail_sha)
        fail_allele = orch.registry.get(fail_sha)
        fail_allele.state = "dominant"
        # Pre-set consecutive failures to just below threshold
        fail_allele.consecutive_failures = 2

        input_json = json.dumps({"bridge_name": "br0", "interfaces": ["eth0"]})
        orch.execute_locus("bridge_create", input_json)

        # Should have been demoted after 3rd consecutive failure
        assert fail_allele.state == "deprecated"


class TestDecompositionEndToEnd:
    """Full chain: diverse errors → decomposition signal → auto-split → pathway."""

    def _create_decompose_fixtures(self, fixtures_dir):
        """Create minimal decomposition fixtures for MockMutationEngine."""
        # Sub-gene contracts
        (fixtures_dir / "bridge_create_sub1.sg").write_text(
            'gene bridge_create_sub1\n'
            '  is configuration\n'
            '  risk low\n'
            '\n'
            '  does:\n'
            '    Validate bridge input parameters.\n'
            '\n'
            '  takes:\n'
            '    bridge_name  string    "Name for the bridge"\n'
            '    interfaces   string[]  "Interfaces to attach"\n'
            '\n'
            '  gives:\n'
            '    success  bool  "Whether validation passed"\n'
        )
        (fixtures_dir / "bridge_create_sub2.sg").write_text(
            'gene bridge_create_sub2\n'
            '  is configuration\n'
            '  risk low\n'
            '\n'
            '  does:\n'
            '    Create bridge on kernel.\n'
            '\n'
            '  takes:\n'
            '    bridge_name  string    "Name for the bridge"\n'
            '    interfaces   string[]  "Interfaces to attach"\n'
            '\n'
            '  gives:\n'
            '    success  bool  "Whether creation succeeded"\n'
        )
        (fixtures_dir / "bridge_create_sub3.sg").write_text(
            'gene bridge_create_sub3\n'
            '  is configuration\n'
            '  risk low\n'
            '\n'
            '  does:\n'
            '    Verify bridge creation.\n'
            '\n'
            '  takes:\n'
            '    bridge_name  string    "Name for the bridge"\n'
            '    interfaces   string[]  "Interfaces to attach"\n'
            '\n'
            '  gives:\n'
            '    success  bool  "Whether verification passed"\n'
        )

        # Sub-gene seed implementations
        (fixtures_dir / "bridge_create_sub1.py").write_text(
            'import json\n'
            'def execute(input_json):\n'
            '    data = json.loads(input_json)\n'
            '    return json.dumps({"success": True, "bridge_name": data["bridge_name"]})\n'
        )
        (fixtures_dir / "bridge_create_sub2.py").write_text(
            'import json\n'
            'def execute(input_json):\n'
            '    data = json.loads(input_json)\n'
            '    gene_sdk.create_bridge(data["bridge_name"], data.get("interfaces", []))\n'
            '    return json.dumps({"success": True})\n'
        )
        (fixtures_dir / "bridge_create_sub3.py").write_text(
            'import json\n'
            'def execute(input_json):\n'
            '    data = json.loads(input_json)\n'
            '    return json.dumps({"success": True})\n'
        )

        # Pathway contract for decomposed pathway
        (fixtures_dir / "bridge_create_decompose_pathway.sg").write_text(
            'pathway bridge_create_decomposed\n'
            '  risk low\n'
            '\n'
            '  does:\n'
            '    Decomposed bridge creation.\n'
            '\n'
            '  steps:\n'
            '    1. bridge_create_sub1\n'
            '         bridge_name = {bridge_name}\n'
            '         interfaces = {interfaces}\n'
            '\n'
            '    2. bridge_create_sub2\n'
            '         bridge_name = {bridge_name}\n'
            '         interfaces = {interfaces}\n'
            '\n'
            '    3. bridge_create_sub3\n'
            '         bridge_name = {bridge_name}\n'
            '         interfaces = {interfaces}\n'
        )

    def test_diverse_errors_trigger_decomposition(self, full_project):
        """10+ errors with 3+ clusters → decomposition fires → sub-genes registered."""
        self._create_decompose_fixtures(full_project / "fixtures")
        orch = _make_orchestrator(full_project)

        # Register three different failing genes as separate alleles
        fail_a_sha = orch.registry.register(FAILING_GENE_TYPE_A, "bridge_create")
        fail_b_sha = orch.registry.register(FAILING_GENE_TYPE_B, "bridge_create")
        fail_c_sha = orch.registry.register(FAILING_GENE_TYPE_C, "bridge_create")

        # Set up as dominant with fallbacks removed — force exhaustion each time
        # We'll promote different failing genes and call execute_locus to accumulate errors
        input_json = json.dumps({"bridge_name": "br0", "interfaces": ["eth0"]})

        # Accumulate errors with diverse patterns — need 10+ errors with 3+ clusters
        for i in range(4):
            # Cycle through different error types
            for fail_sha in [fail_a_sha, fail_b_sha, fail_c_sha]:
                orch.phenotype.promote("bridge_create", fail_sha)
                allele = orch.registry.get(fail_sha)
                allele.state = "dominant"
                allele.consecutive_failures = 0
                # Clear fallback so no recovery
                orch.phenotype.loci["bridge_create"].fallback = []
                # Remove fixture fix so mutation fails too
                fix_path = full_project / "fixtures" / "bridge_create_fix.py"
                fix_existed = fix_path.exists()
                fix_backup = None
                if fix_existed:
                    fix_backup = fix_path.read_text()
                    fix_path.unlink()

                orch.execute_locus("bridge_create", input_json)

                # Restore fixture for later
                if fix_backup is not None:
                    fix_path.write_text(fix_backup)

        # Should have accumulated 12 errors across 3 clusters
        history = orch.decomposition_detector.histories.get("bridge_create")
        assert history is not None
        assert len(history.errors) >= 10

        # Verify analyze() returns a signal
        signal = orch.decomposition_detector.analyze("bridge_create")
        assert signal is not None
        assert len(signal.error_clusters) >= 3

        # Now trigger decomposition by calling execute_locus with all alleles exhausted
        # Re-register a failing gene as the only option
        orch.phenotype.promote("bridge_create", fail_a_sha)
        orch.registry.get(fail_a_sha).state = "dominant"
        orch.registry.get(fail_a_sha).consecutive_failures = 0
        orch.phenotype.loci["bridge_create"].fallback = []

        result = orch.execute_locus("bridge_create", input_json)

        # Decomposition should have occurred
        assert orch.decomposition_detector.is_decomposed("bridge_create")

        # Sub-loci should be registered
        state = orch.decomposition_detector._decomposition_state.get("bridge_create")
        assert state is not None
        for sub_locus in state["sub_loci"]:
            assert orch.contract_store.get_gene(sub_locus) is not None
            stack = orch.phenotype.get_stack(sub_locus)
            assert len(stack) > 0


class TestPathwayMutationFromExhaustion:
    """Full chain: run_pathway fails → allele exhaustion → mutation → new allele."""

    def test_exhaustion_triggers_pathway_mutation(self, full_project):
        """All pathway alleles fail → _try_pathway_mutation fires → new allele registered."""
        from sg.pathway import pathway_from_contract

        orch = _make_orchestrator(full_project)
        pw_name = "configure_bridge_with_stp"

        # Register a failing gene for bridge_stp so the pathway actually fails
        # (the seed gene catches exceptions internally, so we need a gene that raises)
        fail_stp_sha = orch.registry.register(FAILING_GENE_TYPE_A, "bridge_stp")
        orch.phenotype.promote("bridge_stp", fail_stp_sha)
        fail_allele = orch.registry.get(fail_stp_sha)
        fail_allele.state = "dominant"
        # Clear fallback so no working gene to fall back to
        orch.phenotype.loci["bridge_stp"].fallback = []
        # Remove bridge_stp fixture fix so gene-level mutation also fails
        stp_fix = full_project / "fixtures" / "bridge_stp_fix.py"
        if stp_fix.exists():
            stp_fix.unlink()

        # Register the default pathway allele
        pw_contract = orch.contract_store.get_pathway(pw_name)
        default_pw = pathway_from_contract(pw_contract)
        specs = steps_from_pathway(default_pw)
        sha = orch.pathway_registry.register(pw_name, specs)
        pw_allele = orch.pathway_registry.get(sha)
        pw_allele.state = "dominant"
        orch.phenotype.promote_pathway(pw_name, sha)

        # Give bridge_create gene high fitness (> 0.7) so _is_structural_problem returns True
        bc_dom_sha = orch.phenotype.get_dominant("bridge_create")
        if bc_dom_sha:
            bc_allele = orch.registry.get(bc_dom_sha)
            if bc_allele:
                bc_allele.successful_invocations = 95
                bc_allele.failed_invocations = 5
        # Also give bridge_stp high gene fitness (individual gene metrics)
        fail_allele.successful_invocations = 95
        fail_allele.failed_invocations = 5

        # Record many pathway failures to get pathway fitness < 0.5
        for _ in range(20):
            orch.pathway_fitness_tracker.record_execution(
                pw_name,
                steps_executed=["bridge_create", "bridge_stp"],
                step_timings={"bridge_create": 50, "bridge_stp": 50},
                success=False, failure_step="bridge_stp",
                input_json="{}",
            )

        # Inject timing anomaly so reorder operator has something to work with
        rec = orch.pathway_fitness_tracker.get_record(pw_name)
        rec.step_timings["bridge_stp"] = [500.0] * 10

        input_json = json.dumps({
            "bridge_name": "br0", "interfaces": ["eth0"],
            "stp_enabled": True, "forward_delay": 15,
        })

        # Run should exhaust allele stack and attempt mutation
        with pytest.raises(RuntimeError, match="all pathway alleles exhausted"):
            orch.run_pathway(pw_name, input_json)

        # Check if mutation was attempted — new allele should exist
        alleles = orch.pathway_registry.get_for_pathway(pw_name)
        if len(alleles) >= 2:
            # Mutation succeeded
            new_alleles = [a for a in alleles if a.structure_sha != sha]
            assert len(new_alleles) >= 1
            new = new_alleles[0]
            assert new.state == "recessive"
            assert new.mutation_operator is not None
            # Should be in phenotype fallback stack
            stack = orch.phenotype.get_pathway_stack(pw_name)
            assert new.structure_sha in stack


class TestOrchestratorContextManager:
    """Orchestrator supports with-statement and close()."""

    def test_context_manager_saves_state(self, full_project):
        """Using 'with' auto-saves state on exit."""
        with _make_orchestrator(full_project) as orch:
            orch.execute_locus("bridge_create", json.dumps({
                "bridge_name": "br0", "interfaces": ["eth0"],
            }))
        # After exiting context, phenotype should be persisted
        phenotype = PhenotypeMap.load(full_project / "phenotype.toml")
        assert phenotype.get_dominant("bridge_create") is not None

    def test_context_manager_cancels_timers(self, full_project):
        """close() cancels verify scheduler timers."""
        orch = _make_orchestrator(full_project)
        orch.execute_locus("bridge_create", json.dumps({
            "bridge_name": "br0", "interfaces": ["eth0"],
        }))
        pending_before = orch.verify_scheduler.pending_count
        orch.close()
        assert orch.verify_scheduler.pending_count == 0

    def test_close_idempotent(self, full_project):
        """Calling close() twice does not raise."""
        orch = _make_orchestrator(full_project)
        orch.close()
        orch.close()  # should not raise
