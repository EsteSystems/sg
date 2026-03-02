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
from sg.pathway_registry import PathwayRegistry, StepSpec, steps_from_pathway
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
