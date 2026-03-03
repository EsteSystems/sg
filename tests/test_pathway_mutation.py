"""Tests for pathway mutation operators (Phase 5)."""
import json
import shutil
import time
import pytest
from pathlib import Path

from sg.contracts import ContractStore
from sg.pathway_fitness import TimingAnomaly, InputCluster
from sg.pathway_registry import StepSpec, PathwayAllele, PathwayRegistry, steps_from_pathway
from sg.pathway_mutation import (
    PathwayMutationContext,
    PathwayMutationResult,
    PathwayMutationThrottle,
    PathwayMutationOperator,
    ReorderingOperator,
    DeletionOperator,
    StepSubstitutionOperator,
    InsertionOperator,
    select_operator,
    default_operators,
    _reindex_conditionals_after_delete,
    _reindex_conditionals_after_insert,
)


# --- Helpers ---


def _make_ctx(
    steps=None,
    failure_distribution=None,
    timing_anomalies=None,
    per_step_fitness=None,
    gene_fitness_map=None,
    contract=None,
    contract_store=None,
    available_loci=None,
    pathway_fitness=0.3,
    input_clusters=None,
):
    return PathwayMutationContext(
        pathway_name="test_pw",
        current_steps=steps or [],
        pathway_fitness=pathway_fitness,
        per_step_fitness=per_step_fitness or {},
        timing_anomalies=timing_anomalies or [],
        failure_distribution=failure_distribution or {},
        input_clusters=input_clusters or [],
        available_loci=available_loci or ["a", "b", "c"],
        available_pathways=[],
        gene_fitness_map=gene_fitness_map or {},
        contract=contract,
        contract_store=contract_store or ContractStore(),
    )


# --- Framework tests ---


class TestPathwayMutationContext:
    def test_context_creation(self):
        ctx = _make_ctx(
            steps=[StepSpec(step_type="locus", target="a")],
        )
        assert ctx.pathway_name == "test_pw"
        assert len(ctx.current_steps) == 1


class TestPathwayMutationResult:
    def test_result_creation(self):
        result = PathwayMutationResult(
            new_steps=[
                StepSpec(step_type="locus", target="b"),
                StepSpec(step_type="locus", target="a"),
            ],
            operator_name="reorder",
            rationale="test",
        )
        assert result.operator_name == "reorder"
        assert len(result.new_steps) == 2


class TestSelectOperator:
    def test_first_wins(self):
        """select_operator returns the first applicable operator's result."""

        class AlwaysOp(PathwayMutationOperator):
            @property
            def name(self):
                return "always"

            def can_apply(self, ctx):
                return True

            def apply(self, ctx):
                return PathwayMutationResult(
                    new_steps=ctx.current_steps,
                    operator_name=self.name,
                    rationale="always applies",
                )

        class NeverOp(PathwayMutationOperator):
            @property
            def name(self):
                return "never"

            def can_apply(self, ctx):
                return False

            def apply(self, ctx):
                return None

        ctx = _make_ctx(steps=[StepSpec(step_type="locus", target="a")])
        result = select_operator(ctx, [AlwaysOp(), NeverOp()])
        assert result is not None
        assert result.operator_name == "always"

    def test_none_when_no_applicable(self):
        class NeverOp(PathwayMutationOperator):
            @property
            def name(self):
                return "never"

            def can_apply(self, ctx):
                return False

            def apply(self, ctx):
                return None

        ctx = _make_ctx(steps=[StepSpec(step_type="locus", target="a")])
        result = select_operator(ctx, [NeverOp()])
        assert result is None


# --- Throttle tests ---


class TestPathwayMutationThrottle:
    def test_can_mutate_initially(self):
        throttle = PathwayMutationThrottle()
        assert throttle.can_mutate("any_pathway")

    def test_throttled_after_mutation(self):
        throttle = PathwayMutationThrottle(cooldown_seconds=100)
        throttle.record_mutation("pw")
        assert not throttle.can_mutate("pw")

    def test_different_pathways_independent(self):
        throttle = PathwayMutationThrottle(cooldown_seconds=100)
        throttle.record_mutation("pw_a")
        assert throttle.can_mutate("pw_b")

    def test_to_dict_roundtrip(self):
        throttle = PathwayMutationThrottle(cooldown_seconds=300)
        throttle.record_mutation("pw")
        d = throttle.to_dict()
        restored = PathwayMutationThrottle.from_dict(d)
        assert restored.cooldown_seconds == 300
        assert not restored.can_mutate("pw")

    def test_reset_cooldown_clears(self):
        throttle = PathwayMutationThrottle(cooldown_seconds=100)
        throttle.record_mutation("pw")
        assert not throttle.can_mutate("pw")
        throttle.reset_cooldown("pw")
        assert throttle.can_mutate("pw")

    def test_can_mutate_after_reset(self):
        throttle = PathwayMutationThrottle(cooldown_seconds=99999)
        throttle.record_mutation("pw")
        throttle.reset_cooldown("pw")
        # Should be mutable immediately after reset
        assert throttle.can_mutate("pw")

    def test_reset_unknown_noop(self):
        throttle = PathwayMutationThrottle()
        throttle.reset_cooldown("nonexistent")  # no error


# --- ReorderingOperator tests ---


class TestReorderingOperator:
    def test_cannot_apply_single_step(self):
        op = ReorderingOperator()
        ctx = _make_ctx(steps=[StepSpec(step_type="locus", target="a")])
        assert not op.can_apply(ctx)

    def test_can_apply_with_timing_anomaly(self):
        op = ReorderingOperator()
        steps = [
            StepSpec(step_type="locus", target="a"),
            StepSpec(step_type="locus", target="b"),
        ]
        anomalies = [TimingAnomaly(step_name="b", latest_ms=100, avg_ms=10, ratio=10.0)]
        ctx = _make_ctx(steps=steps, timing_anomalies=anomalies)
        assert op.can_apply(ctx)

    def test_can_apply_with_failure_gene_gap(self):
        op = ReorderingOperator()
        steps = [
            StepSpec(step_type="locus", target="a"),
            StepSpec(step_type="locus", target="b"),
        ]
        ctx = _make_ctx(
            steps=steps,
            failure_distribution={"b": 0.5},
            per_step_fitness={"b": 0.9},
        )
        assert op.can_apply(ctx)

    def test_swaps_independent_steps(self):
        op = ReorderingOperator()
        steps = [
            StepSpec(step_type="locus", target="a"),
            StepSpec(step_type="locus", target="b"),
        ]
        anomalies = [TimingAnomaly(step_name="b", latest_ms=100, avg_ms=10, ratio=10.0)]
        ctx = _make_ctx(steps=steps, timing_anomalies=anomalies)
        result = op.apply(ctx)
        assert result is not None
        assert result.operator_name == "reorder"
        assert result.new_steps[0].target == "b"
        assert result.new_steps[1].target == "a"

    def test_respects_requires_dependency(self):
        from sg.parser.types import PathwayContract, Dependency, BlastRadius
        op = ReorderingOperator()
        steps = [
            StepSpec(step_type="locus", target="a"),
            StepSpec(step_type="locus", target="b"),
        ]
        anomalies = [TimingAnomaly(step_name="b", latest_ms=100, avg_ms=10, ratio=10.0)]
        # Step 2 requires step 1 (1-based)
        contract = PathwayContract(
            name="test_pw", risk=BlastRadius.LOW, does="test",
            requires=[Dependency(step=2, needs=1)],
        )
        ctx = _make_ctx(steps=steps, timing_anomalies=anomalies, contract=contract)
        result = op.apply(ctx)
        assert result is None  # Cannot reorder due to dependency

    def test_six_steps_uses_full_permutation(self):
        """6 steps is the boundary — should use full permutations, not swaps."""
        op = ReorderingOperator()
        steps = [StepSpec(step_type="locus", target=f"s{i}") for i in range(6)]
        anomalies = [TimingAnomaly(step_name="s5", latest_ms=100, avg_ms=10, ratio=10.0)]
        ctx = _make_ctx(steps=steps, timing_anomalies=anomalies)
        result = op.apply(ctx)
        # Should find a permutation (6! = 720 candidates minus identity)
        assert result is not None

    def test_seven_steps_uses_adjacent_swap(self):
        """7 steps exceeds boundary — should use adjacent swaps only."""
        op = ReorderingOperator()
        steps = [StepSpec(step_type="locus", target=f"s{i}") for i in range(7)]
        anomalies = [TimingAnomaly(step_name="s6", latest_ms=100, avg_ms=10, ratio=10.0)]
        ctx = _make_ctx(steps=steps, timing_anomalies=anomalies)
        result = op.apply(ctx)
        if result is not None:
            # Adjacent swap: exactly one pair should differ from original
            diffs = sum(
                1 for a, b in zip(steps, result.new_steps) if a.target != b.target
            )
            assert diffs == 2  # one swap = two positions differ

    def test_respects_data_flow_dependency(self):
        """Steps linked by takes/gives data flow cannot be reordered."""
        from sg.parser.types import GeneContract, FieldDef, GeneFamily, BlastRadius

        # Create a contract store with two genes where b depends on a's output
        cs = ContractStore()
        gene_a = GeneContract(
            name="gene_a", family=GeneFamily.CONFIGURATION,
            risk=BlastRadius.LOW, does="produces x",
            gives=[FieldDef(name="x_output", type="string")],
        )
        gene_b = GeneContract(
            name="gene_b", family=GeneFamily.CONFIGURATION,
            risk=BlastRadius.LOW, does="needs x",
            takes=[FieldDef(name="x_output", type="string", required=True)],
        )
        cs.genes = {"gene_a": gene_a, "gene_b": gene_b}

        op = ReorderingOperator()
        steps = [
            StepSpec(step_type="locus", target="gene_a"),
            StepSpec(step_type="locus", target="gene_b"),
        ]
        anomalies = [TimingAnomaly(step_name="gene_b", latest_ms=100, avg_ms=10, ratio=10.0)]
        ctx = _make_ctx(steps=steps, timing_anomalies=anomalies, contract_store=cs)
        result = op.apply(ctx)
        assert result is None  # Cannot reorder: gene_b takes x_output from gene_a


# --- DeletionOperator tests ---


class TestDeletionOperator:
    def test_cannot_delete_only_step(self):
        op = DeletionOperator()
        ctx = _make_ctx(
            steps=[StepSpec(step_type="locus", target="a")],
            per_step_fitness={"a": 1.0},
        )
        assert not op.can_apply(ctx)

    def test_deletes_trivial_step(self):
        op = DeletionOperator()
        steps = [
            StepSpec(step_type="locus", target="a"),
            StepSpec(step_type="locus", target="trivial"),
            StepSpec(step_type="locus", target="b"),
        ]
        ctx = _make_ctx(
            steps=steps,
            per_step_fitness={"a": 0.9, "trivial": 1.0, "b": 0.9},
            failure_distribution={"a": 0.1},
        )
        result = op.apply(ctx)
        assert result is not None
        assert result.operator_name == "deletion"
        assert len(result.new_steps) == 2
        assert all(s.target != "trivial" for s in result.new_steps)

    def test_preserves_step_with_dependents(self):
        """A step whose gives fields are consumed by a later step cannot be deleted."""
        from sg.parser.types import GeneContract, FieldDef, GeneFamily, BlastRadius

        cs = ContractStore()
        cs.genes = {
            "provider": GeneContract(
                name="provider", family=GeneFamily.CONFIGURATION,
                risk=BlastRadius.LOW, does="provides x",
                gives=[FieldDef(name="unique_field", type="string")],
            ),
            "consumer": GeneContract(
                name="consumer", family=GeneFamily.CONFIGURATION,
                risk=BlastRadius.LOW, does="needs x",
                takes=[FieldDef(name="unique_field", type="string", required=True)],
            ),
        }

        op = DeletionOperator()
        steps = [
            StepSpec(step_type="locus", target="provider"),
            StepSpec(step_type="locus", target="consumer"),
        ]
        ctx = _make_ctx(
            steps=steps,
            per_step_fitness={"provider": 1.0, "consumer": 0.9},
            contract_store=cs,
        )
        assert not op.can_apply(ctx)

    def test_reindexes_conditionals(self):
        """condition_step_index adjusted after deletion."""
        steps = [
            StepSpec(step_type="locus", target="a"),
            StepSpec(step_type="locus", target="b"),
            StepSpec(
                step_type="conditional", target="",
                condition_step_index=1, condition_field="status",
                branches={"ok": {"step_type": "locus", "target": "c"}},
            ),
        ]
        result = _reindex_conditionals_after_delete(steps[1:], 0)
        # After deleting index 0, the conditional's reference to index 1 should become 0
        assert result[1].condition_step_index == 0


# --- StepSubstitutionOperator tests ---


class TestStepSubstitutionOperator:
    def test_substitutes_failing_locus(self):
        from sg.parser.types import GeneContract, FieldDef, GeneFamily, BlastRadius

        cs = ContractStore()
        cs.genes = {
            "failing": GeneContract(
                name="failing", family=GeneFamily.CONFIGURATION,
                risk=BlastRadius.LOW, does="fails",
                takes=[FieldDef(name="x", type="string")],
                gives=[FieldDef(name="y", type="string")],
            ),
            "replacement": GeneContract(
                name="replacement", family=GeneFamily.CONFIGURATION,
                risk=BlastRadius.LOW, does="works",
                takes=[FieldDef(name="x", type="string")],
                gives=[FieldDef(name="y", type="string")],
            ),
        }

        op = StepSubstitutionOperator()
        steps = [StepSpec(step_type="locus", target="failing")]
        ctx = _make_ctx(
            steps=steps,
            per_step_fitness={"failing": 0.1},
            failure_distribution={"failing": 0.8},
            gene_fitness_map={"failing": 0.1, "replacement": 0.9},
            available_loci=["failing", "replacement"],
            contract_store=cs,
        )
        assert op.can_apply(ctx)
        result = op.apply(ctx)
        assert result is not None
        assert result.new_steps[0].target == "replacement"

    def test_substitution_picks_highest_fitness(self):
        """With multiple compatible loci, picks the one with highest fitness."""
        from sg.parser.types import GeneContract, FieldDef, GeneFamily, BlastRadius

        cs = ContractStore()
        cs.genes = {
            "failing": GeneContract(
                name="failing", family=GeneFamily.CONFIGURATION,
                risk=BlastRadius.LOW, does="fails",
                takes=[FieldDef(name="x", type="string")],
                gives=[FieldDef(name="y", type="string")],
            ),
            "alt_low": GeneContract(
                name="alt_low", family=GeneFamily.CONFIGURATION,
                risk=BlastRadius.LOW, does="low fitness alt",
                takes=[FieldDef(name="x", type="string")],
                gives=[FieldDef(name="y", type="string")],
            ),
            "alt_high": GeneContract(
                name="alt_high", family=GeneFamily.CONFIGURATION,
                risk=BlastRadius.LOW, does="high fitness alt",
                takes=[FieldDef(name="x", type="string")],
                gives=[FieldDef(name="y", type="string")],
            ),
        }

        op = StepSubstitutionOperator()
        steps = [StepSpec(step_type="locus", target="failing")]
        ctx = _make_ctx(
            steps=steps,
            per_step_fitness={"failing": 0.1},
            failure_distribution={"failing": 0.8},
            gene_fitness_map={"failing": 0.1, "alt_low": 0.5, "alt_high": 0.9},
            available_loci=["failing", "alt_low", "alt_high"],
            contract_store=cs,
        )
        result = op.apply(ctx)
        assert result is not None
        assert result.new_steps[0].target == "alt_high"

    def test_no_substitution_without_compatible(self):
        from sg.parser.types import GeneContract, FieldDef, GeneFamily, BlastRadius

        cs = ContractStore()
        cs.genes = {
            "failing": GeneContract(
                name="failing", family=GeneFamily.CONFIGURATION,
                risk=BlastRadius.LOW, does="fails",
                takes=[FieldDef(name="x", type="string")],
                gives=[FieldDef(name="y", type="string")],
            ),
            "other": GeneContract(
                name="other", family=GeneFamily.DIAGNOSTIC,
                risk=BlastRadius.LOW, does="different family",
                takes=[FieldDef(name="x", type="string")],
                gives=[FieldDef(name="y", type="string")],
            ),
        }

        op = StepSubstitutionOperator()
        steps = [StepSpec(step_type="locus", target="failing")]
        ctx = _make_ctx(
            steps=steps,
            per_step_fitness={"failing": 0.1},
            failure_distribution={"failing": 0.8},
            available_loci=["failing", "other"],
            contract_store=cs,
        )
        assert not op.can_apply(ctx)


# --- InsertionOperator tests ---


class TestInsertionOperator:
    def test_insertion_with_mock_fixture(self, tmp_path):
        fixture = {
            "locus": "check_ready",
            "insert_before_index": 1,
            "params": {},
            "rationale": "Insert check before step 1",
        }
        (tmp_path / "test_pw_insertion.json").write_text(json.dumps(fixture))

        from sg.mutation import MockMutationEngine
        engine = MockMutationEngine(tmp_path)

        op = InsertionOperator(engine)
        steps = [
            StepSpec(step_type="locus", target="a"),
            StepSpec(step_type="locus", target="b"),
        ]
        ctx = _make_ctx(
            steps=steps,
            failure_distribution={"b": 0.6},
            per_step_fitness={"b": 0.9},
            available_loci=["a", "b", "check_ready"],
        )
        assert op.can_apply(ctx)
        result = op.apply(ctx)
        assert result is not None
        assert result.operator_name == "insertion"
        assert len(result.new_steps) == 3
        assert result.new_steps[1].target == "check_ready"

    def test_insertion_unknown_locus_rejected(self, tmp_path):
        fixture = {
            "locus": "nonexistent_locus",
            "insert_before_index": 0,
            "params": {},
            "rationale": "test",
        }
        (tmp_path / "test_pw_insertion.json").write_text(json.dumps(fixture))

        from sg.mutation import MockMutationEngine
        engine = MockMutationEngine(tmp_path)

        op = InsertionOperator(engine)
        steps = [StepSpec(step_type="locus", target="a")]
        ctx = _make_ctx(
            steps=steps,
            failure_distribution={"a": 0.6},
            per_step_fitness={"a": 0.9},
            available_loci=["a", "b"],
        )
        result = op.apply(ctx)
        assert result is None  # nonexistent_locus not in available_loci


# --- Reindexing helpers ---


class TestReindexConditionals:
    def test_delete_reindexes(self):
        steps = [
            StepSpec(step_type="locus", target="x"),
            StepSpec(
                step_type="conditional", target="",
                condition_step_index=2, condition_field="f",
            ),
        ]
        result = _reindex_conditionals_after_delete(steps, 0)
        assert result[1].condition_step_index == 1

    def test_delete_at_zero_reindexes(self):
        """Delete step 0: conditional referencing step 1 → becomes 0."""
        steps = [
            StepSpec(
                step_type="conditional", target="",
                condition_step_index=1, condition_field="f",
            ),
        ]
        result = _reindex_conditionals_after_delete(steps, 0)
        assert result[0].condition_step_index == 0

    def test_insert_at_end_reindexes(self):
        """Insert at end: conditional referencing step before end → unchanged."""
        steps = [
            StepSpec(step_type="locus", target="x"),
            StepSpec(
                step_type="conditional", target="",
                condition_step_index=0, condition_field="f",
            ),
        ]
        # Insert at index 2 (after everything)
        result = _reindex_conditionals_after_insert(steps, 2)
        # condition_step_index=0 < insert_idx=2, so unchanged
        assert result[1].condition_step_index == 0

    def test_insert_reindexes(self):
        steps = [
            StepSpec(step_type="locus", target="x"),
            StepSpec(
                step_type="conditional", target="",
                condition_step_index=0, condition_field="f",
            ),
        ]
        result = _reindex_conditionals_after_insert(steps, 0)
        assert result[1].condition_step_index == 1


# --- Integration tests ---


class TestDefaultOperators:
    def test_default_operators_without_engine(self):
        ops = default_operators()
        assert len(ops) == 3
        names = [op.name for op in ops]
        assert names == ["reorder", "substitution", "deletion"]

    def test_default_operators_with_engine(self):
        ops = default_operators(mutation_engine=object())
        assert len(ops) == 4
        assert ops[-1].name == "insertion"


class TestOrchestratorIntegration:
    @pytest.fixture
    def project(self, tmp_path):
        import sg_network
        CONTRACTS_DIR = sg_network.contracts_path()
        GENES_DIR = sg_network.genes_path()
        FIXTURES_DIR = sg_network.fixtures_path()

        (tmp_path / ".sg").mkdir()
        shutil.copytree(CONTRACTS_DIR, tmp_path / "contracts")
        shutil.copytree(FIXTURES_DIR, tmp_path / "fixtures")

        from sg.registry import Registry
        from sg.phenotype import PhenotypeMap

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

    def _make_orchestrator(self, project):
        from sg.registry import Registry
        from sg.phenotype import PhenotypeMap
        from sg.fusion import FusionTracker
        from sg.mutation import MockMutationEngine
        from sg.orchestrator import Orchestrator
        from sg.pathway_fitness import PathwayFitnessTracker

        import sg_network
        cs = ContractStore.open(project / "contracts")
        registry = Registry.open(project / ".sg" / "registry")
        phenotype = PhenotypeMap.load(project / "phenotype.toml")
        ft = FusionTracker.open(project / "fusion_tracker.json")
        pft = PathwayFitnessTracker.open(project / "pathway_fitness.json")
        kernel = sg_network.MockNetworkKernel()
        mutation_engine = MockMutationEngine(project / "fixtures")
        pathway_registry = PathwayRegistry.open(project / ".sg" / "pathway_registry")

        return Orchestrator(
            registry=registry, phenotype=phenotype,
            mutation_engine=mutation_engine, fusion_tracker=ft,
            kernel=kernel, contract_store=cs, project_root=project,
            pathway_fitness_tracker=pft,
            pathway_registry=pathway_registry,
        )

    def test_pathway_from_stepspecs_reconstructs(self, project):
        """_pathway_from_stepspecs produces executable Pathway from StepSpec."""
        from sg.pathway import pathway_from_contract, Pathway
        orch = self._make_orchestrator(project)
        pw_contract = orch.contract_store.get_pathway("configure_bridge_with_stp")
        default_pw = pathway_from_contract(pw_contract)
        specs = steps_from_pathway(default_pw)

        pw = orch._pathway_from_stepspecs("test", specs, default_pw)
        assert isinstance(pw, Pathway)
        assert len(pw.steps) == len(specs)

    def test_pathway_from_stepspecs_reordered(self, project):
        """Reordered specs produce correct step order."""
        from sg.pathway import pathway_from_contract, Pathway
        orch = self._make_orchestrator(project)
        pw_contract = orch.contract_store.get_pathway("configure_bridge_with_stp")
        default_pw = pathway_from_contract(pw_contract)
        specs = steps_from_pathway(default_pw)

        specs_reversed = list(reversed(specs))
        pw = orch._pathway_from_stepspecs("test", specs_reversed, default_pw)
        assert pw.steps[0].locus == specs_reversed[0].target
        assert pw.steps[-1].locus == specs_reversed[-1].target

    def test_is_structural_problem_detection(self, project):
        """Low pathway fit + high gene fit → structural problem."""
        orch = self._make_orchestrator(project)
        # Give gene alleles high fitness (many successful invocations)
        for locus in ["bridge_create", "bridge_stp"]:
            dom_sha = orch.phenotype.get_dominant(locus)
            if dom_sha:
                allele = orch.registry.get(dom_sha)
                if allele:
                    allele.successful_invocations = 95
                    allele.failed_invocations = 5
        # Set low pathway fitness by recording many failures
        for _ in range(20):
            orch.pathway_fitness_tracker.record_execution(
                "configure_bridge_with_stp",
                steps_executed=["bridge_create", "bridge_stp"],
                step_timings={"bridge_create": 50, "bridge_stp": 50},
                success=False,
                failure_step="bridge_stp",
                input_json="{}",
            )
        assert orch._is_structural_problem("configure_bridge_with_stp")

    def test_mutation_registered_as_recessive(self, project):
        """Full cycle: trigger → register → fallback stack."""
        orch = self._make_orchestrator(project)
        # Register the default allele first
        pw_contract = orch.contract_store.get_pathway("configure_bridge_with_stp")
        from sg.pathway import pathway_from_contract
        default_pw = pathway_from_contract(pw_contract)
        specs = steps_from_pathway(default_pw)
        sha = orch.pathway_registry.register("configure_bridge_with_stp", specs)
        pw_allele = orch.pathway_registry.get(sha)
        pw_allele.state = "dominant"
        orch.phenotype.promote_pathway("configure_bridge_with_stp", sha)

        # Set low pathway fitness + high gene fitness to trigger structural signal
        for _ in range(20):
            orch.pathway_fitness_tracker.record_execution(
                "configure_bridge_with_stp",
                steps_executed=["bridge_create", "bridge_stp"],
                step_timings={"bridge_create": 50, "bridge_stp": 50},
                success=False,
                failure_step="bridge_stp",
                input_json="{}",
            )

        # Also inject a timing anomaly to trigger reordering
        rec = orch.pathway_fitness_tracker.get_record("configure_bridge_with_stp")
        rec.step_timings["bridge_stp"] = [500.0] * 10  # High timing

        new_sha = orch._try_pathway_mutation(
            "configure_bridge_with_stp", default_pw,
        )
        if new_sha is not None:
            # Mutation was registered
            alleles = orch.pathway_registry.get_for_pathway("configure_bridge_with_stp")
            assert len(alleles) >= 2
            new_allele = orch.pathway_registry.get(new_sha)
            assert new_allele.state == "recessive"
            assert new_allele.mutation_operator is not None
            # Should be in phenotype fallback
            stack = orch.phenotype.get_pathway_stack("configure_bridge_with_stp")
            assert new_sha in stack

    def test_mutation_throttled(self, project):
        """Second mutation within cooldown blocked."""
        orch = self._make_orchestrator(project)
        orch._pathway_mutation_throttle.cooldown_seconds = 10000
        orch._pathway_mutation_throttle.record_mutation("configure_bridge_with_stp")

        from sg.pathway import pathway_from_contract
        pw_contract = orch.contract_store.get_pathway("configure_bridge_with_stp")
        default_pw = pathway_from_contract(pw_contract)

        result = orch._try_pathway_mutation("configure_bridge_with_stp", default_pw)
        assert result is None  # throttled

    def test_throttle_persisted_in_save_state(self, project):
        """save_state persists throttle JSON."""
        orch = self._make_orchestrator(project)
        orch._pathway_mutation_throttle.record_mutation("test_pw")
        orch.save_state()

        throttle_path = project / ".sg" / "pathway_mutation_throttle.json"
        assert throttle_path.exists()
        data = json.loads(throttle_path.read_text())
        assert "test_pw" in data["last_mutation"]


# --- Edge-case tests for untested paths ---


class TestThrottleCooldownExpiry:
    def test_can_mutate_after_cooldown(self):
        """After cooldown elapses, pathway is mutable again."""
        throttle = PathwayMutationThrottle(cooldown_seconds=1.0)
        throttle.record_mutation("pw")
        assert not throttle.can_mutate("pw")
        # Manually set last mutation far in the past
        throttle._last_mutation["pw"] = time.time() - 10.0
        assert throttle.can_mutate("pw")


class TestReindexConditionalEqualsRemoved:
    def test_condition_index_equals_removed(self):
        """When condition_step_index == removed_idx, index is unchanged (not decremented)."""
        steps = [
            StepSpec(
                step_type="conditional", target="",
                condition_step_index=2, condition_field="f",
                branches={},
            ),
        ]
        # Removing index 2 — the conditional references index 2 exactly
        result = _reindex_conditionals_after_delete(steps, 2)
        # condition_step_index == removed_idx: not > removed_idx, so unchanged
        assert result[0].condition_step_index == 2


class TestAllAdjacentPairsConstrained:
    def test_returns_none_when_all_pairs_constrained(self):
        """When all adjacent pairs have dependency constraints, reorder returns None."""
        from sg.parser.types import PathwayContract, Dependency, BlastRadius
        op = ReorderingOperator()
        steps = [
            StepSpec(step_type="locus", target="a"),
            StepSpec(step_type="locus", target="b"),
            StepSpec(step_type="locus", target="c"),
        ]
        # Each step depends on the previous (1-based indices)
        contract = PathwayContract(
            name="test_pw", risk=BlastRadius.LOW, does="test",
            requires=[Dependency(step=2, needs=1), Dependency(step=3, needs=2)],
        )
        anomalies = [TimingAnomaly(step_name="c", latest_ms=100, avg_ms=10, ratio=10.0)]
        ctx = _make_ctx(steps=steps, timing_anomalies=anomalies, contract=contract)
        result = op.apply(ctx)
        assert result is None


class TestSelectOperatorFallthrough:
    def test_can_apply_true_but_apply_returns_none(self):
        """Operator says can_apply=True but apply() returns None → tries next."""

        class FalsePositiveOp(PathwayMutationOperator):
            @property
            def name(self):
                return "false_positive"

            def can_apply(self, ctx):
                return True

            def apply(self, ctx):
                return None

        class FallbackOp(PathwayMutationOperator):
            @property
            def name(self):
                return "fallback"

            def can_apply(self, ctx):
                return True

            def apply(self, ctx):
                return PathwayMutationResult(
                    new_steps=ctx.current_steps,
                    operator_name=self.name,
                    rationale="fallback applied",
                )

        ctx = _make_ctx(steps=[StepSpec(step_type="locus", target="a")])
        result = select_operator(ctx, [FalsePositiveOp(), FallbackOp()])
        assert result is not None
        assert result.operator_name == "fallback"


class TestInsertionEdgeCases:
    def test_engine_raises_returns_none(self, tmp_path):
        """When the mutation engine raises an exception, apply() returns None."""
        from sg.mutation import MutationEngine

        class RaisingEngine(MutationEngine):
            def __init__(self):
                pass

            def propose_pathway_insertion(self, **kwargs):
                raise RuntimeError("engine failure")

            def mutate(self, source, locus, context):
                raise NotImplementedError

        op = InsertionOperator(RaisingEngine())
        steps = [
            StepSpec(step_type="locus", target="a"),
            StepSpec(step_type="locus", target="b"),
        ]
        ctx = _make_ctx(
            steps=steps,
            failure_distribution={"b": 0.6},
            per_step_fitness={"b": 0.9},
        )
        result = op.apply(ctx)
        assert result is None

    def test_engine_returns_none_proposal(self, tmp_path):
        """When the mutation engine returns None, apply() returns None."""
        from sg.mutation import MutationEngine

        class NoneEngine(MutationEngine):
            def __init__(self):
                pass

            def propose_pathway_insertion(self, **kwargs):
                return None

            def mutate(self, source, locus, context):
                raise NotImplementedError

        op = InsertionOperator(NoneEngine())
        steps = [
            StepSpec(step_type="locus", target="a"),
            StepSpec(step_type="locus", target="b"),
        ]
        ctx = _make_ctx(
            steps=steps,
            failure_distribution={"b": 0.6},
            per_step_fitness={"b": 0.9},
        )
        result = op.apply(ctx)
        assert result is None

    def test_no_qualifying_step_returns_none(self):
        """When no step meets the gap criteria, apply() returns None."""
        from sg.mutation import MockMutationEngine

        op = InsertionOperator(MockMutationEngine(Path("/tmp")))
        # All steps are non-locus (loop), so no qualifying step found
        steps = [
            StepSpec(step_type="loop", target="body",
                     loop_variable="item", loop_iterable="items"),
        ]
        ctx = _make_ctx(
            steps=steps,
            failure_distribution={"body": 0.6},
            per_step_fitness={"body": 0.9},
        )
        result = op.apply(ctx)
        assert result is None
