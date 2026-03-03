"""Tests for topology mutation operators."""
from sg.contracts import ContractStore
from sg.topology_registry import TopologyStepSpec
from sg.topology_mutation import (
    TopologyMutationContext,
    TopologyMutationResult,
    ActionSubstitutionOperator,
    ResourceReorderingOperator,
    ResourceEliminationOperator,
    LLMDecompositionOperator,
    select_topology_operator,
    default_topology_operators,
)


def _make_ctx(
    steps=None,
    per_resource_success=None,
    per_resource_timing=None,
    topology_fitness=0.3,
    available_pathways=None,
    available_loci=None,
    contract_store=None,
):
    return TopologyMutationContext(
        topology_name="test_topo",
        current_steps=steps or [],
        topology_fitness=topology_fitness,
        per_resource_success=per_resource_success or {},
        per_resource_timing=per_resource_timing or {},
        available_pathways=available_pathways or [],
        available_loci=available_loci or [],
        contract_store=contract_store or ContractStore(),
    )


class TestActionSubstitution:
    def test_swaps_gene_to_pathway(self):
        ctx = _make_ctx(
            steps=[
                TopologyStepSpec(resource_name="storage", action="gene",
                                 target="create_storage"),
            ],
            per_resource_success={"storage": 0.3},
            available_pathways=["create_storage"],
        )
        op = ActionSubstitutionOperator()
        assert op.name == "action_substitution"
        assert op.can_apply(ctx)
        result = op.apply(ctx)
        assert result is not None
        assert result.new_steps[0].action == "pathway"

    def test_swaps_pathway_to_gene(self):
        ctx = _make_ctx(
            steps=[
                TopologyStepSpec(resource_name="net", action="pathway",
                                 target="setup_net"),
            ],
            per_resource_success={"net": 0.2},
            available_loci=["setup_net"],
        )
        op = ActionSubstitutionOperator()
        assert op.can_apply(ctx)
        result = op.apply(ctx)
        assert result is not None
        assert result.new_steps[0].action == "gene"

    def test_no_swap_when_succeeding(self):
        ctx = _make_ctx(
            steps=[
                TopologyStepSpec(resource_name="ok", action="gene",
                                 target="thing"),
            ],
            per_resource_success={"ok": 0.9},
        )
        op = ActionSubstitutionOperator()
        assert not op.can_apply(ctx)


class TestResourceReordering:
    def test_reorders_failing_before_succeeding(self):
        ctx = _make_ctx(
            steps=[
                TopologyStepSpec(resource_name="a", action="gene", target="x"),
                TopologyStepSpec(resource_name="b", action="gene", target="y"),
            ],
            per_resource_success={"a": 0.3, "b": 0.8},
        )
        op = ResourceReorderingOperator()
        assert op.name == "resource_reordering"
        assert op.can_apply(ctx)
        result = op.apply(ctx)
        assert result is not None
        assert result.new_steps[0].resource_name == "b"
        assert result.new_steps[1].resource_name == "a"

    def test_no_reorder_single_step(self):
        ctx = _make_ctx(
            steps=[
                TopologyStepSpec(resource_name="a", action="gene", target="x"),
            ],
        )
        op = ResourceReorderingOperator()
        assert not op.can_apply(ctx)


class TestResourceElimination:
    def test_eliminates_redundant(self):
        ctx = _make_ctx(
            steps=[
                TopologyStepSpec(resource_name="a", action="gene", target="x"),
                TopologyStepSpec(resource_name="b", action="gene", target="y"),
            ],
            per_resource_success={"a": 1.0, "b": 0.5},
            per_resource_timing={"a": 0.001, "b": 0.5},
        )
        op = ResourceEliminationOperator()
        assert op.name == "resource_elimination"
        assert op.can_apply(ctx)
        result = op.apply(ctx)
        assert result is not None
        assert len(result.new_steps) == 1
        assert result.new_steps[0].resource_name == "b"

    def test_no_elimination_single_step(self):
        ctx = _make_ctx(
            steps=[
                TopologyStepSpec(resource_name="a", action="gene", target="x"),
            ],
            per_resource_success={"a": 1.0},
            per_resource_timing={"a": 0.001},
        )
        op = ResourceEliminationOperator()
        assert not op.can_apply(ctx)


class TestLLMDecomposition:
    def test_proposes_via_llm(self):
        class FakeEngine:
            def propose_pathway_insertion(self, **kwargs):
                return {
                    "locus": "verify_state",
                    "insert_before_index": 0,
                    "rationale": "add verification step",
                }

        ctx = _make_ctx(
            steps=[
                TopologyStepSpec(resource_name="a", action="gene", target="x"),
            ],
            topology_fitness=0.2,
            available_loci=["x", "verify_state"],
        )
        op = LLMDecompositionOperator(FakeEngine())
        assert op.name == "llm_decomposition"
        assert op.can_apply(ctx)
        result = op.apply(ctx)
        assert result is not None
        assert len(result.new_steps) == 2
        assert result.new_steps[0].target == "verify_state"

    def test_no_proposal_when_fit(self):
        ctx = _make_ctx(topology_fitness=0.8)
        op = LLMDecompositionOperator(object())
        assert not op.can_apply(ctx)


class TestSelectAndDefaults:
    def test_select_first_applicable(self):
        ctx = _make_ctx(
            steps=[
                TopologyStepSpec(resource_name="a", action="gene", target="x"),
                TopologyStepSpec(resource_name="b", action="gene", target="y"),
            ],
            per_resource_success={"a": 1.0, "b": 0.5},
            per_resource_timing={"a": 0.001, "b": 0.5},
        )
        ops = default_topology_operators()
        result = select_topology_operator(ctx, ops)
        assert result is not None

    def test_default_operators_counts(self):
        ops = default_topology_operators()
        assert len(ops) == 3
        ops_with_engine = default_topology_operators(mutation_engine=object())
        assert len(ops_with_engine) == 4

    def test_none_when_nothing_applies(self):
        ctx = _make_ctx(
            steps=[
                TopologyStepSpec(resource_name="a", action="gene", target="x"),
            ],
            per_resource_success={"a": 0.9},
            topology_fitness=0.9,
        )
        ops = default_topology_operators()
        result = select_topology_operator(ctx, ops)
        assert result is None
