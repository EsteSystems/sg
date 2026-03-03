"""Topology mutation operators — structural evolution for decomposition strategies.

When a topology's overall fitness degrades but individual pathway/gene fitness
stays high, the problem is in the decomposition strategy: wrong resource ordering,
suboptimal action mapping (pathway vs. gene), or missing/redundant resources.

Mutation operators propose alternative TopologyStepSpec arrangements,
registered as recessive topology alleles.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from sg.contracts import ContractStore
from sg.log import get_logger
from sg.topology_registry import TopologyStepSpec

logger = get_logger("topology_mutation")


@dataclass
class TopologyMutationContext:
    """All information an operator needs to decide on a topology mutation."""
    topology_name: str
    current_steps: list[TopologyStepSpec]
    topology_fitness: float
    per_resource_success: dict[str, float]
    per_resource_timing: dict[str, float]
    available_pathways: list[str]
    available_loci: list[str]
    contract_store: ContractStore


@dataclass
class TopologyMutationResult:
    """Output of a successful topology mutation operator."""
    new_steps: list[TopologyStepSpec]
    operator_name: str
    rationale: str


class TopologyMutationOperator(ABC):
    """Base class for topology mutation operators."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def can_apply(self, ctx: TopologyMutationContext) -> bool: ...

    @abstractmethod
    def apply(self, ctx: TopologyMutationContext) -> TopologyMutationResult | None: ...


def select_topology_operator(
    ctx: TopologyMutationContext,
    operators: list[TopologyMutationOperator],
) -> TopologyMutationResult | None:
    """Try each operator in priority order. Returns the first successful result."""
    for op in operators:
        if op.can_apply(ctx):
            logger.info("topology operator '%s' can apply to '%s'",
                        op.name, ctx.topology_name)
            result = op.apply(ctx)
            if result is not None:
                logger.info("operator '%s' produced mutation for '%s': %s",
                            op.name, ctx.topology_name, result.rationale)
                return result
    return None


class ActionSubstitutionOperator(TopologyMutationOperator):
    """Swap a gene action for a pathway or vice versa.

    Signal: a resource mapped to a single gene call fails frequently,
    but a pathway exists that covers the same resource type.
    """

    @property
    def name(self) -> str:
        return "action_substitution"

    def can_apply(self, ctx: TopologyMutationContext) -> bool:
        for step in ctx.current_steps:
            success = ctx.per_resource_success.get(step.resource_name, 1.0)
            if success < 0.5:
                if step.action == "gene" and step.target in ctx.available_pathways:
                    return True
                if step.action == "pathway" and step.target in ctx.available_loci:
                    return True
        return False

    def apply(self, ctx: TopologyMutationContext) -> TopologyMutationResult | None:
        new_steps = list(ctx.current_steps)
        for i, step in enumerate(new_steps):
            success = ctx.per_resource_success.get(step.resource_name, 1.0)
            if success >= 0.5:
                continue
            if step.action == "gene" and step.target in ctx.available_pathways:
                new_steps[i] = TopologyStepSpec(
                    resource_name=step.resource_name,
                    action="pathway",
                    target=step.target,
                    loop_target_count=step.loop_target_count,
                )
                return TopologyMutationResult(
                    new_steps=new_steps,
                    operator_name=self.name,
                    rationale=(
                        f"Switched '{step.resource_name}' from gene to pathway "
                        f"(success={success:.0%})"
                    ),
                )
            if step.action == "pathway" and step.target in ctx.available_loci:
                new_steps[i] = TopologyStepSpec(
                    resource_name=step.resource_name,
                    action="gene",
                    target=step.target,
                    loop_target_count=step.loop_target_count,
                )
                return TopologyMutationResult(
                    new_steps=new_steps,
                    operator_name=self.name,
                    rationale=(
                        f"Switched '{step.resource_name}' from pathway to gene "
                        f"(success={success:.0%})"
                    ),
                )
        return None


class ResourceReorderingOperator(TopologyMutationOperator):
    """Reorder resources when the current order causes cascading failures.

    Signal: an early resource has low success rate and later resources
    fail as a consequence. Swaps adjacent independent resources.
    """

    @property
    def name(self) -> str:
        return "resource_reordering"

    def can_apply(self, ctx: TopologyMutationContext) -> bool:
        if len(ctx.current_steps) < 2:
            return False
        for i in range(len(ctx.current_steps) - 1):
            a = ctx.current_steps[i]
            b = ctx.current_steps[i + 1]
            a_success = ctx.per_resource_success.get(a.resource_name, 1.0)
            b_success = ctx.per_resource_success.get(b.resource_name, 1.0)
            if a_success < 0.5 < b_success and not self._depends_on(b, a, ctx):
                return True
        return False

    def apply(self, ctx: TopologyMutationContext) -> TopologyMutationResult | None:
        new_steps = list(ctx.current_steps)
        for i in range(len(new_steps) - 1):
            a = new_steps[i]
            b = new_steps[i + 1]
            a_success = ctx.per_resource_success.get(a.resource_name, 1.0)
            b_success = ctx.per_resource_success.get(b.resource_name, 1.0)
            if a_success < 0.5 < b_success and not self._depends_on(b, a, ctx):
                new_steps[i], new_steps[i + 1] = new_steps[i + 1], new_steps[i]
                return TopologyMutationResult(
                    new_steps=new_steps,
                    operator_name=self.name,
                    rationale=(
                        f"Reordered: moved '{b.resource_name}' "
                        f"(success={b_success:.0%}) before "
                        f"'{a.resource_name}' (success={a_success:.0%})"
                    ),
                )
        return None

    def _depends_on(
        self,
        step_a: TopologyStepSpec,
        step_b: TopologyStepSpec,
        ctx: TopologyMutationContext,
    ) -> bool:
        """Check if step_a depends on step_b's output."""
        topo = ctx.contract_store.topologies.get(ctx.topology_name)
        if topo is None:
            return False
        resource_names = {r.name for r in topo.has}
        a_res = next((r for r in topo.has if r.name == step_a.resource_name), None)
        if a_res is None:
            return False
        for val in a_res.properties.values():
            if val == step_b.resource_name and val in resource_names:
                return True
        return False


class ResourceEliminationOperator(TopologyMutationOperator):
    """Remove a redundant resource step.

    Signal: a resource step has perfect success and zero timing contribution,
    meaning its work is already handled elsewhere.
    """

    @property
    def name(self) -> str:
        return "resource_elimination"

    def can_apply(self, ctx: TopologyMutationContext) -> bool:
        if len(ctx.current_steps) <= 1:
            return False
        for step in ctx.current_steps:
            success = ctx.per_resource_success.get(step.resource_name, 0.0)
            timing = ctx.per_resource_timing.get(step.resource_name, 1.0)
            if success > 0.99 and timing < 0.01:
                return True
        return False

    def apply(self, ctx: TopologyMutationContext) -> TopologyMutationResult | None:
        for i, step in enumerate(ctx.current_steps):
            success = ctx.per_resource_success.get(step.resource_name, 0.0)
            timing = ctx.per_resource_timing.get(step.resource_name, 1.0)
            if success > 0.99 and timing < 0.01:
                new_steps = [s for j, s in enumerate(ctx.current_steps) if j != i]
                return TopologyMutationResult(
                    new_steps=new_steps,
                    operator_name=self.name,
                    rationale=(
                        f"Eliminated redundant resource '{step.resource_name}' "
                        f"(zero-cost, perfect success)"
                    ),
                )
        return None


class LLMDecompositionOperator(TopologyMutationOperator):
    """Ask the LLM for an alternative decomposition strategy.

    Signal: overall topology fitness is low but no rule-based operator
    can identify a clear fix.
    """

    def __init__(self, mutation_engine: object) -> None:
        self._engine = mutation_engine

    @property
    def name(self) -> str:
        return "llm_decomposition"

    def can_apply(self, ctx: TopologyMutationContext) -> bool:
        return ctx.topology_fitness < 0.5

    def apply(self, ctx: TopologyMutationContext) -> TopologyMutationResult | None:
        try:
            proposal = self._engine.propose_pathway_insertion(
                pathway_name=f"topology:{ctx.topology_name}",
                steps=[s.to_dict() for s in ctx.current_steps],
                problem_step_index=0,
                problem_step_name=ctx.topology_name,
                failure_distribution={
                    s.resource_name: 1.0 - ctx.per_resource_success.get(
                        s.resource_name, 0.0)
                    for s in ctx.current_steps
                },
                available_loci=ctx.available_loci,
                input_clusters=[],
            )
        except (NotImplementedError, Exception) as e:
            logger.warning("LLM decomposition proposal failed: %s", e)
            return None

        if proposal is None:
            return None

        locus = proposal.get("locus")
        if locus is None or locus not in ctx.available_loci:
            return None

        insert_idx = proposal.get("insert_before_index", 0)
        new_step = TopologyStepSpec(
            resource_name=f"llm_{locus}",
            action="gene",
            target=locus,
        )
        new_steps = list(ctx.current_steps)
        new_steps.insert(insert_idx, new_step)

        return TopologyMutationResult(
            new_steps=new_steps,
            operator_name=self.name,
            rationale=proposal.get(
                "rationale",
                f"LLM proposed inserting '{locus}' for topology improvement",
            ),
        )


def default_topology_operators(
    mutation_engine: object | None = None,
) -> list[TopologyMutationOperator]:
    """Return topology operators sorted by conservatism."""
    ops: list[TopologyMutationOperator] = [
        ResourceReorderingOperator(),
        ActionSubstitutionOperator(),
        ResourceEliminationOperator(),
    ]
    if mutation_engine is not None:
        ops.append(LLMDecompositionOperator(mutation_engine))
    return ops
