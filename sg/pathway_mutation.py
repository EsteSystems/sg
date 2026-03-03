"""Pathway mutation operators — structural evolution for pathways.

When pathway fitness degrades but gene fitness stays high, the problem is
structural (wrong step order, missing steps, unnecessary steps). Mutation
operators propose new step arrangements, registered as recessive pathway
alleles that compete through the existing allele stack.
"""
from __future__ import annotations

import itertools
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from sg.contracts import ContractStore, contracts_compatible
from sg.log import get_logger
from sg.pathway_fitness import TimingAnomaly, InputCluster
from sg.pathway_registry import StepSpec

logger = get_logger("pathway_mutation")


DEFAULT_MUTATION_COOLDOWN_HOURS = 4


# --- Data structures ---


@dataclass
class PathwayMutationContext:
    """All information an operator needs to decide on a mutation."""
    pathway_name: str
    current_steps: list[StepSpec]
    pathway_fitness: float
    per_step_fitness: dict[str, float]
    timing_anomalies: list[TimingAnomaly]
    failure_distribution: dict[str, float]
    input_clusters: list[InputCluster]
    available_loci: list[str]
    available_pathways: list[str]
    gene_fitness_map: dict[str, float]
    contract: object | None  # PathwayContract
    contract_store: ContractStore


@dataclass
class PathwayMutationResult:
    """Output of a successful mutation operator."""
    new_steps: list[StepSpec]
    operator_name: str
    rationale: str


# --- Rate limiter ---


@dataclass
class PathwayMutationThrottle:
    """Rate-limits pathway mutation to avoid structural oscillation."""
    cooldown_seconds: float = DEFAULT_MUTATION_COOLDOWN_HOURS * 3600
    _last_mutation: dict[str, float] = field(default_factory=dict)

    def can_mutate(self, pathway_name: str) -> bool:
        last = self._last_mutation.get(pathway_name, 0.0)
        return (time.time() - last) >= self.cooldown_seconds

    def record_mutation(self, pathway_name: str) -> None:
        self._last_mutation[pathway_name] = time.time()

    def reset_cooldown(self, pathway_name: str) -> None:
        """Clear cooldown after a promotion makes the old timer irrelevant."""
        self._last_mutation.pop(pathway_name, None)

    def to_dict(self) -> dict:
        return {
            "cooldown_seconds": self.cooldown_seconds,
            "last_mutation": dict(self._last_mutation),
        }

    @classmethod
    def from_dict(cls, d: dict) -> PathwayMutationThrottle:
        throttle = cls(
            cooldown_seconds=d.get(
                "cooldown_seconds", DEFAULT_MUTATION_COOLDOWN_HOURS * 3600
            ),
        )
        throttle._last_mutation = d.get("last_mutation", {})
        return throttle


# --- Operator ABC ---


class PathwayMutationOperator(ABC):
    """Base class for pathway mutation operators."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def can_apply(self, ctx: PathwayMutationContext) -> bool: ...

    @abstractmethod
    def apply(self, ctx: PathwayMutationContext) -> PathwayMutationResult | None: ...


def select_operator(
    ctx: PathwayMutationContext,
    operators: list[PathwayMutationOperator],
) -> PathwayMutationResult | None:
    """Try each operator in priority order. Returns the first successful result."""
    for op in operators:
        if op.can_apply(ctx):
            logger.info("pathway mutation operator '%s' can apply to '%s'",
                        op.name, ctx.pathway_name)
            result = op.apply(ctx)
            if result is not None:
                logger.info("operator '%s' produced mutation for '%s': %s",
                            op.name, ctx.pathway_name, result.rationale)
                return result
            logger.info("operator '%s' returned None for '%s'",
                        op.name, ctx.pathway_name)
    return None


# --- Conditional reindexing helpers ---


def _reindex_conditionals_after_delete(
    steps: list[StepSpec], removed_idx: int,
) -> list[StepSpec]:
    """Adjust condition_step_index references after a deletion."""
    result = []
    for step in steps:
        if step.step_type == "conditional" and step.condition_step_index is not None:
            new_idx = step.condition_step_index
            if step.condition_step_index > removed_idx:
                new_idx -= 1
            result.append(StepSpec(
                step_type=step.step_type, target=step.target,
                params=step.params,
                loop_variable=step.loop_variable,
                loop_iterable=step.loop_iterable,
                condition_step_index=new_idx,
                condition_field=step.condition_field,
                branches=step.branches,
            ))
        else:
            result.append(step)
    return result


def _reindex_conditionals_after_insert(
    steps: list[StepSpec], insert_idx: int,
) -> list[StepSpec]:
    """Adjust condition_step_index references after an insertion."""
    result = []
    for step in steps:
        if step.step_type == "conditional" and step.condition_step_index is not None:
            new_idx = step.condition_step_index
            if step.condition_step_index >= insert_idx:
                new_idx += 1
            result.append(StepSpec(
                step_type=step.step_type, target=step.target,
                params=step.params,
                loop_variable=step.loop_variable,
                loop_iterable=step.loop_iterable,
                condition_step_index=new_idx,
                condition_field=step.condition_field,
                branches=step.branches,
            ))
        else:
            result.append(step)
    return result


# --- Rule-Based Operators ---


class ReorderingOperator(PathwayMutationOperator):
    """Reorder steps while respecting dependency constraints.

    Signal: timing anomalies or persistent step-level failures despite
    high gene fitness at that step.
    """

    @property
    def name(self) -> str:
        return "reorder"

    def can_apply(self, ctx: PathwayMutationContext) -> bool:
        if len(ctx.current_steps) < 2:
            return False
        # Only consider locus steps for reordering signals
        if ctx.timing_anomalies:
            return True
        for step_name, fail_prob in ctx.failure_distribution.items():
            gene_fit = ctx.per_step_fitness.get(step_name, 0.0)
            if fail_prob > 0.3 and gene_fit > 0.7:
                return True
        return False

    def apply(self, ctx: PathwayMutationContext) -> PathwayMutationResult | None:
        deps = self._extract_dependencies(ctx)
        candidates = self._valid_permutations(ctx.current_steps, deps, limit=1)
        if not candidates:
            return None
        return PathwayMutationResult(
            new_steps=candidates[0],
            operator_name=self.name,
            rationale=self._build_rationale(ctx),
        )

    def _extract_dependencies(
        self, ctx: PathwayMutationContext,
    ) -> list[tuple[int, int]]:
        """Extract ordering constraints from requires + takes/gives data flow."""
        deps: list[tuple[int, int]] = []

        # Explicit requires from contract
        if ctx.contract is not None and hasattr(ctx.contract, "requires"):
            for req in ctx.contract.requires:
                # Contract uses 1-based step numbers, convert to 0-based
                deps.append((req.needs - 1, req.step - 1))

        # Implicit data-flow from takes/gives
        step_gives: dict[int, set[str]] = {}
        for i, step in enumerate(ctx.current_steps):
            if step.step_type == "locus":
                gene_contract = ctx.contract_store.get_gene(step.target)
                if gene_contract and gene_contract.gives:
                    step_gives[i] = {f.name for f in gene_contract.gives}

        for i, step in enumerate(ctx.current_steps):
            if step.step_type == "locus":
                gene_contract = ctx.contract_store.get_gene(step.target)
                if gene_contract and gene_contract.takes:
                    needed = {f.name for f in gene_contract.takes if f.required}
                    for j, gives_set in step_gives.items():
                        if j < i and needed & gives_set:
                            deps.append((j, i))

        return deps

    def _valid_permutations(
        self,
        steps: list[StepSpec],
        deps: list[tuple[int, int]],
        limit: int,
    ) -> list[list[StepSpec]]:
        """Generate valid permutations respecting dependency ordering."""
        n = len(steps)
        if n > 6:
            return self._adjacent_swap_candidates(steps, deps, limit)

        dep_set = set(deps)
        results: list[list[StepSpec]] = []
        for perm in itertools.permutations(range(n)):
            if list(perm) == list(range(n)):
                continue
            pos = {step_idx: perm_pos for perm_pos, step_idx in enumerate(perm)}
            valid = True
            for before, after in dep_set:
                if pos[before] >= pos[after]:
                    valid = False
                    break
            if valid:
                results.append([steps[i] for i in perm])
                if len(results) >= limit:
                    break
        return results

    def _adjacent_swap_candidates(
        self,
        steps: list[StepSpec],
        deps: list[tuple[int, int]],
        limit: int,
    ) -> list[list[StepSpec]]:
        """For pathways with >6 steps, only try swapping adjacent pairs."""
        dep_set = set(deps)
        results: list[list[StepSpec]] = []
        for i in range(len(steps) - 1):
            if (i, i + 1) not in dep_set and (i + 1, i) not in dep_set:
                new_steps = list(steps)
                new_steps[i], new_steps[i + 1] = new_steps[i + 1], new_steps[i]
                results.append(new_steps)
                if len(results) >= limit:
                    break
        return results

    def _build_rationale(self, ctx: PathwayMutationContext) -> str:
        parts = []
        if ctx.timing_anomalies:
            names = [a.step_name for a in ctx.timing_anomalies[:3]]
            parts.append(f"timing anomalies at {', '.join(names)}")
        hotspots = [
            (s, p) for s, p in ctx.failure_distribution.items() if p > 0.3
        ]
        if hotspots:
            names = [s for s, _ in hotspots[:3]]
            parts.append(f"failure hotspots at {', '.join(names)}")
        return "Reordered steps: " + "; ".join(parts) if parts else "Reordered steps"


class DeletionOperator(PathwayMutationOperator):
    """Remove trivial (near-zero failure, perfect fitness) steps.

    Signal: a step that consistently succeeds with perfect gene fitness,
    never fails, and has no unique dependents.
    """

    @property
    def name(self) -> str:
        return "deletion"

    def can_apply(self, ctx: PathwayMutationContext) -> bool:
        if len(ctx.current_steps) <= 1:
            return False
        return len(self._find_trivial_steps(ctx)) > 0

    def apply(self, ctx: PathwayMutationContext) -> PathwayMutationResult | None:
        trivial = self._find_trivial_steps(ctx)
        if not trivial:
            return None
        target_idx = trivial[0]
        target_step = ctx.current_steps[target_idx]
        new_steps = [s for i, s in enumerate(ctx.current_steps) if i != target_idx]
        new_steps = _reindex_conditionals_after_delete(new_steps, target_idx)
        return PathwayMutationResult(
            new_steps=new_steps,
            operator_name=self.name,
            rationale=(
                f"Removed trivial step '{target_step.target}' at index {target_idx}"
            ),
        )

    def _find_trivial_steps(self, ctx: PathwayMutationContext) -> list[int]:
        """Return indices of trivial steps (most trivial first)."""
        trivial = []
        for i, step in enumerate(ctx.current_steps):
            if step.step_type != "locus":
                continue
            fail_prob = ctx.failure_distribution.get(step.target, 0.0)
            gene_fitness = ctx.per_step_fitness.get(step.target, 0.0)
            if fail_prob > 0.05:
                continue
            if gene_fitness < 0.95:
                continue
            if self._has_unique_dependents(i, ctx):
                continue
            trivial.append(i)
        return trivial

    def _has_unique_dependents(
        self, step_idx: int, ctx: PathwayMutationContext,
    ) -> bool:
        """Check if any later step requires output fields only this step gives."""
        step = ctx.current_steps[step_idx]
        if step.step_type != "locus":
            return False
        gene = ctx.contract_store.get_gene(step.target)
        if gene is None or not gene.gives:
            return False
        gives_fields = {f.name for f in gene.gives}

        other_gives: set[str] = set()
        for i, s in enumerate(ctx.current_steps):
            if i == step_idx or s.step_type != "locus":
                continue
            other_gene = ctx.contract_store.get_gene(s.target)
            if other_gene and other_gene.gives:
                other_gives.update(f.name for f in other_gene.gives)

        unique_fields = gives_fields - other_gives
        if not unique_fields:
            return False

        for i, s in enumerate(ctx.current_steps):
            if i <= step_idx or s.step_type != "locus":
                continue
            later_gene = ctx.contract_store.get_gene(s.target)
            if later_gene and later_gene.takes:
                needed = {f.name for f in later_gene.takes if f.required}
                if needed & unique_fields:
                    return True
        return False


class StepSubstitutionOperator(PathwayMutationOperator):
    """Swap a failing locus for a compatible one.

    Signal: a locus at a step consistently fails (gene_fitness < 0.3,
    fail_prob > 0.5) and another locus with compatible takes/gives exists.
    """

    @property
    def name(self) -> str:
        return "substitution"

    def can_apply(self, ctx: PathwayMutationContext) -> bool:
        for step in ctx.current_steps:
            if step.step_type != "locus":
                continue
            gene_fit = ctx.per_step_fitness.get(step.target, 0.0)
            fail_prob = ctx.failure_distribution.get(step.target, 0.0)
            if gene_fit < 0.3 and fail_prob > 0.5:
                if self._find_compatible_loci(step.target, ctx):
                    return True
        return False

    def apply(self, ctx: PathwayMutationContext) -> PathwayMutationResult | None:
        for i, step in enumerate(ctx.current_steps):
            if step.step_type != "locus":
                continue
            gene_fit = ctx.per_step_fitness.get(step.target, 0.0)
            fail_prob = ctx.failure_distribution.get(step.target, 0.0)
            if gene_fit < 0.3 and fail_prob > 0.5:
                compatible = self._find_compatible_loci(step.target, ctx)
                if not compatible:
                    continue
                best = max(
                    compatible, key=lambda l: ctx.gene_fitness_map.get(l, 0.0),
                )
                new_steps = list(ctx.current_steps)
                new_steps[i] = StepSpec(
                    step_type="locus", target=best, params=step.params,
                )
                return PathwayMutationResult(
                    new_steps=new_steps,
                    operator_name=self.name,
                    rationale=(
                        f"Substituted '{step.target}' (fitness={gene_fit:.2f}) "
                        f"with compatible '{best}' "
                        f"(fitness={ctx.gene_fitness_map.get(best, 0.0):.2f})"
                    ),
                )
        return None

    def _find_compatible_loci(
        self, locus: str, ctx: PathwayMutationContext,
    ) -> list[str]:
        """Find loci with compatible takes/gives signatures."""
        source = ctx.contract_store.get_gene(locus)
        if source is None:
            return []
        compatible = []
        for other in ctx.available_loci:
            if other == locus:
                continue
            other_contract = ctx.contract_store.get_gene(other)
            if other_contract is None:
                continue
            if contracts_compatible(source, other_contract):
                compatible.append(other)
        return compatible


# --- LLM-Assisted Operator ---


class InsertionOperator(PathwayMutationOperator):
    """LLM proposes where to insert a locus when a step fails persistently.

    Signal: persistent failure at a step despite high gene fitness —
    preconditions from the previous step are insufficient.
    """

    def __init__(self, mutation_engine: object) -> None:
        self._engine = mutation_engine

    @property
    def name(self) -> str:
        return "insertion"

    def can_apply(self, ctx: PathwayMutationContext) -> bool:
        for step_name, fail_prob in ctx.failure_distribution.items():
            gene_fit = ctx.per_step_fitness.get(step_name, 0.0)
            if fail_prob > 0.4 and gene_fit > 0.7:
                return True
        return False

    def apply(self, ctx: PathwayMutationContext) -> PathwayMutationResult | None:
        # Find the most problematic step
        problem_step = None
        problem_idx = None
        max_gap = 0.0
        for i, step in enumerate(ctx.current_steps):
            if step.step_type != "locus":
                continue
            fail_prob = ctx.failure_distribution.get(step.target, 0.0)
            gene_fit = ctx.per_step_fitness.get(step.target, 0.0)
            gap = fail_prob - (1.0 - gene_fit)
            if gap > max_gap:
                max_gap = gap
                problem_step = step
                problem_idx = i

        if problem_step is None or problem_idx is None:
            return None

        try:
            proposal = self._engine.propose_pathway_insertion(
                pathway_name=ctx.pathway_name,
                steps=[s.to_dict() for s in ctx.current_steps],
                problem_step_index=problem_idx,
                problem_step_name=problem_step.target,
                failure_distribution=ctx.failure_distribution,
                available_loci=ctx.available_loci,
                input_clusters=[c.to_dict() for c in ctx.input_clusters],
            )
        except (NotImplementedError, Exception) as e:
            logger.warning("LLM insertion proposal failed: %s", e)
            return None

        if proposal is None:
            return None

        insert_idx = proposal.get("insert_before_index", problem_idx)
        insert_locus = proposal.get("locus")
        if insert_locus is None or insert_locus not in ctx.available_loci:
            logger.warning("LLM proposed unknown locus '%s'", insert_locus)
            return None

        new_step = StepSpec(
            step_type="locus",
            target=insert_locus,
            params=proposal.get("params", {}),
        )
        new_steps = list(ctx.current_steps)
        new_steps.insert(insert_idx, new_step)
        new_steps = _reindex_conditionals_after_insert(new_steps, insert_idx)

        return PathwayMutationResult(
            new_steps=new_steps,
            operator_name=self.name,
            rationale=proposal.get(
                "rationale",
                f"Inserted '{insert_locus}' before step {insert_idx}",
            ),
        )


# --- Default operator ordering ---


def default_operators(
    mutation_engine: object | None = None,
) -> list[PathwayMutationOperator]:
    """Return operators sorted by conservatism.

    Order: reorder (cheapest) → substitution → deletion → insertion (LLM).
    """
    ops: list[PathwayMutationOperator] = [
        ReorderingOperator(),
        StepSubstitutionOperator(),
        DeletionOperator(),
    ]
    if mutation_engine is not None:
        ops.append(InsertionOperator(mutation_engine))
    return ops
