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
    """Rate-limits pathway mutation to avoid structural oscillation.

    Checks both time-based cooldown and stabilization convergence
    when a stabilization tracker is available.
    """
    cooldown_seconds: float = DEFAULT_MUTATION_COOLDOWN_HOURS * 3600
    _last_mutation: dict[str, float] = field(default_factory=dict)
    _stabilization_tracker: object | None = field(default=None, repr=False)

    def can_mutate(self, pathway_name: str) -> bool:
        last = self._last_mutation.get(pathway_name, 0.0)
        if (time.time() - last) < self.cooldown_seconds:
            return False
        if self._stabilization_tracker is not None:
            if self._stabilization_tracker.is_stabilizing(pathway_name):
                return False
        return True

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


# --- Conditional Wrapping Operator ---


class ConditionalWrappingOperator(PathwayMutationOperator):
    """Wrap a frequently-failing step in a conditional gate.

    Signal: a step fails >40% of the time, and a diagnostic locus exists
    whose output could serve as a precondition check. The operator inserts
    the diagnostic before the failing step and makes the failing step
    conditional on the diagnostic's output.
    """

    @property
    def name(self) -> str:
        return "conditional_wrapping"

    def can_apply(self, ctx: PathwayMutationContext) -> bool:
        for step in ctx.current_steps:
            if step.step_type != "locus":
                continue
            fail_prob = ctx.failure_distribution.get(step.target, 0.0)
            if fail_prob > 0.4:
                if self._find_diagnostic_gate(step.target, ctx):
                    return True
        return False

    def apply(self, ctx: PathwayMutationContext) -> PathwayMutationResult | None:
        worst_idx, worst_step, worst_fail = None, None, 0.0
        for i, step in enumerate(ctx.current_steps):
            if step.step_type != "locus":
                continue
            fail_prob = ctx.failure_distribution.get(step.target, 0.0)
            if fail_prob > worst_fail:
                worst_fail = fail_prob
                worst_idx = i
                worst_step = step

        if worst_step is None or worst_idx is None or worst_fail <= 0.4:
            return None

        diag_locus = self._find_diagnostic_gate(worst_step.target, ctx)
        if not diag_locus:
            return None

        diag_step = StepSpec(step_type="locus", target=diag_locus)
        conditional_step = StepSpec(
            step_type="conditional",
            target=worst_step.target,
            params=worst_step.params,
            condition_step_index=worst_idx,
            condition_field="status",
            branches={"ok": worst_step.to_dict()},
        )

        new_steps = list(ctx.current_steps)
        new_steps[worst_idx] = diag_step
        new_steps.insert(worst_idx + 1, conditional_step)
        new_steps = _reindex_conditionals_after_insert(new_steps, worst_idx + 1)

        return PathwayMutationResult(
            new_steps=new_steps,
            operator_name=self.name,
            rationale=(
                f"Wrapped '{worst_step.target}' (fail={worst_fail:.0%}) "
                f"with diagnostic gate '{diag_locus}'"
            ),
        )

    def _find_diagnostic_gate(
        self, locus: str, ctx: PathwayMutationContext,
    ) -> str | None:
        """Find a diagnostic locus whose gives could gate the target."""
        target_contract = ctx.contract_store.get_gene(locus)
        if target_contract is None:
            return None

        best, best_fitness = None, -1.0
        for other in ctx.available_loci:
            if other == locus:
                continue
            other_contract = ctx.contract_store.get_gene(other)
            if other_contract is None:
                continue
            if getattr(other_contract, "family", None) != "diagnostic":
                continue
            fitness = ctx.gene_fitness_map.get(other, 0.0)
            if fitness > best_fitness:
                best = other
                best_fitness = fitness
        return best


class LoopIntroductionOperator(PathwayMutationOperator):
    """Wrap steps in a retry loop when failures are transient.

    Signal: a step fails intermittently (20-60% failure rate) but succeeds
    on retry, indicated by input clusters where the same or similar inputs
    sometimes succeed and sometimes fail.
    """

    @property
    def name(self) -> str:
        return "loop_introduction"

    def can_apply(self, ctx: PathwayMutationContext) -> bool:
        for step in ctx.current_steps:
            if step.step_type != "locus":
                continue
            if step.loop_variable is not None:
                continue
            fail_prob = ctx.failure_distribution.get(step.target, 0.0)
            gene_fit = ctx.per_step_fitness.get(step.target, 0.0)
            if 0.2 <= fail_prob <= 0.6 and gene_fit > 0.5:
                return True
        return False

    def apply(self, ctx: PathwayMutationContext) -> PathwayMutationResult | None:
        best_idx, best_step, best_score = None, None, 0.0
        for i, step in enumerate(ctx.current_steps):
            if step.step_type != "locus" or step.loop_variable is not None:
                continue
            fail_prob = ctx.failure_distribution.get(step.target, 0.0)
            gene_fit = ctx.per_step_fitness.get(step.target, 0.0)
            if 0.2 <= fail_prob <= 0.6 and gene_fit > 0.5:
                score = fail_prob * gene_fit
                if score > best_score:
                    best_score = score
                    best_idx = i
                    best_step = step

        if best_step is None or best_idx is None:
            return None

        new_steps = list(ctx.current_steps)
        new_steps[best_idx] = StepSpec(
            step_type="loop",
            target=best_step.target,
            params=best_step.params,
            loop_variable="attempt",
            loop_iterable="[1, 2, 3]",
        )
        return PathwayMutationResult(
            new_steps=new_steps,
            operator_name=self.name,
            rationale=(
                f"Wrapped '{best_step.target}' in retry loop "
                f"(fail={ctx.failure_distribution.get(best_step.target, 0):.0%}, "
                f"gene_fitness={ctx.per_step_fitness.get(best_step.target, 0):.2f})"
            ),
        )


class StepSplittingOperator(PathwayMutationOperator):
    """Split a complex step into two sequential sub-loci.

    Signal: a locus has high error diversity (many distinct error types),
    suggesting it handles too many concerns. This works at the pathway level
    by proposing that a new locus be created (via locus discovery) and the
    original step be replaced by two steps in sequence.

    Requires a mutation engine for LLM-assisted splitting proposals.
    """

    def __init__(self, mutation_engine: object) -> None:
        self._engine = mutation_engine

    @property
    def name(self) -> str:
        return "step_splitting"

    def can_apply(self, ctx: PathwayMutationContext) -> bool:
        for step in ctx.current_steps:
            if step.step_type != "locus":
                continue
            fail_prob = ctx.failure_distribution.get(step.target, 0.0)
            if fail_prob > 0.3 and len(ctx.input_clusters) >= 2:
                return True
        return False

    def apply(self, ctx: PathwayMutationContext) -> PathwayMutationResult | None:
        target_idx, target_step = None, None
        max_fail = 0.0
        for i, step in enumerate(ctx.current_steps):
            if step.step_type != "locus":
                continue
            fail_prob = ctx.failure_distribution.get(step.target, 0.0)
            if fail_prob > max_fail and fail_prob > 0.3:
                max_fail = fail_prob
                target_idx = i
                target_step = step

        if target_step is None or target_idx is None:
            return None

        try:
            proposal = self._engine.propose_pathway_insertion(
                pathway_name=ctx.pathway_name,
                steps=[s.to_dict() for s in ctx.current_steps],
                problem_step_index=target_idx,
                problem_step_name=target_step.target,
                failure_distribution=ctx.failure_distribution,
                available_loci=ctx.available_loci,
                input_clusters=[c.to_dict() for c in ctx.input_clusters],
            )
        except (NotImplementedError, Exception) as e:
            logger.warning("LLM step-splitting proposal failed: %s", e)
            return None

        if proposal is None:
            return None

        split_locus = proposal.get("locus")
        if split_locus is None or split_locus not in ctx.available_loci:
            return None

        prep_step = StepSpec(
            step_type="locus",
            target=split_locus,
            params=proposal.get("params", {}),
        )
        new_steps = list(ctx.current_steps)
        new_steps.insert(target_idx, prep_step)
        new_steps = _reindex_conditionals_after_insert(new_steps, target_idx)

        return PathwayMutationResult(
            new_steps=new_steps,
            operator_name=self.name,
            rationale=(
                f"Split '{target_step.target}' by inserting "
                f"'{split_locus}' as preparation step"
            ),
        )


class StepMergingOperator(PathwayMutationOperator):
    """Merge adjacent steps that always succeed together.

    Signal: two adjacent locus steps both have near-perfect fitness
    (>0.95) and near-zero failure probability. If a composed pathway
    or a fused gene exists that covers both, propose merging them
    into a single composed step.
    """

    @property
    def name(self) -> str:
        return "step_merging"

    def can_apply(self, ctx: PathwayMutationContext) -> bool:
        return bool(self._find_mergeable_pairs(ctx))

    def apply(self, ctx: PathwayMutationContext) -> PathwayMutationResult | None:
        pairs = self._find_mergeable_pairs(ctx)
        if not pairs:
            return None
        i, j = pairs[0]
        step_a = ctx.current_steps[i]
        step_b = ctx.current_steps[j]

        composed_name = self._find_composed_pathway(
            step_a.target, step_b.target, ctx,
        )
        if composed_name:
            merged_step = StepSpec(step_type="composed", target=composed_name)
        else:
            merged_step = StepSpec(
                step_type="composed",
                target=f"{step_a.target}+{step_b.target}",
                params={**step_a.params, **step_b.params},
            )

        new_steps = list(ctx.current_steps)
        new_steps[i] = merged_step
        del new_steps[j]
        new_steps = _reindex_conditionals_after_delete(new_steps, j)

        return PathwayMutationResult(
            new_steps=new_steps,
            operator_name=self.name,
            rationale=(
                f"Merged '{step_a.target}' + '{step_b.target}' "
                f"into composed step"
            ),
        )

    def _find_mergeable_pairs(
        self, ctx: PathwayMutationContext,
    ) -> list[tuple[int, int]]:
        pairs = []
        for i in range(len(ctx.current_steps) - 1):
            a = ctx.current_steps[i]
            b = ctx.current_steps[i + 1]
            if a.step_type != "locus" or b.step_type != "locus":
                continue
            a_fail = ctx.failure_distribution.get(a.target, 0.0)
            b_fail = ctx.failure_distribution.get(b.target, 0.0)
            a_fit = ctx.per_step_fitness.get(a.target, 0.0)
            b_fit = ctx.per_step_fitness.get(b.target, 0.0)
            if a_fail < 0.02 and b_fail < 0.02 and a_fit > 0.95 and b_fit > 0.95:
                pairs.append((i, i + 1))
        return pairs

    def _find_composed_pathway(
        self, locus_a: str, locus_b: str, ctx: PathwayMutationContext,
    ) -> str | None:
        """Check if a pathway already composes these two loci in order."""
        for pw_name in ctx.available_pathways:
            pw_contract = ctx.contract_store.get(pw_name)
            if pw_contract is None:
                continue
            steps = getattr(pw_contract, "steps", [])
            targets = [
                getattr(s, "locus", getattr(s, "target", None))
                for s in steps
            ]
            if len(targets) == 2 and targets[0] == locus_a and targets[1] == locus_b:
                return pw_name
        return None


class ParallelizationOperator(PathwayMutationOperator):
    """Mark independent steps for parallel execution.

    Signal: two or more non-adjacent steps have no data dependency between
    them (no takes/gives overlap) and both contribute significant latency.
    The operator groups them into a parallel step.
    """

    @property
    def name(self) -> str:
        return "parallelization"

    def can_apply(self, ctx: PathwayMutationContext) -> bool:
        return len(self._find_parallel_groups(ctx)) > 0

    def apply(self, ctx: PathwayMutationContext) -> PathwayMutationResult | None:
        groups = self._find_parallel_groups(ctx)
        if not groups:
            return None
        group = groups[0]
        indices = sorted(group)

        parallel_step = StepSpec(
            step_type="composed",
            target="parallel:" + ",".join(
                ctx.current_steps[i].target for i in indices
            ),
            params={"parallel": "true"},
        )

        new_steps = []
        inserted = False
        for i, step in enumerate(ctx.current_steps):
            if i in indices:
                if not inserted:
                    new_steps.append(parallel_step)
                    inserted = True
            else:
                new_steps.append(step)

        return PathwayMutationResult(
            new_steps=new_steps,
            operator_name=self.name,
            rationale=(
                f"Parallelized {len(indices)} independent steps: "
                + ", ".join(ctx.current_steps[i].target for i in indices)
            ),
        )

    def _find_parallel_groups(
        self, ctx: PathwayMutationContext,
    ) -> list[list[int]]:
        """Find groups of steps with no data dependency."""
        locus_steps = [
            (i, s) for i, s in enumerate(ctx.current_steps)
            if s.step_type == "locus"
        ]
        if len(locus_steps) < 2:
            return []

        step_gives: dict[int, set[str]] = {}
        step_takes: dict[int, set[str]] = {}
        for i, step in locus_steps:
            gc = ctx.contract_store.get_gene(step.target)
            if gc:
                step_gives[i] = {f.name for f in gc.gives} if gc.gives else set()
                step_takes[i] = {
                    f.name for f in gc.takes if f.required
                } if gc.takes else set()
            else:
                step_gives[i] = set()
                step_takes[i] = set()

        def independent(a_idx: int, b_idx: int) -> bool:
            a_g = step_gives.get(a_idx, set())
            b_t = step_takes.get(b_idx, set())
            b_g = step_gives.get(b_idx, set())
            a_t = step_takes.get(a_idx, set())
            return not (a_g & b_t) and not (b_g & a_t)

        groups: list[list[int]] = []
        used: set[int] = set()
        for ia, (a_idx, _) in enumerate(locus_steps):
            if a_idx in used:
                continue
            group = [a_idx]
            for ib in range(ia + 1, len(locus_steps)):
                b_idx = locus_steps[ib][0]
                if b_idx in used:
                    continue
                if all(independent(b_idx, g) for g in group):
                    group.append(b_idx)
            if len(group) >= 2:
                groups.append(group)
                used.update(group)
        return groups


# --- Default operator ordering ---


def default_operators(
    mutation_engine: object | None = None,
) -> list[PathwayMutationOperator]:
    """Return operators sorted by conservatism.

    Order: reorder (cheapest) → substitution → deletion →
    merging → conditional wrapping → loop introduction →
    parallelization → insertion (LLM) → step splitting (LLM).
    """
    ops: list[PathwayMutationOperator] = [
        ReorderingOperator(),
        StepSubstitutionOperator(),
        DeletionOperator(),
        StepMergingOperator(),
        ConditionalWrappingOperator(),
        LoopIntroductionOperator(),
        ParallelizationOperator(),
    ]
    if mutation_engine is not None:
        ops.append(InsertionOperator(mutation_engine))
        ops.append(StepSplittingOperator(mutation_engine))
    return ops
