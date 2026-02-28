"""Pathway execution — declared sequences of loci with late binding.

Pathways are loaded from .sg contracts or defined programmatically.
Execution is fusion-aware: tries the fused gene first, falls back to
decomposed steps. Input transforms are generated from {reference} syntax.

Supports composed pathway refs (->), for loops, requires dependencies,
and when conditionals.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable

from sg.kernel.base import NetworkKernel
from sg.fusion import FusionTracker, fuse_pathway, try_fused_execution
from sg.mutation import MutationEngine
from sg.phenotype import PhenotypeMap
from sg.registry import Registry
from sg.parser.types import (
    PathwayContract, PathwayStep as ASTPathwayStep,
    ForStep as ASTForStep, ConditionalStep as ASTConditionalStep,
    Dependency,
)


# --- Runtime step types ---

@dataclass
class PathwayStep:
    locus: str
    input_transform: Callable[[str, list[str]], str]
    """Transform (original_input, previous_outputs) -> step_input_json"""


@dataclass
class ComposedStep:
    """Execute another pathway by name."""
    pathway_name: str
    input_transform: Callable[[str, list[str]], str]


@dataclass
class LoopStep:
    """Iterate over an input array, executing the body for each element."""
    variable: str
    iterable_field: str  # field name in pathway input
    body_locus: str
    body_is_composed: bool = False
    body_params: dict[str, str] = field(default_factory=dict)


@dataclass
class ConditionalExecStep:
    """Evaluate a condition from a previous step's output and pick a branch."""
    condition_step_index: int  # 0-based index into previous outputs
    condition_field: str
    branches: dict[str, PathwayStep | ComposedStep]


# Union of all runtime step types
RuntimeStep = PathwayStep | ComposedStep | LoopStep | ConditionalExecStep


@dataclass
class Pathway:
    name: str
    steps: list[RuntimeStep]


def execute_pathway(
    pathway: Pathway,
    input_json: str,
    orchestrator: object,  # duck-typed to avoid circular import
    fusion_tracker: FusionTracker,
    registry: Registry,
    phenotype: PhenotypeMap,
    mutation_engine: MutationEngine,
    kernel: NetworkKernel,
) -> list[str]:
    """Execute a pathway, returning the list of step outputs."""
    fused_result = try_fused_execution(
        pathway.name, input_json, registry, phenotype, fusion_tracker, kernel
    )
    if fused_result is not None:
        return [fused_result]

    outputs: list[str] = []
    allele_shas: list[str] = []

    for step in pathway.steps:
        if isinstance(step, PathwayStep):
            result = _execute_locus_step(
                step, input_json, outputs, orchestrator, pathway.name,
                fusion_tracker,
            )
            output, used_sha = result
            outputs.append(output)
            allele_shas.append(used_sha)

        elif isinstance(step, ComposedStep):
            step_input = step.input_transform(input_json, outputs)
            sub_outputs = orchestrator.run_pathway(step.pathway_name, step_input)
            # Flatten composed outputs — use the last output as this step's output
            combined = sub_outputs[-1] if sub_outputs else "{}"
            outputs.append(combined)
            allele_shas.append(f"pathway:{step.pathway_name}")

        elif isinstance(step, LoopStep):
            loop_outputs = _execute_loop_step(
                step, input_json, outputs, orchestrator, pathway.name,
                fusion_tracker,
            )
            for loop_out, loop_sha in loop_outputs:
                outputs.append(loop_out)
                allele_shas.append(loop_sha)

        elif isinstance(step, ConditionalExecStep):
            result = _execute_conditional_step(
                step, input_json, outputs, orchestrator, pathway.name,
                fusion_tracker,
            )
            if result is not None:
                output, used_sha = result
                outputs.append(output)
                allele_shas.append(used_sha)

    fingerprint = fusion_tracker.record_success(pathway.name, allele_shas)
    if fingerprint is not None:
        print(f"  [pathway] fusion threshold reached for '{pathway.name}'!")
        loci = []
        for step in pathway.steps:
            if isinstance(step, PathwayStep):
                loci.append(step.locus)
            elif isinstance(step, ComposedStep):
                loci.append(step.pathway_name)
        fuse_pathway(
            pathway.name,
            allele_shas,
            loci,
            registry,
            phenotype,
            mutation_engine,
        )

    return outputs


def _execute_locus_step(
    step: PathwayStep,
    input_json: str,
    outputs: list[str],
    orchestrator: object,
    pathway_name: str,
    fusion_tracker: FusionTracker,
) -> tuple[str, str]:
    """Execute a single locus step. Returns (output, sha)."""
    step_input = step.input_transform(input_json, outputs)
    result = orchestrator.execute_locus(step.locus, step_input)

    if result is None:
        fusion_tracker.record_failure(pathway_name)
        raise RuntimeError(
            f"pathway '{pathway_name}' failed at step {step.locus}: "
            f"all alleles exhausted"
        )

    return result


def _execute_loop_step(
    step: LoopStep,
    input_json: str,
    outputs: list[str],
    orchestrator: object,
    pathway_name: str,
    fusion_tracker: FusionTracker,
) -> list[tuple[str, str]]:
    """Execute a for-loop step. Returns list of (output, sha) pairs."""
    data = json.loads(input_json)
    iterable = data.get(step.iterable_field, [])
    if not isinstance(iterable, list):
        raise RuntimeError(
            f"pathway '{pathway_name}' for-loop: "
            f"'{step.iterable_field}' is not a list"
        )

    results: list[tuple[str, str]] = []
    for element in iterable:
        # Build step input: merge pathway input with loop variable
        step_data = dict(data)
        step_data[step.variable] = element

        # Apply body params as transforms
        if step.body_params:
            resolved = {}
            for pname, pval in step.body_params.items():
                match = re.fullmatch(r"\{(\w+)\}", pval)
                if match:
                    key = match.group(1)
                    if key in step_data:
                        resolved[pname] = step_data[key]
                else:
                    resolved[pname] = pval
            step_input = json.dumps(resolved)
        else:
            step_input = json.dumps(step_data)

        if step.body_is_composed:
            sub_outputs = orchestrator.run_pathway(step.body_locus, step_input)
            combined = sub_outputs[-1] if sub_outputs else "{}"
            results.append((combined, f"pathway:{step.body_locus}"))
        else:
            result = orchestrator.execute_locus(step.body_locus, step_input)
            if result is None:
                fusion_tracker.record_failure(pathway_name)
                raise RuntimeError(
                    f"pathway '{pathway_name}' for-loop failed at "
                    f"{step.body_locus}: all alleles exhausted"
                )
            results.append(result)

    return results


def _execute_conditional_step(
    step: ConditionalExecStep,
    input_json: str,
    outputs: list[str],
    orchestrator: object,
    pathway_name: str,
    fusion_tracker: FusionTracker,
) -> tuple[str, str] | None:
    """Execute a conditional step based on a previous step's output."""
    if step.condition_step_index >= len(outputs):
        raise RuntimeError(
            f"pathway '{pathway_name}' when-step references step "
            f"{step.condition_step_index + 1} but only "
            f"{len(outputs)} step(s) completed"
        )

    prev_output = outputs[step.condition_step_index]
    try:
        prev_data = json.loads(prev_output)
    except (json.JSONDecodeError, TypeError):
        return None

    value = prev_data.get(step.condition_field)
    if value is None:
        return None

    # Convert value to string for branch matching
    value_str = str(value).lower() if isinstance(value, bool) else str(value)

    branch = step.branches.get(value_str)
    if branch is None:
        return None

    if isinstance(branch, ComposedStep):
        step_input = branch.input_transform(input_json, outputs)
        sub_outputs = orchestrator.run_pathway(branch.pathway_name, step_input)
        combined = sub_outputs[-1] if sub_outputs else "{}"
        return (combined, f"pathway:{branch.pathway_name}")
    else:
        return _execute_locus_step(
            branch, input_json, outputs, orchestrator, pathway_name,
            fusion_tracker,
        )


# --- Input transform generation from {reference} syntax ---

def _make_reference_transform(params: dict[str, str]) -> Callable[[str, list[str]], str]:
    """Generate an input transform function from parameter bindings.

    params maps step param names to values. Values like {bridge_name}
    are resolved from the pathway's original input.
    """
    def transform(original_input: str, _previous: list[str]) -> str:
        data = json.loads(original_input)
        result = {}
        for param_name, param_value in params.items():
            # {reference} — resolve from pathway input
            match = re.fullmatch(r"\{(\w+)\}", param_value)
            if match:
                key = match.group(1)
                if key in data:
                    result[param_name] = data[key]
            else:
                # Literal value
                result[param_name] = param_value
        return json.dumps(result)
    return transform


def _make_passthrough_transform() -> Callable[[str, list[str]], str]:
    """Pass original input through unchanged."""
    def transform(original_input: str, _previous: list[str]) -> str:
        return original_input
    return transform


def pathway_from_contract(contract: PathwayContract) -> Pathway:
    """Convert a parsed PathwayContract into an executable Pathway."""
    # Validate dependency ordering
    _validate_dependencies(contract.steps, contract.requires)

    steps: list[RuntimeStep] = []
    for ast_step in contract.steps:
        if isinstance(ast_step, ASTPathwayStep):
            if ast_step.is_pathway_ref:
                transform = _make_reference_transform(ast_step.params)
                steps.append(ComposedStep(
                    pathway_name=ast_step.locus,
                    input_transform=transform,
                ))
            else:
                transform = _make_reference_transform(ast_step.params)
                steps.append(PathwayStep(
                    locus=ast_step.locus,
                    input_transform=transform,
                ))

        elif isinstance(ast_step, ASTForStep):
            steps.append(LoopStep(
                variable=ast_step.variable,
                iterable_field=ast_step.iterable,
                body_locus=ast_step.body.locus if ast_step.body else "",
                body_is_composed=ast_step.body.is_pathway_ref if ast_step.body else False,
                body_params=ast_step.body.params if ast_step.body else {},
            ))

        elif isinstance(ast_step, ASTConditionalStep):
            branches: dict[str, PathwayStep | ComposedStep] = {}
            for value, branch_ast in ast_step.branches.items():
                if branch_ast.is_pathway_ref:
                    branches[value] = ComposedStep(
                        pathway_name=branch_ast.locus,
                        input_transform=_make_reference_transform(branch_ast.params),
                    )
                else:
                    branches[value] = PathwayStep(
                        locus=branch_ast.locus,
                        input_transform=_make_reference_transform(branch_ast.params),
                    )
            steps.append(ConditionalExecStep(
                condition_step_index=ast_step.condition_step - 1,  # 1-based → 0-based
                condition_field=ast_step.condition_field,
                branches=branches,
            ))

    return Pathway(name=contract.name, steps=steps)


def _validate_dependencies(
    steps: list, requires: list[Dependency]
) -> None:
    """Validate that step dependencies are satisfied by step ordering."""
    step_indices = {s.index for s in steps if hasattr(s, 'index')}
    for dep in requires:
        if dep.step not in step_indices:
            raise ValueError(f"requires references unknown step {dep.step}")
        if dep.needs not in step_indices:
            raise ValueError(f"requires references unknown step {dep.needs}")
        if dep.needs >= dep.step:
            raise ValueError(
                f"step {dep.step} requires step {dep.needs} but "
                f"it appears after or at the same position"
            )
