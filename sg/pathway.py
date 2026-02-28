"""Pathway execution — declared sequences of loci with late binding.

Pathways are loaded from .sg contracts or defined programmatically.
Execution is fusion-aware: tries the fused gene first, falls back to
decomposed steps. Input transforms are generated from {reference} syntax.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable

from sg.kernel.base import NetworkKernel
from sg.fusion import FusionTracker, fuse_pathway, try_fused_execution
from sg.mutation import MutationEngine
from sg.phenotype import PhenotypeMap
from sg.registry import Registry
from sg.parser.types import PathwayContract, PathwayStep as ASTPathwayStep


@dataclass
class PathwayStep:
    locus: str
    input_transform: Callable[[str, list[str]], str]
    """Transform (original_input, previous_outputs) -> step_input_json"""


@dataclass
class Pathway:
    name: str
    steps: list[PathwayStep]


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
        step_input = step.input_transform(input_json, outputs)
        result = orchestrator.execute_locus(step.locus, step_input)

        if result is None:
            fusion_tracker.record_failure(pathway.name)
            raise RuntimeError(
                f"pathway '{pathway.name}' failed at step {step.locus}: "
                f"all alleles exhausted"
            )

        output, used_sha = result
        outputs.append(output)
        allele_shas.append(used_sha)

    fingerprint = fusion_tracker.record_success(pathway.name, allele_shas)
    if fingerprint is not None:
        print(f"  [pathway] fusion threshold reached for '{pathway.name}'!")
        fuse_pathway(
            pathway.name,
            allele_shas,
            [step.locus for step in pathway.steps],
            registry,
            phenotype,
            mutation_engine,
        )

    return outputs


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


def pathway_from_contract(contract: PathwayContract) -> Pathway:
    """Convert a parsed PathwayContract into an executable Pathway."""
    steps: list[PathwayStep] = []
    for ast_step in contract.steps:
        if isinstance(ast_step, ASTPathwayStep) and not ast_step.is_pathway_ref:
            transform = _make_reference_transform(ast_step.params)
            steps.append(PathwayStep(
                locus=ast_step.locus,
                input_transform=transform,
            ))
    return Pathway(name=contract.name, steps=steps)
