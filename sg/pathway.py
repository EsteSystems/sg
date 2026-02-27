"""Pathway execution — declared sequences of loci with late binding.

A Pathway is an ordered list of steps (loci + input transforms). Execution
is fusion-aware: tries the fused gene first, falls back to decomposed steps.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

from sg.kernel.base import NetworkKernel
from sg.fusion import FusionTracker, fuse_pathway, try_fused_execution
from sg.mutation import MutationEngine
from sg.phenotype import PhenotypeMap
from sg.registry import Registry


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
    """Execute a pathway, returning the list of step outputs.

    1. Try fused execution first
    2. Fall back to decomposed (step-by-step) execution
    3. Track reinforcement for fusion
    """
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


# --- Built-in pathways ---

def _bridge_create_transform(original_input: str, _previous: list[str]) -> str:
    data = json.loads(original_input)
    return json.dumps({
        "bridge_name": data["bridge_name"],
        "interfaces": data["interfaces"],
    })


def _bridge_stp_transform(original_input: str, _previous: list[str]) -> str:
    data = json.loads(original_input)
    return json.dumps({
        "bridge_name": data["bridge_name"],
        "stp_enabled": data["stp_enabled"],
        "forward_delay": data["forward_delay"],
    })


CONFIGURE_BRIDGE_WITH_STP = Pathway(
    name="configure_bridge_with_stp",
    steps=[
        PathwayStep(locus="bridge_create", input_transform=_bridge_create_transform),
        PathwayStep(locus="bridge_stp", input_transform=_bridge_stp_transform),
    ],
)

PATHWAYS: dict[str, Pathway] = {
    "configure_bridge_with_stp": CONFIGURE_BRIDGE_WITH_STP,
}
