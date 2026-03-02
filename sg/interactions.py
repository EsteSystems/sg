"""Cross-locus interaction detection.

Before promoting an allele, verify it doesn't break pathways that
include its locus. Policy controlled by SG_INTERACTION_POLICY env var:
  - "rollback" (default): reject promotion if interactions break
  - "mutate": allow promotion, log broken pathways
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

from sg.contracts import ContractStore
from sg.log import get_logger
from sg.parser.types import (
    PathwayContract, PathwayStep as ASTPathwayStep,
    ForStep, ConditionalStep, FieldDef,
)

logger = get_logger("interactions")

INTERACTION_POLICY = os.environ.get("SG_INTERACTION_POLICY", "rollback")


@dataclass
class InteractionFailure:
    """Records a pathway that broke due to an allele change."""
    pathway_name: str
    failing_step: str
    error: str


def find_affected_pathways(
    locus: str, contract_store: ContractStore,
) -> list[str]:
    """Return pathway names that include the given locus in their steps."""
    affected = []
    for pw_name in contract_store.known_pathways():
        pw = contract_store.get_pathway(pw_name)
        if pw is None:
            continue
        if _pathway_uses_locus(pw, locus, contract_store):
            affected.append(pw_name)
    return affected


def _pathway_uses_locus(
    pw: PathwayContract, locus: str, contract_store: ContractStore,
) -> bool:
    """Check if a pathway directly or transitively references a locus."""
    for step in pw.steps:
        if isinstance(step, ASTPathwayStep):
            if step.is_pathway_ref:
                sub = contract_store.get_pathway(step.locus)
                if sub and _pathway_uses_locus(sub, locus, contract_store):
                    return True
            elif step.locus == locus:
                return True
        elif isinstance(step, ForStep):
            if step.body is not None and step.body.locus == locus:
                return True
        elif isinstance(step, ConditionalStep):
            for branch_step in step.branches.values():
                if branch_step.locus == locus:
                    return True
    return False


def _generate_pathway_input(pw: PathwayContract) -> str:
    """Generate a synthetic input JSON from pathway takes."""
    data: dict = {}
    for f in pw.takes:
        if f.type == "string":
            data[f.name] = f"test-{f.name}"
        elif f.type == "bool":
            data[f.name] = True
        elif f.type == "int":
            data[f.name] = 1
        elif f.type == "float":
            data[f.name] = 1.0
        elif f.type == "string[]":
            data[f.name] = [f"test-{f.name}-1"]
        elif f.type == "int[]":
            data[f.name] = [1]
        else:
            data[f.name] = f"test-{f.name}"
    return json.dumps(data)


def check_interactions(
    locus: str,
    sha: str,
    orchestrator: object,
) -> list[InteractionFailure]:
    """Test an allele against all pathways that include its locus.

    Temporarily promotes the allele to dominant, runs each affected
    pathway with synthetic inputs, then restores the original dominant.

    Returns list of InteractionFailure for pathways that broke.
    """
    failures: list[InteractionFailure] = []
    contract_store = orchestrator.contract_store
    affected = find_affected_pathways(locus, contract_store)

    if not affected:
        return failures

    original_dominant = orchestrator.phenotype.get_dominant(locus)
    orchestrator.phenotype.promote(locus, sha)

    try:
        for pw_name in affected:
            pw_contract = contract_store.get_pathway(pw_name)
            if pw_contract is None:
                continue

            test_input = _generate_pathway_input(pw_contract)

            try:
                orchestrator.run_pathway(pw_name, test_input)
            except Exception as e:
                failures.append(InteractionFailure(
                    pathway_name=pw_name,
                    failing_step=locus,
                    error=str(e),
                ))
    finally:
        if original_dominant:
            orchestrator.phenotype.promote(locus, original_dominant)
        else:
            config = orchestrator.phenotype.loci.get(locus)
            if config:
                config.dominant = None

    return failures
