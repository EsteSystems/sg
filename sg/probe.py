"""Input-space exploration — probing loci with edge-case inputs.

Generates synthetic inputs from contract schemas (empty strings,
boundary ints, missing optionals, etc.) and executes them against
the dominant allele to surface hidden failures.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from sg.contracts import ContractStore, validate_output
from sg.loader import load_gene, call_gene
from sg.log import get_logger
from sg.parser.types import GeneContract, FieldDef

logger = get_logger("probe")


@dataclass
class ProbeResult:
    """Result of a single probe execution."""
    input_json: str
    success: bool
    output: str | None = None
    error: str | None = None


@dataclass
class ProbeReport:
    """Aggregated results from probing a locus."""
    locus: str
    sha: str
    total: int = 0
    passed: int = 0
    failed: int = 0
    results: list[ProbeResult] = field(default_factory=list)

    @property
    def failure_rate(self) -> float:
        return self.failed / max(self.total, 1)


def _generate_valid(fields: list[FieldDef]) -> dict:
    """Generate a valid baseline input from field definitions."""
    data: dict = {}
    for f in fields:
        if f.type == "string":
            data[f.name] = f"test-{f.name}"
        elif f.type == "bool":
            data[f.name] = True
        elif f.type == "int":
            data[f.name] = 1
        elif f.type == "float":
            data[f.name] = 1.0
        elif f.type == "string[]":
            data[f.name] = [f"test-{f.name}-1", f"test-{f.name}-2"]
        elif f.type == "int[]":
            data[f.name] = [1, 2]
        else:
            data[f.name] = f"test-{f.name}"
    return data


def generate_probes(contract: GeneContract, count: int = 10) -> list[str]:
    """Generate edge-case inputs from the contract's takes schema.

    Strategies: valid baseline, empty strings, zero/negative ints,
    empty arrays, missing optionals, boolean inversions, long strings,
    extra unknown fields, empty object.
    """
    probes: list[str] = []
    baseline = _generate_valid(contract.takes)
    probes.append(json.dumps(baseline, sort_keys=True))

    # Empty strings
    for f in contract.takes:
        if f.type == "string":
            variant = dict(baseline)
            variant[f.name] = ""
            probes.append(json.dumps(variant, sort_keys=True))

    # Zero/negative ints
    for f in contract.takes:
        if f.type == "int":
            for val in [0, -1, 2**31 - 1]:
                variant = dict(baseline)
                variant[f.name] = val
                probes.append(json.dumps(variant, sort_keys=True))

    # Empty arrays
    for f in contract.takes:
        if f.type.endswith("[]"):
            variant = dict(baseline)
            variant[f.name] = []
            probes.append(json.dumps(variant, sort_keys=True))

    # Missing optional fields
    optional_fields = [f for f in contract.takes if f.optional]
    if optional_fields:
        variant = dict(baseline)
        for f in optional_fields:
            variant.pop(f.name, None)
        probes.append(json.dumps(variant, sort_keys=True))

    # Boolean inversions
    for f in contract.takes:
        if f.type == "bool":
            variant = dict(baseline)
            variant[f.name] = not baseline.get(f.name, True)
            probes.append(json.dumps(variant, sort_keys=True))

    # Long string (one probe)
    for f in contract.takes:
        if f.type == "string":
            variant = dict(baseline)
            variant[f.name] = "x" * 1000
            probes.append(json.dumps(variant, sort_keys=True))
            break

    # Extra unknown field
    variant = dict(baseline)
    variant["_unknown_extra_field"] = "unexpected"
    probes.append(json.dumps(variant, sort_keys=True))

    # Empty object
    probes.append("{}")

    # Deduplicate
    seen: set[str] = set()
    unique: list[str] = []
    for p in probes:
        if p not in seen:
            seen.add(p)
            unique.append(p)

    return unique[:count]


def probe_locus(
    locus: str,
    orchestrator: object,
    count: int = 10,
) -> ProbeReport:
    """Probe a locus with edge-case inputs and report results.

    Uses the dominant allele. Does NOT trigger mutation on failure —
    this is purely diagnostic.
    """
    contract_store = orchestrator.contract_store
    contract = contract_store.get_gene(locus)
    if contract is None:
        raise ValueError(f"no contract for locus: {locus}")

    dominant_sha = orchestrator.phenotype.get_dominant(locus)
    if dominant_sha is None:
        raise ValueError(f"no dominant allele for locus: {locus}")

    source = orchestrator.registry.load_source(dominant_sha)
    if source is None:
        raise ValueError(f"source not found for {dominant_sha[:12]}")

    execute_fn = load_gene(source, orchestrator.kernel)
    inputs = generate_probes(contract, count)

    report = ProbeReport(locus=locus, sha=dominant_sha[:12])

    for input_json in inputs:
        report.total += 1
        try:
            output = call_gene(execute_fn, input_json)
            if validate_output(locus, output, contract_store):
                report.passed += 1
                report.results.append(ProbeResult(
                    input_json=input_json, success=True, output=output,
                ))
            else:
                report.failed += 1
                report.results.append(ProbeResult(
                    input_json=input_json, success=False, output=output,
                    error="output validation failed",
                ))
        except Exception as e:
            report.failed += 1
            report.results.append(ProbeResult(
                input_json=input_json, success=False, error=str(e),
            ))

    return report
