"""Contract conformance testing — validate genes against their contracts.

Checks that gene source defines execute(), returns valid JSON with
required fields, and field types match the contract schema.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Union

from sg.contracts import ContractStore, validate_output
from sg.kernel.base import Kernel
from sg.loader import load_gene, call_gene
from sg.parser.types import GeneContract, GeneFamily, FieldDef
from sg.phenotype import PhenotypeMap
from sg.registry import Registry


@dataclass
class Check:
    name: str
    passed: bool
    message: str = ""


@dataclass
class ConformanceResult:
    locus: str
    sha: str
    passed: bool
    checks: list[Check] = field(default_factory=list)


def _type_matches(value, expected_type: str) -> bool:
    """Check if a value matches an .sg type."""
    if expected_type == "string":
        return isinstance(value, str)
    elif expected_type == "bool":
        return isinstance(value, bool)
    elif expected_type == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    elif expected_type == "float":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    elif expected_type.endswith("[]"):
        if not isinstance(value, list):
            return False
        base = expected_type[:-2]
        return all(_type_matches(item, base) for item in value)
    return True  # unknown types pass


def generate_test_inputs(contract: GeneContract) -> list[str]:
    """Generate basic test inputs from contract's takes schema."""
    # Build one valid input
    valid = {}
    for f in contract.takes:
        if f.type == "string":
            valid[f.name] = f"test-{f.name}"
        elif f.type == "bool":
            valid[f.name] = True
        elif f.type == "int":
            valid[f.name] = 1
        elif f.type == "float":
            valid[f.name] = 1.0
        elif f.type == "string[]":
            valid[f.name] = [f"test-{f.name}-1", f"test-{f.name}-2"]
        elif f.type == "int[]":
            valid[f.name] = [1, 2]
        else:
            valid[f.name] = f"test-{f.name}"

    inputs = [json.dumps(valid)]

    # Build one input with missing required field (if any)
    required_fields = [f for f in contract.takes if f.required and not f.optional]
    if len(required_fields) > 1:
        incomplete = dict(valid)
        del incomplete[required_fields[0].name]
        inputs.append(json.dumps(incomplete))

    return inputs


def check_gene_conformance(
    source: str,
    contract: GeneContract,
    kernel: Kernel,
    sha: str = "unknown",
    test_inputs: list[str] | None = None,
) -> ConformanceResult:
    """Run conformance checks for a gene against its contract."""
    checks: list[Check] = []
    all_passed = True

    # Check 1: Source defines execute() function
    try:
        execute_fn = load_gene(source, kernel)
        checks.append(Check("defines_execute", True))
    except (ValueError, Exception) as e:
        checks.append(Check("defines_execute", False, str(e)))
        return ConformanceResult(
            locus=contract.name, sha=sha, passed=False, checks=checks,
        )

    # Generate test inputs if not provided
    if test_inputs is None:
        test_inputs = generate_test_inputs(contract)

    # Check 2: Execute returns valid JSON with success field
    for i, test_input in enumerate(test_inputs):
        label = f"execute_input_{i}"
        try:
            result = call_gene(execute_fn, test_input)
        except Exception as e:
            # Genes that catch exceptions and return success=false are valid
            checks.append(Check(label, False, f"execution raised: {e}"))
            all_passed = False
            continue

        # Check valid JSON
        try:
            data = json.loads(result)
        except (json.JSONDecodeError, TypeError):
            checks.append(Check(f"{label}_json", False, "output is not valid JSON"))
            all_passed = False
            continue

        checks.append(Check(f"{label}_json", True))

        # Check success field exists
        if "success" not in data:
            checks.append(Check(f"{label}_success_field", False,
                               "output missing 'success' field"))
            all_passed = False
            continue

        checks.append(Check(f"{label}_success_field", True))

        # Check 3: Output contains all required gives fields
        if data.get("success") is True:
            for field_def in contract.gives:
                if field_def.optional:
                    continue
                if field_def.name not in data:
                    checks.append(Check(
                        f"{label}_field_{field_def.name}", False,
                        f"required output field '{field_def.name}' missing",
                    ))
                    all_passed = False
                else:
                    checks.append(Check(f"{label}_field_{field_def.name}", True))

            # Check 4: Field types match
            for field_def in contract.gives:
                if field_def.name in data:
                    if not _type_matches(data[field_def.name], field_def.type):
                        checks.append(Check(
                            f"{label}_type_{field_def.name}", False,
                            f"field '{field_def.name}' should be {field_def.type}, "
                            f"got {type(data[field_def.name]).__name__}",
                        ))
                        all_passed = False

    return ConformanceResult(
        locus=contract.name, sha=sha, passed=all_passed, checks=checks,
    )


class ConformanceSuite:
    """Run conformance for all loci with dominant alleles."""

    def run_all(
        self,
        contract_store: ContractStore,
        registry: Registry,
        phenotype: PhenotypeMap,
        kernel: Kernel,
    ) -> list[ConformanceResult]:
        results = []
        for locus in contract_store.known_loci():
            contract = contract_store.get_gene(locus)
            if contract is None:
                continue

            dominant_sha = phenotype.get_dominant(locus)
            if dominant_sha is None:
                continue

            source = registry.load_source(dominant_sha)
            if source is None:
                continue

            result = check_gene_conformance(
                source, contract, kernel, sha=dominant_sha[:12],
            )
            results.append(result)
        return results

    def run_locus(
        self,
        locus: str,
        contract_store: ContractStore,
        registry: Registry,
        phenotype: PhenotypeMap,
        kernel: Kernel,
    ) -> ConformanceResult | None:
        contract = contract_store.get_gene(locus)
        if contract is None:
            return None

        dominant_sha = phenotype.get_dominant(locus)
        if dominant_sha is None:
            return None

        source = registry.load_source(dominant_sha)
        if source is None:
            return None

        return check_gene_conformance(
            source, contract, kernel, sha=dominant_sha[:12],
        )
