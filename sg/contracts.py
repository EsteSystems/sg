"""Contract definitions, locus types, and validation.

Temporary bridge: hardcoded contracts for bridge_create and bridge_stp.
Will be replaced by ContractStore loading .sg files in M3.
"""
from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass
class ContractInfo:
    locus: str
    description: str
    input_schema: dict
    output_schema: dict


# Hardcoded contracts — replaced by .sg parsing in M3
_CONTRACTS: dict[str, ContractInfo] = {
    "bridge_create": ContractInfo(
        locus="bridge_create",
        description="Create a network bridge with the given name and interfaces",
        input_schema={
            "type": "object",
            "required": ["bridge_name", "interfaces"],
            "properties": {
                "bridge_name": {"type": "string"},
                "interfaces": {"type": "array", "items": {"type": "string"}},
            },
        },
        output_schema={
            "type": "object",
            "required": ["success"],
            "properties": {"success": {"type": "boolean"}},
        },
    ),
    "bridge_stp": ContractInfo(
        locus="bridge_stp",
        description="Configure STP on an existing bridge",
        input_schema={
            "type": "object",
            "required": ["bridge_name", "stp_enabled", "forward_delay"],
            "properties": {
                "bridge_name": {"type": "string"},
                "stp_enabled": {"type": "boolean"},
                "forward_delay": {"type": "integer"},
            },
        },
        output_schema={
            "type": "object",
            "required": ["success"],
            "properties": {"success": {"type": "boolean"}},
        },
    ),
}


def contract_info(locus: str) -> ContractInfo:
    """Return the contract metadata for a locus."""
    info = _CONTRACTS.get(locus)
    if info is None:
        raise ValueError(f"no contract for locus: {locus}")
    return info


def known_loci() -> list[str]:
    """Return all known locus names."""
    return list(_CONTRACTS.keys())


def validate_output(locus: str, output_json: str) -> bool:
    """Validate that a gene's output conforms to the locus contract.

    Checks that the output is valid JSON containing a 'success' boolean field.
    """
    try:
        data = json.loads(output_json)
    except (json.JSONDecodeError, TypeError):
        return False
    if not isinstance(data, dict):
        return False
    if "success" not in data:
        return False
    if not isinstance(data["success"], bool):
        return False
    return True
