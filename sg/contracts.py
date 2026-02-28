"""Contract store — discovers and parses .sg files.

Adding a new .sg file defines a new locus. Zero Python changes needed.
The ContractStore replaces all hardcoded contract definitions.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from sg.parser.parser import parse_sg
from sg.parser.types import (
    GeneContract, PathwayContract, TopologyContract,
    GeneFamily, BlastRadius, FieldDef,
)


@dataclass
class ContractInfo:
    """Runtime contract metadata derived from a parsed .sg file."""
    locus: str
    description: str
    input_schema: dict
    output_schema: dict
    family: str = ""
    risk: str = "none"
    raw: GeneContract | PathwayContract | TopologyContract | None = None


def _field_to_json_type(field_type: str) -> dict:
    """Convert .sg type notation to JSON schema type."""
    if field_type.endswith("[]"):
        base = field_type[:-2]
        return {"type": "array", "items": _field_to_json_type(base)}
    type_map = {
        "string": {"type": "string"},
        "bool": {"type": "boolean"},
        "int": {"type": "integer"},
        "float": {"type": "number"},
    }
    return type_map.get(field_type, {"type": "string"})


def _fields_to_schema(fields: list[FieldDef]) -> dict:
    """Convert a list of FieldDefs to a JSON schema dict."""
    properties = {}
    required = []
    for f in fields:
        properties[f.name] = _field_to_json_type(f.type)
        if f.description:
            properties[f.name]["description"] = f.description
        if f.required and not f.optional:
            required.append(f.name)
    schema: dict = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def _gene_contract_to_info(contract: GeneContract) -> ContractInfo:
    """Convert a parsed GeneContract to runtime ContractInfo."""
    return ContractInfo(
        locus=contract.name,
        description=contract.does,
        input_schema=_fields_to_schema(contract.takes),
        output_schema=_fields_to_schema(contract.gives),
        family=contract.family.value,
        risk=contract.risk.value,
        raw=contract,
    )


class ContractStore:
    """Discovers and loads .sg contract files from a directory tree.

    Gene contracts define loci. Pathway contracts define pathways.
    Topology contracts define topologies (future use).
    """

    def __init__(self) -> None:
        self.genes: dict[str, GeneContract] = {}
        self.pathways: dict[str, PathwayContract] = {}
        self.topologies: dict[str, TopologyContract] = {}
        self._info_cache: dict[str, ContractInfo] = {}

    def load_directory(self, contracts_dir: Path) -> None:
        """Discover and parse all .sg files in a directory tree."""
        if not contracts_dir.exists():
            return
        for sg_file in sorted(contracts_dir.rglob("*.sg")):
            self.load_file(sg_file)

    def load_file(self, path: Path) -> None:
        """Parse a single .sg file and register the contract."""
        source = path.read_text()
        contract = parse_sg(source)

        if isinstance(contract, GeneContract):
            self.genes[contract.name] = contract
            self._info_cache[contract.name] = _gene_contract_to_info(contract)
        elif isinstance(contract, PathwayContract):
            self.pathways[contract.name] = contract
        elif isinstance(contract, TopologyContract):
            self.topologies[contract.name] = contract

    def contract_info(self, locus: str) -> ContractInfo:
        """Return runtime contract metadata for a gene locus."""
        info = self._info_cache.get(locus)
        if info is None:
            raise ValueError(f"no contract for locus: {locus}")
        return info

    def known_loci(self) -> list[str]:
        """Return all known gene locus names."""
        return list(self.genes.keys())

    def known_pathways(self) -> list[str]:
        """Return all known pathway names."""
        return list(self.pathways.keys())

    def get_gene(self, name: str) -> GeneContract | None:
        return self.genes.get(name)

    def get_pathway(self, name: str) -> PathwayContract | None:
        return self.pathways.get(name)

    def get_topology(self, name: str) -> TopologyContract | None:
        return self.topologies.get(name)

    def known_topologies(self) -> list[str]:
        """Return all known topology names."""
        return list(self.topologies.keys())

    def register_contract(self, source: str, path: Path) -> str:
        """Write a contract source to disk and load it. Returns the contract name."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
        self.load_file(path)
        contract = parse_sg(source)
        return contract.name

    @classmethod
    def open(cls, contracts_dir: Path) -> ContractStore:
        """Create a ContractStore by scanning a contracts directory."""
        store = cls()
        store.load_directory(contracts_dir)
        return store


# --- Output validation (contract-independent) ---

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
