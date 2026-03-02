"""Tests for contracts: ContractStore discovery and validation."""
import json
from pathlib import Path
import pytest
from sg.contracts import ContractStore, validate_output

import sg_network

CONTRACTS_DIR = sg_network.contracts_path()


@pytest.fixture
def store():
    return ContractStore.open(CONTRACTS_DIR)


def test_known_loci(store):
    loci = store.known_loci()
    assert "bridge_create" in loci
    assert "bridge_stp" in loci


def test_known_pathways(store):
    pathways = store.known_pathways()
    assert "configure_bridge_with_stp" in pathways


def test_contract_info_bridge_create(store):
    info = store.contract_info("bridge_create")
    assert info.locus == "bridge_create"
    assert "bridge" in info.description.lower()
    assert "bridge_name" in info.input_schema.get("required", [])


def test_contract_info_bridge_stp(store):
    info = store.contract_info("bridge_stp")
    assert info.locus == "bridge_stp"
    assert "stp" in info.description.lower()


def test_contract_info_unknown(store):
    with pytest.raises(ValueError, match="no contract for locus"):
        store.contract_info("nonexistent")


def test_get_gene(store):
    gene = store.get_gene("bridge_create")
    assert gene is not None
    assert gene.name == "bridge_create"
    assert len(gene.takes) > 0
    assert len(gene.gives) > 0


def test_get_gene_unknown(store):
    assert store.get_gene("nonexistent") is None


def test_get_pathway(store):
    pathway = store.get_pathway("configure_bridge_with_stp")
    assert pathway is not None
    assert pathway.name == "configure_bridge_with_stp"
    assert len(pathway.steps) == 2


def test_get_pathway_unknown(store):
    assert store.get_pathway("nonexistent") is None


def test_empty_store():
    store = ContractStore()
    assert store.known_loci() == []
    assert store.known_pathways() == []


def test_load_nonexistent_directory():
    store = ContractStore.open(Path("/nonexistent"))
    assert store.known_loci() == []


def test_validate_output_valid():
    assert validate_output("bridge_create", json.dumps({"success": True}))
    assert validate_output("bridge_create", json.dumps({"success": False, "error": "x"}))


def test_validate_output_missing_success():
    assert not validate_output("bridge_create", json.dumps({"result": True}))


def test_validate_output_invalid_json():
    assert not validate_output("bridge_create", "not json")


def test_validate_output_not_dict():
    assert not validate_output("bridge_create", json.dumps([1, 2, 3]))


def test_validate_output_success_not_bool():
    assert not validate_output("bridge_create", json.dumps({"success": "yes"}))
