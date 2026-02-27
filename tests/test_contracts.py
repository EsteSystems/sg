"""Tests for contracts: locus definitions and validation."""
import json
import pytest
from sg.contracts import contract_info, validate_output, known_loci


def test_known_loci():
    loci = known_loci()
    assert "bridge_create" in loci
    assert "bridge_stp" in loci


def test_contract_info_bridge_create():
    info = contract_info("bridge_create")
    assert info.locus == "bridge_create"
    assert "bridge" in info.description.lower()
    assert "bridge_name" in info.input_schema["required"]


def test_contract_info_bridge_stp():
    info = contract_info("bridge_stp")
    assert info.locus == "bridge_stp"
    assert "stp" in info.description.lower()


def test_contract_info_unknown():
    with pytest.raises(ValueError, match="no contract for locus"):
        contract_info("nonexistent")


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
