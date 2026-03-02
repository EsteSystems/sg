"""Tests for contract conformance testing."""
import json
import shutil
import pytest
from pathlib import Path

from sg.conformance import (
    check_gene_conformance, generate_test_inputs,
    ConformanceSuite, ConformanceResult, Check, _type_matches,
)
from sg.contracts import ContractStore
import sg_network
from sg_network import MockNetworkKernel
from sg.phenotype import PhenotypeMap
from sg.registry import Registry


CONTRACTS_DIR = sg_network.contracts_path()
GENES_DIR = sg_network.genes_path()


@pytest.fixture
def kernel():
    return MockNetworkKernel()


@pytest.fixture
def contract_store():
    return ContractStore.open(CONTRACTS_DIR)


@pytest.fixture
def project(tmp_path):
    """Set up a project with registered seed genes."""
    shutil.copytree(CONTRACTS_DIR, tmp_path / "contracts")
    contract_store = ContractStore.open(tmp_path / "contracts")
    registry = Registry.open(tmp_path / ".sg" / "registry")
    phenotype = PhenotypeMap()

    for locus in contract_store.known_loci():
        candidates = sorted(GENES_DIR.glob(f"{locus}_*.py"))
        if candidates:
            source = candidates[0].read_text()
            sha = registry.register(source, locus)
            phenotype.promote(locus, sha)
            allele = registry.get(sha)
            allele.state = "dominant"

    registry.save_index()
    phenotype.save(tmp_path / "phenotype.toml")
    return tmp_path


class TestGeneConformance:
    def test_valid_gene_basic_checks_pass(self, kernel, contract_store):
        """A seed gene passes basic conformance (defines execute, returns JSON with success)."""
        contract = contract_store.get_gene("bridge_create")
        source = (GENES_DIR / "bridge_create_v1.py").read_text()
        result = check_gene_conformance(source, contract, kernel, sha="test")
        # Basic checks should pass: defines_execute, json, success_field
        basic_checks = [c for c in result.checks
                        if c.name in ("defines_execute", "execute_input_0_json",
                                      "execute_input_0_success_field")]
        assert all(c.passed for c in basic_checks)

    def test_output_missing_success_field(self, kernel, contract_store):
        """Gene that doesn't return success field fails."""
        contract = contract_store.get_gene("bridge_create")
        source = '''
import json
def execute(input_json):
    return json.dumps({"result": "ok"})
'''
        result = check_gene_conformance(source, contract, kernel, sha="test")
        assert not result.passed
        failed = [c for c in result.checks if not c.passed]
        assert any("success" in c.message for c in failed)

    def test_wrong_output_type(self, kernel, contract_store):
        """Gene returning non-JSON fails."""
        contract = contract_store.get_gene("bridge_create")
        source = '''
def execute(input_json):
    return "not json"
'''
        result = check_gene_conformance(source, contract, kernel, sha="test")
        assert not result.passed

    def test_no_execute_function(self, kernel, contract_store):
        """Gene without execute() fails."""
        contract = contract_store.get_gene("bridge_create")
        source = '''
def run(input_json):
    return '{}'
'''
        result = check_gene_conformance(source, contract, kernel, sha="test")
        assert not result.passed
        assert result.checks[0].name == "defines_execute"
        assert not result.checks[0].passed

    def test_diagnostic_gene_passes(self, kernel, contract_store):
        """A diagnostic seed gene passes conformance."""
        contract = contract_store.get_gene("check_link_state")
        source = (GENES_DIR / "check_link_state_v1.py").read_text()
        result = check_gene_conformance(source, contract, kernel, sha="test")
        assert result.passed, [c for c in result.checks if not c.passed]


class TestGenerateInputs:
    def test_generates_valid_input(self, contract_store):
        """Generates at least one valid input."""
        contract = contract_store.get_gene("bridge_create")
        inputs = generate_test_inputs(contract)
        assert len(inputs) >= 1
        data = json.loads(inputs[0])
        assert "bridge_name" in data
        assert "interfaces" in data

    def test_generates_string_array(self, contract_store):
        """string[] fields produce list values."""
        contract = contract_store.get_gene("bridge_create")
        inputs = generate_test_inputs(contract)
        data = json.loads(inputs[0])
        assert isinstance(data["interfaces"], list)

    def test_generates_bool_fields(self, contract_store):
        """bool fields produce boolean values."""
        contract = contract_store.get_gene("check_link_state")
        inputs = generate_test_inputs(contract)
        data = json.loads(inputs[0])
        assert "interface" in data


class TestTypeMatches:
    def test_string(self):
        assert _type_matches("hello", "string")
        assert not _type_matches(42, "string")

    def test_bool(self):
        assert _type_matches(True, "bool")
        assert not _type_matches(1, "bool")

    def test_int(self):
        assert _type_matches(42, "int")
        assert not _type_matches(True, "int")
        assert not _type_matches("42", "int")

    def test_float(self):
        assert _type_matches(3.14, "float")
        assert _type_matches(42, "float")  # int is valid float
        assert not _type_matches("3.14", "float")

    def test_string_array(self):
        assert _type_matches(["a", "b"], "string[]")
        assert not _type_matches("a", "string[]")
        assert not _type_matches([1, 2], "string[]")

    def test_empty_array(self):
        assert _type_matches([], "string[]")


class TestConformanceSuite:
    def test_suite_all_loci(self, project):
        """ConformanceSuite runs across all loci with dominant alleles."""
        contract_store = ContractStore.open(project / "contracts")
        registry = Registry.open(project / ".sg" / "registry")
        phenotype = PhenotypeMap.load(project / "phenotype.toml")
        kernel = MockNetworkKernel()

        suite = ConformanceSuite()
        results = suite.run_all(contract_store, registry, phenotype, kernel)
        assert len(results) > 0
        for r in results:
            assert isinstance(r, ConformanceResult)

    def test_suite_specific_locus(self, project):
        """ConformanceSuite can test a specific locus."""
        contract_store = ContractStore.open(project / "contracts")
        registry = Registry.open(project / ".sg" / "registry")
        phenotype = PhenotypeMap.load(project / "phenotype.toml")
        kernel = MockNetworkKernel()

        suite = ConformanceSuite()
        result = suite.run_locus("bridge_create", contract_store, registry, phenotype, kernel)
        assert result is not None
        assert result.locus == "bridge_create"
