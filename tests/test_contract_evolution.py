"""Tests for contract evolution: LLM-generated contracts."""
import json
import shutil
import pytest
from pathlib import Path

from sg.contracts import ContractStore
from sg.mutation import MockMutationEngine, MutationEngine
from sg.parser.parser import parse_sg

import sg_network

CONTRACTS_DIR = sg_network.contracts_path()
FIXTURES_DIR = sg_network.fixtures_path()


class TestGenerateContract:
    def test_mock_engine_returns_fixture(self):
        """MockMutationEngine.generate_contract() returns fixture content."""
        engine = MockMutationEngine(FIXTURES_DIR)
        source = engine.generate_contract("diagnostic", "check VLAN trunk", [])
        assert "gene check_vlan_trunk" in source
        assert "is diagnostic" in source

    def test_mock_engine_missing_fixture(self):
        """MockMutationEngine raises if fixture missing."""
        engine = MockMutationEngine(Path("/nonexistent"))
        with pytest.raises(FileNotFoundError):
            engine.generate_contract("diagnostic", "anything", [])

    def test_base_engine_raises(self):
        """MutationEngine.generate_contract() raises NotImplementedError."""
        class Stub(MutationEngine):
            def mutate(self, ctx):
                return ""
        with pytest.raises(NotImplementedError):
            Stub().generate_contract("diagnostic", "test", [])


class TestRegisterContract:
    def test_register_contract(self, tmp_path):
        """ContractStore.register_contract writes and loads the contract."""
        shutil.copytree(CONTRACTS_DIR, tmp_path / "contracts")
        cs = ContractStore.open(tmp_path / "contracts")

        source = (FIXTURES_DIR / "generated_contract.sg").read_text()
        name = cs.register_contract(source, tmp_path / "contracts" / "genes" / "test_gen.sg")

        assert name == "check_vlan_trunk"
        assert cs.get_gene("check_vlan_trunk") is not None

    def test_generated_contract_parses(self):
        """The fixture contract parses successfully."""
        source = (FIXTURES_DIR / "generated_contract.sg").read_text()
        contract = parse_sg(source)
        assert contract.name == "check_vlan_trunk"
        assert contract.family.value == "diagnostic"
        assert contract.risk.value == "none"
        assert len(contract.takes) >= 2
        assert len(contract.gives) >= 2


class TestEvolveCommand:
    def test_cmd_evolve(self, tmp_path, capsys):
        """cmd_evolve generates and registers a contract."""
        import os
        import argparse
        from sg.cli import cmd_evolve

        shutil.copytree(CONTRACTS_DIR, tmp_path / "contracts")
        shutil.copytree(FIXTURES_DIR, tmp_path / "fixtures")
        (tmp_path / ".sg" / "registry").mkdir(parents=True)

        old = os.environ.get("SG_PROJECT_ROOT")
        os.environ["SG_PROJECT_ROOT"] = str(tmp_path)
        try:
            args = argparse.Namespace(
                family="diagnostic",
                context="check VLAN trunk connectivity",
                mutation_engine="mock",
                model=None,
            )
            cmd_evolve(args)
        finally:
            if old is None:
                os.environ.pop("SG_PROJECT_ROOT", None)
            else:
                os.environ["SG_PROJECT_ROOT"] = old

        captured = capsys.readouterr()
        assert "Generated contract: check_vlan_trunk" in captured.out
        # File should exist
        assert (tmp_path / "contracts" / "genes" / "check_vlan_trunk.sg").exists()
