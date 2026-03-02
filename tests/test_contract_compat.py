"""Tests for contract structural compatibility and domain validation."""
import pytest
from pathlib import Path

from sg.contracts import contracts_compatible, ContractStore
from sg.parser.types import (
    GeneContract, GeneFamily, BlastRadius, FieldDef,
)


def _make_gene(name, takes=None, gives=None, domain=None):
    """Helper to create a minimal GeneContract."""
    return GeneContract(
        name=name,
        family=GeneFamily.CONFIGURATION,
        risk=BlastRadius.LOW,
        does="test",
        domain=domain,
        takes=takes or [],
        gives=gives or [],
    )


class TestContractsCompatible:
    def test_identical_contracts(self):
        a = _make_gene("x", takes=[
            FieldDef(name="name", type="string"),
        ], gives=[
            FieldDef(name="success", type="bool"),
        ])
        b = _make_gene("y", takes=[
            FieldDef(name="name", type="string"),
        ], gives=[
            FieldDef(name="success", type="bool"),
        ])
        assert contracts_compatible(a, b)

    def test_missing_required_takes_field(self):
        a = _make_gene("x", takes=[
            FieldDef(name="name", type="string"),
            FieldDef(name="count", type="int"),
        ])
        b = _make_gene("y", takes=[
            FieldDef(name="name", type="string"),
        ])
        assert not contracts_compatible(a, b)

    def test_missing_required_gives_field(self):
        a = _make_gene("x", gives=[
            FieldDef(name="success", type="bool"),
            FieldDef(name="result", type="string"),
        ])
        b = _make_gene("y", gives=[
            FieldDef(name="success", type="bool"),
        ])
        assert not contracts_compatible(a, b)

    def test_type_mismatch(self):
        a = _make_gene("x", takes=[
            FieldDef(name="count", type="int"),
        ])
        b = _make_gene("y", takes=[
            FieldDef(name="count", type="string"),
        ])
        assert not contracts_compatible(a, b)

    def test_extra_optional_fields_compatible(self):
        """b can have extra fields — only a's required fields matter."""
        a = _make_gene("x", takes=[
            FieldDef(name="name", type="string"),
        ])
        b = _make_gene("y", takes=[
            FieldDef(name="name", type="string"),
            FieldDef(name="extra", type="int"),
        ])
        assert contracts_compatible(a, b)

    def test_optional_field_not_required_in_b(self):
        """Optional fields in a don't need to be in b."""
        a = _make_gene("x", takes=[
            FieldDef(name="name", type="string"),
            FieldDef(name="hint", type="string", optional=True),
        ])
        b = _make_gene("y", takes=[
            FieldDef(name="name", type="string"),
        ])
        assert contracts_compatible(a, b)

    def test_empty_contracts_compatible(self):
        a = _make_gene("x")
        b = _make_gene("y")
        assert contracts_compatible(a, b)

    def test_array_type_match(self):
        a = _make_gene("x", takes=[
            FieldDef(name="items", type="string[]"),
        ])
        b = _make_gene("y", takes=[
            FieldDef(name="items", type="string[]"),
        ])
        assert contracts_compatible(a, b)

    def test_array_type_mismatch(self):
        a = _make_gene("x", takes=[
            FieldDef(name="items", type="string[]"),
        ])
        b = _make_gene("y", takes=[
            FieldDef(name="items", type="int[]"),
        ])
        assert not contracts_compatible(a, b)

    def test_family_mismatch_rejected(self):
        """Configuration and diagnostic genes are not compatible."""
        a = GeneContract(
            name="x", family=GeneFamily.CONFIGURATION,
            risk=BlastRadius.LOW, does="test", domain=None,
            takes=[FieldDef(name="name", type="string")],
            gives=[FieldDef(name="success", type="bool")],
        )
        b = GeneContract(
            name="y", family=GeneFamily.DIAGNOSTIC,
            risk=BlastRadius.NONE, does="test", domain=None,
            takes=[FieldDef(name="name", type="string")],
            gives=[FieldDef(name="success", type="bool")],
        )
        assert not contracts_compatible(a, b)

    def test_same_family_compatible(self):
        """Same family with matching fields is compatible."""
        a = GeneContract(
            name="x", family=GeneFamily.DIAGNOSTIC,
            risk=BlastRadius.NONE, does="test", domain="data",
            takes=[FieldDef(name="name", type="string")],
            gives=[FieldDef(name="healthy", type="bool")],
        )
        b = GeneContract(
            name="y", family=GeneFamily.DIAGNOSTIC,
            risk=BlastRadius.NONE, does="test", domain="network",
            takes=[FieldDef(name="name", type="string")],
            gives=[FieldDef(name="healthy", type="bool")],
        )
        assert contracts_compatible(a, b)


class TestDomainValidation:
    def test_load_file_warns_domain_mismatch(self, tmp_path, caplog):
        """Loading a contract with wrong domain logs a warning."""
        import logging
        sg_file = tmp_path / "test.sg"
        sg_file.write_text("""\
gene bridge_create for network
  is configuration
  risk low

  does:
    Create a bridge.
""")
        store = ContractStore()
        with caplog.at_level(logging.WARNING, logger="sg.contracts"):
            store.load_file(sg_file, kernel_domain="storage")
        assert "network" in caplog.text
        assert "storage" in caplog.text
        # Contract is still loaded despite warning
        assert "bridge_create" in store.genes

    def test_load_file_no_warning_matching_domain(self, tmp_path, capsys):
        """No warning when domain matches kernel."""
        sg_file = tmp_path / "test.sg"
        sg_file.write_text("""\
gene bridge_create for network
  is configuration
  risk low

  does:
    Create a bridge.
""")
        store = ContractStore()
        store.load_file(sg_file, kernel_domain="network")
        captured = capsys.readouterr()
        assert "warning" not in captured.err

    def test_load_file_no_warning_no_domain(self, tmp_path, capsys):
        """No warning for contracts without domain clause."""
        sg_file = tmp_path / "test.sg"
        sg_file.write_text("""\
gene bridge_create
  is configuration
  risk low

  does:
    Create a bridge.
""")
        store = ContractStore()
        store.load_file(sg_file, kernel_domain="storage")
        captured = capsys.readouterr()
        assert "warning" not in captured.err

    def test_contract_info_includes_domain(self, tmp_path):
        """ContractInfo includes the domain field."""
        sg_file = tmp_path / "test.sg"
        sg_file.write_text("""\
gene bridge_create for network
  is configuration
  risk low

  does:
    Create a bridge.
""")
        store = ContractStore()
        store.load_file(sg_file)
        info = store.contract_info("bridge_create")
        assert info.domain == "network"

    def test_contract_info_domain_none(self, tmp_path):
        """ContractInfo has domain=None for contracts without domain clause."""
        sg_file = tmp_path / "test.sg"
        sg_file.write_text("""\
gene bridge_create
  is configuration
  risk low

  does:
    Create a bridge.
""")
        store = ContractStore()
        store.load_file(sg_file)
        info = store.contract_info("bridge_create")
        assert info.domain is None
