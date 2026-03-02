"""Tests for the .sg contract parser."""
import pytest
from pathlib import Path

from sg.parser.lexer import tokenize, TokenType
from sg.parser.parser import parse_sg, ParseError
from sg.parser.types import (
    GeneFamily, BlastRadius, GeneContract, PathwayContract,
    TopologyContract, FieldDef, VerifyStep, FeedsDef,
    PathwayStep, ForStep, Dependency, TopologyResource, TypeDef,
)


CONTRACTS_DIR = Path(__file__).parent.parent / "contracts"


# --- Lexer tests ---

class TestLexer:
    def test_tokenize_gene_header(self):
        tokens = tokenize("gene bridge_create")
        types = [t.type for t in tokens if t.type not in (TokenType.NEWLINE, TokenType.EOF)]
        assert TokenType.KEYWORD in types
        assert TokenType.IDENTIFIER in types

    def test_tokenize_string(self):
        tokens = tokenize('  name  string  "A description"')
        strs = [t for t in tokens if t.type == TokenType.STRING]
        assert len(strs) == 1
        assert strs[0].value == "A description"

    def test_tokenize_reference(self):
        tokens = tokenize("  bridge_name = {bridge_name}")
        refs = [t for t in tokens if t.type == TokenType.REFERENCE]
        assert len(refs) == 1
        assert refs[0].value == "bridge_name"

    def test_tokenize_arrow(self):
        tokens = tokenize("  -> provision_management_bridge")
        arrows = [t for t in tokens if t.type == TokenType.ARROW]
        assert len(arrows) == 1

    def test_tokenize_dash(self):
        tokens = tokenize("  - Bridge already exists")
        dashes = [t for t in tokens if t.type == TokenType.DASH]
        assert len(dashes) == 1

    def test_tokenize_two_word_keyword(self):
        tokens = tokenize("  fails when:")
        kws = [t for t in tokens if t.type == TokenType.KEYWORD]
        assert any(t.value == "fails when" for t in kws)

    def test_tokenize_number_dot(self):
        tokens = tokenize("  1. bridge_create")
        nums = [t for t in tokens if t.type == TokenType.NUMBER]
        dots = [t for t in tokens if t.type == TokenType.DOT]
        assert len(nums) == 1
        assert len(dots) == 1

    def test_tokenize_optional_type(self):
        tokens = tokenize('  error  string?  "Error msg"')
        qs = [t for t in tokens if t.type == TokenType.QUESTION]
        assert len(qs) == 1


# --- Gene contract parsing ---

class TestGeneParser:
    SIMPLE_GENE = """\
gene bridge_create
  is configuration
  risk low

  does:
    Create a network bridge.

  takes:
    bridge_name  string    "Name for the bridge"
    interfaces   string[]  "Interfaces to attach"

  gives:
    success  bool  "Whether it worked"
    error    string?  "Error message"

  before:
    - Bridge does not already exist

  after:
    - Bridge is UP

  fails when:
    - bridge already exists -> success=false
"""

    def test_parse_simple_gene(self):
        contract = parse_sg(self.SIMPLE_GENE)
        assert isinstance(contract, GeneContract)
        assert contract.name == "bridge_create"
        assert contract.family == GeneFamily.CONFIGURATION
        assert contract.risk == BlastRadius.LOW

    def test_does_prose(self):
        contract = parse_sg(self.SIMPLE_GENE)
        assert "Create a network bridge" in contract.does

    def test_takes_fields(self):
        contract = parse_sg(self.SIMPLE_GENE)
        assert len(contract.takes) == 2
        assert contract.takes[0].name == "bridge_name"
        assert contract.takes[0].type == "string"
        assert contract.takes[0].description == "Name for the bridge"
        assert contract.takes[1].name == "interfaces"

    def test_gives_fields(self):
        contract = parse_sg(self.SIMPLE_GENE)
        assert len(contract.gives) == 2
        assert contract.gives[0].name == "success"
        assert contract.gives[0].type == "bool"
        assert contract.gives[1].name == "error"
        assert contract.gives[1].optional is True

    def test_before_after(self):
        contract = parse_sg(self.SIMPLE_GENE)
        assert len(contract.before) == 1
        assert "already exist" in contract.before[0]
        assert len(contract.after) == 1

    def test_fails_when(self):
        contract = parse_sg(self.SIMPLE_GENE)
        assert len(contract.fails_when) == 1
        assert "bridge already exists" in contract.fails_when[0]

    def test_diagnostic_gene(self):
        source = """\
gene check_fdb_stability
  is diagnostic
  risk none

  does:
    Monitor FDB for MAC flapping.

  takes:
    bridge  string  "Bridge to monitor"

  gives:
    healthy  bool  "True if stable"

  unhealthy when:
    - Any MAC appears on more than one port

  feeds:
    bridge_create  convergence
    bridge_uplink  convergence
"""
        contract = parse_sg(source)
        assert isinstance(contract, GeneContract)
        assert contract.family == GeneFamily.DIAGNOSTIC
        assert contract.risk == BlastRadius.NONE
        assert len(contract.unhealthy_when) == 1
        assert len(contract.feeds) == 2
        assert contract.feeds[0].target_locus == "bridge_create"
        assert contract.feeds[0].timescale == "convergence"

    def test_verify_block(self):
        source = """\
gene bridge_stp
  is configuration
  risk low

  does:
    Configure STP.

  takes:
    bridge_name  string  "Bridge"

  gives:
    success  bool  "OK"

  verify:
    check_link_state device={bridge_name}
    within 30s
"""
        contract = parse_sg(source)
        assert len(contract.verify) == 1
        assert contract.verify[0].locus == "check_link_state"
        assert contract.verify[0].params["device"] == "{bridge_name}"
        assert contract.verify_within == "30s"

    def test_field_with_default(self):
        source = """\
gene test_gene
  is configuration
  risk none

  does:
    Test.

  takes:
    name     string      "Name"
    enabled  bool = true "Enable it"
"""
        contract = parse_sg(source)
        field = contract.takes[1]
        assert field.name == "enabled"
        assert field.default == "true"
        assert field.optional is True

    def test_types_block(self):
        source = """\
gene check_fdb
  is diagnostic
  risk none

  does:
    Check FDB.

  takes:
    bridge  string  "Bridge"

  gives:
    flapping_macs  mac_flap[]  "Flapping MACs"

  types:
    mac_flap:
      mac          string   "The MAC address"
      ports        string[] "Ports seen on"
      transitions  int      "Number of changes"
"""
        contract = parse_sg(source)
        assert len(contract.types) == 1
        td = contract.types[0]
        assert td.name == "mac_flap"
        assert len(td.fields) == 3
        assert td.fields[0].name == "mac"
        assert td.fields[2].name == "transitions"


# --- Pathway contract parsing ---

class TestPathwayParser:
    SIMPLE_PATHWAY = """\
pathway configure_bridge_with_stp
  risk low

  does:
    Create a bridge and configure STP.

  takes:
    bridge_name    string    "Bridge name"
    interfaces     string[]  "Interfaces"
    stp_enabled    bool      "Enable STP"
    forward_delay  int       "Delay"

  steps:
    1. bridge_create
         bridge_name = {bridge_name}
         interfaces = {interfaces}

    2. bridge_stp
         bridge_name = {bridge_name}
         stp_enabled = {stp_enabled}
         forward_delay = {forward_delay}

  on failure:
    rollback all
"""

    def test_parse_simple_pathway(self):
        contract = parse_sg(self.SIMPLE_PATHWAY)
        assert isinstance(contract, PathwayContract)
        assert contract.name == "configure_bridge_with_stp"
        assert contract.risk == BlastRadius.LOW

    def test_pathway_steps(self):
        contract = parse_sg(self.SIMPLE_PATHWAY)
        assert len(contract.steps) == 2

        step1 = contract.steps[0]
        assert isinstance(step1, PathwayStep)
        assert step1.locus == "bridge_create"
        assert step1.index == 1
        assert step1.is_pathway_ref is False
        assert step1.params["bridge_name"] == "{bridge_name}"

        step2 = contract.steps[1]
        assert step2.locus == "bridge_stp"
        assert step2.params["stp_enabled"] == "{stp_enabled}"

    def test_pathway_on_failure(self):
        contract = parse_sg(self.SIMPLE_PATHWAY)
        assert "rollback" in contract.on_failure

    def test_composed_pathway(self):
        source = """\
pathway deploy_server_network
  risk critical

  does:
    Deploy complete network topology.

  takes:
    management_nic  string    "Management NIC"
    storage_nics    string[]  "Storage NICs"

  steps:
    1. -> provision_management_bridge
         management_bridge = "Management"
         uplink_interface = {management_nic}

    2. -> provision_bond
         bond_name = "storage"
         members = {storage_nics}

  requires:
    step 2 needs step 1
"""
        contract = parse_sg(source)
        assert isinstance(contract, PathwayContract)
        assert len(contract.steps) == 2

        step1 = contract.steps[0]
        assert step1.is_pathway_ref is True
        assert step1.locus == "provision_management_bridge"

        assert len(contract.requires) == 1
        assert contract.requires[0].step == 2
        assert contract.requires[0].needs == 1

    def test_for_loop_step(self):
        source = """\
pathway deploy_vlans
  risk medium

  does:
    Deploy VLAN bridges.

  takes:
    vm_vlans  int[]  "VLAN IDs"

  steps:
    1. for vlan in {vm_vlans}:
         -> provision_vlan_bridge
             vlan_id = {vlan}
"""
        contract = parse_sg(source)
        assert len(contract.steps) == 1

        step = contract.steps[0]
        assert isinstance(step, ForStep)
        assert step.variable == "vlan"
        assert step.iterable == "vm_vlans"
        assert step.body is not None
        assert step.body.locus == "provision_vlan_bridge"
        assert step.body.is_pathway_ref is True


# --- Topology contract parsing ---

class TestTopologyParser:
    def test_simple_topology(self):
        source = """\
topology production_server

  does:
    Standard production server network.

  has:
    management:
      is bridge
      uplink eth0
      stp enabled

    storage:
      is bond
      mode active-backup
"""
        contract = parse_sg(source)
        assert isinstance(contract, TopologyContract)
        assert contract.name == "production_server"
        assert "production server" in contract.does

        assert len(contract.has) == 2
        assert contract.has[0].name == "management"
        assert contract.has[0].resource_type == "bridge"
        assert contract.has[0].properties["uplink"] == "eth0"

        assert contract.has[1].name == "storage"
        assert contract.has[1].resource_type == "bond"
        assert contract.has[1].properties["mode"] == "active-backup"


# --- File parsing tests ---

class TestFileContracts:
    def test_parse_bridge_create_sg(self):
        path = CONTRACTS_DIR / "genes" / "bridge_create.sg"
        if not path.exists():
            pytest.skip("contract file not found")
        source = path.read_text()
        contract = parse_sg(source)
        assert isinstance(contract, GeneContract)
        assert contract.name == "bridge_create"
        assert contract.family == GeneFamily.CONFIGURATION
        assert contract.risk == BlastRadius.LOW
        assert len(contract.takes) >= 2
        assert len(contract.gives) >= 1
        assert len(contract.before) >= 1
        assert len(contract.after) >= 1
        assert len(contract.fails_when) >= 1

    def test_parse_bridge_stp_sg(self):
        path = CONTRACTS_DIR / "genes" / "bridge_stp.sg"
        if not path.exists():
            pytest.skip("contract file not found")
        source = path.read_text()
        contract = parse_sg(source)
        assert isinstance(contract, GeneContract)
        assert contract.name == "bridge_stp"
        assert len(contract.verify) >= 1
        assert contract.verify_within is not None

    def test_parse_pathway_sg(self):
        path = CONTRACTS_DIR / "pathways" / "configure_bridge_with_stp.sg"
        if not path.exists():
            pytest.skip("contract file not found")
        source = path.read_text()
        contract = parse_sg(source)
        assert isinstance(contract, PathwayContract)
        assert contract.name == "configure_bridge_with_stp"
        assert len(contract.steps) == 2
        assert contract.steps[0].locus == "bridge_create"
        assert contract.steps[1].locus == "bridge_stp"


# --- Error handling ---

class TestParseErrors:
    def test_missing_is_declaration(self):
        source = """\
gene broken
  risk low

  does:
    Missing family.
"""
        with pytest.raises(ParseError, match="missing 'is'"):
            parse_sg(source)

    def test_unknown_top_level(self):
        source = "widget broken"
        with pytest.raises(ParseError):
            parse_sg(source)


class TestDomainClause:
    def test_gene_with_domain(self):
        source = """\
gene bridge_create for network
  is configuration
  risk low

  does:
    Create a bridge.
"""
        contract = parse_sg(source)
        assert contract.name == "bridge_create"
        assert contract.domain == "network"

    def test_gene_without_domain(self):
        source = """\
gene bridge_create
  is configuration
  risk low

  does:
    Create a bridge.
"""
        contract = parse_sg(source)
        assert contract.name == "bridge_create"
        assert contract.domain is None

    def test_pathway_with_domain(self):
        source = """\
pathway deploy_bridge for network
  risk medium

  does:
    Deploy a bridge.

  steps:
    1. bridge_create
"""
        contract = parse_sg(source)
        assert contract.name == "deploy_bridge"
        assert contract.domain == "network"

    def test_pathway_without_domain(self):
        source = """\
pathway deploy_bridge
  risk medium

  does:
    Deploy a bridge.

  steps:
    1. bridge_create
"""
        contract = parse_sg(source)
        assert contract.domain is None

    def test_topology_with_domain(self):
        source = """\
topology production_server for network

  does:
    Full production server topology.

  has:
    management bridge
      uplink = {uplink}
"""
        contract = parse_sg(source)
        assert contract.name == "production_server"
        assert contract.domain == "network"

    def test_topology_without_domain(self):
        source = """\
topology production_server

  does:
    Full production server topology.
"""
        contract = parse_sg(source)
        assert contract.domain is None

    def test_existing_contracts_have_no_domain(self):
        """All existing .sg contracts parse with domain=None."""
        from pathlib import Path
        from sg.contracts import ContractStore
        contracts_dir = Path(__file__).parent.parent / "contracts"
        store = ContractStore.open(contracts_dir)
        for name, gene in store.genes.items():
            assert gene.domain is None, f"gene {name} has unexpected domain"
        for name, pathway in store.pathways.items():
            assert pathway.domain is None, f"pathway {name} has unexpected domain"
        for name, topology in store.topologies.items():
            assert topology.domain is None, f"topology {name} has unexpected domain"
