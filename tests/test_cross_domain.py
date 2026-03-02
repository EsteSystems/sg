"""End-to-end cross-domain federation tests.

Uses real .sg contract parsing (not mocked dicts) to verify the full
cross-domain allele flow: push from one domain, pull from another.
"""
import json
import pytest
from pathlib import Path

from sg.contracts import (
    ContractStore, contracts_compatible, contract_to_pool_metadata,
)
from sg.parser.types import GeneFamily, BlastRadius
from sg.pool_server import create_pool_app


@pytest.fixture
def pool_client(tmp_path):
    """Pool test client with no auth."""
    app = create_pool_app(tmp_path / "pool", token=None, reciprocity=0)
    from starlette.testclient import TestClient
    return TestClient(app)


def _write_contract(tmp_path, filename, content):
    """Write a .sg file and return its path."""
    p = tmp_path / filename
    p.write_text(content)
    return p


NETWORK_BRIDGE_CREATE = """\
gene bridge_create for network
  is configuration
  risk low

  does:
    Create a network bridge.

  takes:
    name: string
    mtu: int

  gives:
    success: bool
    bridge_id: string
"""

DATA_TABLE_CREATE = """\
gene table_create for data
  is configuration
  risk low

  does:
    Create a data table.

  takes:
    name: string
    mtu: int

  gives:
    success: bool
    bridge_id: string
"""

NETWORK_BRIDGE_VERIFY = """\
gene bridge_verify for network
  is diagnostic
  risk none

  does:
    Verify a bridge exists.

  takes:
    name: string

  gives:
    success: bool
    healthy: bool
"""

DATA_TABLE_VERIFY = """\
gene table_verify for data
  is diagnostic
  risk none

  does:
    Verify a table exists.

  takes:
    name: string

  gives:
    success: bool
    healthy: bool
"""


class TestCrossDomainContractParsing:
    """Test cross-domain compatibility using real parsed contracts."""

    def test_compatible_across_domains(self, tmp_path):
        """Configuration genes with matching signatures are compatible."""
        store = ContractStore()
        store.load_file(_write_contract(tmp_path, "net.sg", NETWORK_BRIDGE_CREATE))
        store.load_file(_write_contract(tmp_path, "data.sg", DATA_TABLE_CREATE))

        net_gene = store.get_gene("bridge_create")
        data_gene = store.get_gene("table_create")
        assert net_gene.domain == "network"
        assert data_gene.domain == "data"
        assert contracts_compatible(net_gene, data_gene)

    def test_diagnostic_compatible_across_domains(self, tmp_path):
        """Diagnostic genes with matching signatures are compatible."""
        store = ContractStore()
        store.load_file(_write_contract(tmp_path, "net.sg", NETWORK_BRIDGE_VERIFY))
        store.load_file(_write_contract(tmp_path, "data.sg", DATA_TABLE_VERIFY))

        net_gene = store.get_gene("bridge_verify")
        data_gene = store.get_gene("table_verify")
        assert contracts_compatible(net_gene, data_gene)

    def test_family_mismatch_rejected(self, tmp_path):
        """Configuration and diagnostic genes are not compatible."""
        store = ContractStore()
        store.load_file(_write_contract(tmp_path, "net.sg", NETWORK_BRIDGE_CREATE))
        store.load_file(_write_contract(tmp_path, "diag.sg", NETWORK_BRIDGE_VERIFY))

        config_gene = store.get_gene("bridge_create")
        diag_gene = store.get_gene("bridge_verify")
        assert config_gene.family == GeneFamily.CONFIGURATION
        assert diag_gene.family == GeneFamily.DIAGNOSTIC
        assert not contracts_compatible(config_gene, diag_gene)

    def test_pool_metadata_roundtrip(self, tmp_path):
        """Contract metadata serializes correctly for pool transport."""
        store = ContractStore()
        store.load_file(_write_contract(tmp_path, "net.sg", NETWORK_BRIDGE_CREATE))

        meta = contract_to_pool_metadata(store, "bridge_create")
        assert meta is not None
        assert meta["domain"] == "network"
        assert meta["family"] == "configuration"
        assert len(meta["takes"]) == 2
        assert len(meta["gives"]) == 2


class TestCrossDomainPoolFlow:
    """End-to-end: push from one domain, pull from another via pool server."""

    def _push_allele(self, client, locus, domain, family, takes, gives,
                     sha=None, fitness=0.90):
        sha = sha or "a" * 64
        payload = {
            "locus": locus,
            "sha256": sha,
            "source": "def execute(i): return '{\"success\": true}'",
            "generation": 1,
            "fitness": fitness,
            "successful_invocations": 100,
            "total_invocations": 105,
            "domain": domain,
            "contract": {
                "family": family,
                "takes": takes,
                "gives": gives,
            },
        }
        resp = client.post("/pool/push", json=payload)
        assert resp.status_code == 200
        return sha

    def _pull_cross_domain(self, client, locus, domain, family, takes, gives):
        contract = json.dumps({
            "domain": domain,
            "family": family,
            "takes": takes,
            "gives": gives,
        })
        resp = client.get(
            f"/pool/pull/{locus}?cross_domain=true&contract={contract}"
        )
        assert resp.status_code == 200
        return resp.json()["alleles"]

    def test_push_network_pull_data(self, pool_client):
        """Push a network allele, pull it from data domain."""
        takes = [{"name": "name", "type": "string", "required": True}]
        gives = [{"name": "success", "type": "bool", "required": True}]

        self._push_allele(pool_client, "bridge_create", "network",
                          "configuration", takes, gives)

        alleles = self._pull_cross_domain(
            pool_client, "table_create", "data",
            "configuration", takes, gives,
        )
        assert len(alleles) == 1
        assert alleles[0]["sha256"] == "a" * 64

    def test_family_mismatch_blocks_pull(self, pool_client):
        """Configuration allele cannot be pulled by diagnostic consumer."""
        takes = [{"name": "name", "type": "string", "required": True}]
        gives = [{"name": "success", "type": "bool", "required": True}]

        self._push_allele(pool_client, "bridge_create", "network",
                          "configuration", takes, gives)

        alleles = self._pull_cross_domain(
            pool_client, "table_verify", "data",
            "diagnostic", takes, gives,
        )
        assert len(alleles) == 0

    def test_same_domain_excluded(self, pool_client):
        """Cross-domain pull excludes alleles from the same domain."""
        takes = [{"name": "name", "type": "string", "required": True}]
        gives = [{"name": "success", "type": "bool", "required": True}]

        self._push_allele(pool_client, "bridge_create", "network",
                          "configuration", takes, gives)

        alleles = self._pull_cross_domain(
            pool_client, "bridge_verify_alt", "network",
            "configuration", takes, gives,
        )
        assert len(alleles) == 0

    def test_field_type_mismatch_blocks_pull(self, pool_client):
        """Incompatible field types prevent cross-domain pull."""
        takes_str = [{"name": "name", "type": "string", "required": True}]
        takes_int = [{"name": "name", "type": "int", "required": True}]
        gives = [{"name": "success", "type": "bool", "required": True}]

        self._push_allele(pool_client, "bridge_create", "network",
                          "configuration", takes_str, gives)

        alleles = self._pull_cross_domain(
            pool_client, "table_create", "data",
            "configuration", takes_int, gives,
        )
        assert len(alleles) == 0

    def test_multiple_domains_mixed(self, pool_client):
        """Alleles from multiple domains, only compatible ones returned."""
        takes = [{"name": "name", "type": "string", "required": True}]
        gives = [{"name": "success", "type": "bool", "required": True}]

        # Push from network (configuration)
        self._push_allele(pool_client, "bridge_create", "network",
                          "configuration", takes, gives, sha="a" * 64)

        # Push from storage (configuration, compatible)
        self._push_allele(pool_client, "vol_create", "storage",
                          "configuration", takes, gives, sha="b" * 64)

        # Push from network (diagnostic, incompatible family)
        self._push_allele(pool_client, "bridge_verify", "network",
                          "diagnostic", takes, gives, sha="c" * 64)

        # Pull from data domain as configuration
        alleles = self._pull_cross_domain(
            pool_client, "table_create", "data",
            "configuration", takes, gives,
        )
        shas = {a["sha256"] for a in alleles}
        assert "a" * 64 in shas  # network configuration
        assert "b" * 64 in shas  # storage configuration
        assert "c" * 64 not in shas  # network diagnostic excluded

    def test_normalized_fitness_in_cross_domain(self, pool_client):
        """Cross-domain alleles include normalized fitness scores."""
        takes = [{"name": "name", "type": "string", "required": True}]
        gives = [{"name": "success", "type": "bool", "required": True}]

        # Push several alleles to build domain stats
        for i in range(5):
            self._push_allele(
                pool_client, "bridge_create", "network",
                "configuration", takes, gives,
                sha=f"{i}" * 64, fitness=0.7 + i * 0.05,
            )

        alleles = self._pull_cross_domain(
            pool_client, "table_create", "data",
            "configuration", takes, gives,
        )
        assert len(alleles) == 5
        for a in alleles:
            assert "normalized_fitness" in a
