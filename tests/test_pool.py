"""Tests for gene pool client — push eligibility, pull integration, membership tracking."""
import json
import time
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from sg.pool import (
    PoolConfig, PoolClient, PoolMembership, MembershipStore,
    is_push_eligible, load_pool_configs,
    DEFAULT_MIN_FITNESS, DEFAULT_MIN_INVOCATIONS,
)
from sg.registry import Registry, AlleleMetadata
from sg.phenotype import PhenotypeMap
from sg import arena


# --- Pool configuration ---


class TestPoolConfig:
    def test_load_pool_configs(self, tmp_path):
        pools_toml = tmp_path / "pools.toml"
        pools_toml.write_text("""\
[[pool]]
name = "public"
url = "https://pool.example.com/v1"

[[pool]]
name = "private"
url = "https://sg.acme.corp/v1"
token_env = "SG_POOL_TOKEN"
""")
        configs = load_pool_configs(pools_toml)
        assert len(configs) == 2
        assert configs[0].name == "public"
        assert configs[0].url == "https://pool.example.com/v1"
        assert configs[0].token_env == ""
        assert configs[1].name == "private"
        assert configs[1].token_env == "SG_POOL_TOKEN"

    def test_load_missing_file(self, tmp_path):
        configs = load_pool_configs(tmp_path / "nonexistent.toml")
        assert configs == []

    def test_token_from_env(self, monkeypatch):
        monkeypatch.setenv("MY_TOKEN", "secret123")
        cfg = PoolConfig(name="test", url="http://localhost", token_env="MY_TOKEN")
        assert cfg.token == "secret123"

    def test_token_missing_env(self):
        cfg = PoolConfig(name="test", url="http://localhost", token_env="NONEXISTENT_VAR")
        assert cfg.token is None

    def test_token_no_env_key(self):
        cfg = PoolConfig(name="test", url="http://localhost")
        assert cfg.token is None


# --- Push eligibility ---


class TestPushEligibility:
    def _make_allele(self, fitness_ratio=0.9, invocations=100, state="dominant"):
        allele = AlleleMetadata(sha256="a" * 64, locus="test_gene")
        allele.state = state
        allele.successful_invocations = int(invocations * fitness_ratio)
        allele.failed_invocations = invocations - allele.successful_invocations
        return allele

    def test_eligible_dominant(self):
        allele = self._make_allele(fitness_ratio=0.95, invocations=100)
        assert is_push_eligible(allele)

    def test_eligible_recessive(self):
        """Recessive alleles can be pushed if they meet criteria."""
        allele = self._make_allele(fitness_ratio=0.95, invocations=100, state="recessive")
        assert is_push_eligible(allele)

    def test_not_eligible_low_fitness(self):
        allele = self._make_allele(fitness_ratio=0.5, invocations=100)
        assert not is_push_eligible(allele)

    def test_not_eligible_low_invocations(self):
        allele = self._make_allele(fitness_ratio=0.95, invocations=10)
        assert not is_push_eligible(allele)

    def test_not_eligible_deprecated(self):
        allele = self._make_allele(fitness_ratio=0.95, invocations=100, state="deprecated")
        assert not is_push_eligible(allele)

    def test_custom_thresholds(self):
        allele = self._make_allele(fitness_ratio=0.7, invocations=30)
        assert not is_push_eligible(allele)
        assert is_push_eligible(allele, min_fitness=0.5, min_invocations=20)


# --- Membership store ---


class TestMembershipStore:
    def test_create_and_save(self, tmp_path):
        path = tmp_path / ".sg" / "pool_memberships.json"
        store = MembershipStore(path)
        m = store.get_or_create("public", "https://pool.example.com")
        m.total_pushed = 5
        m.total_pulled = 3
        m.last_push = 1000.0
        store.save()

        # Reload
        store2 = MembershipStore(path)
        m2 = store2.get("public")
        assert m2 is not None
        assert m2.total_pushed == 5
        assert m2.total_pulled == 3
        assert m2.last_push == 1000.0

    def test_get_missing(self, tmp_path):
        store = MembershipStore(tmp_path / "memberships.json")
        assert store.get("nonexistent") is None

    def test_get_or_create_idempotent(self, tmp_path):
        store = MembershipStore(tmp_path / "memberships.json")
        m1 = store.get_or_create("test", "http://localhost")
        m1.total_pushed = 10
        m2 = store.get_or_create("test", "http://localhost")
        assert m2.total_pushed == 10  # same object


# --- Pool client (with mocked HTTP) ---


@pytest.fixture
def pool_project(tmp_path):
    """Set up a minimal project with pool config, registry, and phenotype."""
    # Create pools.toml
    (tmp_path / "pools.toml").write_text("""\
[[pool]]
name = "test-pool"
url = "http://localhost:9999"
""")

    # Set up registry with a high-fitness allele
    registry = Registry.open(tmp_path / ".sg" / "registry")
    source = "def execute(i): return '{\"success\": true}'"
    sha = registry.register(source, "bridge_create", generation=0)
    allele = registry.get(sha)
    allele.state = "dominant"
    allele.successful_invocations = 100
    allele.failed_invocations = 5
    registry.save_index()

    phenotype = PhenotypeMap()
    phenotype.promote("bridge_create", sha)
    phenotype.save(tmp_path / "phenotype.toml")

    return tmp_path, registry, phenotype, sha


class TestPoolClientPush:
    def test_push_eligible_allele(self, pool_project):
        root, registry, phenotype, sha = pool_project
        client = PoolClient(root)

        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch("sg.pool.httpx") as mock_httpx:
            mock_httpx.post.return_value = mock_resp
            result = client.push("bridge_create", registry, phenotype, "test-pool")

        assert result is True
        # Verify membership updated
        m = client.memberships.get("test-pool")
        assert m is not None
        assert m.total_pushed == 1
        assert m.last_push is not None

    def test_push_ineligible_allele(self, pool_project):
        root, registry, phenotype, sha = pool_project
        # Make allele ineligible by lowering invocations
        allele = registry.get(sha)
        allele.successful_invocations = 5
        allele.failed_invocations = 0

        client = PoolClient(root)
        with patch("sg.pool.httpx") as mock_httpx:
            result = client.push("bridge_create", registry, phenotype, "test-pool")

        assert result is False
        mock_httpx.post.assert_not_called()

    def test_push_no_dominant(self, pool_project):
        root, registry, phenotype, sha = pool_project
        # Clear phenotype
        phenotype.loci.clear()

        client = PoolClient(root)
        result = client.push("bridge_create", registry, phenotype, "test-pool")
        assert result is False

    def test_push_server_error(self, pool_project):
        root, registry, phenotype, sha = pool_project
        client = PoolClient(root)

        mock_resp = MagicMock()
        mock_resp.status_code = 500

        with patch("sg.pool.httpx") as mock_httpx:
            mock_httpx.post.return_value = mock_resp
            result = client.push("bridge_create", registry, phenotype, "test-pool")

        assert result is False

    def test_push_unknown_pool(self, pool_project):
        root, registry, phenotype, sha = pool_project
        client = PoolClient(root)

        with pytest.raises(ValueError, match="no pool configured"):
            client.push("bridge_create", registry, phenotype, "nonexistent")


class TestPoolClientPull:
    def test_pull_alleles(self, pool_project):
        root, registry, phenotype, sha = pool_project
        client = PoolClient(root)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "alleles": [{
                "source": "def execute(i): return '{\"success\": true}' # variant",
                "locus": "bridge_create",
                "generation": 1,
            }]
        }

        with patch("sg.pool.httpx") as mock_httpx:
            mock_httpx.get.return_value = mock_resp
            shas = client.pull("bridge_create", registry, phenotype, "test-pool")

        assert len(shas) == 1
        # Verify allele is recessive
        imported = registry.get(shas[0])
        assert imported is not None
        assert imported.state == "recessive"

        # Verify membership updated
        m = client.memberships.get("test-pool")
        assert m is not None
        assert m.total_pulled == 1

    def test_pull_empty_response(self, pool_project):
        root, registry, phenotype, sha = pool_project
        client = PoolClient(root)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"alleles": []}

        with patch("sg.pool.httpx") as mock_httpx:
            mock_httpx.get.return_value = mock_resp
            shas = client.pull("bridge_create", registry, phenotype, "test-pool")

        assert shas == []

    def test_pull_server_error(self, pool_project):
        root, registry, phenotype, sha = pool_project
        client = PoolClient(root)

        mock_resp = MagicMock()
        mock_resp.status_code = 500

        with patch("sg.pool.httpx") as mock_httpx:
            mock_httpx.get.return_value = mock_resp
            shas = client.pull("bridge_create", registry, phenotype, "test-pool")

        assert shas == []


class TestPoolClientAuto:
    def test_auto_push_and_pull(self, pool_project):
        root, registry, phenotype, sha = pool_project

        # Create a minimal ContractStore stand-in
        class FakeContractStore:
            def known_loci(self):
                return ["bridge_create"]

        client = PoolClient(root)

        push_resp = MagicMock()
        push_resp.status_code = 200

        pull_resp = MagicMock()
        pull_resp.status_code = 200
        pull_resp.json.return_value = {
            "alleles": [{
                "source": "def execute(i): return '{\"success\": true}' # pulled",
                "locus": "bridge_create",
                "generation": 2,
            }]
        }

        with patch("sg.pool.httpx") as mock_httpx:
            mock_httpx.post.return_value = push_resp
            mock_httpx.get.return_value = pull_resp
            result = client.auto(registry, phenotype, FakeContractStore(), "test-pool")

        assert result["pushed"] == 1
        assert result["pulled"] == 1
        assert result["push_errors"] == []
        assert result["pull_errors"] == []


# --- CLI pool command ---


class TestCmdPool:
    def test_cmd_pool_list(self, pool_project, capsys):
        root, registry, phenotype, sha = pool_project
        import argparse
        import os
        from sg.cli import cmd_pool

        old = os.environ.get("SG_PROJECT_ROOT")
        os.environ["SG_PROJECT_ROOT"] = str(root)
        try:
            args = argparse.Namespace(pool_command="list")
            cmd_pool(args)
        finally:
            if old is None:
                os.environ.pop("SG_PROJECT_ROOT", None)
            else:
                os.environ["SG_PROJECT_ROOT"] = old

        captured = capsys.readouterr()
        assert "test-pool" in captured.out
        assert "localhost:9999" in captured.out

    def test_cmd_pool_no_subcommand(self, pool_project, capsys):
        root, registry, phenotype, sha = pool_project
        import argparse
        import os
        from sg.cli import cmd_pool

        old = os.environ.get("SG_PROJECT_ROOT")
        os.environ["SG_PROJECT_ROOT"] = str(root)
        try:
            args = argparse.Namespace(pool_command=None)
            cmd_pool(args)
        finally:
            if old is None:
                os.environ.pop("SG_PROJECT_ROOT", None)
            else:
                os.environ["SG_PROJECT_ROOT"] = old

        captured = capsys.readouterr()
        assert "usage" in captured.out
