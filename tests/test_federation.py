"""Tests for multi-organism federation."""
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from sg.federation import (
    PeerConfig, load_peers, export_allele, import_allele,
    push_allele, pull_alleles,
)
from sg.registry import Registry


class TestExportImport:
    def test_export_allele(self, tmp_path):
        """export_allele packages metadata + source."""
        registry = Registry.open(tmp_path / ".sg" / "registry")
        source = "def execute(i): return '{\"success\": true}'"
        sha = registry.register(source, "bridge_create")

        data = export_allele(registry, sha)
        assert data is not None
        assert data["sha256"] == sha
        assert data["locus"] == "bridge_create"
        assert data["source"] == source
        assert "fitness" in data

    def test_export_nonexistent(self, tmp_path):
        """export_allele returns None for missing SHA."""
        registry = Registry.open(tmp_path / ".sg" / "registry")
        assert export_allele(registry, "nonexistent") is None

    def test_import_allele(self, tmp_path):
        """import_allele registers the gene and returns SHA."""
        registry = Registry.open(tmp_path / ".sg" / "registry")
        data = {
            "source": "def execute(i): return '{\"success\": true}'",
            "locus": "bridge_create",
            "generation": 3,
        }
        sha = import_allele(registry, data)
        assert sha is not None

        allele = registry.get(sha)
        assert allele is not None
        assert allele.locus == "bridge_create"
        assert allele.generation == 3

    def test_round_trip(self, tmp_path):
        """Export then import produces the same source."""
        reg1 = Registry.open(tmp_path / "reg1")
        source = "def execute(i): return '{\"success\": true}'"
        sha1 = reg1.register(source, "bridge_create")

        data = export_allele(reg1, sha1)
        assert data is not None

        reg2 = Registry.open(tmp_path / "reg2")
        sha2 = import_allele(reg2, data)

        assert reg2.load_source(sha2) == source


class TestPeerConfig:
    def test_load_peers(self, tmp_path):
        """load_peers reads peers.json correctly."""
        peers_json = tmp_path / "peers.json"
        peers_json.write_text(json.dumps({
            "peers": [
                {"url": "http://host1:8420", "name": "server1"},
                {"url": "http://host2:8420"},
            ]
        }))
        peers = load_peers(peers_json)
        assert len(peers) == 2
        assert peers[0].url == "http://host1:8420"
        assert peers[0].name == "server1"
        assert peers[1].name == ""

    def test_load_peers_missing_file(self, tmp_path):
        """load_peers returns empty list if file doesn't exist."""
        peers = load_peers(tmp_path / "nonexistent.json")
        assert peers == []


class TestNetworkOps:
    def test_push_allele_success(self):
        """push_allele POSTs to peer and returns success."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch("sg.federation.httpx") as mock_httpx:
            mock_httpx.post.return_value = mock_resp
            peer = PeerConfig(url="http://peer:8420")
            result = push_allele(peer, {"source": "test", "locus": "test"})
            assert result is True
            mock_httpx.post.assert_called_once()

    def test_push_allele_failure(self):
        """push_allele returns False on error."""
        with patch("sg.federation.httpx") as mock_httpx:
            mock_httpx.post.side_effect = Exception("connection refused")
            peer = PeerConfig(url="http://down:8420")
            result = push_allele(peer, {"source": "test", "locus": "test"})
            assert result is False

    def test_pull_alleles_success(self):
        """pull_alleles GETs from peer and returns alleles."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "alleles": [{"source": "test", "locus": "bridge_create", "generation": 0}]
        }

        with patch("sg.federation.httpx") as mock_httpx:
            mock_httpx.get.return_value = mock_resp
            peer = PeerConfig(url="http://peer:8420")
            alleles = pull_alleles(peer, "bridge_create")
            assert len(alleles) == 1

    def test_pull_alleles_failure(self):
        """pull_alleles returns empty on error."""
        with patch("sg.federation.httpx") as mock_httpx:
            mock_httpx.get.side_effect = Exception("timeout")
            peer = PeerConfig(url="http://down:8420")
            alleles = pull_alleles(peer, "bridge_create")
            assert alleles == []
