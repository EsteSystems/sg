"""Tests for multi-organism federation."""
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from sg.federation import (
    PeerConfig, load_peers, export_allele, import_allele,
    push_allele, pull_alleles,
    compute_source_sha, sign_payload, verify_signature,
    verify_allele_integrity,
)
from sg.registry import Registry


class TestExportImport:
    def test_export_allele(self, tmp_path):
        """export_allele packages metadata + source + integrity hash."""
        registry = Registry.open(tmp_path / ".sg" / "registry")
        source = "def execute(i): return '{\"success\": true}'"
        sha = registry.register(source, "bridge_create")

        data = export_allele(registry, sha)
        assert data is not None
        assert data["sha256"] == sha
        assert data["locus"] == "bridge_create"
        assert data["source"] == source
        assert "fitness" in data
        assert "source_sha256" in data
        assert data["source_sha256"] == compute_source_sha(source)

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

    def test_push_with_secret_includes_signature(self):
        """push_allele sends X-SG-Signature header when peer has secret."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch("sg.federation.httpx") as mock_httpx:
            mock_httpx.post.return_value = mock_resp
            peer = PeerConfig(url="http://peer:8420", secret="s3cret")
            data = {"source": "test", "locus": "test"}
            push_allele(peer, data)
            call_kwargs = mock_httpx.post.call_args
            assert "X-SG-Signature" in call_kwargs.kwargs.get("headers", {})

    def test_pull_with_secret_includes_signature(self):
        """pull_alleles sends X-SG-Signature header when peer has secret."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"alleles": []}

        with patch("sg.federation.httpx") as mock_httpx:
            mock_httpx.get.return_value = mock_resp
            peer = PeerConfig(url="http://peer:8420", secret="s3cret")
            pull_alleles(peer, "bridge_create")
            call_kwargs = mock_httpx.get.call_args
            assert "X-SG-Signature" in call_kwargs.kwargs.get("headers", {})


class TestIntegrity:
    def test_source_sha_deterministic(self):
        """compute_source_sha is deterministic."""
        source = "def execute(i): return 'ok'"
        assert compute_source_sha(source) == compute_source_sha(source)

    def test_verify_allele_integrity_valid(self):
        """verify_allele_integrity passes for matching hash."""
        source = "def execute(i): return 'ok'"
        data = {"source": source, "source_sha256": compute_source_sha(source)}
        assert verify_allele_integrity(data) is True

    def test_verify_allele_integrity_tampered(self):
        """verify_allele_integrity fails for tampered source."""
        data = {
            "source": "def execute(i): return 'evil'",
            "source_sha256": compute_source_sha("def execute(i): return 'ok'"),
        }
        assert verify_allele_integrity(data) is False

    def test_verify_allele_integrity_no_hash(self):
        """verify_allele_integrity passes when no hash provided (backwards compat)."""
        data = {"source": "anything"}
        assert verify_allele_integrity(data) is True

    def test_import_rejects_tampered(self, tmp_path):
        """import_allele raises on integrity mismatch."""
        registry = Registry.open(tmp_path / ".sg" / "registry")
        data = {
            "source": "def execute(i): return 'evil'",
            "source_sha256": compute_source_sha("def execute(i): return 'ok'"),
            "locus": "bridge_create",
        }
        with pytest.raises(ValueError, match="integrity check failed"):
            import_allele(registry, data)


class TestSignature:
    def test_sign_and_verify(self):
        """sign_payload + verify_signature round-trips."""
        payload = {"locus": "bridge_create", "source": "test"}
        secret = "my-secret-key"
        sig = sign_payload(payload, secret)
        assert verify_signature(payload, sig, secret)

    def test_verify_wrong_secret(self):
        """Wrong secret fails verification."""
        payload = {"locus": "bridge_create"}
        sig = sign_payload(payload, "correct-secret")
        assert not verify_signature(payload, sig, "wrong-secret")

    def test_verify_tampered_payload(self):
        """Tampered payload fails verification."""
        payload = {"locus": "bridge_create"}
        sig = sign_payload(payload, "secret")
        payload["locus"] = "evil_locus"
        assert not verify_signature(payload, sig, "secret")

    def test_load_peers_with_secret(self, tmp_path):
        """Peers can have a secret field."""
        peers_json = tmp_path / "peers.json"
        peers_json.write_text(json.dumps({
            "peers": [
                {"url": "http://host:8420", "name": "s1", "secret": "abc123"},
            ]
        }))
        peers = load_peers(peers_json)
        assert peers[0].secret == "abc123"
