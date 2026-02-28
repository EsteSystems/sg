"""Tests for web dashboard API endpoints."""
import json
import shutil
import os
import pytest
from pathlib import Path

pytest.importorskip("fastapi", reason="FastAPI not available (requires sg[dashboard])", exc_type=ImportError)

from sg.contracts import ContractStore
from sg.fusion import FusionTracker
from sg.phenotype import PhenotypeMap
from sg.registry import Registry


CONTRACTS_DIR = Path(__file__).parent.parent / "contracts"
GENES_DIR = Path(__file__).parent.parent / "genes"
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def dashboard_project(tmp_path):
    """Set up a project and configure dashboard."""
    shutil.copytree(FIXTURES_DIR, tmp_path / "fixtures")
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
    FusionTracker.open(tmp_path / "fusion_tracker.json").save(tmp_path / "fusion_tracker.json")
    return tmp_path


@pytest.fixture
def client(dashboard_project):
    """Create a TestClient with project root configured."""
    from fastapi.testclient import TestClient
    import sg.dashboard as dash
    dash._project_root = dashboard_project
    return TestClient(dash.app)


class TestStatusEndpoint:
    def test_status_returns_counts(self, client):
        """GET /api/status returns genome counts."""
        resp = client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "loci_count" in data
        assert "allele_count" in data
        assert "pathway_count" in data
        assert data["loci_count"] > 0
        assert data["allele_count"] > 0


class TestLociEndpoint:
    def test_loci_list(self, client):
        """GET /api/loci returns list of loci."""
        resp = client.get("/api/loci")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) > 0
        locus = data[0]
        assert "name" in locus
        assert "dominant_sha" in locus
        assert "allele_count" in locus
        assert "dominant_fitness" in locus


class TestLocusDetail:
    def test_locus_detail(self, client):
        """GET /api/locus/{name} returns alleles and contract."""
        resp = client.get("/api/locus/bridge_create")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "bridge_create"
        assert len(data["alleles"]) >= 1
        assert data["contract"] is not None
        assert "does" in data["contract"]

    def test_locus_allele_fields(self, client):
        """Alleles have all expected fields."""
        resp = client.get("/api/locus/bridge_create")
        data = resp.json()
        allele = data["alleles"][0]
        for field in ["sha", "generation", "fitness", "state",
                      "successful_invocations", "failed_invocations"]:
            assert field in allele


class TestPathwaysEndpoint:
    def test_pathways_list(self, client):
        """GET /api/pathways returns pathway list."""
        resp = client.get("/api/pathways")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) > 0
        pw = data[0]
        assert "name" in pw
        assert "fused" in pw
        assert "reinforcement_count" in pw


class TestAlleleSource:
    def test_allele_source(self, client, dashboard_project):
        """GET /api/allele/{sha}/source returns source code."""
        registry = Registry.open(dashboard_project / ".sg" / "registry")
        sha = list(registry.alleles.keys())[0]

        resp = client.get(f"/api/allele/{sha}/source")
        assert resp.status_code == 200
        data = resp.json()
        assert "source" in data
        assert "def execute" in data["source"]

    def test_allele_source_prefix(self, client, dashboard_project):
        """GET /api/allele/{sha_prefix}/source works with prefix."""
        registry = Registry.open(dashboard_project / ".sg" / "registry")
        sha = list(registry.alleles.keys())[0]

        resp = client.get(f"/api/allele/{sha[:12]}/source")
        assert resp.status_code == 200

    def test_allele_source_not_found(self, client):
        """GET /api/allele/{bad_sha}/source returns 404."""
        resp = client.get("/api/allele/nonexistent/source")
        assert resp.status_code == 404


class TestDashboardHTML:
    def test_html_serves(self, client):
        """GET / returns HTML page."""
        resp = client.get("/")
        assert resp.status_code == 200
        assert "Software Genome Dashboard" in resp.text
        assert "<script>" in resp.text


class TestFederationEndpoints:
    def test_receive_allele(self, client, dashboard_project):
        """POST /api/federation/receive registers an allele."""
        resp = client.post("/api/federation/receive", json={
            "source": "def execute(i): return '{\"success\": true}'",
            "locus": "bridge_create",
            "generation": 5,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "sha" in data

    def test_serve_alleles(self, client):
        """GET /api/federation/alleles/{locus} serves alleles."""
        resp = client.get("/api/federation/alleles/bridge_create")
        assert resp.status_code == 200
        data = resp.json()
        assert "alleles" in data
        assert len(data["alleles"]) >= 1
        assert "source" in data["alleles"][0]
