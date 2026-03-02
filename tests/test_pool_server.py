"""Tests for the central genome pool server."""
import json
import pytest
from pathlib import Path

from sg.pool_server import (
    PoolStore, ContractMetadata, DomainStats,
    _contracts_compatible_dicts, create_pool_app,
)


# --- PoolStore unit tests ---


class TestPoolStore:
    def test_store_and_retrieve(self, tmp_path):
        store = PoolStore(tmp_path / "pool")
        store.ensure_dirs()

        data = {
            "locus": "bridge_create",
            "sha256": "a" * 64,
            "source": "def execute(i): pass",
            "fitness": 0.95,
            "generation": 1,
            "domain": "network",
        }
        sha = store.store_allele(data, "org-1")
        assert sha == "a" * 64

        alleles = store.get_alleles_for_locus("bridge_create")
        assert len(alleles) == 1
        assert alleles[0].sha256 == "a" * 64
        assert alleles[0].fitness == 0.95
        assert alleles[0].organism_id == "org-1"

    def test_idempotent_store(self, tmp_path):
        store = PoolStore(tmp_path / "pool")
        store.ensure_dirs()

        data = {
            "locus": "bridge_create",
            "sha256": "a" * 64,
            "source": "def execute(i): pass",
            "fitness": 0.95,
            "domain": "network",
        }
        store.store_allele(data, "org-1")
        store.store_allele(data, "org-1")

        alleles = store.get_alleles_for_locus("bridge_create")
        assert len(alleles) == 1

    def test_locus_lookup_multiple(self, tmp_path):
        store = PoolStore(tmp_path / "pool")
        store.ensure_dirs()

        for i in range(3):
            sha = f"{i}" * 64
            store.store_allele({
                "locus": "bridge_create", "sha256": sha,
                "source": f"# variant {i}", "fitness": 0.5 + i * 0.1,
                "domain": "network",
            }, "org-1")

        alleles = store.get_alleles_for_locus("bridge_create")
        assert len(alleles) == 3
        # Sorted by fitness descending
        assert alleles[0].fitness >= alleles[1].fitness >= alleles[2].fitness

    def test_locus_lookup_with_limit(self, tmp_path):
        store = PoolStore(tmp_path / "pool")
        store.ensure_dirs()

        for i in range(5):
            store.store_allele({
                "locus": "test_gene", "sha256": f"{i}" * 64,
                "source": f"# v{i}", "fitness": 0.5 + i * 0.1,
                "domain": "network",
            }, "org-1")

        assert len(store.get_alleles_for_locus("test_gene", limit=2)) == 2

    def test_contract_storage(self, tmp_path):
        store = PoolStore(tmp_path / "pool")
        store.ensure_dirs()

        meta = ContractMetadata(
            locus="bridge_create", domain="network", family="configuration",
            takes=[{"name": "bridge_name", "type": "string", "required": True}],
            gives=[{"name": "success", "type": "bool", "required": True}],
        )
        store.store_contract("bridge_create", meta)

        assert "bridge_create" in store._contracts
        assert store._contracts["bridge_create"].domain == "network"

    def test_domain_stats(self, tmp_path):
        store = PoolStore(tmp_path / "pool")
        store.ensure_dirs()

        for i in range(4):
            store.store_allele({
                "locus": "test_gene", "sha256": f"{i}" * 64,
                "source": f"# v{i}", "fitness": 0.8 + i * 0.05,
                "domain": "network",
            }, "org-1")

        stats = store._domains["network"]
        assert stats.allele_count == 4
        assert stats.avg > 0
        assert stats.stddev >= 0

    def test_save_and_load(self, tmp_path):
        store = PoolStore(tmp_path / "pool")
        store.ensure_dirs()

        store.store_allele({
            "locus": "bridge_create", "sha256": "a" * 64,
            "source": "def execute(i): pass", "fitness": 0.9,
            "domain": "network",
        }, "org-1")
        store.record_push("org-1")
        store.save()

        store2 = PoolStore(tmp_path / "pool")
        store2.load()
        assert "a" * 64 in store2._alleles
        assert "bridge_create" in store2._loci
        assert "org-1" in store2._organisms
        assert "network" in store2._domains


# --- Contract compatibility ---


class TestContractCompatibility:
    def test_compatible(self):
        a = {"takes": [{"name": "x", "type": "string", "required": True}],
             "gives": [{"name": "success", "type": "bool", "required": True}]}
        b = {"takes": [{"name": "x", "type": "string", "required": True}],
             "gives": [{"name": "success", "type": "bool", "required": True}]}
        assert _contracts_compatible_dicts(a, b)

    def test_incompatible_type(self):
        a = {"takes": [{"name": "x", "type": "string", "required": True}],
             "gives": []}
        b = {"takes": [{"name": "x", "type": "int", "required": True}],
             "gives": []}
        assert not _contracts_compatible_dicts(a, b)

    def test_missing_field(self):
        a = {"takes": [{"name": "x", "type": "string", "required": True}],
             "gives": []}
        b = {"takes": [{"name": "y", "type": "string", "required": True}],
             "gives": []}
        assert not _contracts_compatible_dicts(a, b)

    def test_optional_field_not_required_in_target(self):
        a = {"takes": [{"name": "x", "type": "string", "required": True, "optional": True}],
             "gives": []}
        b = {"takes": [], "gives": []}
        assert _contracts_compatible_dicts(a, b)

    def test_superset_compatible(self):
        a = {"takes": [{"name": "x", "type": "string", "required": True}],
             "gives": []}
        b = {"takes": [
            {"name": "x", "type": "string", "required": True},
            {"name": "y", "type": "int", "required": True},
        ], "gives": []}
        assert _contracts_compatible_dicts(a, b)


# --- DomainStats ---


class TestDomainStats:
    def test_empty_stats(self):
        ds = DomainStats(domain="test")
        assert ds.avg == 0.0
        assert ds.stddev == 0.0

    def test_computed_stats(self):
        ds = DomainStats(domain="test", allele_count=4,
                         fitness_sum=3.6, fitness_sum_sq=3.28)
        assert abs(ds.avg - 0.9) < 0.01
        assert ds.stddev > 0


# --- FastAPI endpoint tests ---


@pytest.fixture
def pool_app(tmp_path):
    """Create a pool app with no auth, reciprocity=0."""
    app = create_pool_app(tmp_path / "pool", token=None, reciprocity=0)
    from starlette.testclient import TestClient
    return TestClient(app)


@pytest.fixture
def pool_app_auth(tmp_path):
    """Create a pool app with auth enabled."""
    app = create_pool_app(tmp_path / "pool", token="secret-token", reciprocity=0)
    from starlette.testclient import TestClient
    return TestClient(app)


@pytest.fixture
def pool_app_reciprocity(tmp_path):
    """Create a pool app with reciprocity=1."""
    app = create_pool_app(tmp_path / "pool", token=None, reciprocity=1)
    from starlette.testclient import TestClient
    return TestClient(app)


def _push_payload(locus="bridge_create", sha=None, domain="network",
                  fitness=0.95, contract=None):
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
    }
    if contract is not None:
        payload["contract"] = contract
    return payload


class TestPoolPush:
    def test_basic_push(self, pool_app):
        resp = pool_app.post("/pool/push", json=_push_payload())
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["sha256"] == "a" * 64

    def test_push_with_contract(self, pool_app):
        payload = _push_payload(contract={
            "family": "configuration",
            "takes": [{"name": "bridge_name", "type": "string", "required": True}],
            "gives": [{"name": "success", "type": "bool", "required": True}],
        })
        resp = pool_app.post("/pool/push", json=payload)
        assert resp.status_code == 200

    def test_push_missing_fields(self, pool_app):
        resp = pool_app.post("/pool/push", json={"locus": "test"})
        assert resp.status_code == 400
        assert "missing" in resp.json()["error"]

    def test_push_with_organism_header(self, pool_app):
        resp = pool_app.post(
            "/pool/push", json=_push_payload(),
            headers={"X-SG-Organism": "org-42"},
        )
        assert resp.status_code == 200

        # Check organism tracking
        orgs = pool_app.get("/pool/organisms").json()
        ids = [o["organism_id"] for o in orgs["organisms"]]
        assert "org-42" in ids


class TestPoolPull:
    def test_push_then_pull(self, pool_app):
        pool_app.post("/pool/push", json=_push_payload())
        resp = pool_app.get("/pool/pull/bridge_create")
        assert resp.status_code == 200
        alleles = resp.json()["alleles"]
        assert len(alleles) == 1
        assert alleles[0]["sha256"] == "a" * 64
        assert "source" in alleles[0]

    def test_pull_empty_locus(self, pool_app):
        resp = pool_app.get("/pool/pull/nonexistent")
        assert resp.status_code == 200
        assert resp.json()["alleles"] == []

    def test_pull_limit(self, pool_app):
        for i in range(5):
            pool_app.post("/pool/push", json=_push_payload(
                sha=f"{i}" * 64, fitness=0.5 + i * 0.1,
            ))
        resp = pool_app.get("/pool/pull/bridge_create?limit=2")
        assert len(resp.json()["alleles"]) == 2


class TestReciprocity:
    def test_blocked_without_push(self, pool_app_reciprocity):
        resp = pool_app_reciprocity.get("/pool/pull/bridge_create")
        assert resp.status_code == 403
        assert "reciprocity" in resp.json()["error"]

    def test_allowed_after_push(self, pool_app_reciprocity):
        pool_app_reciprocity.post(
            "/pool/push", json=_push_payload(),
            headers={"X-SG-Organism": "org-1"},
        )
        resp = pool_app_reciprocity.get(
            "/pool/pull/bridge_create",
            headers={"X-SG-Organism": "org-1"},
        )
        assert resp.status_code == 200

    def test_disabled_with_zero(self, pool_app):
        # pool_app has reciprocity=0
        resp = pool_app.get("/pool/pull/bridge_create")
        assert resp.status_code == 200


class TestCrossDomain:
    def _push_network_allele(self, client):
        """Push a network allele with contract metadata."""
        payload = _push_payload(
            locus="bridge_create", domain="network",
            contract={
                "family": "configuration",
                "takes": [{"name": "name", "type": "string", "required": True}],
                "gives": [{"name": "success", "type": "bool", "required": True}],
            },
        )
        client.post("/pool/push", json=payload)

    def test_cross_domain_compatible(self, pool_app):
        self._push_network_allele(pool_app)

        # Pull from data domain with compatible contract
        contract = json.dumps({
            "domain": "data",
            "family": "configuration",
            "takes": [{"name": "name", "type": "string", "required": True}],
            "gives": [{"name": "success", "type": "bool", "required": True}],
        })
        resp = pool_app.get(
            f"/pool/pull/data_transform?cross_domain=true&contract={contract}"
        )
        assert resp.status_code == 200
        assert len(resp.json()["alleles"]) == 1

    def test_cross_domain_incompatible(self, pool_app):
        self._push_network_allele(pool_app)

        # Pull with incompatible contract (different field type)
        contract = json.dumps({
            "domain": "data",
            "family": "configuration",
            "takes": [{"name": "name", "type": "int", "required": True}],
            "gives": [{"name": "success", "type": "bool", "required": True}],
        })
        resp = pool_app.get(
            f"/pool/pull/data_transform?cross_domain=true&contract={contract}"
        )
        assert resp.status_code == 200
        assert len(resp.json()["alleles"]) == 0

    def test_cross_domain_missing_contract(self, pool_app):
        resp = pool_app.get("/pool/pull/test?cross_domain=true")
        assert resp.status_code == 400
        assert "contract" in resp.json()["error"]

    def test_same_domain_excluded(self, pool_app):
        self._push_network_allele(pool_app)

        # Cross-domain pull from same domain should find nothing
        contract = json.dumps({
            "domain": "network",
            "family": "configuration",
            "takes": [{"name": "name", "type": "string", "required": True}],
            "gives": [{"name": "success", "type": "bool", "required": True}],
        })
        resp = pool_app.get(
            f"/pool/pull/bridge_verify?cross_domain=true&contract={contract}"
        )
        assert resp.status_code == 200
        assert len(resp.json()["alleles"]) == 0


class TestFitnessNormalization:
    def test_normalized_fitness_computed(self, pool_app):
        # Push several alleles to build domain stats
        for i in range(5):
            pool_app.post("/pool/push", json=_push_payload(
                sha=f"{i}" * 64, fitness=0.7 + i * 0.05,
            ))

        resp = pool_app.get("/pool/pull/bridge_create")
        alleles = resp.json()["alleles"]
        # All should have normalized_fitness
        for a in alleles:
            assert "normalized_fitness" in a

    def test_normalization_values(self, tmp_path):
        store = PoolStore(tmp_path / "pool")
        store.ensure_dirs()

        # Manually set domain stats
        store._domains["test"] = DomainStats(
            domain="test", allele_count=100,
            fitness_sum=90.0,  # avg = 0.9
            fitness_sum_sq=81.1,  # variance = 0.001, stddev ≈ 0.0316
        )
        normalized = store.normalize_fitness(0.95, "test")
        # (0.95 - 0.9) / max(stddev, 0.001)
        assert normalized > 0


class TestAuthentication:
    def test_rejected_without_token(self, pool_app_auth):
        resp = pool_app_auth.post("/pool/push", json=_push_payload())
        assert resp.status_code == 401

    def test_rejected_wrong_token(self, pool_app_auth):
        resp = pool_app_auth.post(
            "/pool/push", json=_push_payload(),
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert resp.status_code == 401

    def test_accepted_correct_token(self, pool_app_auth):
        resp = pool_app_auth.post(
            "/pool/push", json=_push_payload(),
            headers={"Authorization": "Bearer secret-token"},
        )
        assert resp.status_code == 200

    def test_status_requires_auth(self, pool_app_auth):
        resp = pool_app_auth.get("/pool/status")
        assert resp.status_code == 401

    def test_status_with_auth(self, pool_app_auth):
        resp = pool_app_auth.get(
            "/pool/status",
            headers={"Authorization": "Bearer secret-token"},
        )
        assert resp.status_code == 200


class TestPoolStatus:
    def test_empty_pool(self, pool_app):
        resp = pool_app.get("/pool/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["allele_count"] == 0
        assert data["locus_count"] == 0
        assert data["organism_count"] == 0
        assert data["domain_count"] == 0

    def test_status_after_pushes(self, pool_app):
        pool_app.post("/pool/push", json=_push_payload(
            locus="gene_a", sha="a" * 64, domain="network",
        ))
        pool_app.post("/pool/push", json=_push_payload(
            locus="gene_b", sha="b" * 64, domain="data",
        ))

        resp = pool_app.get("/pool/status")
        data = resp.json()
        assert data["allele_count"] == 2
        assert data["locus_count"] == 2
        assert data["domain_count"] == 2


class TestPoolOrganisms:
    def test_organisms_tracking(self, pool_app):
        pool_app.post(
            "/pool/push", json=_push_payload(),
            headers={"X-SG-Organism": "org-A"},
        )
        pool_app.post(
            "/pool/push", json=_push_payload(sha="b" * 64),
            headers={"X-SG-Organism": "org-B"},
        )

        resp = pool_app.get("/pool/organisms")
        assert resp.status_code == 200
        organisms = resp.json()["organisms"]
        assert len(organisms) == 2
        ids = {o["organism_id"] for o in organisms}
        assert ids == {"org-A", "org-B"}


class TestBackwardCompat:
    def test_payload_without_new_fields(self, pool_app):
        """Existing PoolClient payload format works without domain/contract."""
        payload = {
            "locus": "bridge_create",
            "sha256": "c" * 64,
            "source": "def execute(i): return '{\"success\": true}'",
            "generation": 0,
            "fitness": 0.9,
            "successful_invocations": 100,
            "total_invocations": 105,
        }
        resp = pool_app.post("/pool/push", json=payload)
        assert resp.status_code == 200

        resp = pool_app.get("/pool/pull/bridge_create")
        alleles = resp.json()["alleles"]
        assert len(alleles) == 1
        assert alleles[0]["domain"] == "unknown"

    def test_pull_without_cross_domain(self, pool_app):
        """Normal pull works without cross_domain param."""
        pool_app.post("/pool/push", json=_push_payload())
        resp = pool_app.get("/pool/pull/bridge_create")
        assert resp.status_code == 200
        assert len(resp.json()["alleles"]) == 1
