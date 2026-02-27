"""Tests for the phenotype map."""
import pytest
from sg.phenotype import PhenotypeMap


@pytest.fixture
def pm():
    return PhenotypeMap()


def test_promote(pm):
    pm.promote("bridge_create", "sha1")
    assert pm.get_dominant("bridge_create") == "sha1"
    assert pm.get_stack("bridge_create") == ["sha1"]


def test_promote_replaces_dominant(pm):
    pm.promote("bridge_create", "sha1")
    pm.promote("bridge_create", "sha2")
    assert pm.get_dominant("bridge_create") == "sha2"
    assert pm.get_stack("bridge_create") == ["sha2", "sha1"]


def test_add_to_fallback(pm):
    pm.promote("bridge_create", "sha1")
    pm.add_to_fallback("bridge_create", "sha2")
    assert pm.get_stack("bridge_create") == ["sha1", "sha2"]


def test_add_to_fallback_no_duplicate(pm):
    pm.promote("bridge_create", "sha1")
    pm.add_to_fallback("bridge_create", "sha2")
    pm.add_to_fallback("bridge_create", "sha2")
    assert pm.get_stack("bridge_create") == ["sha1", "sha2"]


def test_empty_stack(pm):
    assert pm.get_stack("bridge_create") == []


def test_fusion_state(pm):
    assert pm.get_fused("test_pathway") is None
    pm.set_fused("test_pathway", "fused_sha", "fingerprint")
    config = pm.get_fused("test_pathway")
    assert config.fused_sha == "fused_sha"
    assert config.composition_fingerprint == "fingerprint"


def test_clear_fusion(pm):
    pm.set_fused("test_pathway", "fused_sha", "fp")
    pm.clear_fused("test_pathway")
    assert pm.get_fused("test_pathway") is None


def test_save_and_load(pm, tmp_path):
    pm.promote("bridge_create", "sha1")
    pm.add_to_fallback("bridge_create", "sha2")
    pm.set_fused("test_pathway", "fused_sha", "fp123")

    path = tmp_path / "phenotype.toml"
    pm.save(path)

    pm2 = PhenotypeMap.load(path)
    assert pm2.get_dominant("bridge_create") == "sha1"
    assert pm2.get_stack("bridge_create") == ["sha1", "sha2"]
    config = pm2.get_fused("test_pathway")
    assert config.fused_sha == "fused_sha"
    assert config.composition_fingerprint == "fp123"
