"""Tests for the content-addressed registry."""
import pytest
from sg.registry import Registry, AlleleMetadata


@pytest.fixture
def registry(tmp_path):
    return Registry.open(tmp_path / "registry")


def test_register_and_retrieve(registry):
    source = "def execute(x): return x"
    sha = registry.register(source, "bridge_create")
    assert len(sha) == 64
    allele = registry.get(sha)
    assert allele is not None
    assert allele.locus == "bridge_create"
    assert allele.generation == 0


def test_register_idempotent(registry):
    source = "def execute(x): return x"
    sha1 = registry.register(source, "bridge_create")
    sha2 = registry.register(source, "bridge_create")
    assert sha1 == sha2


def test_register_different_sources(registry):
    sha1 = registry.register("def execute(x): return x", "bridge_create")
    sha2 = registry.register("def execute(x): return '!'", "bridge_create")
    assert sha1 != sha2


def test_load_source(registry):
    source = "def execute(x): return x"
    sha = registry.register(source, "bridge_create")
    loaded = registry.load_source(sha)
    assert loaded == source


def test_source_not_found(registry):
    assert registry.load_source("nonexistent") is None


def test_alleles_for_locus(registry):
    sha1 = registry.register("def execute(x): return 'a'", "bridge_create")
    sha2 = registry.register("def execute(x): return 'b'", "bridge_create")
    sha3 = registry.register("def execute(x): return 'c'", "bridge_stp")
    alleles = registry.alleles_for_locus("bridge_create")
    shas = [a.sha256 for a in alleles]
    assert sha1 in shas
    assert sha2 in shas
    assert sha3 not in shas


def test_save_and_load_index(registry):
    sha = registry.register("def execute(x): return x", "bridge_create")
    allele = registry.get(sha)
    allele.successful_invocations = 5
    registry.save_index()

    reg2 = Registry.open(registry.root)
    allele2 = reg2.get(sha)
    assert allele2 is not None
    assert allele2.successful_invocations == 5


def test_lineage(registry):
    parent_sha = registry.register("def execute(x): return 'v1'", "bridge_create")
    child_sha = registry.register(
        "def execute(x): return 'v2'", "bridge_create",
        generation=1, parent_sha=parent_sha,
    )
    child = registry.get(child_sha)
    assert child.generation == 1
    assert child.parent_sha == parent_sha
