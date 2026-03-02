"""Tests for proactive allele generation (sg generate)."""
import json
import shutil
import pytest
from pathlib import Path

from sg import arena
from sg.contracts import ContractStore
from sg.mutation import MockMutationEngine, MutationEngine
from sg.phenotype import PhenotypeMap
from sg.registry import Registry

import sg_network

CONTRACTS_DIR = sg_network.contracts_path()
GENES_DIR = sg_network.genes_path()
FIXTURES_DIR = sg_network.fixtures_path()


class TestMockMutationEngineGenerate:
    def test_generate_returns_fixture(self, tmp_path):
        fixtures = tmp_path / "fixtures"
        fixtures.mkdir()
        (fixtures / "bridge_create_fix.py").write_text("# generated\ndef execute(input_json): return '{}'")

        engine = MockMutationEngine(fixtures)
        sources = engine.generate("bridge_create", "contract prompt", count=1)
        assert len(sources) == 1
        assert "execute" in sources[0]

    def test_generate_missing_fixture_raises(self, tmp_path):
        engine = MockMutationEngine(tmp_path)
        with pytest.raises(FileNotFoundError):
            engine.generate("nonexistent", "contract prompt")

    def test_generate_ignores_count(self, tmp_path):
        fixtures = tmp_path / "fixtures"
        fixtures.mkdir()
        (fixtures / "bridge_create_fix.py").write_text("def execute(i): return '{}'")

        engine = MockMutationEngine(fixtures)
        # Mock always returns 1 regardless of count
        sources = engine.generate("bridge_create", "contract prompt", count=3)
        assert len(sources) == 1


class TestGenerateIntegration:
    @pytest.fixture
    def project(self, tmp_path):
        shutil.copytree(FIXTURES_DIR, tmp_path / "fixtures")
        shutil.copytree(CONTRACTS_DIR, tmp_path / "contracts")

        contract_store = ContractStore.open(tmp_path / "contracts")
        registry = Registry.open(tmp_path / ".sg" / "registry")
        phenotype = PhenotypeMap()

        # Register seed genes
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
        return tmp_path

    def test_generate_single_locus(self, project):
        contract_store = ContractStore.open(project / "contracts")
        registry = Registry.open(project / ".sg" / "registry")
        phenotype = PhenotypeMap.load(project / "phenotype.toml")
        engine = MockMutationEngine(project / "fixtures")

        locus = "bridge_create"
        dominant_sha = phenotype.get_dominant(locus)
        alleles_before = len(registry.alleles_for_locus(locus))

        contract_prompt = "test prompt"
        sources = engine.generate(locus, contract_prompt)

        for source in sources:
            sha = registry.register(
                source, locus,
                generation=1, parent_sha=dominant_sha,
            )
            allele = registry.get(sha)
            allele.state = "recessive"
            phenotype.add_to_fallback(locus, sha)

        alleles_after = len(registry.alleles_for_locus(locus))
        assert alleles_after > alleles_before

        # New allele should be in fallback, not dominant
        stack = phenotype.get_stack(locus)
        assert stack[0] == dominant_sha  # dominant unchanged
        assert len(stack) > 1  # fallback has the new allele

    def test_generate_sets_lineage(self, project):
        registry = Registry.open(project / ".sg" / "registry")
        phenotype = PhenotypeMap.load(project / "phenotype.toml")
        engine = MockMutationEngine(project / "fixtures")

        locus = "bridge_create"
        dominant_sha = phenotype.get_dominant(locus)
        dominant = registry.get(dominant_sha)

        sources = engine.generate(locus, "prompt")
        sha = registry.register(
            sources[0], locus,
            generation=dominant.generation + 1,
            parent_sha=dominant_sha,
        )

        allele = registry.get(sha)
        assert allele.parent_sha == dominant_sha
        assert allele.generation == dominant.generation + 1

    def test_generate_all_loci(self, project):
        contract_store = ContractStore.open(project / "contracts")
        registry = Registry.open(project / ".sg" / "registry")
        phenotype = PhenotypeMap.load(project / "phenotype.toml")
        engine = MockMutationEngine(project / "fixtures")

        generated = 0
        for locus in contract_store.known_loci():
            dominant_sha = phenotype.get_dominant(locus)
            if dominant_sha is None:
                continue
            try:
                sources = engine.generate(locus, "prompt")
            except FileNotFoundError:
                continue
            for source in sources:
                sha = registry.register(source, locus, generation=1, parent_sha=dominant_sha)
                allele = registry.get(sha)
                allele.state = "recessive"
                phenotype.add_to_fallback(locus, sha)
                generated += 1

        assert generated > 0

    def test_duplicate_source_same_sha(self, project):
        """Generating the same source twice produces the same SHA — no duplicates."""
        registry = Registry.open(project / ".sg" / "registry")
        phenotype = PhenotypeMap.load(project / "phenotype.toml")
        engine = MockMutationEngine(project / "fixtures")

        locus = "bridge_create"
        sources1 = engine.generate(locus, "prompt")
        sources2 = engine.generate(locus, "prompt")

        sha1 = registry.register(sources1[0], locus)
        sha2 = registry.register(sources2[0], locus)
        assert sha1 == sha2  # content-addressed — same source, same SHA
