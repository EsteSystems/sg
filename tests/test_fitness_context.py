"""Phase E.1a: Gene fitness context tagging tests.

Verifies that FitnessRecord carries structure_hash, that
compute_temporal_fitness weights old-structure records lower,
and that records are tagged on pathway promotion.
"""
from __future__ import annotations

import json
import shutil
import pytest

import sg_network
from sg import arena
from sg.contracts import ContractStore
from sg.fitness import (
    FitnessRecord, record_feedback, compute_temporal_fitness,
    _score_for_timescale, OLD_STRUCTURE_WEIGHT,
)
from sg.fusion import FusionTracker
from sg_network import MockNetworkKernel
from sg.mutation import MockMutationEngine
from sg.orchestrator import Orchestrator
from sg.pathway_fitness import PathwayFitnessTracker
from sg.pathway_registry import PathwayRegistry
from sg.phenotype import PhenotypeMap
from sg.registry import Registry, AlleleMetadata


CONTRACTS_DIR = sg_network.contracts_path()
GENES_DIR = sg_network.genes_path()
FIXTURES_DIR = sg_network.fixtures_path()


class TestFitnessRecordStructureHash:
    """FitnessRecord stores and round-trips structure_hash."""

    def test_fitness_record_structure_hash_roundtrip(self):
        rec = FitnessRecord(
            timescale="convergence",
            success=True,
            source_locus="check_conn",
            structure_hash="abc123",
        )
        d = rec.to_dict()
        assert d["structure_hash"] == "abc123"
        restored = FitnessRecord.from_dict(d)
        assert restored.structure_hash == "abc123"

    def test_fitness_record_backward_compat(self):
        """Old records without structure_hash deserialize with empty string."""
        d = {"timescale": "convergence", "success": True, "source": "x"}
        rec = FitnessRecord.from_dict(d)
        assert rec.structure_hash == ""

    def test_to_dict_omits_empty_structure_hash(self):
        """Empty structure_hash is not included in serialized dict."""
        rec = FitnessRecord(
            timescale="convergence", success=True, source_locus="x",
        )
        d = rec.to_dict()
        assert "structure_hash" not in d


class TestRecordFeedback:
    """record_feedback stores structure_hash."""

    def test_record_feedback_stores_structure_hash(self):
        allele = AlleleMetadata(sha256="aaa", locus="test")
        record_feedback(allele, "convergence", True, "check_x",
                        structure_hash="hash_abc")
        assert len(allele.fitness_records) == 1
        assert allele.fitness_records[0]["structure_hash"] == "hash_abc"

    def test_record_feedback_default_no_hash(self):
        allele = AlleleMetadata(sha256="aaa", locus="test")
        record_feedback(allele, "convergence", True, "check_x")
        assert "structure_hash" not in allele.fitness_records[0]


class TestStructureWeightedFitness:
    """compute_temporal_fitness weights old-structure records lower."""

    def test_score_for_timescale_weights_old_structure_lower(self):
        """Records with mismatched structure_hash get weight 0.5."""
        records = [
            FitnessRecord("convergence", True, "x", structure_hash="new"),
            FitnessRecord("convergence", True, "x", structure_hash="new"),
            FitnessRecord("convergence", False, "x", structure_hash="old"),
            FitnessRecord("convergence", False, "x", structure_hash="old"),
        ]
        # Without structure weighting: 2/4 = 0.5
        score_unweighted = _score_for_timescale(records, "convergence")
        assert abs(score_unweighted - 0.5) < 0.01

        # With structure weighting: successes=2*1.0, failures=2*0.5
        # total_weight = 2 + 1 = 3, weighted_success = 2
        # score = 2/3 = 0.667
        score_weighted = _score_for_timescale(
            records, "convergence", current_structure_hash="new",
        )
        assert score_weighted > score_unweighted
        assert abs(score_weighted - 2.0 / 3.0) < 0.01

    def test_compute_temporal_fitness_backward_compat(self):
        """Without structure_hash arg, behavior is unchanged."""
        allele = AlleleMetadata(sha256="a", locus="test")
        allele.successful_invocations = 80
        allele.failed_invocations = 20
        record_feedback(allele, "convergence", True, "check_x")

        f1 = compute_temporal_fitness(allele)
        f2 = compute_temporal_fitness(allele, current_structure_hash="")
        assert f1 == f2


class TestTagGeneRecords:
    """_tag_gene_fitness_records tags untagged records on pathway promotion."""

    @pytest.fixture
    def full_project(self, tmp_path):
        shutil.copytree(FIXTURES_DIR, tmp_path / "fixtures")
        shutil.copytree(CONTRACTS_DIR, tmp_path / "contracts")
        (tmp_path / ".sg").mkdir(exist_ok=True)
        contract_store = ContractStore.open(tmp_path / "contracts")
        registry = Registry.open(tmp_path / ".sg" / "registry")
        phenotype = PhenotypeMap()
        for locus in contract_store.known_loci():
            candidates = sorted(GENES_DIR.glob(f"{locus}_*.py"))
            if candidates:
                source = candidates[0].read_text()
                sha = registry.register(source, locus)
                phenotype.promote(locus, sha)
                registry.get(sha).state = "dominant"
        registry.save_index()
        phenotype.save(tmp_path / "phenotype.toml")
        return tmp_path

    def test_tag_gene_fitness_records(self, full_project):
        from sg.parser.types import GeneFamily
        contract_store = ContractStore.open(full_project / "contracts")
        for locus in contract_store.known_loci():
            gc = contract_store.get_gene(locus)
            if gc and gc.family == GeneFamily.CONFIGURATION and gc.verify_within:
                gc.verify_within = "0.01s"

        registry = Registry.open(full_project / ".sg" / "registry")
        phenotype = PhenotypeMap.load(full_project / "phenotype.toml")
        fusion_tracker = FusionTracker.open(full_project / "fusion_tracker.json")
        pft = PathwayFitnessTracker.open(full_project / "pathway_fitness.json")
        pr = PathwayRegistry.open(full_project / ".sg" / "pathway_registry")
        mutation_engine = MockMutationEngine(full_project / "fixtures")
        kernel = MockNetworkKernel()

        orch = Orchestrator(
            registry=registry, phenotype=phenotype,
            mutation_engine=mutation_engine, fusion_tracker=fusion_tracker,
            kernel=kernel, contract_store=contract_store,
            project_root=full_project,
            pathway_fitness_tracker=pft, pathway_registry=pr,
        )

        # Add untagged fitness records to a gene in the pathway
        locus = "bridge_create"
        dom_sha = phenotype.get_dominant(locus)
        allele = registry.get(dom_sha)
        record_feedback(allele, "convergence", True, "check_conn")
        record_feedback(allele, "convergence", False, "check_conn")
        assert all(
            "structure_hash" not in r for r in allele.fitness_records
        )

        # Tag them
        orch._tag_gene_fitness_records(
            "configure_bridge_with_stp", "old_hash_abc",
        )

        for r in allele.fitness_records:
            assert r.get("structure_hash") == "old_hash_abc"

    def test_already_tagged_not_overwritten(self, full_project):
        from sg.parser.types import GeneFamily
        contract_store = ContractStore.open(full_project / "contracts")
        for locus in contract_store.known_loci():
            gc = contract_store.get_gene(locus)
            if gc and gc.family == GeneFamily.CONFIGURATION and gc.verify_within:
                gc.verify_within = "0.01s"

        registry = Registry.open(full_project / ".sg" / "registry")
        phenotype = PhenotypeMap.load(full_project / "phenotype.toml")
        fusion_tracker = FusionTracker.open(full_project / "fusion_tracker.json")
        pft = PathwayFitnessTracker.open(full_project / "pathway_fitness.json")
        pr = PathwayRegistry.open(full_project / ".sg" / "pathway_registry")
        mutation_engine = MockMutationEngine(full_project / "fixtures")
        kernel = MockNetworkKernel()

        orch = Orchestrator(
            registry=registry, phenotype=phenotype,
            mutation_engine=mutation_engine, fusion_tracker=fusion_tracker,
            kernel=kernel, contract_store=contract_store,
            project_root=full_project,
            pathway_fitness_tracker=pft, pathway_registry=pr,
        )

        locus = "bridge_create"
        dom_sha = phenotype.get_dominant(locus)
        allele = registry.get(dom_sha)
        record_feedback(allele, "convergence", True, "check_conn",
                        structure_hash="existing_hash")

        orch._tag_gene_fitness_records(
            "configure_bridge_with_stp", "new_hash",
        )

        # Already-tagged record should keep its original hash
        assert allele.fitness_records[0]["structure_hash"] == "existing_hash"
