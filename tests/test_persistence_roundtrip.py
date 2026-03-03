"""Phase D: Persistence round-trip tests.

Verifies that all stateful components survive save/load cycles
without data loss or corruption.
"""
import json
import shutil
import time
import pytest
from pathlib import Path

import sg_network
from sg import arena
from sg.contracts import ContractStore
from sg.fitness import record_feedback
from sg.fusion import FusionTracker
from sg_network import MockNetworkKernel
from sg.mutation import MockMutationEngine
from sg.orchestrator import Orchestrator
from sg.pathway_fitness import PathwayFitnessTracker
from sg.pathway_mutation import PathwayMutationThrottle
from sg.pathway_registry import PathwayRegistry
from sg.phenotype import PhenotypeMap
from sg.registry import Registry


CONTRACTS_DIR = sg_network.contracts_path()
GENES_DIR = sg_network.genes_path()
FIXTURES_DIR = sg_network.fixtures_path()


@pytest.fixture
def full_project(tmp_path):
    """Full project with real contracts and gene fixtures."""
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
            allele = registry.get(sha)
            allele.state = "dominant"

    registry.save_index()
    phenotype.save(tmp_path / "phenotype.toml")
    return tmp_path


def _make_orchestrator(project_root):
    from sg.parser.types import GeneFamily
    contract_store = ContractStore.open(project_root / "contracts")
    for locus in contract_store.known_loci():
        gc = contract_store.get_gene(locus)
        if gc and gc.family == GeneFamily.CONFIGURATION and gc.verify_within:
            gc.verify_within = "0.01s"

    registry = Registry.open(project_root / ".sg" / "registry")
    phenotype = PhenotypeMap.load(project_root / "phenotype.toml")
    fusion_tracker = FusionTracker.open(project_root / "fusion_tracker.json")
    pft = PathwayFitnessTracker.open(project_root / "pathway_fitness.json")
    pr = PathwayRegistry.open(project_root / ".sg" / "pathway_registry")
    mutation_engine = MockMutationEngine(project_root / "fixtures")
    kernel = MockNetworkKernel()

    return Orchestrator(
        registry=registry, phenotype=phenotype,
        mutation_engine=mutation_engine, fusion_tracker=fusion_tracker,
        kernel=kernel, contract_store=contract_store,
        project_root=project_root,
        pathway_fitness_tracker=pft, pathway_registry=pr,
    )


class TestFitnessRecordsRoundtrip:
    """fitness_records with all three timescale fields survive save/reload."""

    def test_fitness_records_roundtrip(self, full_project):
        orch = _make_orchestrator(full_project)
        locus = "bridge_create"
        dom_sha = orch.phenotype.get_dominant(locus)
        allele = orch.registry.get(dom_sha)

        # Add fitness records for all three timescales
        record_feedback(allele, "immediate", True, locus)
        record_feedback(allele, "convergence", True, "check_connectivity")
        record_feedback(allele, "resilience", False, "check_connectivity")

        assert len(allele.fitness_records) == 3

        orch.save_state()

        # Reload
        orch2 = _make_orchestrator(full_project)
        allele2 = orch2.registry.get(dom_sha)

        assert len(allele2.fitness_records) == 3
        timescales = [r["timescale"] for r in allele2.fitness_records]
        assert "immediate" in timescales
        assert "convergence" in timescales
        assert "resilience" in timescales
        # Check detailed field preservation
        convergence_rec = [r for r in allele2.fitness_records if r["timescale"] == "convergence"][0]
        assert convergence_rec["success"] is True
        assert convergence_rec["source"] == "check_connectivity"


class TestPathwayAllelePhenotypeRoundtrip:
    """Pathway dominant + fallback in phenotype survive save/reload."""

    def test_pathway_allele_phenotype_roundtrip(self, full_project):
        orch = _make_orchestrator(full_project)
        pathway_name = "configure_bridge_with_stp"

        # Set up pathway alleles
        sha1 = "aaa111"
        sha2 = "bbb222"
        sha3 = "ccc333"
        orch.phenotype.promote_pathway(pathway_name, sha1)
        orch.phenotype.add_pathway_fallback(pathway_name, sha2)
        orch.phenotype.add_pathway_fallback(pathway_name, sha3)

        stack_before = orch.phenotype.get_pathway_stack(pathway_name)
        assert stack_before == [sha1, sha2, sha3]

        orch.save_state()

        # Reload
        orch2 = _make_orchestrator(full_project)
        stack_after = orch2.phenotype.get_pathway_stack(pathway_name)
        assert stack_after == [sha1, sha2, sha3]
        assert orch2.phenotype.get_pathway_dominant(pathway_name) == sha1


class TestFusionConstituentAllelesRoundtrip:
    """constituent_alleles and composition_fingerprint survive save/reload."""

    def test_fusion_constituent_alleles_roundtrip(self, full_project):
        orch = _make_orchestrator(full_project)
        pathway_name = "configure_bridge_with_stp"

        allele_shas = ["sha_bridge_create", "sha_stp_set"]
        # Record enough successes to build up the track
        for _ in range(5):
            orch.fusion_tracker.record_success(pathway_name, allele_shas)

        track_before = orch.fusion_tracker.get_track(pathway_name)
        assert track_before is not None
        assert track_before.constituent_alleles == allele_shas
        assert track_before.composition_fingerprint is not None
        fp_before = track_before.composition_fingerprint

        orch.save_state()

        # Reload
        orch2 = _make_orchestrator(full_project)
        track_after = orch2.fusion_tracker.get_track(pathway_name)
        assert track_after is not None
        assert track_after.constituent_alleles == allele_shas
        assert track_after.composition_fingerprint == fp_before
        assert track_after.reinforcement_count == 5


class TestPathwayFitnessRecentOutcomesRoundtrip:
    """_recent_outcomes list preserved across save/reload."""

    def test_pathway_fitness_recent_outcomes_roundtrip(self, tmp_path):
        pft = PathwayFitnessTracker()
        pathway_name = "test_pathway"

        # Record a mix of successes and failures
        outcomes = [True, True, False, True, False, True, True, True, False, True]
        for i, success in enumerate(outcomes):
            pft.record_execution(
                pathway_name,
                steps_executed=["step_a", "step_b"],
                step_timings={"step_a": 10.0, "step_b": 20.0},
                success=success,
                failure_step="step_b" if not success else None,
                input_json='{"key": "val"}',
            )

        rec_before = pft.get_record(pathway_name)
        assert rec_before._recent_outcomes == outcomes

        path = tmp_path / "pathway_fitness.json"
        pft.save(path)

        # Reload
        pft2 = PathwayFitnessTracker.open(path)
        rec_after = pft2.get_record(pathway_name)
        assert rec_after is not None
        assert rec_after._recent_outcomes == outcomes
        assert rec_after.total_executions == 10
        assert rec_after.successful_executions == 7
        assert rec_after.failed_executions == 3


class TestMutationThrottleCooldownRoundtrip:
    """Throttle cooldown + mutation timestamps survive save/reload."""

    def test_mutation_throttle_cooldown_roundtrip(self, full_project):
        orch = _make_orchestrator(full_project)
        pathway_name = "configure_bridge_with_stp"

        # Record a mutation
        orch._pathway_mutation_throttle.record_mutation(pathway_name)
        assert not orch._pathway_mutation_throttle.can_mutate(pathway_name)

        orch.save_state()

        # Reload and check throttle state persists
        orch2 = _make_orchestrator(full_project)
        assert not orch2._pathway_mutation_throttle.can_mutate(pathway_name), \
            "Throttle cooldown lost across save/load cycle"
