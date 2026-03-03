"""Phase E.4: Meta-evolution parameter tracking tests.

Verifies EvolutionaryParams, ParamSnapshot, MetaParamTracker lifecycle,
persistence, bounded growth, and orchestrator integration.
"""
from __future__ import annotations

import shutil

import pytest

import sg_network
from sg.contracts import ContractStore
from sg.fusion import FusionTracker
from sg_network import MockNetworkKernel
from sg.meta_params import (
    EvolutionaryParams,
    MetaParamTracker,
    ParamSnapshot,
    MAX_SNAPSHOTS_PER_ENTITY,
)
from sg.mutation import MockMutationEngine
from sg.orchestrator import Orchestrator
from sg.pathway_fitness import PathwayFitnessTracker
from sg.pathway_registry import PathwayRegistry
from sg.phenotype import PhenotypeMap
from sg.registry import Registry


CONTRACTS_DIR = sg_network.contracts_path()
GENES_DIR = sg_network.genes_path()
FIXTURES_DIR = sg_network.fixtures_path()


class TestEvolutionaryParams:
    """Parameter defaults and serialization."""

    def test_defaults(self):
        p = EvolutionaryParams()
        assert p.immediate_weight == 0.30
        assert p.convergence_weight == 0.50
        assert p.resilience_weight == 0.20
        assert p.promotion_advantage == 0.1
        assert p.promotion_min_invocations == 50
        assert p.demotion_consecutive_failures == 3
        assert p.pathway_promotion_advantage == 0.15
        assert p.topology_promotion_advantage == 0.20
        assert p.fusion_threshold == 10
        assert p.cv_threshold == 0.05

    def test_to_dict_roundtrip(self):
        p = EvolutionaryParams()
        d = p.to_dict()
        p2 = EvolutionaryParams.from_dict(d)
        assert p2.immediate_weight == p.immediate_weight
        assert p2.topology_promotion_min_executions == p.topology_promotion_min_executions

    def test_from_dict_partial(self):
        p = EvolutionaryParams.from_dict({"promotion_advantage": 0.2})
        assert p.promotion_advantage == 0.2
        # Other fields should have defaults
        assert p.immediate_weight == 0.30
        assert p.fusion_threshold == 10


class TestParamSnapshot:
    """Snapshot serialization."""

    def test_to_dict_roundtrip(self):
        snap = ParamSnapshot(
            entity_name="bridge_create",
            entity_type="gene",
            params={"promotion_advantage": 0.1},
            outcome_fitness=0.85,
            allele_sha="sha123",
            allele_survived=True,
        )
        d = snap.to_dict()
        restored = ParamSnapshot.from_dict(d)
        assert restored.entity_name == "bridge_create"
        assert restored.entity_type == "gene"
        assert restored.outcome_fitness == 0.85
        assert restored.allele_survived is True

    def test_timestamp_auto_populated(self):
        snap = ParamSnapshot(entity_name="x", entity_type="gene")
        assert snap.timestamp > 0


class TestMetaParamTracker:
    """Tracker operations."""

    def test_defaults_returned_without_overrides(self):
        tracker = MetaParamTracker()
        params = tracker.get_params("unknown_entity")
        assert params.immediate_weight == 0.30

    def test_overrides_applied(self):
        tracker = MetaParamTracker()
        tracker.overrides["bridge_create"] = {"promotion_advantage": 0.2}
        params = tracker.get_params("bridge_create")
        assert params.promotion_advantage == 0.2
        # Other fields still default
        assert params.immediate_weight == 0.30

    def test_record_and_retrieve_snapshot(self):
        tracker = MetaParamTracker()
        tracker.record_snapshot(
            entity_name="bridge_create",
            entity_type="gene",
            outcome_fitness=0.85,
            allele_sha="sha123",
            allele_survived=True,
        )
        snaps = tracker.get_snapshots("bridge_create")
        assert len(snaps) == 1
        assert snaps[0].outcome_fitness == 0.85

    def test_gene_snapshot_params(self):
        tracker = MetaParamTracker()
        tracker.record_snapshot(
            entity_name="bridge_create",
            entity_type="gene",
            outcome_fitness=0.85,
            allele_sha="sha1",
            allele_survived=True,
        )
        snap = tracker.get_snapshots("bridge_create")[0]
        assert "promotion_advantage" in snap.params
        assert "immediate_weight" in snap.params
        assert "pathway_promotion_advantage" not in snap.params

    def test_pathway_snapshot_params(self):
        tracker = MetaParamTracker()
        tracker.record_snapshot(
            entity_name="configure_bridge",
            entity_type="pathway",
            outcome_fitness=0.9,
            allele_sha="sha1",
            allele_survived=True,
        )
        snap = tracker.get_snapshots("configure_bridge")[0]
        assert "pathway_promotion_advantage" in snap.params
        assert "fusion_threshold" in snap.params
        assert "promotion_advantage" not in snap.params

    def test_topology_snapshot_params(self):
        tracker = MetaParamTracker()
        tracker.record_snapshot(
            entity_name="production_server",
            entity_type="topology",
            outcome_fitness=0.75,
            allele_sha="sha1",
            allele_survived=False,
        )
        snap = tracker.get_snapshots("production_server")[0]
        assert "topology_promotion_advantage" in snap.params
        assert "promotion_advantage" not in snap.params

    def test_snapshot_sliding_window(self):
        tracker = MetaParamTracker()
        for i in range(MAX_SNAPSHOTS_PER_ENTITY + 10):
            tracker.record_snapshot(
                entity_name="bridge_create",
                entity_type="gene",
                outcome_fitness=0.5,
                allele_sha=f"sha{i}",
                allele_survived=True,
            )
        assert len(tracker.get_snapshots("bridge_create")) == MAX_SNAPSHOTS_PER_ENTITY

    def test_survival_rate_calculation(self):
        tracker = MetaParamTracker()
        for i in range(10):
            tracker.record_snapshot(
                entity_name="locus_a",
                entity_type="gene",
                outcome_fitness=0.5,
                allele_sha=f"sha{i}",
                allele_survived=(i % 2 == 0),  # 5 survived, 5 didn't
            )
        rate = tracker.survival_rate("locus_a")
        assert rate == pytest.approx(0.5)

    def test_survival_rate_no_data(self):
        tracker = MetaParamTracker()
        assert tracker.survival_rate("unknown") is None

    def test_persistence_roundtrip(self, tmp_path):
        tracker = MetaParamTracker()
        tracker.overrides["bridge_create"] = {"promotion_advantage": 0.2}
        tracker.record_snapshot(
            entity_name="bridge_create",
            entity_type="gene",
            outcome_fitness=0.85,
            allele_sha="sha123",
            allele_survived=True,
        )
        path = tmp_path / "meta_params.json"
        tracker.save(path)

        tracker2 = MetaParamTracker.open(path)
        assert tracker2.overrides["bridge_create"]["promotion_advantage"] == 0.2
        snaps = tracker2.get_snapshots("bridge_create")
        assert len(snaps) == 1
        assert snaps[0].allele_sha == "sha123"

    def test_empty_load(self, tmp_path):
        path = tmp_path / "nonexistent.json"
        tracker = MetaParamTracker.open(path)
        assert tracker.defaults.immediate_weight == 0.30
        assert len(tracker.snapshots) == 0


class TestMetaParamsIntegration:
    """Meta-param tracking through orchestrator."""

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

    def _make_orchestrator(self, project_root):
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
        meta = MetaParamTracker()
        return Orchestrator(
            registry=registry, phenotype=phenotype,
            mutation_engine=mutation_engine, fusion_tracker=fusion_tracker,
            kernel=kernel, contract_store=contract_store,
            project_root=project_root,
            pathway_fitness_tracker=pft, pathway_registry=pr,
            meta_param_tracker=meta,
        )

    def test_orchestrator_saves_meta_params(self, full_project):
        """save_state() persists meta_params.json."""
        orch = self._make_orchestrator(full_project)
        orch._meta_param_tracker.record_snapshot(
            entity_name="bridge_create",
            entity_type="gene",
            outcome_fitness=0.85,
            allele_sha="sha_test",
            allele_survived=True,
        )
        orch.save_state()

        path = full_project / ".sg" / "meta_params.json"
        assert path.exists()
        loaded = MetaParamTracker.open(path)
        snaps = loaded.get_snapshots("bridge_create")
        assert len(snaps) == 1
