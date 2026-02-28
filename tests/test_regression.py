"""Tests for fitness regression detection."""
import json
import shutil
import pytest
from pathlib import Path

from sg import arena
from sg.contracts import ContractStore
from sg.fusion import FusionTracker
from sg.kernel.mock import MockNetworkKernel
from sg.mutation import MockMutationEngine
from sg.orchestrator import Orchestrator
from sg.phenotype import PhenotypeMap
from sg.regression import (
    RegressionDetector, FitnessHistory,
    REGRESSION_THRESHOLD, SEVERE_REGRESSION, MIN_INVOCATIONS,
)
from sg.registry import Registry, AlleleMetadata


CONTRACTS_DIR = Path(__file__).parent.parent / "contracts"
GENES_DIR = Path(__file__).parent.parent / "genes"
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


class TestRegressionDetector:
    def test_no_regression_first_record(self):
        """First record of an allele never triggers regression."""
        det = RegressionDetector()
        allele = AlleleMetadata(sha256="abc", locus="test",
                                successful_invocations=5, failed_invocations=0)
        assert det.record(allele) is None

    def test_no_regression_below_min_invocations(self):
        """Regression not detected with too few invocations."""
        det = RegressionDetector()
        allele = AlleleMetadata(sha256="abc", locus="test",
                                successful_invocations=5, failed_invocations=0)
        det.record(allele)  # peak = 0.5

        # Now fitness drops but not enough invocations
        allele.failed_invocations = 5
        assert det.record(allele) is None

    def test_mild_regression_detected(self):
        """Fitness drop >= 0.2 from peak triggers mild regression."""
        det = RegressionDetector()
        allele = AlleleMetadata(sha256="abc", locus="test",
                                successful_invocations=10, failed_invocations=0)
        det.record(allele)  # peak = 1.0

        # Drop to 0.7 (drop of 0.3 > 0.2 threshold)
        allele.failed_invocations = 3
        assert allele.total_invocations >= MIN_INVOCATIONS
        result = det.record(allele)
        # 10/13 = 0.77, drop = 0.23 — mild
        assert result == "mild"

    def test_severe_regression_detected(self):
        """Fitness drop >= 0.4 from peak triggers severe regression."""
        det = RegressionDetector()
        allele = AlleleMetadata(sha256="abc", locus="test",
                                successful_invocations=10, failed_invocations=0)
        det.record(allele)  # peak = 1.0

        # Major drop
        allele.failed_invocations = 10
        result = det.record(allele)
        # 10/20 = 0.5, drop = 0.5 — severe
        assert result == "severe"

    def test_peak_tracks_highest(self):
        """Peak fitness tracks the historical maximum."""
        det = RegressionDetector()
        allele = AlleleMetadata(sha256="abc", locus="test",
                                successful_invocations=8, failed_invocations=2)
        det.record(allele)  # peak = 0.8

        allele.successful_invocations = 10
        allele.failed_invocations = 0
        det.record(allele)  # peak = 1.0

        allele.successful_invocations = 9
        allele.failed_invocations = 1
        det.record(allele)  # 0.9, no regression (drop = 0.1 < 0.2)

        h = det.get_history("abc")
        assert h.peak_fitness == 1.0
        assert h.samples == 3

    def test_no_regression_when_fitness_stable(self):
        """Small drops don't trigger regression."""
        det = RegressionDetector()
        allele = AlleleMetadata(sha256="abc", locus="test",
                                successful_invocations=10, failed_invocations=1)
        det.record(allele)  # ~0.91

        allele.failed_invocations = 2
        result = det.record(allele)
        # 10/12 = 0.83, drop = ~0.08 < 0.2
        assert result is None


class TestRegressionPersistence:
    def test_save_and_load(self, tmp_path):
        """RegressionDetector round-trips through JSON."""
        det = RegressionDetector()
        allele = AlleleMetadata(sha256="abc", locus="test",
                                successful_invocations=10, failed_invocations=0)
        det.record(allele)

        path = tmp_path / "regression.json"
        det.save(path)
        assert path.exists()

        det2 = RegressionDetector.open(path)
        h = det2.get_history("abc")
        assert h is not None
        assert h.peak_fitness == 1.0
        assert h.samples == 1

    def test_open_missing_file(self, tmp_path):
        """RegressionDetector.open on missing file starts empty."""
        det = RegressionDetector.open(tmp_path / "nonexistent.json")
        assert det.history == {}

    def test_persistence_across_sessions(self, tmp_path):
        """Peak fitness persists, regression detected in later session."""
        path = tmp_path / "regression.json"

        # Session 1: record high fitness
        det1 = RegressionDetector()
        allele = AlleleMetadata(sha256="abc", locus="test",
                                successful_invocations=10, failed_invocations=0)
        det1.record(allele)
        det1.save(path)

        # Session 2: load state, fitness has dropped
        det2 = RegressionDetector.open(path)
        allele.failed_invocations = 5
        result = det2.record(allele)
        # 10/15 = 0.67, drop = 0.33 from peak 1.0 → mild
        assert result == "mild"


class TestRegressionIntegration:
    @pytest.fixture
    def project(self, tmp_path):
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
        return tmp_path

    def test_orchestrator_has_regression_detector(self, project):
        """Orchestrator creates a RegressionDetector by default."""
        cs = ContractStore.open(project / "contracts")
        registry = Registry.open(project / ".sg" / "registry")
        phenotype = PhenotypeMap.load(project / "phenotype.toml")
        ft = FusionTracker.open(project / "fusion_tracker.json")
        kernel = MockNetworkKernel()
        mutation_engine = MockMutationEngine(project / "fixtures")

        orch = Orchestrator(
            registry=registry, phenotype=phenotype,
            mutation_engine=mutation_engine, fusion_tracker=ft,
            kernel=kernel, contract_store=cs, project_root=project,
        )
        assert isinstance(orch.regression_detector, RegressionDetector)

    def test_regression_records_after_success(self, project):
        """After execute_locus success, regression detector records the allele."""
        cs = ContractStore.open(project / "contracts")
        registry = Registry.open(project / ".sg" / "registry")
        phenotype = PhenotypeMap.load(project / "phenotype.toml")
        ft = FusionTracker.open(project / "fusion_tracker.json")
        kernel = MockNetworkKernel()
        mutation_engine = MockMutationEngine(project / "fixtures")

        orch = Orchestrator(
            registry=registry, phenotype=phenotype,
            mutation_engine=mutation_engine, fusion_tracker=ft,
            kernel=kernel, contract_store=cs, project_root=project,
        )

        result = orch.execute_locus("bridge_create", json.dumps({
            "bridge_name": "br0",
            "interfaces": ["eth0"],
        }))
        assert result is not None

        dominant_sha = phenotype.get_dominant("bridge_create")
        h = orch.regression_detector.get_history(dominant_sha)
        assert h is not None
        assert h.samples == 1

    def test_regression_state_persisted_to_disk(self, project):
        """After execute_locus, regression state is saved to .sg/regression.json."""
        cs = ContractStore.open(project / "contracts")
        registry = Registry.open(project / ".sg" / "registry")
        phenotype = PhenotypeMap.load(project / "phenotype.toml")
        ft = FusionTracker.open(project / "fusion_tracker.json")
        kernel = MockNetworkKernel()
        mutation_engine = MockMutationEngine(project / "fixtures")

        orch = Orchestrator(
            registry=registry, phenotype=phenotype,
            mutation_engine=mutation_engine, fusion_tracker=ft,
            kernel=kernel, contract_store=cs, project_root=project,
        )

        orch.execute_locus("bridge_create", json.dumps({
            "bridge_name": "br0", "interfaces": ["eth0"],
        }))

        # Regression state should be saved to disk
        regression_path = project / ".sg" / "regression.json"
        assert regression_path.exists()

        # Loading a fresh detector should recover the state
        det2 = RegressionDetector.open(regression_path)
        dominant_sha = phenotype.get_dominant("bridge_create")
        h = det2.get_history(dominant_sha)
        assert h is not None
        assert h.samples == 1
