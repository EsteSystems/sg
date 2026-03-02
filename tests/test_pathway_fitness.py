"""Tests for pathway-level fitness tracking (Phase 2)."""
import json
import shutil
from pathlib import Path

import pytest

import sg_network
from sg.contracts import ContractStore
from sg.fusion import FusionTracker
from sg_network import MockNetworkKernel
from sg.mutation import MockMutationEngine
from sg.orchestrator import Orchestrator
from sg.pathway_fitness import (
    InputCluster,
    PathwayFitnessRecord,
    PathwayFitnessTracker,
    TimingAnomaly,
    MAX_STEP_TIMINGS,
)
from sg.phenotype import PhenotypeMap
from sg.registry import Registry

CONTRACTS_DIR = sg_network.contracts_path()
GENES_DIR = sg_network.genes_path()
FIXTURES_DIR = sg_network.fixtures_path()


class TestPathwayFitnessRecord:
    def test_record_creation(self):
        rec = PathwayFitnessRecord(pathway_name="test_pw")
        assert rec.total_executions == 0
        assert rec.successful_executions == 0
        assert rec.failed_executions == 0
        assert rec.consecutive_failures == 0
        assert rec.step_timings == {}

    def test_to_dict_roundtrip(self):
        rec = PathwayFitnessRecord(
            pathway_name="p1",
            total_executions=5,
            successful_executions=3,
            failed_executions=2,
            failure_step_distribution={"step_a": 2},
            avg_execution_time_ms=42.5,
            step_timings={"step_a": [10.0, 12.0]},
            input_failure_clusters=[InputCluster(failure_step="step_a", count=2)],
            consecutive_failures=1,
            last_structure_hash="abc123",
        )
        rec._recent_outcomes = [True, True, False, True, False]

        d = rec.to_dict()
        restored = PathwayFitnessRecord.from_dict(d)

        assert restored.pathway_name == "p1"
        assert restored.total_executions == 5
        assert restored.successful_executions == 3
        assert restored.failed_executions == 2
        assert restored.failure_step_distribution == {"step_a": 2}
        assert restored.avg_execution_time_ms == 42.5
        assert restored.step_timings == {"step_a": [10.0, 12.0]}
        assert len(restored.input_failure_clusters) == 1
        assert restored.consecutive_failures == 1
        assert restored.last_structure_hash == "abc123"
        assert restored._recent_outcomes == [True, True, False, True, False]


class TestPathwayFitnessTracker:
    def test_record_success(self):
        tracker = PathwayFitnessTracker()
        tracker.record_execution(
            "pw1", ["step_a", "step_b"],
            {"step_a": 10.0, "step_b": 20.0},
            success=True, failure_step=None, input_json='{"x": 1}',
        )
        rec = tracker.get_record("pw1")
        assert rec.total_executions == 1
        assert rec.successful_executions == 1
        assert rec.failed_executions == 0
        assert rec.consecutive_failures == 0
        assert "step_a" in rec.step_timings
        assert "step_b" in rec.step_timings

    def test_record_failure(self):
        tracker = PathwayFitnessTracker()
        tracker.record_execution(
            "pw1", ["step_a"],
            {"step_a": 10.0},
            success=False, failure_step="step_a", input_json='{}',
        )
        rec = tracker.get_record("pw1")
        assert rec.total_executions == 1
        assert rec.failed_executions == 1
        assert rec.consecutive_failures == 1
        assert rec.failure_step_distribution == {"step_a": 1}

    def test_consecutive_failures_reset_on_success(self):
        tracker = PathwayFitnessTracker()
        for _ in range(3):
            tracker.record_execution("pw1", [], {}, False, "s", '{}')
        assert tracker.get_record("pw1").consecutive_failures == 3

        tracker.record_execution("pw1", [], {}, True, None, '{}')
        assert tracker.get_record("pw1").consecutive_failures == 0

    def test_compute_fitness_empty(self):
        tracker = PathwayFitnessTracker()
        assert tracker.compute_fitness("nonexistent") == 0.0

    def test_compute_fitness_all_success(self):
        tracker = PathwayFitnessTracker()
        for _ in range(10):
            tracker.record_execution("pw1", [], {}, True, None, '{}')
        assert tracker.compute_fitness("pw1") == pytest.approx(1.0)

    def test_compute_fitness_all_failure(self):
        tracker = PathwayFitnessTracker()
        for _ in range(10):
            tracker.record_execution("pw1", [], {}, False, "s", '{}')
        assert tracker.compute_fitness("pw1") == pytest.approx(0.0)

    def test_compute_fitness_recency_bias(self):
        """Recent failures should drag fitness down more than old ones."""
        # 10 successes then 5 failures
        tracker1 = PathwayFitnessTracker()
        for _ in range(10):
            tracker1.record_execution("pw1", [], {}, True, None, '{}')
        for _ in range(5):
            tracker1.record_execution("pw1", [], {}, False, "s", '{}')
        fitness_failures_recent = tracker1.compute_fitness("pw1")

        # 5 failures then 10 successes
        tracker2 = PathwayFitnessTracker()
        for _ in range(5):
            tracker2.record_execution("pw1", [], {}, False, "s", '{}')
        for _ in range(10):
            tracker2.record_execution("pw1", [], {}, True, None, '{}')
        fitness_successes_recent = tracker2.compute_fitness("pw1")

        assert fitness_successes_recent > fitness_failures_recent

    def test_failure_distribution(self):
        tracker = PathwayFitnessTracker()
        tracker.record_execution("pw1", [], {}, False, "step_a", '{}')
        tracker.record_execution("pw1", [], {}, False, "step_a", '{}')
        tracker.record_execution("pw1", [], {}, False, "step_b", '{}')

        dist = tracker.get_failure_distribution("pw1")
        assert dist["step_a"] == pytest.approx(2 / 3)
        assert dist["step_b"] == pytest.approx(1 / 3)

    def test_failure_distribution_empty(self):
        tracker = PathwayFitnessTracker()
        assert tracker.get_failure_distribution("nonexistent") == {}

    def test_timing_anomaly_detected(self):
        tracker = PathwayFitnessTracker()
        # 10 normal executions at ~10ms
        for _ in range(10):
            tracker.record_execution(
                "pw1", ["s1"], {"s1": 10.0}, True, None, '{}',
            )
        # One very slow execution (5x)
        tracker.record_execution(
            "pw1", ["s1"], {"s1": 50.0}, True, None, '{}',
        )
        anomalies = tracker.get_timing_anomalies("pw1")
        assert len(anomalies) == 1
        assert anomalies[0].step_name == "s1"
        assert anomalies[0].ratio > 2.0

    def test_no_anomaly_for_normal_timings(self):
        tracker = PathwayFitnessTracker()
        for _ in range(5):
            tracker.record_execution(
                "pw1", ["s1"], {"s1": 10.0}, True, None, '{}',
            )
        assert tracker.get_timing_anomalies("pw1") == []

    def test_timing_anomalies_empty(self):
        tracker = PathwayFitnessTracker()
        assert tracker.get_timing_anomalies("nonexistent") == []

    def test_step_timings_windowed(self):
        tracker = PathwayFitnessTracker()
        for i in range(60):
            tracker.record_execution(
                "pw1", ["s1"], {"s1": float(i)}, True, None, '{}',
            )
        rec = tracker.get_record("pw1")
        assert len(rec.step_timings["s1"]) == MAX_STEP_TIMINGS

    def test_input_clusters_grouped_by_failure_step(self):
        tracker = PathwayFitnessTracker()
        tracker.record_execution("pw1", [], {}, False, "s1", '{"a": 1}')
        tracker.record_execution("pw1", [], {}, False, "s1", '{"a": 2}')
        tracker.record_execution("pw1", [], {}, False, "s2", '{"b": 1}')
        tracker.record_execution("pw1", [], {}, True, None, '{"c": 1}')

        clusters = tracker.get_input_clusters("pw1")
        assert len(clusters) == 3  # s1, s2, None (success)

        s1_cluster = [c for c in clusters if c.failure_step == "s1"][0]
        assert s1_cluster.count == 2
        assert len(s1_cluster.recent_inputs) == 2

    def test_input_clusters_empty(self):
        tracker = PathwayFitnessTracker()
        assert tracker.get_input_clusters("nonexistent") == []

    def test_save_and_load(self, tmp_path):
        tracker = PathwayFitnessTracker()
        tracker.record_execution(
            "pw1", ["s1", "s2"], {"s1": 10.0, "s2": 20.0},
            True, None, '{"x": 1}',
        )
        tracker.record_execution(
            "pw1", ["s1"], {"s1": 15.0},
            False, "s2", '{"x": 2}',
        )

        path = tmp_path / "pathway_fitness.json"
        tracker.save(path)

        tracker2 = PathwayFitnessTracker.open(path)
        rec = tracker2.get_record("pw1")
        assert rec is not None
        assert rec.total_executions == 2
        assert rec.successful_executions == 1
        assert rec.failed_executions == 1
        assert "s1" in rec.step_timings
        assert rec.failure_step_distribution == {"s2": 1}

    def test_open_nonexistent_file(self, tmp_path):
        tracker = PathwayFitnessTracker.open(tmp_path / "does_not_exist.json")
        assert len(tracker.records) == 0

    def test_structure_hash_tracked(self):
        tracker = PathwayFitnessTracker()
        tracker.record_execution(
            "pw1", [], {}, True, None, '{}', structure_hash="abc123",
        )
        assert tracker.get_record("pw1").last_structure_hash == "abc123"

        tracker.record_execution(
            "pw1", [], {}, True, None, '{}', structure_hash="def456",
        )
        assert tracker.get_record("pw1").last_structure_hash == "def456"

    def test_get_record_none(self):
        tracker = PathwayFitnessTracker()
        assert tracker.get_record("nonexistent") is None

    def test_avg_time_ema(self):
        """Average time uses exponential moving average."""
        tracker = PathwayFitnessTracker()
        tracker.record_execution("pw1", ["s1"], {"s1": 100.0}, True, None, '{}')
        assert tracker.get_record("pw1").avg_execution_time_ms == 100.0

        # Second execution much faster — EMA should move slowly
        tracker.record_execution("pw1", ["s1"], {"s1": 0.0}, True, None, '{}')
        avg = tracker.get_record("pw1").avg_execution_time_ms
        assert avg == pytest.approx(90.0)  # 0.9 * 100 + 0.1 * 0

    def test_anomaly_at_exactly_threshold_not_detected(self):
        """Exactly 2.0x ratio uses >, so 2.0 should NOT be anomaly."""
        from sg.pathway_fitness import ANOMALY_THRESHOLD
        tracker = PathwayFitnessTracker()
        # 10 normal executions at 10ms
        for _ in range(10):
            tracker.record_execution(
                "pw1", ["s1"], {"s1": 10.0}, True, None, '{}',
            )
        # One execution at exactly 2x average — ratio = 20.0/10.0 = 2.0
        tracker.record_execution(
            "pw1", ["s1"], {"s1": 20.0}, True, None, '{}',
        )
        anomalies = tracker.get_timing_anomalies("pw1")
        assert len(anomalies) == 0  # exactly 2.0 not > 2.0

    def test_execution_window_at_max(self):
        """At exactly MAX_EXECUTION_WINDOW, adding one more trims oldest."""
        from sg.pathway_fitness import MAX_EXECUTION_WINDOW
        tracker = PathwayFitnessTracker()
        # Fill to exactly MAX
        for _ in range(MAX_EXECUTION_WINDOW):
            tracker.record_execution("pw1", [], {}, True, None, '{}')
        rec = tracker.get_record("pw1")
        assert len(rec._recent_outcomes) == MAX_EXECUTION_WINDOW
        # One more
        tracker.record_execution("pw1", [], {}, False, "s", '{}')
        assert len(rec._recent_outcomes) == MAX_EXECUTION_WINDOW

    def test_single_timing_no_anomaly(self):
        """Only 1 timing entry → no anomaly possible (needs >= 2)."""
        tracker = PathwayFitnessTracker()
        tracker.record_execution(
            "pw1", ["s1"], {"s1": 1000.0}, True, None, '{}',
        )
        anomalies = tracker.get_timing_anomalies("pw1")
        assert len(anomalies) == 0

    def test_timing_anomaly_to_dict(self):
        a = TimingAnomaly(step_name="s1", latest_ms=50.0, avg_ms=10.0, ratio=5.0)
        d = a.to_dict()
        assert d["step_name"] == "s1"
        assert d["latest_ms"] == 50.0
        assert d["ratio"] == 5.0


class TestPathwayFitnessIntegration:
    """Integration tests: fitness tracking through real pathway execution."""

    @pytest.fixture
    def project(self, tmp_path):
        fixtures_dst = tmp_path / "fixtures"
        shutil.copytree(FIXTURES_DIR, fixtures_dst)
        contracts_dst = tmp_path / "contracts"
        shutil.copytree(CONTRACTS_DIR, contracts_dst)

        contract_store = ContractStore.open(contracts_dst)
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

    def _make_orch(self, project_root):
        contract_store = ContractStore.open(project_root / "contracts")
        registry = Registry.open(project_root / ".sg" / "registry")
        phenotype = PhenotypeMap.load(project_root / "phenotype.toml")
        fusion_tracker = FusionTracker.open(project_root / "fusion_tracker.json")
        pathway_fitness_tracker = PathwayFitnessTracker.open(
            project_root / "pathway_fitness.json"
        )
        mutation_engine = MockMutationEngine(project_root / "fixtures")
        kernel = MockNetworkKernel()
        return Orchestrator(
            registry=registry,
            phenotype=phenotype,
            mutation_engine=mutation_engine,
            fusion_tracker=fusion_tracker,
            kernel=kernel,
            contract_store=contract_store,
            project_root=project_root,
            pathway_fitness_tracker=pathway_fitness_tracker,
        )

    def test_successful_pathway_records_fitness(self, project):
        orch = self._make_orch(project)
        orch.run_pathway("provision_management_bridge", json.dumps({
            "bridge_name": "br0",
            "interfaces": ["eth0"],
            "uplink": "eth1",
            "stp_enabled": True,
            "forward_delay": 15,
        }))
        rec = orch.pathway_fitness_tracker.get_record("provision_management_bridge")
        assert rec is not None
        assert rec.total_executions == 1
        assert rec.successful_executions == 1
        assert rec.failed_executions == 0
        assert len(rec.step_timings) > 0

    def test_step_timings_have_positive_values(self, project):
        orch = self._make_orch(project)
        orch.run_pathway("provision_management_bridge", json.dumps({
            "bridge_name": "br0",
            "interfaces": ["eth0"],
            "uplink": "eth1",
            "stp_enabled": True,
            "forward_delay": 15,
        }))
        rec = orch.pathway_fitness_tracker.get_record("provision_management_bridge")
        for step, timings in rec.step_timings.items():
            assert all(t >= 0 for t in timings), f"negative timing for {step}"

    def test_fitness_persists_via_save_state(self, project):
        orch = self._make_orch(project)
        orch.run_pathway("provision_management_bridge", json.dumps({
            "bridge_name": "br0",
            "interfaces": ["eth0"],
            "uplink": "eth1",
            "stp_enabled": True,
            "forward_delay": 15,
        }))
        orch.save_state()

        # Reload from disk
        orch2 = self._make_orch(project)
        rec = orch2.pathway_fitness_tracker.get_record("provision_management_bridge")
        assert rec is not None
        assert rec.total_executions == 1
