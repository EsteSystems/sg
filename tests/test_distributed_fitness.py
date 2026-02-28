"""Tests for distributed fitness scoring."""
import json
import pytest
from pathlib import Path

from sg.arena import compute_fitness, compute_distributed_fitness, record_success
from sg.federation import merge_peer_observation
from sg.registry import Registry, AlleleMetadata


class TestDistributedFitness:
    def test_no_peers_returns_local(self):
        """With no peer observations, distributed fitness equals local."""
        allele = AlleleMetadata(sha256="abc", locus="bridge_create")
        allele.successful_invocations = 8
        allele.failed_invocations = 2

        local = compute_fitness(allele)
        distributed = compute_distributed_fitness(allele)
        assert distributed == local

    def test_with_peers_weighted(self):
        """Distributed fitness is 70% local + 30% peer."""
        allele = AlleleMetadata(sha256="abc", locus="bridge_create")
        allele.successful_invocations = 10
        allele.failed_invocations = 0

        # Peer reports 10 successes, 0 failures
        allele.peer_observations = [
            {"peer": "peer1", "successes": 10, "failures": 0, "timestamp": 0},
        ]

        local = compute_fitness(allele)  # 10/10 = 1.0
        distributed = compute_distributed_fitness(allele)
        # 0.7 * 1.0 + 0.3 * 1.0 = 1.0
        assert distributed == pytest.approx(1.0)

    def test_peers_lower_fitness(self):
        """Peer failures reduce distributed fitness."""
        allele = AlleleMetadata(sha256="abc", locus="bridge_create")
        allele.successful_invocations = 10
        allele.failed_invocations = 0

        # Peer reports 5 successes, 5 failures
        allele.peer_observations = [
            {"peer": "peer1", "successes": 5, "failures": 5, "timestamp": 0},
        ]

        local = compute_fitness(allele)  # 1.0
        distributed = compute_distributed_fitness(allele)
        peer_fitness = 5 / 10  # 0.5
        expected = 0.7 * local + 0.3 * peer_fitness
        assert distributed == pytest.approx(expected)

    def test_insufficient_peer_data_uses_local(self):
        """Below MIN_INVOCATIONS_FOR_SCORE peer data, use local only."""
        allele = AlleleMetadata(sha256="abc", locus="bridge_create")
        allele.successful_invocations = 10
        allele.failed_invocations = 0

        # Peer reports only 3 total invocations (below threshold of 10)
        allele.peer_observations = [
            {"peer": "peer1", "successes": 2, "failures": 1, "timestamp": 0},
        ]

        local = compute_fitness(allele)
        distributed = compute_distributed_fitness(allele)
        assert distributed == local

    def test_multiple_peers_aggregated(self):
        """Multiple peer observations are aggregated."""
        allele = AlleleMetadata(sha256="abc", locus="bridge_create")
        allele.successful_invocations = 10
        allele.failed_invocations = 0

        allele.peer_observations = [
            {"peer": "peer1", "successes": 5, "failures": 0, "timestamp": 0},
            {"peer": "peer2", "successes": 5, "failures": 0, "timestamp": 0},
        ]

        distributed = compute_distributed_fitness(allele)
        # Total peer: 10s + 0f = 1.0 fitness
        assert distributed == pytest.approx(0.7 * 1.0 + 0.3 * 1.0)


class TestMergePeerObservation:
    def test_merge_adds_observation(self):
        """merge_peer_observation adds to peer_observations list."""
        allele = AlleleMetadata(sha256="abc", locus="bridge_create")
        assert len(allele.peer_observations) == 0

        data = {"successful_invocations": 10, "total_invocations": 12}
        merge_peer_observation(allele, "peer1", data)

        assert len(allele.peer_observations) == 1
        obs = allele.peer_observations[0]
        assert obs["peer"] == "peer1"
        assert obs["successes"] == 10
        assert obs["failures"] == 2
        assert "timestamp" in obs

    def test_merge_multiple_peers(self):
        """Multiple merges from different peers."""
        allele = AlleleMetadata(sha256="abc", locus="bridge_create")

        merge_peer_observation(allele, "peer1", {"successful_invocations": 5, "total_invocations": 5})
        merge_peer_observation(allele, "peer2", {"successful_invocations": 3, "total_invocations": 10})

        assert len(allele.peer_observations) == 2
        assert allele.peer_observations[0]["peer"] == "peer1"
        assert allele.peer_observations[1]["peer"] == "peer2"


class TestPeerObservationsPersist:
    def test_peer_observations_saved_and_loaded(self, tmp_path):
        """peer_observations persist through save/load cycle."""
        reg = Registry.open(tmp_path / "reg")
        sha = reg.register("def execute(i): return '{}'", "bridge_create")
        allele = reg.get(sha)

        merge_peer_observation(allele, "peer1", {
            "successful_invocations": 10, "total_invocations": 12,
        })
        reg.save_index()

        reg2 = Registry.open(tmp_path / "reg")
        allele2 = reg2.get(sha)
        assert len(allele2.peer_observations) == 1
        assert allele2.peer_observations[0]["peer"] == "peer1"
        assert allele2.peer_observations[0]["successes"] == 10
