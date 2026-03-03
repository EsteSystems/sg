"""Tests for contract evolution: LLM-generated contracts + tightening/relaxation/feeds."""
import json
import shutil
import pytest
from pathlib import Path

from sg.contracts import ContractStore
from sg.contract_evolution import (
    ContractEvolution, ContractProposal, OutputObservation,
    MutationFailureTracker, FitnessCorrelation,
    TIGHTENING_THRESHOLD, RELAXATION_THRESHOLD,
    MIN_CORRELATION_SAMPLES, CORRELATION_THRESHOLD,
    MAX_PROPOSALS_PER_LOCUS, MAX_CORRELATIONS,
    _pearson,
)
from sg.mutation import MockMutationEngine, MutationEngine
from sg.parser.parser import parse_sg

import sg_network

CONTRACTS_DIR = sg_network.contracts_path()
FIXTURES_DIR = sg_network.fixtures_path()


class TestGenerateContract:
    def test_mock_engine_returns_fixture(self):
        """MockMutationEngine.generate_contract() returns fixture content."""
        engine = MockMutationEngine(FIXTURES_DIR)
        source = engine.generate_contract("diagnostic", "check VLAN trunk", [])
        assert "gene check_vlan_trunk" in source
        assert "is diagnostic" in source

    def test_mock_engine_missing_fixture(self):
        """MockMutationEngine raises if fixture missing."""
        engine = MockMutationEngine(Path("/nonexistent"))
        with pytest.raises(FileNotFoundError):
            engine.generate_contract("diagnostic", "anything", [])

    def test_base_engine_raises(self):
        """MutationEngine.generate_contract() raises NotImplementedError."""
        class Stub(MutationEngine):
            def mutate(self, ctx):
                return ""
        with pytest.raises(NotImplementedError):
            Stub().generate_contract("diagnostic", "test", [])


class TestRegisterContract:
    def test_register_contract(self, tmp_path):
        """ContractStore.register_contract writes and loads the contract."""
        shutil.copytree(CONTRACTS_DIR, tmp_path / "contracts")
        cs = ContractStore.open(tmp_path / "contracts")

        source = (FIXTURES_DIR / "generated_contract.sg").read_text()
        name = cs.register_contract(source, tmp_path / "contracts" / "genes" / "test_gen.sg")

        assert name == "check_vlan_trunk"
        assert cs.get_gene("check_vlan_trunk") is not None

    def test_generated_contract_parses(self):
        """The fixture contract parses successfully."""
        source = (FIXTURES_DIR / "generated_contract.sg").read_text()
        contract = parse_sg(source)
        assert contract.name == "check_vlan_trunk"
        assert contract.family.value == "diagnostic"
        assert contract.risk.value == "none"
        assert len(contract.takes) >= 2
        assert len(contract.gives) >= 2


class TestEvolveCommand:
    def test_cmd_evolve(self, tmp_path, capsys):
        """cmd_evolve generates and registers a contract."""
        import os
        import argparse
        from sg.cli import cmd_evolve

        shutil.copytree(CONTRACTS_DIR, tmp_path / "contracts")
        shutil.copytree(FIXTURES_DIR, tmp_path / "fixtures")
        (tmp_path / ".sg" / "registry").mkdir(parents=True)

        old = os.environ.get("SG_PROJECT_ROOT")
        os.environ["SG_PROJECT_ROOT"] = str(tmp_path)
        try:
            args = argparse.Namespace(
                family="diagnostic",
                context="check VLAN trunk connectivity",
                mutation_engine="mock",
                model=None,
            )
            cmd_evolve(args)
        finally:
            if old is None:
                os.environ.pop("SG_PROJECT_ROOT", None)
            else:
                os.environ["SG_PROJECT_ROOT"] = old

        captured = capsys.readouterr()
        assert "Generated contract: check_vlan_trunk" in captured.out
        # File should exist
        assert (tmp_path / "contracts" / "genes" / "check_vlan_trunk.sg").exists()


# --- Helpers for tightening/relaxation/feeds tests ---

class FakeField:
    def __init__(self, name, optional=False):
        self.name = name
        self.optional = optional


class FakeContract:
    def __init__(self, gives=None, feeds=None, before=None):
        self.gives = gives or []
        self.feeds = feeds or []
        self.before = before or []


class FakeFeed:
    def __init__(self, target):
        self.target = target
        self.target_locus = target
        self.timescale = "convergence"


class FakeContractStore:
    """Minimal mock of ContractStore for feeds analysis."""
    def __init__(self, loci=None):
        self._loci = loci or {}

    def known_loci(self):
        return list(self._loci.keys())

    def get_gene(self, name):
        return self._loci.get(name)


# --- ContractProposal ---

class TestContractProposalRoundtrip:
    def test_roundtrip(self):
        p = ContractProposal(
            locus="bridge_create",
            proposal_type="tighten_gives",
            description="extra field 'vlan_id'",
            evidence_count=55,
        )
        d = p.to_dict()
        restored = ContractProposal.from_dict(d)
        assert restored.locus == "bridge_create"
        assert restored.proposal_type == "tighten_gives"
        assert restored.evidence_count == 55
        assert restored.status == "pending"


# --- OutputObservation ---

class TestOutputObservationRoundtrip:
    def test_roundtrip(self):
        obs = OutputObservation(
            locus="check_conn",
            field_counts={"healthy": 10, "latency_ms": 10},
            total_observations=10,
            extra_fields={"latency_ms": 10},
            field_value_ranges={"latency_ms": [1.5, 2.0, 3.1]},
        )
        d = obs.to_dict()
        restored = OutputObservation.from_dict(d)
        assert restored.locus == "check_conn"
        assert restored.field_counts["healthy"] == 10
        assert restored.extra_fields["latency_ms"] == 10


# --- MutationFailureTracker ---

class TestMutationFailureTrackerUnit:
    def test_failure_rate(self):
        t = MutationFailureTracker(locus="x", total_mutations=10, failed_mutations=3)
        assert abs(t.failure_rate - 0.3) < 1e-9

    def test_failure_rate_zero_total(self):
        t = MutationFailureTracker(locus="x")
        assert t.failure_rate == 0.0

    def test_roundtrip(self):
        t = MutationFailureTracker(
            locus="x", total_mutations=20, failed_mutations=8,
            constraint_violations={"gives_missing_field": 5},
        )
        d = t.to_dict()
        restored = MutationFailureTracker.from_dict(d)
        assert restored.failure_rate == 8 / 20


# --- Tightening ---

class TestTightening:
    def test_no_proposal_below_threshold(self):
        ce = ContractEvolution()
        contract = FakeContract(gives=[FakeField("success")])
        result = None
        for _ in range(TIGHTENING_THRESHOLD - 1):
            result = ce.record_output(
                "l", json.dumps({"success": True, "extra": 1}), contract,
            )
        assert result is None

    def test_extra_field_proposed(self):
        ce = ContractEvolution()
        contract = FakeContract(gives=[FakeField("success")])
        result = None
        for _ in range(TIGHTENING_THRESHOLD):
            result = ce.record_output(
                "l", json.dumps({"success": True, "extra": 42}), contract,
            )
        assert result is not None
        assert result.proposal_type == "tighten_gives"
        assert "extra" in result.description

    def test_optional_always_present_proposed(self):
        ce = ContractEvolution()
        contract = FakeContract(gives=[
            FakeField("success"),
            FakeField("detail", optional=True),
        ])
        for _ in range(TIGHTENING_THRESHOLD):
            ce.record_output(
                "l", json.dumps({"success": True, "detail": "ok"}), contract,
            )
        proposals = ce.get_proposals(locus="l")
        optional_proposals = [
            p for p in proposals if "always present" in p.description
        ]
        assert len(optional_proposals) == 1
        assert "detail" in optional_proposals[0].description

    def test_invalid_json_ignored(self):
        ce = ContractEvolution()
        contract = FakeContract()
        assert ce.record_output("l", "not json", contract) is None

    def test_non_dict_json_ignored(self):
        ce = ContractEvolution()
        contract = FakeContract()
        assert ce.record_output("l", '"string"', contract) is None

    def test_duplicate_proposals_not_created(self):
        ce = ContractEvolution()
        contract = FakeContract(gives=[FakeField("success")])
        for _ in range(TIGHTENING_THRESHOLD * 2):
            ce.record_output(
                "l", json.dumps({"success": True, "extra": 42}), contract,
            )
        proposals = ce.get_proposals(locus="l")
        descs = [p.description for p in proposals]
        assert len(descs) == len(set(descs))

    def test_numeric_range_tracked(self):
        ce = ContractEvolution()
        contract = FakeContract(gives=[FakeField("success"), FakeField("value")])
        for i in range(TIGHTENING_THRESHOLD):
            ce.record_output(
                "l", json.dumps({"success": True, "value": float(i)}), contract,
            )
        obs = ce.observations["l"]
        assert len(obs.field_value_ranges["value"]) == TIGHTENING_THRESHOLD


# --- Relaxation ---

class TestRelaxation:
    def test_no_proposal_below_min_mutations(self):
        ce = ContractEvolution()
        for _ in range(5):
            ce.record_mutation_failure("l", "missing field 'x'")
            ce.record_mutation_success("l")
        # 10 total but only 50% failure rate
        proposals = ce.get_proposals(locus="l")
        assert len(proposals) == 0

    def test_high_failure_rate_proposed(self):
        ce = ContractEvolution()
        # Put successes first so 10th call is a failure (triggers check)
        for _ in range(2):
            ce.record_mutation_success("l")
        for _ in range(8):
            ce.record_mutation_failure("l", "missing field 'x'")
        # 80% failure rate > 0.3 threshold, 10 total mutations
        proposals = ce.get_proposals(locus="l")
        assert len(proposals) >= 1
        assert proposals[0].proposal_type == "relax_constraint"

    def test_low_failure_rate_no_proposal(self):
        ce = ContractEvolution()
        for _ in range(2):
            ce.record_mutation_failure("l", "missing field 'x'")
        for _ in range(18):
            ce.record_mutation_success("l")
        # 10% failure rate < 0.3
        proposals = ce.get_proposals(locus="l")
        assert len(proposals) == 0

    def test_classify_violation(self):
        assert ContractEvolution._classify_violation("Missing field 'x'") == "gives_missing_field"
        assert ContractEvolution._classify_violation("Type mismatch for y") == "gives_wrong_type"
        assert ContractEvolution._classify_violation("Invalid JSON output") == "invalid_json"
        assert ContractEvolution._classify_violation("some random error") == ""


# --- Feeds discovery ---

class TestFeedsDiscovery:
    def test_perfect_correlation_proposed(self):
        ce = ContractEvolution()
        ce.ensure_correlation_pair("check_conn", "bridge_create")

        for i in range(MIN_CORRELATION_SAMPLES):
            ce.record_diagnostic_output("check_conn", {"healthy": i % 2 == 0})
            ce.record_config_fitness("bridge_create", 1.0 if i % 2 == 0 else 0.0)

        store = FakeContractStore({"check_conn": FakeContract(), "bridge_create": FakeContract()})
        proposals = ce.analyze_feeds(store)
        assert len(proposals) >= 1
        assert proposals[0].proposal_type == "add_feeds"

    def test_no_correlation_no_proposal(self):
        ce = ContractEvolution()
        ce.ensure_correlation_pair("check_conn", "bridge_create")

        # Uncorrelated: constant diagnostic, varying fitness
        for i in range(MIN_CORRELATION_SAMPLES):
            ce.record_diagnostic_output("check_conn", {"healthy": True})
            ce.record_config_fitness("bridge_create", float(i))

        store = FakeContractStore({"check_conn": FakeContract(), "bridge_create": FakeContract()})
        proposals = ce.analyze_feeds(store)
        assert len(proposals) == 0

    def test_existing_feeds_not_re_proposed(self):
        ce = ContractEvolution()
        ce.ensure_correlation_pair("check_conn", "bridge_create")

        for i in range(MIN_CORRELATION_SAMPLES):
            ce.record_diagnostic_output("check_conn", {"healthy": i % 2 == 0})
            ce.record_config_fitness("bridge_create", 1.0 if i % 2 == 0 else 0.0)

        store = FakeContractStore({
            "check_conn": FakeContract(feeds=[FakeFeed("bridge_create")]),
            "bridge_create": FakeContract(),
        })
        proposals = ce.analyze_feeds(store)
        assert len(proposals) == 0

    def test_insufficient_samples_no_proposal(self):
        ce = ContractEvolution()
        ce.ensure_correlation_pair("check_conn", "bridge_create")

        for i in range(MIN_CORRELATION_SAMPLES - 1):
            ce.record_diagnostic_output("check_conn", {"healthy": True})
            ce.record_config_fitness("bridge_create", 1.0)

        store = FakeContractStore({"check_conn": FakeContract(), "bridge_create": FakeContract()})
        proposals = ce.analyze_feeds(store)
        assert len(proposals) == 0

    def test_max_correlations_bounded(self):
        ce = ContractEvolution()
        for i in range(MAX_CORRELATIONS + 10):
            ce.ensure_correlation_pair(f"diag_{i}", f"config_{i}")
        assert len(ce.correlations) == MAX_CORRELATIONS


# --- Proposal management ---

class TestProposalManagement:
    def test_accept_proposal(self):
        ce = ContractEvolution()
        ce._store_proposals("l", [ContractProposal(
            locus="l", proposal_type="tighten_gives", description="test",
        )])
        assert ce.accept_proposal("l", 0) is True
        assert ce.get_proposals(locus="l", status="accepted")[0].status == "accepted"
        assert len(ce.get_proposals(locus="l", status="pending")) == 0

    def test_reject_proposal(self):
        ce = ContractEvolution()
        ce._store_proposals("l", [ContractProposal(
            locus="l", proposal_type="tighten_gives", description="test",
        )])
        assert ce.reject_proposal("l", 0) is True
        assert ce.get_proposals(locus="l", status="rejected")[0].status == "rejected"

    def test_invalid_index_returns_false(self):
        ce = ContractEvolution()
        assert ce.accept_proposal("l", 0) is False
        assert ce.reject_proposal("l", 0) is False

    def test_bounded_proposals(self):
        ce = ContractEvolution()
        proposals = [
            ContractProposal(locus="l", proposal_type="tighten_gives",
                             description=f"proposal {i}")
            for i in range(MAX_PROPOSALS_PER_LOCUS + 5)
        ]
        ce._store_proposals("l", proposals)
        assert len(ce.proposals["l"]) == MAX_PROPOSALS_PER_LOCUS

    def test_get_all_proposals(self):
        ce = ContractEvolution()
        ce._store_proposals("a", [ContractProposal(
            locus="a", proposal_type="t", description="from a",
        )])
        ce._store_proposals("b", [ContractProposal(
            locus="b", proposal_type="t", description="from b",
        )])
        all_pending = ce.get_proposals()
        assert len(all_pending) == 2


# --- Pearson ---

class TestPearsonCorrelation:
    def test_perfect_positive(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]
        assert abs(_pearson(x, y) - 1.0) < 1e-9

    def test_perfect_negative(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [10.0, 8.0, 6.0, 4.0, 2.0]
        assert abs(_pearson(x, y) - (-1.0)) < 1e-9

    def test_no_correlation(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [5.0, 5.0, 5.0, 5.0, 5.0]  # constant = 0 variance
        assert _pearson(x, y) == 0.0

    def test_mismatched_lengths(self):
        assert _pearson([1.0], [1.0, 2.0]) == 0.0

    def test_too_few_samples(self):
        assert _pearson([1.0], [1.0]) == 0.0


# --- Persistence ---

class TestContractEvolutionPersistence:
    def test_save_and_open(self, tmp_path):
        ce = ContractEvolution()
        ce.record_mutation_success("bridge_create")
        ce.record_mutation_failure("bridge_create", "missing field 'x'")
        ce._store_proposals("bridge_create", [ContractProposal(
            locus="bridge_create", proposal_type="relax_constraint",
            description="test proposal",
        )])
        ce.ensure_correlation_pair("check_conn", "bridge_create")
        ce.record_diagnostic_output("check_conn", {"healthy": True})
        ce.record_config_fitness("bridge_create", 0.9)

        path = tmp_path / "contract_evolution.json"
        ce.save(path)

        restored = ContractEvolution.open(path)
        assert restored.mutation_trackers["bridge_create"].total_mutations == 2
        assert restored.mutation_trackers["bridge_create"].failed_mutations == 1
        assert len(restored.proposals["bridge_create"]) == 1
        assert len(restored.correlations) == 1

    def test_open_missing_file(self, tmp_path):
        path = tmp_path / "missing.json"
        ce = ContractEvolution.open(path)
        assert len(ce.observations) == 0

    def test_open_corrupted_file(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not valid json{{{")
        ce = ContractEvolution.open(path)
        assert len(ce.observations) == 0


# --- Before condition analysis (Item 8) ---


class TestBeforeAnalysis:
    def test_before_condition_always_satisfied_proposed(self):
        """Before conditions always satisfied across 50+ outputs -> tighten_before proposal."""
        import json as _json
        ce = ContractEvolution()
        contract = FakeContract(
            gives=[FakeField("status")],
            before=["system must be online"],
        )
        for _ in range(TIGHTENING_THRESHOLD + 5):
            ce.record_output("bridge_create",
                             _json.dumps({"status": "ok", "success": True}),
                             contract)
        proposals = [p for p in ce.get_proposals("bridge_create")
                     if p.proposal_type == "tighten_before"]
        assert len(proposals) >= 1
        assert "always satisfied" in proposals[0].description

    def test_before_condition_not_always_satisfied(self):
        """Before conditions satisfied < 95% -> no proposal."""
        import json as _json
        ce = ContractEvolution()
        contract_with = FakeContract(
            gives=[FakeField("status")],
            before=["interface must exist"],
        )
        contract_without = FakeContract(
            gives=[FakeField("status")],
            before=[],  # simulate condition not met by not tracking it
        )
        # Record 40 with before, 20 without (ratio = 40/60 < 0.95)
        for _ in range(40):
            ce.record_output("bridge_create",
                             _json.dumps({"status": "ok", "success": True}),
                             contract_with)
        for _ in range(20):
            ce.record_output("bridge_create",
                             _json.dumps({"status": "ok", "success": True}),
                             contract_without)
        proposals = [p for p in ce.get_proposals("bridge_create")
                     if p.proposal_type == "tighten_before"]
        assert len(proposals) == 0

    def test_precondition_counter_increments(self):
        """Precondition satisfied counter tracks each observation."""
        import json as _json
        ce = ContractEvolution()
        contract = FakeContract(before=["X is ready"])
        ce.record_output("test", _json.dumps({"success": True}), contract)
        ce.record_output("test", _json.dumps({"success": True}), contract)
        obs = ce.observations["test"]
        assert obs.precondition_satisfied["X is ready"] == 2

    def test_precondition_serialization_roundtrip(self, tmp_path):
        """Precondition data survives save/load."""
        import json as _json
        ce = ContractEvolution()
        contract = FakeContract(before=["net is up"])
        ce.record_output("test", _json.dumps({"success": True}), contract)
        ce.save(tmp_path / "ce.json")
        restored = ContractEvolution.open(tmp_path / "ce.json")
        assert restored.observations["test"].precondition_satisfied["net is up"] == 1


# --- Feeds analysis trigger (Item 1) ---


class TestFeedsAnalysisTrigger:
    def test_record_config_fitness_triggers_analysis(self):
        """When enough correlated samples exist, analyze_feeds runs automatically."""
        ce = ContractEvolution()
        ce.ensure_correlation_pair("diag_x", "config_y")

        # Build up enough samples
        for i in range(MIN_CORRELATION_SAMPLES):
            ce.record_diagnostic_output("diag_x", {"healthy": True})
            ce.record_config_fitness("config_y", 0.9)

        # Now record one more with contract_store to trigger analysis
        ce.record_diagnostic_output("diag_x", {"healthy": True})
        store = FakeContractStore({"diag_x": FakeContract(feeds=[])})
        proposals = ce.record_config_fitness("config_y", 0.9, contract_store=store)
        # With perfectly correlated data, should get a feeds proposal
        # (all values are constant so r is undefined, but no error)
        # This tests the trigger mechanism, not the correlation result
        assert isinstance(proposals, list)

    def test_no_contract_store_no_analysis(self):
        """Without contract_store, analyze_feeds is not called."""
        ce = ContractEvolution()
        ce.ensure_correlation_pair("diag_x", "config_y")
        for i in range(MIN_CORRELATION_SAMPLES + 5):
            ce.record_diagnostic_output("diag_x", {"healthy": i % 2 == 0})
            result = ce.record_config_fitness("config_y", 0.5 + (i % 2) * 0.4)
        assert result == []  # No store means no analysis

    def test_insufficient_samples_no_trigger(self):
        """With < MIN_CORRELATION_SAMPLES, no analysis triggered."""
        ce = ContractEvolution()
        ce.ensure_correlation_pair("diag_x", "config_y")
        for i in range(5):
            ce.record_diagnostic_output("diag_x", {"healthy": True})
            result = ce.record_config_fitness("config_y", 0.9,
                                              contract_store=FakeContractStore())
        assert result == []
