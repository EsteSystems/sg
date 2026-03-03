"""Tests for cross-locus failure analysis."""
from __future__ import annotations

import pytest

from sg.locus_discovery import (
    CrossLocusFailureAnalyzer, CrossLocusPattern, LocusProposal,
    MIN_LOCI_FOR_PROPOSAL, MIN_OCCURRENCES_FOR_PROPOSAL,
    MAX_CROSS_LOCUS_PATTERNS, MAX_PROPOSALS,
)


class TestCrossLocusPattern:
    def test_roundtrip(self):
        p = CrossLocusPattern(
            pattern="resource '<STR>' not found",
            loci={"bridge_create": 3, "vlan_set": 2},
            total_count=5,
        )
        d = p.to_dict()
        restored = CrossLocusPattern.from_dict(d)
        assert restored.pattern == p.pattern
        assert restored.loci == p.loci
        assert restored.total_count == 5


class TestLocusProposal:
    def test_roundtrip(self):
        p = LocusProposal(
            proposed_name="check_resource",
            family="diagnostic",
            description="test",
            evidence_pattern="not found",
            affected_loci=["a", "b"],
        )
        d = p.to_dict()
        restored = LocusProposal.from_dict(d)
        assert restored.proposed_name == "check_resource"
        assert restored.status == "pending"


class TestCrossLocusFailureAnalyzer:
    def test_no_proposal_below_loci_threshold(self):
        analyzer = CrossLocusFailureAnalyzer()
        # Same error from only 2 loci (need 3)
        for _ in range(MIN_OCCURRENCES_FOR_PROPOSAL):
            analyzer.record_error("locus_a", "resource 'br0' not found")
            analyzer.record_error("locus_b", "resource 'br1' not found")
        proposals = analyzer.get_proposals()
        assert len(proposals) == 0

    def test_no_proposal_below_count_threshold(self):
        analyzer = CrossLocusFailureAnalyzer()
        # 3 loci but under 10 total
        for locus in ["a", "b", "c"]:
            analyzer.record_error(locus, "resource 'x' not found")
        proposals = analyzer.get_proposals()
        assert len(proposals) == 0

    def test_proposal_when_thresholds_met(self):
        analyzer = CrossLocusFailureAnalyzer()
        loci = ["locus_a", "locus_b", "locus_c", "locus_d"]
        for i in range(MIN_OCCURRENCES_FOR_PROPOSAL):
            locus = loci[i % len(loci)]
            result = analyzer.record_error(locus, f"resource 'br{i}' not found")
        # After enough calls, should have a proposal
        proposals = analyzer.get_proposals()
        assert len(proposals) >= 1
        assert proposals[0].status == "pending"

    def test_duplicate_proposal_not_created(self):
        analyzer = CrossLocusFailureAnalyzer()
        loci = ["a", "b", "c"]
        # First round: creates proposal
        for i in range(MIN_OCCURRENCES_FOR_PROPOSAL):
            analyzer.record_error(loci[i % 3], f"resource 'x{i}' not found")
        # Second round: same pattern shouldn't create another
        for i in range(MIN_OCCURRENCES_FOR_PROPOSAL):
            analyzer.record_error(loci[i % 3], f"resource 'y{i}' not found")
        proposals = analyzer.get_proposals()
        assert len(proposals) == 1

    def test_error_normalization(self):
        analyzer = CrossLocusFailureAnalyzer()
        loci = ["a", "b", "c"]
        # Different specific values should normalize to same pattern
        for i in range(MIN_OCCURRENCES_FOR_PROPOSAL):
            locus = loci[i % 3]
            analyzer.record_error(locus, f"resource 'item_{i}' not found")
        # All should be same pattern after normalization
        assert len(analyzer.patterns) == 1

    def test_bounded_patterns(self):
        analyzer = CrossLocusFailureAnalyzer()
        for i in range(MAX_CROSS_LOCUS_PATTERNS + 50):
            # Each unique error creates a unique pattern
            analyzer.record_error("locus_a", f"unique_error_{i} happened")
        assert len(analyzer.patterns) <= MAX_CROSS_LOCUS_PATTERNS

    def test_accept_reject_proposals(self):
        analyzer = CrossLocusFailureAnalyzer()
        loci = ["a", "b", "c"]
        for i in range(MIN_OCCURRENCES_FOR_PROPOSAL):
            analyzer.record_error(loci[i % 3], f"resource 'x{i}' not found")

        assert analyzer.accept_proposal(0) is True
        assert len(analyzer.get_proposals(status="accepted")) == 1
        assert len(analyzer.get_proposals(status="pending")) == 0

    def test_reject_proposal(self):
        analyzer = CrossLocusFailureAnalyzer()
        loci = ["a", "b", "c"]
        for i in range(MIN_OCCURRENCES_FOR_PROPOSAL):
            analyzer.record_error(loci[i % 3], f"resource 'x{i}' not found")

        assert analyzer.reject_proposal(0) is True
        assert len(analyzer.get_proposals(status="rejected")) == 1

    def test_invalid_index(self):
        analyzer = CrossLocusFailureAnalyzer()
        assert analyzer.accept_proposal(0) is False
        assert analyzer.reject_proposal(0) is False

    def test_proposal_family_detection(self):
        analyzer = CrossLocusFailureAnalyzer()
        loci = ["a", "b", "c"]
        # "not found" pattern → diagnostic
        for i in range(MIN_OCCURRENCES_FOR_PROPOSAL):
            analyzer.record_error(loci[i % 3], f"resource 'x{i}' not found")
        proposals = analyzer.get_proposals()
        assert proposals[0].family == "diagnostic"


class TestCrossLocusPersistence:
    def test_save_and_open(self, tmp_path):
        analyzer = CrossLocusFailureAnalyzer()
        loci = ["a", "b", "c"]
        for i in range(MIN_OCCURRENCES_FOR_PROPOSAL):
            analyzer.record_error(loci[i % 3], f"resource 'x{i}' not found")

        path = tmp_path / "locus_discovery.json"
        analyzer.save(path)

        restored = CrossLocusFailureAnalyzer.open(path)
        assert len(restored.patterns) == len(analyzer.patterns)
        assert len(restored.proposals) == len(analyzer.proposals)

    def test_open_missing(self, tmp_path):
        analyzer = CrossLocusFailureAnalyzer.open(tmp_path / "missing.json")
        assert len(analyzer.patterns) == 0

    def test_open_corrupted(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{invalid")
        analyzer = CrossLocusFailureAnalyzer.open(path)
        assert len(analyzer.patterns) == 0
