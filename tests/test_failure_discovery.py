"""Phase E.2: Failure mode discovery tests.

Verifies novel error tracking, coverage heuristic, proposal lifecycle,
persistence, bounded growth, and orchestrator integration.
"""
from __future__ import annotations

import shutil

import pytest

import sg_network
from sg.contracts import ContractStore
from sg.failure_discovery import (
    FailureDiscovery,
    FailureProposal,
    NovelErrorPattern,
    MAX_TRACKED_PATTERNS,
    PROPOSAL_THRESHOLD,
)
from sg.fusion import FusionTracker
from sg_network import MockNetworkKernel
from sg.mutation import MockMutationEngine
from sg.orchestrator import Orchestrator
from sg.pathway_fitness import PathwayFitnessTracker
from sg.pathway_registry import PathwayRegistry
from sg.phenotype import PhenotypeMap
from sg.registry import Registry


CONTRACTS_DIR = sg_network.contracts_path()
GENES_DIR = sg_network.genes_path()
FIXTURES_DIR = sg_network.fixtures_path()


class TestNovelErrorTracking:
    """Record errors and accumulate counts."""

    def test_records_novel_error(self):
        fd = FailureDiscovery()
        fd.record_error("bridge_create", "sha1", "timeout after 30s", [])
        assert "bridge_create" in fd.patterns
        patterns = fd.patterns["bridge_create"]
        assert len(patterns) == 1
        entry = next(iter(patterns.values()))
        assert entry.count == 1
        assert "timeout after 30s" in entry.representative_messages

    def test_covered_error_ignored(self):
        fd = FailureDiscovery()
        # "connection timeout" words overlap >50% with known fails_when
        known = ["connection timeout failure"]
        result = fd.record_error(
            "bridge_create", "sha1", "connection timeout", known,
        )
        assert result is None
        # Pattern should not be tracked since it's covered
        assert len(fd.patterns.get("bridge_create", {})) == 0

    def test_is_covered_heuristic(self):
        # More than 50% of significant words overlap → covered
        assert FailureDiscovery._is_covered(
            "connection timeout error", ["timeout error on device"],
        )
        # No overlap → not covered
        assert not FailureDiscovery._is_covered(
            "connection timeout error", ["disk full"],
        )
        # Empty pattern words (all stop words) → not covered
        assert not FailureDiscovery._is_covered("the a an", ["foo"])


class TestProposalLifecycle:
    """Proposal generation, dedup, accept, reject."""

    def test_no_proposal_below_threshold(self):
        fd = FailureDiscovery()
        for i in range(PROPOSAL_THRESHOLD - 1):
            result = fd.record_error(
                "bridge_create", f"sha{i}", "segfault at address 0xdead", [],
            )
            assert result is None

    def test_proposal_at_threshold(self):
        fd = FailureDiscovery()
        result = None
        for i in range(PROPOSAL_THRESHOLD):
            result = fd.record_error(
                "bridge_create", f"sha{i}", "segfault at address 0xdead", [],
            )
        assert result is not None
        assert isinstance(result, FailureProposal)
        assert result.locus == "bridge_create"
        assert result.occurrence_count >= PROPOSAL_THRESHOLD
        assert result.status == "pending"

    def test_no_duplicate_proposal(self):
        fd = FailureDiscovery()
        proposals = []
        for i in range(PROPOSAL_THRESHOLD * 2):
            result = fd.record_error(
                "bridge_create", f"sha{i}", "segfault at address 0xdead", [],
            )
            if result:
                proposals.append(result)
        # Only one proposal for the same normalized pattern
        assert len(proposals) == 1

    def test_proposal_text_readable(self):
        fd = FailureDiscovery()
        for i in range(PROPOSAL_THRESHOLD):
            fd.record_error(
                "stp_set", f"sha{i}",
                "failed to set priority 42 on /sys/class/net/br0", [],
            )
        proposals = fd.get_proposals("stp_set")
        assert len(proposals) == 1
        text = proposals[0].proposed_text
        # Placeholders should be replaced with readable words
        assert "<N>" not in text
        assert "<PATH>" not in text

    def test_accept_proposal(self):
        fd = FailureDiscovery()
        for i in range(PROPOSAL_THRESHOLD):
            fd.record_error("locus_a", f"sha{i}", "disk full error", [])
        proposals = fd.get_proposals("locus_a")
        assert len(proposals) == 1
        pattern = proposals[0].pattern
        assert fd.accept_proposal("locus_a", pattern)
        # After accept, no longer pending
        assert len(fd.get_proposals("locus_a")) == 0

    def test_reject_proposal(self):
        fd = FailureDiscovery()
        for i in range(PROPOSAL_THRESHOLD):
            fd.record_error("locus_a", f"sha{i}", "disk full error", [])
        proposals = fd.get_proposals("locus_a")
        pattern = proposals[0].pattern
        assert fd.reject_proposal("locus_a", pattern)
        assert len(fd.get_proposals("locus_a")) == 0


class TestPersistence:
    """Save/load round-trip."""

    def test_persistence_roundtrip(self, tmp_path):
        fd = FailureDiscovery()
        for i in range(PROPOSAL_THRESHOLD):
            fd.record_error("bridge_create", f"sha{i}", "timeout error", [])

        path = tmp_path / "fd.json"
        fd.save(path)

        fd2 = FailureDiscovery.open(path)
        assert "bridge_create" in fd2.patterns
        proposals = fd2.get_proposals("bridge_create")
        assert len(proposals) == 1
        assert proposals[0].occurrence_count >= PROPOSAL_THRESHOLD


class TestBoundedGrowth:
    """Pattern map stays within MAX_TRACKED_PATTERNS."""

    def test_max_patterns_capped(self):
        fd = FailureDiscovery()
        for i in range(MAX_TRACKED_PATTERNS + 20):
            fd.record_error(
                "locus_a", "sha1", f"unique error number {i}", [],
            )
        assert len(fd.patterns["locus_a"]) <= MAX_TRACKED_PATTERNS


class TestFailureDiscoveryIntegration:
    """Failure discovery through orchestrator execute_locus."""

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
        return Orchestrator(
            registry=registry, phenotype=phenotype,
            mutation_engine=mutation_engine, fusion_tracker=fusion_tracker,
            kernel=kernel, contract_store=contract_store,
            project_root=project_root,
            pathway_fitness_tracker=pft, pathway_registry=pr,
        )

    def test_orchestrator_records_failure_discovery(self, full_project):
        """Errors in execute_locus feed into failure discovery."""
        orch = self._make_orchestrator(full_project)

        # Register a gene that always fails with a distinctive error
        failing_source = (
            'def execute(input_json: str) -> str:\n'
            '    raise RuntimeError("custom unique failure xyz")\n'
        )
        locus = "bridge_create"
        sha = orch.registry.register(failing_source, locus)
        orch.phenotype.promote(locus, sha)
        orch.registry.get(sha).state = "dominant"
        # Clear fallback so mutation path triggers
        orch.phenotype.loci[locus].fallback.clear()

        # Execute enough times to accumulate errors
        for _ in range(PROPOSAL_THRESHOLD + 1):
            try:
                orch.execute_locus(locus, '{"bridge": "br0"}')
            except Exception:
                pass

        # The failure discovery should have tracked this pattern
        patterns = orch._failure_discovery.patterns.get(locus, {})
        assert len(patterns) > 0


# --- LLM-assisted failure summarization (Item 7) ---


class FakeSummarizingEngine:
    """A mutation engine that supports failure summarization."""

    def summarize_failure(self, locus, pattern, messages):
        return f"Gene '{locus}' fails when {pattern} occurs ({len(messages)} examples)"


class FakeFailingEngine:
    """A mutation engine that raises on summarization."""

    def summarize_failure(self, locus, pattern, messages):
        raise RuntimeError("LLM unavailable")


class TestLLMFailureSummarization:
    def test_with_engine_uses_llm(self):
        """When engine supports summarize_failure, uses LLM output."""
        fd = FailureDiscovery(mutation_engine=FakeSummarizingEngine())
        for _ in range(PROPOSAL_THRESHOLD + 1):
            fd.record_error("bridge_create", "abc", "timeout at 0xFF",
                            known_fails_when=[])
        proposals = fd.proposals.get("bridge_create", [])
        assert len(proposals) >= 1
        assert "Gene 'bridge_create' fails" in proposals[0].proposed_text

    def test_engine_fallback_on_error(self):
        """When engine raises, falls back to placeholder."""
        fd = FailureDiscovery(mutation_engine=FakeFailingEngine())
        for _ in range(PROPOSAL_THRESHOLD + 1):
            fd.record_error("bridge_create", "abc", "timeout at 0xFF",
                            known_fails_when=[])
        proposals = fd.proposals.get("bridge_create", [])
        assert len(proposals) >= 1
        # Should have used placeholder substitution
        assert "Gene 'bridge_create' fails" not in proposals[0].proposed_text

    def test_no_engine_uses_placeholder(self):
        """Without engine, uses placeholder substitution."""
        fd = FailureDiscovery(mutation_engine=None)
        for _ in range(PROPOSAL_THRESHOLD + 1):
            fd.record_error("bridge_create", "abc", "error at /path/to/file",
                            known_fails_when=[])
        proposals = fd.proposals.get("bridge_create", [])
        assert len(proposals) >= 1
        # Placeholder should have replaced <PATH> with "path"
        assert proposals[0].proposed_text  # non-empty

    def test_open_with_engine(self, tmp_path):
        """open() passes engine to instance."""
        fd = FailureDiscovery.open(tmp_path / "fd.json",
                                   mutation_engine=FakeSummarizingEngine())
        assert fd._mutation_engine is not None
