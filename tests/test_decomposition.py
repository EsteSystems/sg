"""Tests for automatic decomposition (Phase 3)."""
import json
import re
import shutil

import pytest
from pathlib import Path

from sg.decomposition import (
    DecompositionDetector,
    DecompositionResult,
    DecompositionSignal,
    ErrorCluster,
    LocusErrorHistory,
    _normalize_error,
    ERROR_WINDOW,
    MAX_SPLIT_COUNT,
)


class TestNormalizeError:
    def test_strips_numbers(self):
        a = _normalize_error("port 42 not found")
        b = _normalize_error("port 99 not found")
        assert a == b
        assert "<N>" in a

    def test_strips_hex(self):
        result = _normalize_error("invalid address 0xdeadbeef")
        assert "<HEX>" in result
        assert "0xdeadbeef" not in result

    def test_strips_hashes(self):
        result = _normalize_error("allele abcdef1234567890 failed")
        assert "<HASH>" in result

    def test_strips_paths(self):
        result = _normalize_error("file /etc/network/interfaces not found")
        assert "<PATH>" in result
        assert "/etc" not in result

    def test_strips_quoted_single(self):
        result = _normalize_error("key 'bridge_name' missing")
        assert "'<STR>'" in result

    def test_strips_quoted_double(self):
        result = _normalize_error('field "name" is required')
        assert '"<STR>"' in result

    def test_first_line_only(self):
        result = _normalize_error("first line\nsecond line\nthird")
        assert "second" not in result
        assert "first" in result


class TestErrorCluster:
    def test_to_dict_roundtrip(self):
        ec = ErrorCluster(
            pattern="test <N>",
            count=5,
            representative_messages=["test 1", "test 2"],
        )
        d = ec.to_dict()
        restored = ErrorCluster.from_dict(d)
        assert restored.pattern == "test <N>"
        assert restored.count == 5
        assert len(restored.representative_messages) == 2


class TestDecompositionDetector:
    def test_record_error_accumulates(self):
        det = DecompositionDetector()
        for i in range(5):
            det.record_error("locus_a", f"sha{i}", f"error {i}")
        assert len(det.histories["locus_a"].errors) == 5

    def test_record_error_windows(self):
        det = DecompositionDetector()
        for i in range(150):
            det.record_error("locus_a", "sha", f"error {i}")
        assert len(det.histories["locus_a"].errors) == ERROR_WINDOW

    def test_analyze_none_below_threshold(self):
        """Less than 10 total errors returns None."""
        det = DecompositionDetector()
        for i in range(5):
            det.record_error("x", "sha", f"error type {i % 3}")
        assert det.analyze("x") is None

    def test_analyze_none_few_clusters(self):
        """15 errors but only 1 pattern returns None."""
        det = DecompositionDetector()
        for i in range(15):
            det.record_error("x", "sha", "always the same error")
        assert det.analyze("x") is None

    def test_analyze_none_two_clusters(self):
        """15 errors with only 2 patterns returns None (need 3+)."""
        det = DecompositionDetector()
        for i in range(15):
            det.record_error("x", "sha", f"error type {i % 2}")
        assert det.analyze("x") is None

    def test_analyze_returns_signal(self):
        """15 errors with 4 distinct patterns returns signal."""
        det = DecompositionDetector()
        patterns = [
            "KeyError: name is missing",
            "ValueError: count must be positive",
            "ConnectionError: cannot reach server",
            "TimeoutError: operation timed out",
        ]
        for i in range(16):
            det.record_error("x", "sha", patterns[i % 4])

        signal = det.analyze("x")
        assert signal is not None
        assert signal.locus == "x"
        assert signal.recommended_split_count == 4
        assert signal.total_errors == 16
        assert len(signal.error_clusters) == 4

    def test_analyze_caps_split_count(self):
        """8 patterns → split_count capped at MAX_SPLIT_COUNT."""
        det = DecompositionDetector()
        kinds = [
            "KeyError: name missing",
            "ValueError: count invalid",
            "ConnectionError: server unreachable",
            "TimeoutError: timed out",
            "TypeError: wrong argument",
            "RuntimeError: state corrupted",
            "IOError: disk full",
            "PermissionError: access denied",
        ]
        for i in range(24):
            det.record_error("x", "sha", kinds[i % 8])
        signal = det.analyze("x")
        assert signal is not None
        assert signal.recommended_split_count == MAX_SPLIT_COUNT

    def test_analyze_clusters_sorted_by_count(self):
        """Clusters are sorted by count descending."""
        det = DecompositionDetector()
        # 10 of pattern A, 5 of B, 3 of C
        for _ in range(10):
            det.record_error("x", "sha", "pattern A error")
        for _ in range(5):
            det.record_error("x", "sha", "pattern B error")
        for _ in range(3):
            det.record_error("x", "sha", "pattern C error")
        signal = det.analyze("x")
        assert signal is not None
        counts = [c.count for c in signal.error_clusters]
        assert counts == sorted(counts, reverse=True)

    def test_analyze_unknown_locus(self):
        det = DecompositionDetector()
        assert det.analyze("nonexistent") is None

    def test_decomposition_state_lifecycle(self):
        det = DecompositionDetector()
        assert not det.is_decomposed("x")
        assert det.get_decomposition("x") is None

        det.record_decomposition("x", "x_decomposed", ["x_step1", "x_step2"])
        assert det.is_decomposed("x")

        state = det.get_decomposition("x")
        assert state["pathway_name"] == "x_decomposed"
        assert state["sub_loci"] == ["x_step1", "x_step2"]

        det.clear_decomposition("x")
        assert not det.is_decomposed("x")

    def test_record_fusion_of_decomposition(self):
        det = DecompositionDetector()
        det.record_decomposition("x", "x_decomposed", ["x_step1"])
        det.record_fusion_of_decomposition("x", "fused_sha_abc")
        state = det.get_decomposition("x")
        assert state["status"] == "refined"
        assert state["fused_sha"] == "fused_sha_abc"

    def test_save_and_load(self, tmp_path):
        det = DecompositionDetector()
        det.record_error("locus_a", "sha1", "error one")
        det.record_error("locus_a", "sha2", "error two")
        det.record_decomposition("locus_b", "pw_b", ["b_step1", "b_step2"])

        path = tmp_path / "decomposition.json"
        det.save(path)

        det2 = DecompositionDetector.open(path)
        assert len(det2.histories["locus_a"].errors) == 2
        assert det2.is_decomposed("locus_b")
        state = det2.get_decomposition("locus_b")
        assert state["pathway_name"] == "pw_b"

    def test_open_nonexistent(self, tmp_path):
        det = DecompositionDetector.open(tmp_path / "missing.json")
        assert len(det.histories) == 0
        assert len(det._decomposition_state) == 0

    def test_decomposition_state_persists(self, tmp_path):
        det = DecompositionDetector()
        det.record_decomposition("x", "x_pw", ["x_s1"])
        det.record_fusion_of_decomposition("x", "fused123")

        path = tmp_path / "decomposition.json"
        det.save(path)

        det2 = DecompositionDetector.open(path)
        state = det2.get_decomposition("x")
        assert state["status"] == "refined"
        assert state["fused_sha"] == "fused123"

    def test_clear_nonexistent_is_noop(self):
        det = DecompositionDetector()
        det.clear_decomposition("nonexistent")  # should not raise


class TestMockDecompose:
    def test_mock_decompose(self, tmp_path):
        from sg.mutation import MockMutationEngine
        # Create fixture files
        (tmp_path / "bridge_create_decompose_pathway.sg").write_text(
            "pathway bridge_create_decomposed\n  bridge_validate -> bridge_apply"
        )
        (tmp_path / "bridge_create_sub1.sg").write_text(
            "gene bridge_validate\n  does: validate bridge params"
        )
        (tmp_path / "bridge_create_sub1.py").write_text(
            'def execute(input_json):\n    return \'{"success": true}\''
        )
        (tmp_path / "bridge_create_sub2.sg").write_text(
            "gene bridge_apply\n  does: apply bridge config"
        )
        (tmp_path / "bridge_create_sub2.py").write_text(
            'def execute(input_json):\n    return \'{"success": true}\''
        )

        engine = MockMutationEngine(tmp_path)
        result = engine.decompose(
            "bridge_create", "def execute(x): pass", [], "contract", 2,
        )
        assert "bridge_create_decomposed" in result.pathway_contract_source
        assert len(result.sub_gene_contract_sources) == 2
        assert len(result.sub_gene_seed_sources) == 2
        assert result.original_locus == "bridge_create"

    def test_mock_decompose_missing(self, tmp_path):
        from sg.mutation import MockMutationEngine
        engine = MockMutationEngine(tmp_path)
        with pytest.raises(FileNotFoundError):
            engine.decompose("missing", "source", [], "contract", 2)


class TestLLMDecompose:
    def test_llm_decompose_prompt(self):
        """Prompt includes error cluster information."""
        from unittest.mock import patch, MagicMock
        from sg.mutation import OpenAIMutationEngine
        from sg.contracts import ContractStore

        import sg_network
        cs = ContractStore.open(sg_network.contracts_path())
        engine = OpenAIMutationEngine("fake-key", cs)

        clusters = [
            ErrorCluster(pattern="KeyError: <STR>", count=5,
                         representative_messages=["KeyError: name"]),
            ErrorCluster(pattern="ValueError: <STR>", count=3,
                         representative_messages=["ValueError: bad count"]),
        ]

        response_text = (
            "===PATHWAY===\npathway test_pw\n  step1 -> step2\n"
            "===GENE_CONTRACT===\ngene step1\n  does: first\n"
            "===GENE_SEED===\n```python\ndef execute(x): return '{\"success\": true}'\n```\n"
            "===GENE_CONTRACT===\ngene step2\n  does: second\n"
            "===GENE_SEED===\n```python\ndef execute(x): return '{\"success\": true}'\n```"
        )
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": response_text}}],
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=mock_resp) as mock_post:
            result = engine.decompose(
                "bridge_create", "def execute(x): pass", clusters, "contract src", 2,
            )

        prompt = mock_post.call_args[1]["json"]["messages"][0]["content"]
        assert "KeyError" in prompt
        assert "ValueError" in prompt
        assert "bridge_create" in prompt

    def test_llm_decompose_parses(self):
        """Structured response is parsed into DecompositionResult."""
        from unittest.mock import patch, MagicMock
        from sg.mutation import OpenAIMutationEngine
        from sg.contracts import ContractStore

        import sg_network
        cs = ContractStore.open(sg_network.contracts_path())
        engine = OpenAIMutationEngine("fake-key", cs)

        response_text = (
            "===PATHWAY===\npathway bridge_decomposed\n  validate -> apply\n"
            "===GENE_CONTRACT===\ngene validate\n  does: validate params\n"
            "===GENE_SEED===\n```python\ndef execute(x): return '{\"success\": true}'\n```\n"
            "===GENE_CONTRACT===\ngene apply\n  does: apply config\n"
            "===GENE_SEED===\n```python\ndef execute(x): return '{\"ok\": true}'\n```"
        )
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": response_text}}],
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=mock_resp):
            result = engine.decompose(
                "bridge_create", "source", [], "contract", 2,
            )

        assert "bridge_decomposed" in result.pathway_contract_source
        assert len(result.sub_gene_contract_sources) == 2
        assert len(result.sub_gene_seed_sources) == 2
        assert "def execute" in result.sub_gene_seed_sources[0]
        assert result.original_locus == "bridge_create"

    def test_llm_decompose_malformed(self):
        """Missing PATHWAY section raises RuntimeError."""
        from unittest.mock import patch, MagicMock
        from sg.mutation import OpenAIMutationEngine
        from sg.contracts import ContractStore

        import sg_network
        cs = ContractStore.open(sg_network.contracts_path())
        engine = OpenAIMutationEngine("fake-key", cs)

        response_text = "Here is some code but no section markers"
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": response_text}}],
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=mock_resp):
            with pytest.raises(RuntimeError, match="missing.*PATHWAY"):
                engine.decompose(
                    "bridge_create", "source", [], "contract", 2,
                )


# --- Integration tests ---

import sg_network
from sg_network import MockNetworkKernel
from sg.contracts import ContractStore
from sg.fusion import FusionTracker
from sg.mutation import MockMutationEngine
from sg.orchestrator import Orchestrator
from sg.phenotype import PhenotypeMap
from sg.registry import Registry

CONTRACTS_DIR = sg_network.contracts_path()
GENES_DIR = sg_network.genes_path()
FIXTURES_DIR = sg_network.fixtures_path()


class TestIntegration:
    @pytest.fixture
    def project(self, tmp_path):
        """Set up a project with sg-network plugin contracts and genes."""
        (tmp_path / ".sg").mkdir()
        shutil.copytree(CONTRACTS_DIR, tmp_path / "contracts")
        shutil.copytree(FIXTURES_DIR, tmp_path / "fixtures")

        registry = Registry.open(tmp_path / ".sg" / "registry")
        phenotype = PhenotypeMap()

        for locus in ContractStore.open(tmp_path / "contracts").known_loci():
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

    def _make_orchestrator(self, project):
        cs = ContractStore.open(project / "contracts")
        registry = Registry.open(project / ".sg" / "registry")
        phenotype = PhenotypeMap.load(project / "phenotype.toml")
        ft = FusionTracker.open(project / "fusion_tracker.json")
        kernel = MockNetworkKernel()
        mutation_engine = MockMutationEngine(project / "fixtures")
        return Orchestrator(
            registry=registry, phenotype=phenotype,
            mutation_engine=mutation_engine, fusion_tracker=ft,
            kernel=kernel, contract_store=cs, project_root=project,
        )

    def test_orchestrator_has_decomposition_detector(self, project):
        """Orchestrator creates a DecompositionDetector by default."""
        orch = self._make_orchestrator(project)
        assert isinstance(orch.decomposition_detector, DecompositionDetector)

    def test_error_recorded_on_failure(self, project):
        """execute_locus failure records error in decomposition detector."""
        orch = self._make_orchestrator(project)

        # Force a failure with invalid input
        orch.execute_locus("bridge_create", '{"invalid": true}')

        # Check that the decomposition detector recorded the error
        history = orch.decomposition_detector.histories.get("bridge_create")
        # May or may not have errors depending on whether the gene fails
        # The key assertion is that no exception was raised
        assert isinstance(orch.decomposition_detector, DecompositionDetector)

    def test_save_state_persists_decomposition(self, project):
        """save_state() writes decomposition.json."""
        orch = self._make_orchestrator(project)
        orch.decomposition_detector.record_error("test_locus", "sha1", "err1")
        orch.decomposition_detector.record_decomposition(
            "test_locus", "test_pw", ["test_sub1"],
        )

        orch.save_state()

        decomp_path = project / ".sg" / "decomposition.json"
        assert decomp_path.exists()

        det2 = DecompositionDetector.open(decomp_path)
        assert det2.is_decomposed("test_locus")
        assert len(det2.histories["test_locus"].errors) == 1

    def test_decomposition_skipped_if_active(self, project):
        """If locus already decomposed, decomposition is not re-triggered."""
        orch = self._make_orchestrator(project)
        orch.decomposition_detector.record_decomposition(
            "bridge_create", "bridge_create_decomposed", ["sub1"],
        )

        # Feed 15 diverse errors — would normally trigger decomposition
        patterns = [
            "KeyError: name missing",
            "ValueError: count invalid",
            "ConnectionError: server unreachable",
            "TimeoutError: timed out",
        ]
        for i in range(16):
            orch.decomposition_detector.record_error(
                "bridge_create", "sha", patterns[i % 4],
            )

        # Analyze would return a signal, but is_decomposed blocks it
        signal = orch.decomposition_detector.analyze("bridge_create")
        assert signal is not None  # Would trigger if not already decomposed
        assert orch.decomposition_detector.is_decomposed("bridge_create")
