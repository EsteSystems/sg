"""Tests for parallel diverse mutation (Phase 1.2)."""
import json
import pytest
from unittest.mock import patch, MagicMock

from sg.contracts import ContractStore
from sg.fusion import FusionTracker
from sg_network import MockNetworkKernel
from sg.mutation import (
    MutationEngine, MockMutationEngine, MutationContext, LLMMutationEngine,
)
from sg.orchestrator import Orchestrator
from sg.phenotype import PhenotypeMap
from sg.registry import Registry


SIMPLE_GENE = '''
import json

def execute(input_json):
    data = json.loads(input_json)
    gene_sdk.create_bridge(data["bridge_name"], data["interfaces"])
    return json.dumps({"success": True})
'''

FAILING_GENE = '''
import json

def execute(input_json):
    raise ValueError("always fails")
'''


@pytest.fixture
def project(tmp_path):
    fixtures_dir = tmp_path / "fixtures"
    fixtures_dir.mkdir()
    (fixtures_dir / "bridge_create_fix.py").write_text(SIMPLE_GENE)

    registry = Registry.open(tmp_path / ".sg" / "registry")
    phenotype = PhenotypeMap()
    fusion_tracker = FusionTracker()
    mutation_engine = MockMutationEngine(fixtures_dir)
    kernel = MockNetworkKernel()
    contract_store = ContractStore()

    return {
        "root": tmp_path,
        "registry": registry,
        "phenotype": phenotype,
        "fusion_tracker": fusion_tracker,
        "mutation_engine": mutation_engine,
        "kernel": kernel,
        "contract_store": contract_store,
    }


@pytest.fixture
def orch(project):
    project["kernel"].reset()
    return Orchestrator(
        registry=project["registry"],
        phenotype=project["phenotype"],
        mutation_engine=project["mutation_engine"],
        fusion_tracker=project["fusion_tracker"],
        kernel=project["kernel"],
        contract_store=project["contract_store"],
        project_root=project["root"],
    )


class TestMutateBatchDefault:
    """Default mutate_batch calls mutate() N times."""

    def test_default_batch_calls_mutate_n_times(self):
        call_count = 0

        class CountingEngine(MutationEngine):
            def mutate(self, ctx):
                nonlocal call_count
                call_count += 1
                return f"# variant {call_count}\ndef execute(i): pass"

        engine = CountingEngine()
        results = engine.mutate_batch(MutationContext(
            gene_source="", locus="test", failing_input="{}", error_message="err",
        ), count=3)
        assert call_count == 3
        assert len(results) == 3

    def test_default_batch_tolerates_failures(self):
        call_count = 0

        class FlakeyEngine(MutationEngine):
            def mutate(self, ctx):
                nonlocal call_count
                call_count += 1
                if call_count % 2 == 0:
                    raise RuntimeError("intermittent failure")
                return f"# variant {call_count}\ndef execute(i): pass"

        engine = FlakeyEngine()
        results = engine.mutate_batch(MutationContext(
            gene_source="", locus="test", failing_input="{}", error_message="err",
        ), count=4)
        # 4 calls, 2 fail, 2 succeed
        assert len(results) == 2

    def test_default_batch_all_fail_returns_empty(self):
        class BrokenEngine(MutationEngine):
            def mutate(self, ctx):
                raise RuntimeError("always fails")

        engine = BrokenEngine()
        results = engine.mutate_batch(MutationContext(
            gene_source="", locus="test", failing_input="{}", error_message="err",
        ), count=3)
        assert results == []


class TestMockMutateBatch:
    def test_mock_returns_single_fixture(self, project):
        engine = project["mutation_engine"]
        ctx = MutationContext(
            gene_source="", locus="bridge_create",
            failing_input="{}", error_message="err",
        )
        results = engine.mutate_batch(ctx, count=3)
        assert len(results) == 1
        assert "execute" in results[0]


class TestLLMMutateBatch:
    def _make_engine(self, contract_store, response_text):
        class MockLLM(LLMMutationEngine):
            captured_prompt = None

            def _call_api(self, prompt):
                MockLLM.captured_prompt = prompt
                return response_text

        return MockLLM(contract_store)

    def test_batch_prompt_requests_count_variants(self):
        engine = self._make_engine(
            ContractStore(),
            "```python\ndef execute(i): pass\n```"
        )
        ctx = MutationContext(
            gene_source="def execute(i): pass",
            locus="bridge_create",
            failing_input="{}",
            error_message="broken",
        )
        engine.mutate_batch(ctx, count=3)
        assert "3 DIFFERENT" in engine.captured_prompt
        assert "---VARIANT---" in engine.captured_prompt

    def test_batch_parses_variant_separator(self):
        response = (
            "```python\ndef execute(i): return 'v1'\n```\n"
            "---VARIANT---\n"
            "```python\ndef execute(i): return 'v2'\n```\n"
            "---VARIANT---\n"
            "```python\ndef execute(i): return 'v3'\n```"
        )
        engine = self._make_engine(ContractStore(), response)
        ctx = MutationContext(
            gene_source="", locus="test",
            failing_input="{}", error_message="err",
        )
        results = engine.mutate_batch(ctx, count=3)
        assert len(results) == 3
        assert "v1" in results[0]
        assert "v2" in results[1]
        assert "v3" in results[2]

    def test_batch_single_result_fallback(self):
        """LLM returns only 1 variant despite asking for 3."""
        engine = self._make_engine(
            ContractStore(),
            "```python\ndef execute(i): return 'only_one'\n```"
        )
        ctx = MutationContext(
            gene_source="", locus="test",
            failing_input="{}", error_message="err",
        )
        results = engine.mutate_batch(ctx, count=3)
        assert len(results) == 1
        assert "only_one" in results[0]

    def test_batch_includes_enriched_context(self):
        engine = self._make_engine(
            ContractStore(),
            "```python\ndef execute(i): pass\n```"
        )
        ctx = MutationContext(
            gene_source="def execute(i): pass",
            locus="bridge_create",
            failing_input="{}",
            error_message="broken",
            kernel_state="Tracked resources:\n  bridge: br0",
            prior_mutations=["gen0 (abc): fitness=0.50"],
        )
        engine.mutate_batch(ctx, count=2)
        assert "## Current kernel state:" in engine.captured_prompt
        assert "## Prior mutation attempts" in engine.captured_prompt


class TestOrchestratorBatchMutation:
    def test_batch_returns_first_passing(self, orch, project):
        """Mutation via batch returns a passing result."""
        sha = project["registry"].register(FAILING_GENE, "bridge_create")
        project["phenotype"].promote("bridge_create", sha)

        input_json = json.dumps({"bridge_name": "br0", "interfaces": ["eth0"]})
        result = orch.execute_locus("bridge_create", input_json)
        # MockMutationEngine.mutate_batch returns [fixture], which should pass
        assert result is not None
        output, used_sha = result
        assert json.loads(output)["success"] is True

    def test_batch_fallback_to_sequential(self, project):
        """If mutate_batch raises, falls back to sequential mutate()."""
        call_log = []

        class BatchFailEngine(MutationEngine):
            def mutate(self, ctx):
                call_log.append("mutate")
                return SIMPLE_GENE

            def mutate_batch(self, ctx, count=3):
                call_log.append("batch_fail")
                raise RuntimeError("batch unavailable")

        orch = Orchestrator(
            registry=project["registry"],
            phenotype=project["phenotype"],
            mutation_engine=BatchFailEngine(),
            fusion_tracker=project["fusion_tracker"],
            kernel=project["kernel"],
            contract_store=project["contract_store"],
            project_root=project["root"],
        )

        sha = project["registry"].register(FAILING_GENE, "bridge_create")
        project["phenotype"].promote("bridge_create", sha)

        input_json = json.dumps({"bridge_name": "br0", "interfaces": ["eth0"]})
        result = orch.execute_locus("bridge_create", input_json)
        assert result is not None
        assert "batch_fail" in call_log
        assert "mutate" in call_log

    def test_batch_registers_all_passing(self, project):
        """All passing candidates are registered in the registry."""

        class MultiEngine(MutationEngine):
            def mutate(self, ctx):
                raise RuntimeError("should not be called")

            def mutate_batch(self, ctx, count=3):
                return [SIMPLE_GENE, SIMPLE_GENE]

        orch = Orchestrator(
            registry=project["registry"],
            phenotype=project["phenotype"],
            mutation_engine=MultiEngine(),
            fusion_tracker=project["fusion_tracker"],
            kernel=project["kernel"],
            contract_store=project["contract_store"],
            project_root=project["root"],
        )

        sha = project["registry"].register(FAILING_GENE, "bridge_create")
        project["phenotype"].promote("bridge_create", sha)

        input_json = json.dumps({"bridge_name": "br0", "interfaces": ["eth0"]})
        result = orch.execute_locus("bridge_create", input_json)
        assert result is not None

        # Both identical candidates register as same SHA, but the mutation path works
        alleles = project["registry"].alleles_for_locus("bridge_create")
        # At least the failing gene + at least one mutation
        assert len(alleles) >= 2
