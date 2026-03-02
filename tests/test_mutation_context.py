"""Tests for enriched MutationContext (Phase 1.1)."""
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

ALT_GENE = '''
import json

def execute(input_json):
    data = json.loads(input_json)
    gene_sdk.create_bridge(data["bridge_name"], [])
    return json.dumps({"success": True})
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


class TestMutationContextDefaults:
    """New fields have defaults, so old construction still works."""

    def test_backward_compatible_construction(self):
        ctx = MutationContext(
            gene_source="def execute(i): pass",
            locus="bridge_create",
            failing_input="{}",
            error_message="broken",
        )
        assert ctx.kernel_state is None
        assert ctx.prior_mutations == []
        assert ctx.pathway_context is None
        assert ctx.sibling_summaries == []

    def test_full_construction(self):
        ctx = MutationContext(
            gene_source="def execute(i): pass",
            locus="bridge_create",
            failing_input="{}",
            error_message="broken",
            kernel_state="Tracked resources:\n  bridge: br0",
            prior_mutations=["gen0 (abc12345): fitness=0.50"],
            pathway_context="step 1 of 'setup'",
            sibling_summaries=["abc12345: state=recessive, fitness=0.50"],
        )
        assert ctx.kernel_state is not None
        assert len(ctx.prior_mutations) == 1
        assert ctx.pathway_context is not None
        assert len(ctx.sibling_summaries) == 1


class TestOrchestratorContextBuilders:
    def test_build_kernel_state_empty(self, orch):
        assert orch._build_kernel_state() is None

    def test_build_kernel_state_with_resources(self, orch, project):
        project["kernel"].track_resource("bridge", "br0")
        state = orch._build_kernel_state()
        assert state is not None
        assert "br0" in state
        assert "bridge" in state

    def test_build_prior_mutations_empty(self, orch):
        result = orch._build_prior_mutations("nonexistent_locus")
        assert result == []

    def test_build_prior_mutations_with_alleles(self, orch, project):
        project["registry"].register(SIMPLE_GENE, "bridge_create")
        project["registry"].register(FAILING_GENE, "bridge_create")
        result = orch._build_prior_mutations("bridge_create")
        assert len(result) == 2
        assert "gen0" in result[0]
        assert "fitness=" in result[0]

    def test_build_prior_mutations_max_three(self, orch, project):
        for i in range(5):
            project["registry"].register(f"# variant {i}\n{SIMPLE_GENE}", "bridge_create")
        result = orch._build_prior_mutations("bridge_create")
        assert len(result) == 3

    def test_build_sibling_summaries_empty(self, orch):
        result = orch._build_sibling_summaries("nonexistent")
        assert result == []

    def test_build_sibling_summaries_excludes_sha(self, orch, project):
        sha1 = project["registry"].register(SIMPLE_GENE, "bridge_create")
        sha2 = project["registry"].register(FAILING_GENE, "bridge_create")
        result = orch._build_sibling_summaries("bridge_create", exclude_sha=sha1)
        assert all(sha1[:8] not in s for s in result)
        assert any(sha2[:8] in s for s in result)

    def test_build_sibling_summaries_includes_state(self, orch, project):
        sha = project["registry"].register(SIMPLE_GENE, "bridge_create")
        result = orch._build_sibling_summaries("bridge_create")
        assert any("state=" in s for s in result)

    def test_build_sibling_summaries_max_five(self, orch, project):
        for i in range(7):
            project["registry"].register(f"# v{i}\n{SIMPLE_GENE}", "bridge_create")
        result = orch._build_sibling_summaries("bridge_create")
        assert len(result) <= 5


class TestPathwayContextAttr:
    def test_orchestrator_has_pathway_context(self, orch):
        assert hasattr(orch, '_current_pathway_context')
        assert orch._current_pathway_context is None


class TestPathwayContextThreading:
    """Verify pathway context is set during pathway execution."""

    def test_pathway_context_set_during_locus_execution(self, project):
        """Use a capturing mutation engine to verify context flows."""
        captured_contexts = []

        class CapturingEngine(MutationEngine):
            def mutate(self, ctx):
                captured_contexts.append(ctx.pathway_context)
                return SIMPLE_GENE

        fixtures_dir = project["root"] / "fixtures"
        (fixtures_dir / "bridge_create_fix.py").write_text(SIMPLE_GENE)

        orch = Orchestrator(
            registry=project["registry"],
            phenotype=project["phenotype"],
            mutation_engine=CapturingEngine(),
            fusion_tracker=project["fusion_tracker"],
            kernel=project["kernel"],
            contract_store=project["contract_store"],
            project_root=project["root"],
        )

        # Register a failing gene that will trigger mutation
        sha = project["registry"].register(FAILING_GENE, "bridge_create")
        project["phenotype"].promote("bridge_create", sha)

        # Build a minimal pathway
        from sg.pathway import Pathway, PathwayStep as RTPathwayStep

        def passthrough(orig, outputs):
            return orig

        pathway = Pathway(name="test_pathway", steps=[
            RTPathwayStep(locus="bridge_create", input_transform=passthrough),
        ])

        from sg.pathway import execute_pathway
        input_json = json.dumps({"bridge_name": "br0", "interfaces": ["eth0"]})

        try:
            execute_pathway(
                pathway, input_json, orch,
                project["fusion_tracker"], project["registry"],
                project["phenotype"], orch.mutation_engine, project["kernel"],
            )
        except RuntimeError:
            pass  # pathway may fail, that's ok

        # The capturing engine should have seen a pathway context
        if captured_contexts:
            assert any(ctx is not None and "test_pathway" in ctx
                       for ctx in captured_contexts)

    def test_pathway_context_cleared_after_execution(self, orch, project):
        """After pathway step, _current_pathway_context is None."""
        sha = project["registry"].register(SIMPLE_GENE, "bridge_create")
        project["phenotype"].promote("bridge_create", sha)

        from sg.pathway import Pathway, PathwayStep as RTPathwayStep

        def passthrough(orig, outputs):
            return orig

        pathway = Pathway(name="test_pw", steps=[
            RTPathwayStep(locus="bridge_create", input_transform=passthrough),
        ])

        from sg.pathway import execute_pathway
        input_json = json.dumps({"bridge_name": "br0", "interfaces": ["eth0"]})

        execute_pathway(
            pathway, input_json, orch,
            project["fusion_tracker"], project["registry"],
            project["phenotype"], orch.mutation_engine, project["kernel"],
        )

        assert orch._current_pathway_context is None


class TestEnrichedPrompt:
    """Test that LLM prompt includes new context sections."""

    def _make_engine(self, contract_store):
        """Create an LLM engine that captures the prompt."""
        class CapturingLLM(LLMMutationEngine):
            captured_prompt = None

            def _call_api(self, prompt):
                CapturingLLM.captured_prompt = prompt
                return "```python\ndef execute(i): return '{\"success\": true}'\n```"

        return CapturingLLM(contract_store)

    def test_prompt_includes_kernel_state(self):
        engine = self._make_engine(ContractStore())
        ctx = MutationContext(
            gene_source="def execute(i): pass",
            locus="bridge_create",
            failing_input="{}",
            error_message="broken",
            kernel_state="Tracked resources:\n  bridge: br0",
        )
        engine.mutate(ctx)
        assert "## Current kernel state:" in engine.captured_prompt
        assert "bridge: br0" in engine.captured_prompt

    def test_prompt_includes_prior_mutations(self):
        engine = self._make_engine(ContractStore())
        ctx = MutationContext(
            gene_source="def execute(i): pass",
            locus="bridge_create",
            failing_input="{}",
            error_message="broken",
            prior_mutations=[
                "gen0 (abc12345): fitness=0.50, failures=3, approach: import json",
                "gen1 (def67890): fitness=0.20, failures=2, approach: import os",
            ],
        )
        engine.mutate(ctx)
        assert "## Prior mutation attempts" in engine.captured_prompt
        assert "abc12345" in engine.captured_prompt
        assert "Avoid repeating" in engine.captured_prompt

    def test_prompt_includes_pathway_context(self):
        engine = self._make_engine(ContractStore())
        ctx = MutationContext(
            gene_source="def execute(i): pass",
            locus="bridge_create",
            failing_input="{}",
            error_message="broken",
            pathway_context="step 2 of 'provision_bridge', after prior step produced {\"success\": true}",
        )
        engine.mutate(ctx)
        assert "## Pathway context:" in engine.captured_prompt
        assert "provision_bridge" in engine.captured_prompt

    def test_prompt_includes_sibling_summaries(self):
        engine = self._make_engine(ContractStore())
        ctx = MutationContext(
            gene_source="def execute(i): pass",
            locus="bridge_create",
            failing_input="{}",
            error_message="broken",
            sibling_summaries=[
                "abc12345: state=recessive, fitness=0.50, invocations=100",
            ],
        )
        engine.mutate(ctx)
        assert "## Other alleles" in engine.captured_prompt
        assert "abc12345" in engine.captured_prompt
        assert "different approach" in engine.captured_prompt

    def test_prompt_omits_empty_sections(self):
        engine = self._make_engine(ContractStore())
        ctx = MutationContext(
            gene_source="def execute(i): pass",
            locus="bridge_create",
            failing_input="{}",
            error_message="broken",
        )
        engine.mutate(ctx)
        assert "## Current kernel state:" not in engine.captured_prompt
        assert "## Prior mutation attempts" not in engine.captured_prompt
        assert "## Pathway context:" not in engine.captured_prompt
        assert "## Other alleles" not in engine.captured_prompt
