"""Tests for the orchestrator — core execution loop."""
import json
import pytest

from sg.fusion import FusionTracker
from sg.kernel.mock import MockNetworkKernel
from sg.mutation import MockMutationEngine, MutationEngine, MutationContext
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

    return {
        "root": tmp_path,
        "registry": registry,
        "phenotype": phenotype,
        "fusion_tracker": fusion_tracker,
        "mutation_engine": mutation_engine,
        "kernel": kernel,
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
        project_root=project["root"],
    )


def test_execute_locus_success(orch, project):
    sha = project["registry"].register(SIMPLE_GENE, "bridge_create")
    project["phenotype"].promote("bridge_create", sha)

    input_json = json.dumps({"bridge_name": "br0", "interfaces": ["eth0"]})
    result = orch.execute_locus("bridge_create", input_json)
    assert result is not None
    output, used_sha = result
    assert json.loads(output)["success"] is True
    assert used_sha == sha


def test_execute_locus_fallback(orch, project):
    fail_sha = project["registry"].register(FAILING_GENE, "bridge_create")
    project["phenotype"].promote("bridge_create", fail_sha)

    good_sha = project["registry"].register(SIMPLE_GENE, "bridge_create")
    project["phenotype"].add_to_fallback("bridge_create", good_sha)

    input_json = json.dumps({"bridge_name": "br0", "interfaces": ["eth0"]})
    result = orch.execute_locus("bridge_create", input_json)
    assert result is not None
    output, used_sha = result
    assert json.loads(output)["success"] is True
    assert used_sha == good_sha


def test_execute_locus_mutation_on_exhaustion(orch, project):
    fail_sha = project["registry"].register(FAILING_GENE, "bridge_create")
    project["phenotype"].promote("bridge_create", fail_sha)

    input_json = json.dumps({"bridge_name": "br0", "interfaces": ["eth0"]})
    result = orch.execute_locus("bridge_create", input_json)
    assert result is not None
    output, used_sha = result
    assert json.loads(output)["success"] is True
    assert used_sha != fail_sha


def test_execute_locus_records_fitness(orch, project):
    sha = project["registry"].register(SIMPLE_GENE, "bridge_create")
    project["phenotype"].promote("bridge_create", sha)

    input_json = json.dumps({"bridge_name": "br0", "interfaces": ["eth0"]})
    orch.execute_locus("bridge_create", input_json)

    allele = project["registry"].get(sha)
    assert allele.successful_invocations == 1
    assert allele.failed_invocations == 0


def test_empty_stack_returns_none(orch):
    class FailingMutationEngine(MutationEngine):
        def mutate(self, ctx: MutationContext) -> str:
            raise RuntimeError("no mutation available")
    orch.mutation_engine = FailingMutationEngine()

    result = orch.execute_locus("bridge_create", '{}')
    assert result is None
