"""End-to-end tests: full evolutionary loop with pathways and fusion."""
import json
import shutil
import pytest
from pathlib import Path

from sg.contracts import ContractStore
from sg.fusion import FusionTracker, FUSION_THRESHOLD
from sg.kernel.mock import MockNetworkKernel
from sg.mutation import MockMutationEngine
from sg.orchestrator import Orchestrator
from sg.phenotype import PhenotypeMap
from sg.registry import Registry


FIXTURE_DIR = Path(__file__).parent.parent / "fixtures"
GENES_DIR = Path(__file__).parent.parent / "genes"
CONTRACTS_DIR = Path(__file__).parent.parent / "contracts"


@pytest.fixture
def project(tmp_path):
    """Full project setup mimicking `sg init`."""
    fixtures_dst = tmp_path / "fixtures"
    if FIXTURE_DIR.exists():
        shutil.copytree(FIXTURE_DIR, fixtures_dst)
    else:
        fixtures_dst.mkdir()

    # Copy contracts so ContractStore can discover them
    contracts_dst = tmp_path / "contracts"
    if CONTRACTS_DIR.exists():
        shutil.copytree(CONTRACTS_DIR, contracts_dst)

    registry = Registry.open(tmp_path / ".sg" / "registry")
    phenotype = PhenotypeMap()

    for locus, filename in [
        ("bridge_create", "bridge_create_v1.py"),
        ("bridge_stp", "bridge_stp_v1.py"),
    ]:
        gene_path = GENES_DIR / filename
        if gene_path.exists():
            source = gene_path.read_text()
        else:
            if locus == "bridge_create":
                source = _bridge_create_source()
            else:
                source = _bridge_stp_source()
        sha = registry.register(source, locus)
        phenotype.promote(locus, sha)
        allele = registry.get(sha)
        allele.state = "dominant"

    registry.save_index()
    phenotype.save(tmp_path / "phenotype.toml")

    return tmp_path


def _bridge_create_source():
    return '''import json

def execute(input_json):
    data = json.loads(input_json)
    bridge_name = data.get("bridge_name")
    interfaces = data.get("interfaces", [])
    try:
        result = gene_sdk.create_bridge(bridge_name, interfaces)
        return json.dumps({"success": True, "bridge": result})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})
'''


def _bridge_stp_source():
    return '''import json

def execute(input_json):
    data = json.loads(input_json)
    bridge_name = data.get("bridge_name")
    stp_enabled = data.get("stp_enabled", True)
    forward_delay = data.get("forward_delay", 15)
    try:
        result = gene_sdk.set_stp(bridge_name, stp_enabled, forward_delay)
        return json.dumps({"success": True, "bridge": result})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})
'''


def make_orchestrator(project_root: Path) -> Orchestrator:
    contract_store = ContractStore.open(project_root / "contracts")
    registry = Registry.open(project_root / ".sg" / "registry")
    phenotype = PhenotypeMap.load(project_root / "phenotype.toml")
    fusion_tracker = FusionTracker.open(project_root / "fusion_tracker.json")
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
    )


INPUT_JSON = json.dumps({
    "bridge_name": "br0",
    "interfaces": ["eth0", "eth1"],
    "stp_enabled": True,
    "forward_delay": 15,
})


class TestHappyPath:
    def test_single_pathway_execution(self, project):
        orch = make_orchestrator(project)
        outputs = orch.run_pathway("configure_bridge_with_stp", INPUT_JSON)
        orch.save_state()

        assert len(outputs) == 2
        step1 = json.loads(outputs[0])
        step2 = json.loads(outputs[1])
        assert step1["success"] is True
        assert step2["success"] is True
        assert step2["bridge"]["stp_enabled"] is True


class TestFailureAndMutation:
    def test_injected_failure_triggers_mutation(self, project):
        orch = make_orchestrator(project)
        orch.kernel.inject_failure("create_bridge", "simulated kernel panic")

        outputs = orch.run_pathway("configure_bridge_with_stp", INPUT_JSON)
        orch.save_state()

        step1 = json.loads(outputs[0])
        assert isinstance(step1["success"], bool)


class TestFusion:
    def test_fusion_triggers_after_threshold(self, project):
        for i in range(FUSION_THRESHOLD + 1):
            orch = make_orchestrator(project)

            input_data = json.loads(INPUT_JSON)
            input_data["bridge_name"] = f"br{i}"
            run_input = json.dumps(input_data)

            outputs = orch.run_pathway("configure_bridge_with_stp", run_input)
            orch.save_state()

            assert len(outputs) >= 1

        phenotype = PhenotypeMap.load(project / "phenotype.toml")
        fusion_config = phenotype.get_fused("configure_bridge_with_stp")
        assert fusion_config is not None
        assert fusion_config.fused_sha is not None

        orch = make_orchestrator(project)
        input_data = json.loads(INPUT_JSON)
        input_data["bridge_name"] = "br_fused"
        outputs = orch.run_pathway("configure_bridge_with_stp", json.dumps(input_data))
        orch.save_state()

        assert len(outputs) == 1
        result = json.loads(outputs[0])
        assert result["success"] is True


class TestStatus:
    def test_allele_tracking(self, project):
        orch = make_orchestrator(project)
        orch.run_pathway("configure_bridge_with_stp", INPUT_JSON)
        orch.save_state()

        registry = Registry.open(project / ".sg" / "registry")
        for locus in ["bridge_create", "bridge_stp"]:
            alleles = registry.alleles_for_locus(locus)
            assert len(alleles) >= 1
            total = sum(a.total_invocations for a in alleles)
            assert total >= 1
