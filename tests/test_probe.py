"""Tests for input-space exploration (Phase 1.4)."""
import json
import pytest

from sg.contracts import ContractStore
from sg.fusion import FusionTracker
from sg_network import MockNetworkKernel
from sg.mutation import MockMutationEngine
from sg.orchestrator import Orchestrator
from sg.phenotype import PhenotypeMap
from sg.probe import generate_probes, probe_locus, ProbeReport, ProbeResult
from sg.parser.types import GeneContract, GeneFamily, BlastRadius, FieldDef
from sg.registry import Registry


ROBUST_GENE = '''
import json

def execute(input_json):
    data = json.loads(input_json)
    return json.dumps({"success": True})
'''

FRAGILE_GENE = '''
import json

def execute(input_json):
    data = json.loads(input_json)
    # Fails if name is empty or missing
    name = data["name"]
    if not name:
        raise ValueError("name cannot be empty")
    return json.dumps({"success": True})
'''


def _write_sg(path, content):
    path.write_text(content)
    return path


@pytest.fixture
def contract_store_with_gene(tmp_path):
    genes_dir = tmp_path / "contracts" / "genes"
    genes_dir.mkdir(parents=True)

    _write_sg(genes_dir / "test_action.sg", """\
gene test_action
  is configuration
  risk low

  does:
    A test action gene.

  takes:
    name: string
    count: int
    enabled: bool
    tags: string[]
    hint: string (optional)

  gives:
    success: bool
""")
    store = ContractStore()
    store.load_directory(tmp_path / "contracts")
    return store


@pytest.fixture
def project(tmp_path, contract_store_with_gene):
    fixtures_dir = tmp_path / "fixtures"
    fixtures_dir.mkdir()
    (fixtures_dir / "test_action_fix.py").write_text(ROBUST_GENE)

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
        "contract_store": contract_store_with_gene,
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


class TestGenerateProbes:
    def _make_contract(self, takes):
        return GeneContract(
            name="test",
            family=GeneFamily.CONFIGURATION,
            risk=BlastRadius.NONE,
            does="test",
            takes=takes,
            gives=[FieldDef(name="success", type="bool")],
        )

    def test_generates_at_least_one(self):
        contract = self._make_contract([
            FieldDef(name="name", type="string"),
        ])
        probes = generate_probes(contract, count=5)
        assert len(probes) >= 1
        assert all(isinstance(p, str) for p in probes)

    def test_includes_empty_string_probe(self):
        contract = self._make_contract([
            FieldDef(name="name", type="string"),
        ])
        probes = generate_probes(contract, count=20)
        assert any(json.loads(p).get("name") == "" for p in probes)

    def test_includes_zero_int_probe(self):
        contract = self._make_contract([
            FieldDef(name="count", type="int"),
        ])
        probes = generate_probes(contract, count=20)
        assert any(json.loads(p).get("count") == 0 for p in probes)

    def test_includes_negative_int_probe(self):
        contract = self._make_contract([
            FieldDef(name="count", type="int"),
        ])
        probes = generate_probes(contract, count=20)
        assert any(json.loads(p).get("count") == -1 for p in probes)

    def test_includes_empty_array_probe(self):
        contract = self._make_contract([
            FieldDef(name="items", type="string[]"),
        ])
        probes = generate_probes(contract, count=20)
        assert any(json.loads(p).get("items") == [] for p in probes)

    def test_includes_empty_object_probe(self):
        contract = self._make_contract([
            FieldDef(name="name", type="string"),
        ])
        probes = generate_probes(contract, count=20)
        assert "{}" in probes

    def test_includes_boolean_inversion(self):
        contract = self._make_contract([
            FieldDef(name="enabled", type="bool"),
        ])
        probes = generate_probes(contract, count=20)
        assert any(json.loads(p).get("enabled") is False for p in probes)

    def test_includes_extra_field(self):
        contract = self._make_contract([
            FieldDef(name="name", type="string"),
        ])
        probes = generate_probes(contract, count=20)
        assert any("_unknown_extra_field" in json.loads(p) for p in probes)

    def test_respects_count_limit(self):
        contract = self._make_contract([
            FieldDef(name="a", type="string"),
            FieldDef(name="b", type="int"),
            FieldDef(name="c", type="bool"),
            FieldDef(name="d", type="string[]"),
        ])
        probes = generate_probes(contract, count=3)
        assert len(probes) <= 3

    def test_no_duplicates(self):
        contract = self._make_contract([
            FieldDef(name="x", type="string"),
        ])
        probes = generate_probes(contract, count=20)
        assert len(probes) == len(set(probes))

    def test_all_valid_json(self):
        contract = self._make_contract([
            FieldDef(name="name", type="string"),
            FieldDef(name="count", type="int"),
        ])
        probes = generate_probes(contract, count=20)
        for p in probes:
            json.loads(p)  # should not raise


class TestProbeLocus:
    def test_probe_passing_locus(self, orch, project):
        """A robust gene passes all probes."""
        sha = project["registry"].register(ROBUST_GENE, "test_action")
        project["phenotype"].promote("test_action", sha)

        report = probe_locus("test_action", orch, count=5)
        assert report.total >= 1
        assert report.passed == report.total
        assert report.failed == 0

    def test_probe_failing_locus(self, orch, project):
        """A fragile gene fails on some probes."""
        sha = project["registry"].register(FRAGILE_GENE, "test_action")
        project["phenotype"].promote("test_action", sha)

        report = probe_locus("test_action", orch, count=10)
        assert report.total >= 1
        assert report.failed > 0
        # Should have some failures (empty string, empty object, etc.)
        failure_errors = [r.error for r in report.results if not r.success]
        assert len(failure_errors) > 0

    def test_probe_unknown_locus_raises(self, orch):
        with pytest.raises(ValueError, match="no contract"):
            probe_locus("nonexistent", orch)

    def test_probe_no_dominant_raises(self, orch):
        """Locus has contract but no dominant allele."""
        with pytest.raises(ValueError, match="no dominant"):
            probe_locus("test_action", orch)

    def test_probe_report_structure(self, orch, project):
        """Verify ProbeReport fields are populated correctly."""
        sha = project["registry"].register(ROBUST_GENE, "test_action")
        project["phenotype"].promote("test_action", sha)

        report = probe_locus("test_action", orch, count=5)
        assert report.locus == "test_action"
        assert report.sha == sha[:12]
        assert report.total == len(report.results)
        assert report.passed + report.failed == report.total


class TestProbeReport:
    def test_failure_rate_zero(self):
        report = ProbeReport(locus="x", sha="abc", total=5, passed=5, failed=0)
        assert report.failure_rate == 0.0

    def test_failure_rate_half(self):
        report = ProbeReport(locus="x", sha="abc", total=10, passed=5, failed=5)
        assert report.failure_rate == 0.5

    def test_failure_rate_empty(self):
        report = ProbeReport(locus="x", sha="abc")
        assert report.failure_rate == 0.0
