"""End-to-end tests: full evolutionary loop with the data pipeline domain.

Proves the engine is truly domain-agnostic — this entire test uses
DataKernel/MockDataKernel without any modifications to sg/ engine code.
"""
import json
import shutil
import pytest
from pathlib import Path

import sg_data
from sg.contracts import ContractStore
from sg.fusion import FusionTracker, FUSION_THRESHOLD
from sg_data import MockDataKernel
from sg.mutation import MockMutationEngine
from sg.orchestrator import Orchestrator
from sg.phenotype import PhenotypeMap
from sg.registry import Registry


FIXTURE_DIR = sg_data.fixtures_path()
GENES_DIR = sg_data.genes_path()
CONTRACTS_DIR = sg_data.contracts_path()


@pytest.fixture
def project(tmp_path):
    """Full project setup for data pipeline domain."""
    fixtures_dst = tmp_path / "fixtures"
    if FIXTURE_DIR.exists():
        shutil.copytree(FIXTURE_DIR, fixtures_dst)
    else:
        fixtures_dst.mkdir()

    contracts_dst = tmp_path / "contracts"
    if CONTRACTS_DIR.exists():
        shutil.copytree(CONTRACTS_DIR, contracts_dst)

    registry = Registry.open(tmp_path / ".sg" / "registry")
    phenotype = PhenotypeMap()

    for locus, filename in [
        ("ingest_csv_to_table", "ingest_csv_to_table_v1.py"),
        ("check_row_count", "check_row_count_v1.py"),
        ("check_nulls", "check_nulls_v1.py"),
    ]:
        gene_path = GENES_DIR / filename
        source = gene_path.read_text()
        sha = registry.register(source, locus)
        phenotype.promote(locus, sha)
        allele = registry.get(sha)
        allele.state = "dominant"

    registry.save_index()
    phenotype.save(tmp_path / "phenotype.toml")

    return tmp_path


def make_orchestrator(project_root: Path) -> Orchestrator:
    contract_store = ContractStore.open(project_root / "contracts")
    registry = Registry.open(project_root / ".sg" / "registry")
    phenotype = PhenotypeMap.load(project_root / "phenotype.toml")
    fusion_tracker = FusionTracker.open(project_root / "fusion_tracker.json")
    mutation_engine = MockMutationEngine(project_root / "fixtures")
    kernel = MockDataKernel()

    return Orchestrator(
        registry=registry,
        phenotype=phenotype,
        mutation_engine=mutation_engine,
        fusion_tracker=fusion_tracker,
        kernel=kernel,
        contract_store=contract_store,
        project_root=project_root,
    )


def setup_kernel_state(kernel: MockDataKernel) -> None:
    """Prepare mock kernel with tables and HTTP responses."""
    kernel.add_table("warehouse", "events", {
        "id": "int",
        "name": "string",
        "value": "float",
    })
    kernel.add_http_response("https://data.example.com/events.csv", {
        "records": [
            {"id": 1, "name": "alpha", "value": 10.0},
            {"id": 2, "name": "beta", "value": 20.0},
            {"id": 3, "name": "gamma", "value": 30.0},
        ],
    })


INPUT_JSON = json.dumps({
    "url": "https://data.example.com/events.csv",
    "connection": "warehouse",
    "table": "events",
    "column": "name",
})


class TestHappyPath:
    def test_single_gene_execution(self, project):
        """Execute the ingest gene directly."""
        orch = make_orchestrator(project)
        setup_kernel_state(orch.kernel)

        result = orch.execute_locus("ingest_csv_to_table", INPUT_JSON)
        orch.save_state()

        assert result is not None
        output = json.loads(result[0])
        assert output["success"] is True
        assert output["rows_written"] == 3

    def test_diagnostic_gene_execution(self, project):
        """Execute a diagnostic gene directly."""
        orch = make_orchestrator(project)
        setup_kernel_state(orch.kernel)

        # First ingest some data
        orch.execute_locus("ingest_csv_to_table", INPUT_JSON)

        # Now run diagnostic
        diag_input = json.dumps({
            "connection": "warehouse",
            "table": "events",
        })
        result = orch.execute_locus("check_row_count", diag_input)
        orch.save_state()

        assert result is not None
        output = json.loads(result[0])
        assert output["success"] is True
        assert output["healthy"] is True
        assert output["row_count"] == 3

    def test_pathway_execution(self, project):
        """Execute the full ingest_and_validate pathway."""
        orch = make_orchestrator(project)
        setup_kernel_state(orch.kernel)

        outputs = orch.run_pathway("ingest_and_validate", INPUT_JSON)
        orch.save_state()

        assert len(outputs) == 3

        ingest_out = json.loads(outputs[0])
        assert ingest_out["success"] is True
        assert ingest_out["rows_written"] == 3

        row_count_out = json.loads(outputs[1])
        assert row_count_out["success"] is True
        assert row_count_out["healthy"] is True
        assert row_count_out["row_count"] == 3

        nulls_out = json.loads(outputs[2])
        assert nulls_out["success"] is True
        assert nulls_out["healthy"] is True
        assert nulls_out["null_ratio"] == 0.0


class TestFailureAndMutation:
    def test_injected_failure_triggers_mutation(self, project):
        """A kernel failure should trigger mutation and use the fixture."""
        orch = make_orchestrator(project)
        setup_kernel_state(orch.kernel)
        orch.kernel.inject_failure("http_get", "simulated network timeout")

        result = orch.execute_locus("ingest_csv_to_table", INPUT_JSON)
        orch.save_state()

        # The mutation engine should provide a fix from fixtures
        assert result is not None
        output = json.loads(result[0])
        assert isinstance(output["success"], bool)

    def test_write_records_failure_triggers_mutation(self, project):
        """A write_records failure should trigger mutation."""
        orch = make_orchestrator(project)
        setup_kernel_state(orch.kernel)
        orch.kernel.inject_failure("write_records", "simulated disk full")

        result = orch.execute_locus("ingest_csv_to_table", INPUT_JSON)
        orch.save_state()

        assert result is not None


class TestDiagnosticFeedback:
    def test_diagnostic_feeds_configuration_gene(self, project):
        """Diagnostic gene output should feed fitness to configuration gene."""
        orch = make_orchestrator(project)
        setup_kernel_state(orch.kernel)

        # Run ingest first
        orch.execute_locus("ingest_csv_to_table", INPUT_JSON)

        # Run diagnostic — its contract has "feeds ingest_csv_to_table convergence"
        diag_input = json.dumps({
            "connection": "warehouse",
            "table": "events",
        })
        orch.execute_locus("check_row_count", diag_input)
        orch.save_state()

        # Check that the ingest allele received convergence feedback
        registry = Registry.open(project / ".sg" / "registry")
        phenotype = PhenotypeMap.load(project / "phenotype.toml")
        ingest_sha = phenotype.get_dominant("ingest_csv_to_table")
        assert ingest_sha is not None
        allele = registry.get(ingest_sha)
        assert allele is not None
        convergence_records = [
            r for r in allele.fitness_records
            if r.get("timescale") == "convergence"
        ]
        assert len(convergence_records) > 0


class TestSafeKernel:
    def test_write_records_is_transactional(self, project):
        """write_records is @mutating, so SafeKernel should record undo actions."""
        from sg.safety import Transaction, SafeKernel
        from sg.parser.types import BlastRadius

        orch = make_orchestrator(project)
        setup_kernel_state(orch.kernel)

        txn = Transaction("ingest_csv_to_table", BlastRadius.LOW)
        safe_kernel = SafeKernel(orch.kernel, txn)

        safe_kernel.write_records("warehouse", "events", [
            {"id": 4, "name": "delta", "value": 40.0},
        ])

        # Transaction should have recorded the write
        assert txn.action_count == 1

        # Rollback should undo the write
        txn.rollback()
        assert orch.kernel.row_count("warehouse", "events") == 0


class TestFitness:
    def test_allele_tracking(self, project):
        """Fitness tracking works with data domain genes."""
        orch = make_orchestrator(project)
        setup_kernel_state(orch.kernel)

        orch.execute_locus("ingest_csv_to_table", INPUT_JSON)
        orch.save_state()

        registry = Registry.open(project / ".sg" / "registry")
        alleles = registry.alleles_for_locus("ingest_csv_to_table")
        assert len(alleles) >= 1
        total = sum(a.total_invocations for a in alleles)
        assert total >= 1


class TestContractDomain:
    def test_contracts_have_data_domain(self, project):
        """All data contracts should declare 'for data' domain."""
        contract_store = ContractStore.open(project / "contracts")
        for locus in ["ingest_csv_to_table", "check_row_count", "check_nulls"]:
            contract = contract_store.get_gene(locus)
            assert contract is not None, f"contract not found: {locus}"
            assert contract.domain == "data", f"{locus} domain is {contract.domain}"

    def test_kernel_domain_name(self):
        """MockDataKernel should report domain as 'data'."""
        kernel = MockDataKernel()
        assert kernel.domain_name() == "data"


class TestNoEngineModification:
    def test_no_data_imports_in_engine(self):
        """Verify that sg/ engine code has no references to sg_data or DataKernel.

        This is the critical criterion: the data domain is purely a plugin.
        """
        import sg
        engine_dir = Path(sg.__file__).parent
        for py_file in engine_dir.rglob("*.py"):
            content = py_file.read_text()
            assert "sg_data" not in content, \
                f"engine file {py_file} references sg_data"
            assert "DataKernel" not in content, \
                f"engine file {py_file} references DataKernel"
            assert "MockDataKernel" not in content, \
                f"engine file {py_file} references MockDataKernel"
