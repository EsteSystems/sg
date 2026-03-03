"""End-to-end tests: full evolutionary loop with the data pipeline domain.

Proves the engine is truly domain-agnostic — this entire test uses
DataKernel/MockDataKernel without any modifications to sg/ engine code.
"""
import json
import shutil
import pytest
from pathlib import Path

sg_data = pytest.importorskip("sg_data", reason="sg_data plugin not installed")

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


ALL_SEED_GENES = [
    ("ingest_csv_to_table", "ingest_csv_to_table_v1.py"),
    ("check_row_count", "check_row_count_v1.py"),
    ("check_nulls", "check_nulls_v1.py"),
    ("validate_schema", "validate_schema_v1.py"),
    ("clean_records", "clean_records_v1.py"),
    ("transform_records", "transform_records_v1.py"),
]


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

    for locus, filename in ALL_SEED_GENES:
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
        all_loci = [
            "ingest_csv_to_table", "check_row_count", "check_nulls",
            "validate_schema", "clean_records", "transform_records",
        ]
        for locus in all_loci:
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


# --- Phase 1: New gene tests ---


def setup_multi_source_kernel(kernel):
    """Set up mock data representing 3 sources with known quality issues."""
    # POS data — stable, clean
    kernel.add_table("warehouse", "pos_sales", {
        "id": "int", "product": "string", "amount": "float", "date": "string",
    }, rows=[
        {"id": 1, "product": "Widget", "amount": 10.0, "date": "2024-01-01"},
        {"id": 2, "product": "Gadget", "amount": 25.0, "date": "2024-01-01"},
    ])
    kernel.add_http_response("mock://pos/daily_sales.csv", {
        "records": [
            {"id": 3, "product": "Widget", "amount": 15.0, "date": "2024-01-02"},
        ],
    })

    # E-commerce data — schema drift (extra column "source_id")
    kernel.add_table("warehouse", "ecom_sales", {
        "id": "int", "product": "string", "amount": "float",
    })
    kernel.add_http_response("mock://ecom/orders.csv", {
        "records": [
            {"id": 1, "product": "Widget", "amount": 10.0, "source_id": "ecom"},
        ],
    })

    # Partner feed — nulls and duplicates
    kernel.add_table("warehouse", "partner_sales", {
        "id": "int", "product": "string", "amount": "float",
    }, rows=[
        {"id": 1, "product": None, "amount": 10.0},
        {"id": 1, "product": None, "amount": 10.0},  # duplicate
        {"id": 2, "product": "Gadget", "amount": None},
    ])

    # Target warehouse table
    kernel.add_table("warehouse", "daily_sales", {
        "id": "int", "product": "string", "amount": "float", "date": "string",
    })


class TestValidateSchema:
    def test_schema_matches(self, project):
        """validate_schema reports healthy when columns match."""
        orch = make_orchestrator(project)
        setup_kernel_state(orch.kernel)

        result = orch.execute_locus("validate_schema", json.dumps({
            "connection": "warehouse",
            "table": "events",
            "expected_columns": ["id", "name", "value"],
        }))
        assert result is not None
        output = json.loads(result[0])
        assert output["success"] is True
        assert output["healthy"] is True
        assert output["missing_columns"] == []
        assert output["extra_columns"] == []

    def test_schema_drift_extra_columns(self, project):
        """validate_schema detects extra columns not in expected list."""
        orch = make_orchestrator(project)
        setup_kernel_state(orch.kernel)

        result = orch.execute_locus("validate_schema", json.dumps({
            "connection": "warehouse",
            "table": "events",
            "expected_columns": ["id", "name"],  # missing "value"
        }))
        output = json.loads(result[0])
        assert output["success"] is True
        assert output["healthy"] is True  # no missing columns
        assert output["extra_columns"] == ["value"]
        assert output["missing_columns"] == []

    def test_schema_missing_columns(self, project):
        """validate_schema detects missing expected columns."""
        orch = make_orchestrator(project)
        setup_kernel_state(orch.kernel)

        result = orch.execute_locus("validate_schema", json.dumps({
            "connection": "warehouse",
            "table": "events",
            "expected_columns": ["id", "name", "value", "timestamp"],
        }))
        output = json.loads(result[0])
        assert output["success"] is True
        assert output["healthy"] is False
        assert output["missing_columns"] == ["timestamp"]


class TestCleanRecords:
    def test_clean_drops_nulls(self, project):
        """clean_records removes rows with nulls in specified columns."""
        orch = make_orchestrator(project)
        setup_multi_source_kernel(orch.kernel)

        result = orch.execute_locus("clean_records", json.dumps({
            "connection": "warehouse",
            "table": "partner_sales",
            "rules": {"drop_nulls": ["product"]},
        }))
        output = json.loads(result[0])
        assert output["success"] is True
        assert output["dropped_count"] == 2  # two rows with product=None
        assert output["cleaned_count"] == 1  # one row remains

    def test_clean_deduplicates(self, project):
        """clean_records removes duplicate rows."""
        orch = make_orchestrator(project)
        setup_multi_source_kernel(orch.kernel)

        result = orch.execute_locus("clean_records", json.dumps({
            "connection": "warehouse",
            "table": "partner_sales",
            "rules": {"dedup_columns": ["id", "product"]},
        }))
        output = json.loads(result[0])
        assert output["success"] is True
        assert output["dropped_count"] == 1  # one duplicate removed
        assert output["cleaned_count"] == 2

    def test_clean_fills_defaults(self, project):
        """clean_records fills null values with defaults."""
        orch = make_orchestrator(project)
        setup_multi_source_kernel(orch.kernel)

        result = orch.execute_locus("clean_records", json.dumps({
            "connection": "warehouse",
            "table": "partner_sales",
            "rules": {"fill_defaults": {"product": "unknown", "amount": 0.0}},
        }))
        output = json.loads(result[0])
        assert output["success"] is True
        assert output["cleaned_count"] == 3  # all rows kept
        assert output["dropped_count"] == 0

        # Verify nulls are filled
        count = orch.kernel.row_count("warehouse", "partner_sales")
        assert count == 3
        nulls = orch.kernel.check_nulls("warehouse", "partner_sales", "product")
        assert nulls["null_count"] == 0


class TestTransformRecords:
    def test_transform_with_mapping(self, project):
        """transform_records copies rows with renamed columns."""
        orch = make_orchestrator(project)
        setup_multi_source_kernel(orch.kernel)

        result = orch.execute_locus("transform_records", json.dumps({
            "connection": "warehouse",
            "source_table": "pos_sales",
            "target_table": "daily_sales",
            "mapping": {"id": "id", "product": "product", "amount": "amount", "date": "date"},
        }))
        output = json.loads(result[0])
        assert output["success"] is True
        assert output["transformed_count"] == 2

        # Verify target has the rows
        assert orch.kernel.row_count("warehouse", "daily_sales") == 2

    def test_transform_empty_source(self, project):
        """transform_records with empty source produces zero rows."""
        orch = make_orchestrator(project)
        setup_multi_source_kernel(orch.kernel)

        result = orch.execute_locus("transform_records", json.dumps({
            "connection": "warehouse",
            "source_table": "ecom_sales",  # empty table
            "target_table": "daily_sales",
            "mapping": {"id": "id", "product": "product", "amount": "amount"},
        }))
        output = json.loads(result[0])
        assert output["success"] is True
        assert output["transformed_count"] == 0


class TestExtendedPathway:
    def test_ingest_clean_and_load_happy(self, project):
        """Full 5-step pipeline: ingest, validate, clean, transform, check."""
        orch = make_orchestrator(project)
        setup_kernel_state(orch.kernel)
        # Add target table for transform step
        orch.kernel.add_table("warehouse", "warehouse_events", {
            "id": "int", "name": "string", "value": "float",
        })

        outputs = orch.run_pathway("ingest_clean_and_load", json.dumps({
            "url": "https://data.example.com/events.csv",
            "connection": "warehouse",
            "table": "events",
            "target_table": "warehouse_events",
            "column": "name",
            "expected_columns": ["id", "name", "value"],
            "clean_rules": {},
            "transform_mapping": {"id": "id", "name": "name", "value": "value"},
        }))

        assert len(outputs) == 5

        # Step 1: ingest
        ingest_out = json.loads(outputs[0])
        assert ingest_out["success"] is True
        assert ingest_out["rows_written"] == 3

        # Step 2: validate_schema
        schema_out = json.loads(outputs[1])
        assert schema_out["success"] is True
        assert schema_out["healthy"] is True

        # Step 3: clean_records
        clean_out = json.loads(outputs[2])
        assert clean_out["success"] is True

        # Step 4: transform_records
        transform_out = json.loads(outputs[3])
        assert transform_out["success"] is True
        assert transform_out["transformed_count"] == 3

        # Step 5: check_nulls on target
        nulls_out = json.loads(outputs[4])
        assert nulls_out["success"] is True
        assert nulls_out["healthy"] is True


class TestNewFailureInjection:
    def test_http_timeout_triggers_mutation(self, project):
        """Source timeout: inject http_get failure, mutation should recover."""
        orch = make_orchestrator(project)
        setup_kernel_state(orch.kernel)
        orch.kernel.inject_failure("http_get", "connection timeout")

        result = orch.execute_locus("ingest_csv_to_table", INPUT_JSON)
        assert result is not None
        output = json.loads(result[0])
        assert isinstance(output["success"], bool)

    def test_clean_failure_triggers_mutation(self, project):
        """clean_records failure should trigger mutation."""
        orch = make_orchestrator(project)
        setup_multi_source_kernel(orch.kernel)
        orch.kernel.inject_failure("clean_records", "simulated error")

        result = orch.execute_locus("clean_records", json.dumps({
            "connection": "warehouse",
            "table": "partner_sales",
            "rules": {"drop_nulls": ["product"]},
        }))
        assert result is not None
        output = json.loads(result[0])
        assert isinstance(output["success"], bool)

    def test_transform_failure_triggers_mutation(self, project):
        """transform_records failure should trigger mutation."""
        orch = make_orchestrator(project)
        setup_multi_source_kernel(orch.kernel)
        orch.kernel.inject_failure("transform_records", "simulated error")

        result = orch.execute_locus("transform_records", json.dumps({
            "connection": "warehouse",
            "source_table": "pos_sales",
            "target_table": "daily_sales",
            "mapping": {"id": "id", "product": "product"},
        }))
        assert result is not None
        output = json.loads(result[0])
        assert isinstance(output["success"], bool)

    def test_write_capacity_failure(self, project):
        """write_records capacity failure triggers mutation."""
        orch = make_orchestrator(project)
        setup_kernel_state(orch.kernel)
        orch.kernel.inject_failure("write_records", "table full: row limit exceeded")

        result = orch.execute_locus("ingest_csv_to_table", INPUT_JSON)
        assert result is not None


class TestMultiSource:
    def test_partner_feed_null_detection(self, project):
        """check_nulls detects unhealthy null ratio in partner data."""
        orch = make_orchestrator(project)
        setup_multi_source_kernel(orch.kernel)

        result = orch.execute_locus("check_nulls", json.dumps({
            "connection": "warehouse",
            "table": "partner_sales",
            "column": "product",
        }))
        output = json.loads(result[0])
        assert output["success"] is True
        assert output["healthy"] is False
        assert output["null_count"] == 2
        assert output["null_ratio"] > 0.1

    def test_schema_drift_detection(self, project):
        """validate_schema detects schema drift in e-commerce data."""
        orch = make_orchestrator(project)
        setup_multi_source_kernel(orch.kernel)
        # Ingest ecom data first so records have extra "source_id" column
        # but the table schema itself only has id/product/amount
        result = orch.execute_locus("validate_schema", json.dumps({
            "connection": "warehouse",
            "table": "ecom_sales",
            "expected_columns": ["id", "product", "amount", "date"],
        }))
        output = json.loads(result[0])
        assert output["success"] is True
        assert output["healthy"] is False
        assert "date" in output["missing_columns"]
