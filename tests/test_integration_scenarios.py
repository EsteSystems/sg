"""Phase 3 integration scenarios: exercise all evolutionary subsystems.

Each scenario uses the data pipeline plugin with a real Orchestrator
and all subsystems wired (event bus, metrics, audit log, contract
evolution, etc). Proves the wired subsystems work end-to-end.
"""
import json
import shutil
import pytest
from pathlib import Path

sg_data = pytest.importorskip("sg_data", reason="sg_data plugin not installed")

from sg.audit import AuditLog
from sg.contracts import ContractStore
from sg.contract_evolution import (
    ContractEvolution,
    TIGHTENING_THRESHOLD,
    MIN_CORRELATION_SAMPLES,
)
from sg.daemon import Daemon, DaemonConfig
from sg.events import EventBus
from sg.fusion import FusionTracker
from sg.meta_params import MetaParamTracker
from sg.metrics import MetricsCollector
from sg.mutation import MockMutationEngine
from sg.orchestrator import Orchestrator
from sg.pathway_fitness import PathwayFitnessTracker
from sg.pathway_registry import PathwayRegistry
from sg.phenotype import PhenotypeMap
from sg.registry import Registry
from sg_data import MockDataKernel


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
def scenario(tmp_path):
    """Full data pipeline project with all subsystems wired."""
    # Copy plugin files
    fixtures_dst = tmp_path / "fixtures"
    if FIXTURE_DIR.exists():
        shutil.copytree(FIXTURE_DIR, fixtures_dst)
    else:
        fixtures_dst.mkdir()

    contracts_dst = tmp_path / "contracts"
    if CONTRACTS_DIR.exists():
        shutil.copytree(CONTRACTS_DIR, contracts_dst)

    # Register all seed genes
    registry = Registry.open(tmp_path / ".sg" / "registry")
    phenotype = PhenotypeMap()

    for locus, filename in ALL_SEED_GENES:
        source = (GENES_DIR / filename).read_text()
        sha = registry.register(source, locus)
        phenotype.promote(locus, sha)
        allele = registry.get(sha)
        allele.state = "dominant"

    registry.save_index()
    phenotype.save(tmp_path / "phenotype.toml")

    # Create subsystems
    event_bus = EventBus()
    metrics = MetricsCollector()
    metrics.attach(event_bus)
    audit_log = AuditLog(tmp_path / ".sg" / "audit.jsonl")
    meta_tracker = MetaParamTracker.open(tmp_path / ".sg" / "meta_params.json")

    contract_store = ContractStore.open(tmp_path / "contracts")
    registry = Registry.open(tmp_path / ".sg" / "registry")
    phenotype = PhenotypeMap.load(tmp_path / "phenotype.toml")
    fusion_tracker = FusionTracker.open(tmp_path / "fusion_tracker.json")
    mutation_engine = MockMutationEngine(tmp_path / "fixtures")
    kernel = MockDataKernel()
    pft = PathwayFitnessTracker.open(tmp_path / "pathway_fitness.json")
    pr = PathwayRegistry.open(tmp_path / ".sg" / "pathway_registry")

    orch = Orchestrator(
        registry=registry,
        phenotype=phenotype,
        mutation_engine=mutation_engine,
        fusion_tracker=fusion_tracker,
        kernel=kernel,
        contract_store=contract_store,
        project_root=tmp_path,
        audit_log=audit_log,
        pathway_fitness_tracker=pft,
        pathway_registry=pr,
        event_bus=event_bus,
        meta_param_tracker=meta_tracker,
    )

    return {
        "orch": orch,
        "event_bus": event_bus,
        "metrics": metrics,
        "audit_log": audit_log,
        "meta_tracker": meta_tracker,
        "project_root": tmp_path,
    }


def setup_kernel(kernel):
    """Prepare mock kernel with tables and HTTP responses."""
    kernel.add_table("warehouse", "events", {
        "id": "int", "name": "string", "value": "float",
    })
    kernel.add_http_response("https://data.example.com/events.csv", {
        "records": [
            {"id": 1, "name": "alpha", "value": 10.0},
            {"id": 2, "name": "beta", "value": 20.0},
            {"id": 3, "name": "gamma", "value": 30.0},
        ],
    })


INGEST_INPUT = json.dumps({
    "url": "https://data.example.com/events.csv",
    "connection": "warehouse",
    "table": "events",
    "column": "name",
})


class TestScenarioA_GeneMutation:
    """Gene mutation cycle: failure handling, mutation, fitness tracking."""

    def test_graceful_failure_returns_success_false(self, scenario):
        """Genes that catch errors return success=false (no mutation needed)."""
        orch = scenario["orch"]
        setup_kernel(orch.kernel)

        # inject_failure causes RuntimeError which the gene catches
        orch.kernel.inject_failure("http_get", "simulated network timeout")
        result = orch.execute_locus("ingest_csv_to_table", INGEST_INPUT)

        assert result is not None
        output = json.loads(result[0])
        # Gene caught the error gracefully — valid output, no mutation
        assert output["success"] is False
        assert "timeout" in output.get("error", "")

    def test_mutation_on_uncaught_error(self, scenario):
        """When all alleles raise, mutation triggers and fixture recovers."""
        orch = scenario["orch"]
        setup_kernel(orch.kernel)
        event_bus = scenario["event_bus"]
        metrics = scenario["metrics"]

        # Replace the entire stack with a bad gene (clear fallbacks)
        bad_source = '''
import json
def execute(input_json: str) -> str:
    raise RuntimeError("always fails")
'''
        bad_sha = orch.registry.register(bad_source, "ingest_csv_to_table")
        # Clear the phenotype stack and set only the bad gene
        config = orch.phenotype.loci["ingest_csv_to_table"]
        config.dominant = bad_sha
        config.fallback = []
        orch.registry.get(bad_sha).state = "dominant"

        # Execute — bad gene raises, no fallbacks, triggers mutation
        result = orch.execute_locus("ingest_csv_to_table", INGEST_INPUT)
        assert result is not None

        # Verify mutation event published
        mutation_events = event_bus.recent(count=10, event_type="mutation_generated")
        assert len(mutation_events) >= 1

        # Verify metrics tracked mutation
        assert metrics.sg_mutations_total.value >= 1

    def test_fitness_tracking(self, scenario):
        """Successful executions accumulate fitness records."""
        orch = scenario["orch"]
        setup_kernel(orch.kernel)

        for _ in range(5):
            result = orch.execute_locus("ingest_csv_to_table", INGEST_INPUT)
            assert result is not None

        sha = orch.phenotype.get_dominant("ingest_csv_to_table")
        allele = orch.registry.get(sha)
        assert allele.successful_invocations >= 5
        assert allele.total_invocations >= 5


class TestScenarioB_ContractTightening:
    """Contract evolution: extra fields → tighten proposal."""

    def test_tightening_from_observations(self, scenario):
        orch = scenario["orch"]
        ce = orch._contract_evolution
        contract_store = orch.contract_store

        # Simulate outputs with an extra field not in contract
        gene_contract = contract_store.get_gene("ingest_csv_to_table")
        for _ in range(TIGHTENING_THRESHOLD):
            ce.record_output(
                "ingest_csv_to_table",
                json.dumps({
                    "success": True,
                    "rows_written": 3,
                    "source_id": "pos",  # extra field not in contract
                }),
                gene_contract,
            )

        # record_output calls analyze_tightening internally at threshold,
        # storing proposals.  Retrieve from the stored proposals.
        proposals = ce.get_proposals("ingest_csv_to_table")
        assert len(proposals) >= 1
        tighten = [p for p in proposals if p.proposal_type == "tighten_gives"]
        assert len(tighten) >= 1
        assert "source_id" in tighten[0].description


class TestScenarioC_ContractRelaxation:
    """Contract evolution: high mutation failure rate → relax proposal."""

    def test_relaxation_from_failures(self, scenario):
        orch = scenario["orch"]
        ce = orch._contract_evolution

        # Record mutation successes and failures to hit 30% failure rate
        for _ in range(3):
            ce.record_mutation_success("ingest_csv_to_table")
        for _ in range(7):
            ce.record_mutation_failure(
                "ingest_csv_to_table",
                "missing field 'estimated_delivery' in output",
            )

        # record_mutation_failure calls analyze_relaxation internally at threshold.
        # Retrieve from stored proposals.
        proposals = ce.get_proposals("ingest_csv_to_table")
        assert len(proposals) >= 1
        relax = [p for p in proposals if p.proposal_type == "relax_constraint"]
        assert len(relax) >= 1


class TestScenarioD_FeedsDiscovery:
    """Feeds discovery: correlated diagnostic + config fitness → add_feeds proposal."""

    def test_feeds_correlation_detected(self, scenario):
        orch = scenario["orch"]
        ce = orch._contract_evolution
        contract_store = orch.contract_store

        # Test discovery of a NEW feeds link: validate_schema → clean_records
        # (validate_schema already feeds ingest_csv_to_table, but not clean_records)
        ce.ensure_correlation_pair("validate_schema", "clean_records")

        # Record correlated observations
        for i in range(MIN_CORRELATION_SAMPLES):
            healthy = i % 2 == 0
            ce.record_diagnostic_output("validate_schema", {"healthy": healthy})
            fitness = 0.9 if healthy else 0.1
            ce.record_config_fitness("clean_records", fitness)

        # Analyze feeds
        proposals = ce.analyze_feeds(contract_store)
        feeds_proposals = [p for p in proposals if p.proposal_type == "add_feeds"]
        assert len(feeds_proposals) >= 1
        # Should propose validate_schema feeds clean_records
        desc = feeds_proposals[0].description
        assert "validate_schema" in desc or "clean_records" in desc


class TestScenarioE_DaemonOperation:
    """Daemon continuous operation: ticks, workload, metrics, state persistence."""

    def test_daemon_with_workload(self, scenario):
        orch = scenario["orch"]
        event_bus = scenario["event_bus"]
        metrics = scenario["metrics"]
        project_root = scenario["project_root"]
        setup_kernel(orch.kernel)

        def workload(orchestrator, tick_number):
            # Ensure kernel state is fresh each tick
            orchestrator.kernel.add_http_response(
                "https://data.example.com/events.csv", {
                    "records": [
                        {"id": 1, "name": "alpha", "value": 10.0},
                    ],
                })
            orchestrator.kernel.add_table("warehouse", "events", {
                "id": "int", "name": "string", "value": "float",
            })
            orchestrator.execute_locus("ingest_csv_to_table", INGEST_INPUT)

        config = DaemonConfig(
            tick_interval=0.0,
            max_ticks=15,
            health_check_interval=5,
            auto_tune_interval=100,
            workload=workload,
        )
        daemon = Daemon(orch, event_bus=event_bus, config=config,
                        metrics_collector=metrics)
        daemon.start()

        # Verify ticks completed
        assert daemon.tick_count == 15
        assert metrics.sg_daemon_ticks_total.value == 15

        # Verify tick events published
        tick_events = event_bus.recent(count=20, event_type="tick_complete")
        assert len(tick_events) == 15

        # Verify state persisted
        assert (project_root / "phenotype.toml").exists()
        assert (project_root / ".sg" / "registry" / "registry.json").exists()

        # Verify metrics snapshot written
        assert (project_root / ".sg" / "metrics.json").exists()

    def test_daemon_state_survives_restart(self, scenario):
        orch = scenario["orch"]
        event_bus = scenario["event_bus"]
        metrics = scenario["metrics"]
        project_root = scenario["project_root"]
        setup_kernel(orch.kernel)

        def workload(orchestrator, tick_number):
            orchestrator.kernel.add_http_response(
                "https://data.example.com/events.csv", {
                    "records": [{"id": 1, "name": "a", "value": 1.0}],
                })
            orchestrator.kernel.add_table("warehouse", "events", {
                "id": "int", "name": "string", "value": "float",
            })
            orchestrator.execute_locus("ingest_csv_to_table", INGEST_INPUT)

        # First run: 5 ticks
        config = DaemonConfig(tick_interval=0.0, max_ticks=5, workload=workload)
        daemon = Daemon(orch, event_bus=event_bus, config=config,
                        metrics_collector=metrics)
        daemon.start()
        assert daemon.tick_count == 5

        # Reload state from disk
        registry2 = Registry.open(project_root / ".sg" / "registry")
        phenotype2 = PhenotypeMap.load(project_root / "phenotype.toml")
        # State should have persisted
        sha = phenotype2.get_dominant("ingest_csv_to_table")
        assert sha is not None
        allele = registry2.get(sha)
        assert allele is not None
        assert allele.total_invocations >= 5


class TestScenarioF_FullPipeline:
    """Multi-step pipeline with mutation recovery and audit trail."""

    def test_full_pipeline_happy_path(self, scenario):
        orch = scenario["orch"]
        setup_kernel(orch.kernel)
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
        for out_json in outputs:
            out = json.loads(out_json)
            assert out["success"] is True

    def test_pipeline_with_mutation_recovery(self, scenario):
        """When sole allele raises, pipeline recovers via mutation."""
        orch = scenario["orch"]
        event_bus = scenario["event_bus"]
        setup_kernel(orch.kernel)
        orch.kernel.add_table("warehouse", "warehouse_events", {
            "id": "int", "name": "string", "value": "float",
        })

        # Replace ingest allele stack with a bad gene only
        bad_source = '''
import json
def execute(input_json: str) -> str:
    raise RuntimeError("broken ingest")
'''
        bad_sha = orch.registry.register(bad_source, "ingest_csv_to_table")
        config = orch.phenotype.loci["ingest_csv_to_table"]
        config.dominant = bad_sha
        config.fallback = []
        orch.registry.get(bad_sha).state = "dominant"

        # Pipeline should recover via mutation (fixture provides fix)
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

        # Should complete all 5 steps (mutation fixes ingest)
        assert len(outputs) == 5

        # Mutation event should have fired
        mutation_events = event_bus.recent(count=10, event_type="mutation_generated")
        assert len(mutation_events) >= 1

    def test_audit_log_records_activity(self, scenario):
        orch = scenario["orch"]
        audit_log = scenario["audit_log"]

        # Audit log records specific events (mutations, promotions, etc.)
        # Record directly to verify infrastructure works
        audit_log.record("test_event", locus="ingest_csv_to_table",
                         sha="abc123", detail="integration test")

        assert audit_log.path.exists()
        entries = audit_log.path.read_text().strip().split("\n")
        assert len(entries) >= 1
        entry = json.loads(entries[0])
        assert entry["event"] == "test_event"
        assert entry["locus"] == "ingest_csv_to_table"

    def test_diagnostic_feedback_loop(self, scenario):
        """Diagnostic → feeds → convergence fitness on config gene."""
        orch = scenario["orch"]
        event_bus = scenario["event_bus"]
        setup_kernel(orch.kernel)

        # Execute ingest first
        orch.execute_locus("ingest_csv_to_table", INGEST_INPUT)

        # Execute diagnostic — check_row_count feeds ingest_csv_to_table convergence
        result = orch.execute_locus("check_row_count", json.dumps({
            "connection": "warehouse",
            "table": "events",
        }))
        output = json.loads(result[0])
        assert output["healthy"] is True

        # Verify convergence feedback was recorded
        sha = orch.phenotype.get_dominant("ingest_csv_to_table")
        allele = orch.registry.get(sha)
        convergence = [r for r in allele.fitness_records
                       if r.get("timescale") == "convergence"]
        assert len(convergence) >= 1

        # Verify fitness_feedback event published
        fb_events = event_bus.recent(count=10, event_type="fitness_feedback")
        assert len(fb_events) >= 1
