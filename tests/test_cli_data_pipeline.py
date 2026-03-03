"""CLI smoke test: bootstrap and run a data pipeline project."""
import json
import shutil
import pytest
from pathlib import Path

sg_data = pytest.importorskip("sg_data", reason="sg_data plugin not installed")

from sg.contracts import ContractStore
from sg_data import MockDataKernel
from sg.mutation import MockMutationEngine
from sg.orchestrator import Orchestrator
from sg.phenotype import PhenotypeMap
from sg.registry import Registry
from sg.fusion import FusionTracker


FIXTURE_DIR = sg_data.fixtures_path()
GENES_DIR = sg_data.genes_path()
CONTRACTS_DIR = sg_data.contracts_path()


@pytest.fixture
def cli_project(tmp_path):
    """Simulate 'sg init' for the data pipeline plugin."""
    # Copy contracts
    contracts_dst = tmp_path / "contracts"
    shutil.copytree(CONTRACTS_DIR, contracts_dst)

    # Copy seed genes (sg init discovers these from genes/)
    genes_dst = tmp_path / "genes"
    shutil.copytree(GENES_DIR, genes_dst)

    # Copy fixtures for mutation engine
    fixtures_dst = tmp_path / "fixtures"
    shutil.copytree(FIXTURE_DIR, fixtures_dst)

    return tmp_path


class TestBootstrap:
    def test_init_registers_all_seed_genes(self, cli_project):
        """sg init should discover and register all 6 seed genes."""
        from sg.cli import _discover_seed_genes

        contract_store = ContractStore.open(cli_project / "contracts")
        seeds = _discover_seed_genes(
            cli_project / "genes", contract_store.known_loci(),
        )

        assert len(seeds) == 6
        expected = {
            "ingest_csv_to_table", "check_row_count", "check_nulls",
            "validate_schema", "clean_records", "transform_records",
        }
        assert set(seeds.keys()) == expected

    def test_init_and_run_pathway(self, cli_project):
        """Bootstrap project, then run ingest_and_validate pathway."""
        root = cli_project
        contract_store = ContractStore.open(root / "contracts")

        # Simulate sg init: register seed genes
        from sg.cli import _discover_seed_genes
        seeds = _discover_seed_genes(root / "genes", contract_store.known_loci())

        registry = Registry.open(root / ".sg" / "registry")
        phenotype = PhenotypeMap()
        for locus, gene_path in seeds.items():
            source = gene_path.read_text()
            sha = registry.register(source, locus)
            phenotype.promote(locus, sha)
            allele = registry.get(sha)
            allele.state = "dominant"
        registry.save_index()
        phenotype.save(root / "phenotype.toml")

        # Create orchestrator
        registry = Registry.open(root / ".sg" / "registry")
        phenotype = PhenotypeMap.load(root / "phenotype.toml")
        fusion_tracker = FusionTracker.open(root / "fusion_tracker.json")
        mutation_engine = MockMutationEngine(root / "fixtures")
        kernel = MockDataKernel()

        orch = Orchestrator(
            registry=registry,
            phenotype=phenotype,
            mutation_engine=mutation_engine,
            fusion_tracker=fusion_tracker,
            kernel=kernel,
            contract_store=contract_store,
            project_root=root,
        )

        # Setup kernel state
        kernel.add_table("warehouse", "events", {
            "id": "int", "name": "string", "value": "float",
        })
        kernel.add_http_response("https://data.example.com/events.csv", {
            "records": [
                {"id": 1, "name": "alpha", "value": 10.0},
                {"id": 2, "name": "beta", "value": 20.0},
            ],
        })

        # Run pathway
        outputs = orch.run_pathway("ingest_and_validate", json.dumps({
            "url": "https://data.example.com/events.csv",
            "connection": "warehouse",
            "table": "events",
            "column": "name",
        }))

        assert len(outputs) == 3
        ingest_out = json.loads(outputs[0])
        assert ingest_out["success"] is True
        assert ingest_out["rows_written"] == 2

    def test_status_shows_all_loci(self, cli_project):
        """After init, sg status should show all 6 loci."""
        root = cli_project
        contract_store = ContractStore.open(root / "contracts")

        from sg.cli import _discover_seed_genes
        seeds = _discover_seed_genes(root / "genes", contract_store.known_loci())

        registry = Registry.open(root / ".sg" / "registry")
        phenotype = PhenotypeMap()
        for locus, gene_path in seeds.items():
            source = gene_path.read_text()
            sha = registry.register(source, locus)
            phenotype.promote(locus, sha)
            allele = registry.get(sha)
            allele.state = "dominant"
        registry.save_index()
        phenotype.save(root / "phenotype.toml")

        # Verify all loci have dominant alleles
        phenotype = PhenotypeMap.load(root / "phenotype.toml")
        for locus in ["ingest_csv_to_table", "check_row_count", "check_nulls",
                       "validate_schema", "clean_records", "transform_records"]:
            dom = phenotype.get_dominant(locus)
            assert dom is not None, f"no dominant for {locus}"
