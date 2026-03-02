"""Tests for cross-locus interaction detection (Phase 1.3)."""
import json
import pytest
from unittest.mock import patch

from sg.contracts import ContractStore
from sg.fusion import FusionTracker
from sg_network import MockNetworkKernel
from sg.interactions import (
    find_affected_pathways, check_interactions, InteractionFailure,
)
from sg.mutation import MockMutationEngine, MutationEngine
from sg.orchestrator import Orchestrator
from sg.phenotype import PhenotypeMap
from sg.registry import Registry


SIMPLE_GENE = '''
import json

def execute(input_json):
    data = json.loads(input_json)
    gene_sdk.create_bridge(data.get("bridge_name", "br0"),
                           data.get("interfaces", []))
    return json.dumps({"success": True})
'''

FAILING_GENE = '''
import json

def execute(input_json):
    raise ValueError("always fails")
'''


def _write_sg(path, content):
    path.write_text(content)
    return path


@pytest.fixture
def contract_store_with_pathways(tmp_path):
    """Contract store with genes and pathways for interaction testing."""
    genes_dir = tmp_path / "contracts" / "genes"
    genes_dir.mkdir(parents=True)
    pathways_dir = tmp_path / "contracts" / "pathways"
    pathways_dir.mkdir(parents=True)

    _write_sg(genes_dir / "bridge_create.sg", """\
gene bridge_create for network
  is configuration
  risk low

  does:
    Create a bridge.

  takes:
    bridge_name: string
    interfaces: string[]

  gives:
    success: bool
""")
    _write_sg(genes_dir / "bridge_stp.sg", """\
gene bridge_stp for network
  is configuration
  risk low

  does:
    Configure STP on a bridge.

  takes:
    bridge_name: string
    stp_enabled: bool

  gives:
    success: bool
""")
    _write_sg(genes_dir / "standalone.sg", """\
gene standalone for network
  is diagnostic
  risk none

  does:
    A standalone gene not in any pathway.

  takes:
    name: string

  gives:
    success: bool
    healthy: bool
""")

    _write_sg(pathways_dir / "setup_bridge.sg", """\
pathway setup_bridge
  risk low

  does:
    Create a bridge and configure STP.

  takes:
    bridge_name: string
    interfaces: string[]
    stp_enabled: bool

  steps:
    1. bridge_create
         bridge_name = {bridge_name}
         interfaces = {interfaces}

    2. bridge_stp
         bridge_name = {bridge_name}
         stp_enabled = {stp_enabled}

  on failure:
    rollback all
""")

    store = ContractStore()
    store.load_directory(tmp_path / "contracts")
    return store


@pytest.fixture
def project_with_pathways(tmp_path, contract_store_with_pathways):
    """Full project setup with pathway contracts."""
    fixtures_dir = tmp_path / "fixtures"
    fixtures_dir.mkdir()
    (fixtures_dir / "bridge_create_fix.py").write_text(SIMPLE_GENE)
    (fixtures_dir / "bridge_stp_fix.py").write_text(SIMPLE_GENE)

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
        "contract_store": contract_store_with_pathways,
    }


@pytest.fixture
def orch_with_pathways(project_with_pathways):
    p = project_with_pathways
    p["kernel"].reset()
    return Orchestrator(
        registry=p["registry"],
        phenotype=p["phenotype"],
        mutation_engine=p["mutation_engine"],
        fusion_tracker=p["fusion_tracker"],
        kernel=p["kernel"],
        contract_store=p["contract_store"],
        project_root=p["root"],
    )


class TestFindAffectedPathways:
    def test_locus_in_pathway(self, contract_store_with_pathways):
        result = find_affected_pathways(
            "bridge_create", contract_store_with_pathways)
        assert "setup_bridge" in result

    def test_second_step_found(self, contract_store_with_pathways):
        result = find_affected_pathways(
            "bridge_stp", contract_store_with_pathways)
        assert "setup_bridge" in result

    def test_locus_not_in_any_pathway(self, contract_store_with_pathways):
        result = find_affected_pathways(
            "standalone", contract_store_with_pathways)
        assert result == []

    def test_nonexistent_locus(self, contract_store_with_pathways):
        result = find_affected_pathways(
            "nonexistent_locus", contract_store_with_pathways)
        assert result == []


class TestFindAffectedPathwaysComposed:
    """Test composed pathway detection with -> refs."""

    def test_composed_pathway_ref(self, tmp_path):
        """Locus found transitively through composed pathway ref."""
        genes_dir = tmp_path / "contracts" / "genes"
        genes_dir.mkdir(parents=True)
        pathways_dir = tmp_path / "contracts" / "pathways"
        pathways_dir.mkdir(parents=True)

        _write_sg(genes_dir / "leaf_gene.sg", """\
gene leaf_gene
  is configuration
  risk low

  does:
    A leaf gene.

  takes:
    name: string

  gives:
    success: bool
""")
        _write_sg(pathways_dir / "inner.sg", """\
pathway inner
  risk low

  does:
    Inner pathway containing the leaf gene.

  takes:
    name: string

  steps:
    1. leaf_gene
         name = {name}

  on failure:
    rollback all
""")
        _write_sg(pathways_dir / "outer.sg", """\
pathway outer
  risk low

  does:
    Outer pathway composing inner.

  takes:
    name: string

  steps:
    1. -> inner
         name = {name}

  on failure:
    rollback all
""")

        store = ContractStore()
        store.load_directory(tmp_path / "contracts")

        result = find_affected_pathways("leaf_gene", store)
        assert "inner" in result
        assert "outer" in result


class TestFindAffectedPathwaysForLoop:
    def test_for_loop_body(self, tmp_path):
        genes_dir = tmp_path / "contracts" / "genes"
        genes_dir.mkdir(parents=True)
        pathways_dir = tmp_path / "contracts" / "pathways"
        pathways_dir.mkdir(parents=True)

        _write_sg(genes_dir / "item_process.sg", """\
gene item_process
  is configuration
  risk low

  does:
    Process one item.

  takes:
    item: string

  gives:
    success: bool
""")
        _write_sg(pathways_dir / "batch.sg", """\
pathway batch
  risk low

  does:
    Process items in a loop.

  takes:
    items: string[]

  steps:
    1. for item in items
         item_process
           item = {item}

  on failure:
    rollback all
""")

        store = ContractStore()
        store.load_directory(tmp_path / "contracts")

        result = find_affected_pathways("item_process", store)
        assert "batch" in result


class TestInteractionTesting:
    def test_no_affected_pathways_returns_empty(self, orch_with_pathways):
        result = check_interactions("standalone", "sha123", orch_with_pathways)
        assert result == []

    def test_original_dominant_restored(self, orch_with_pathways, project_with_pathways):
        """After interaction test, phenotype is back to original state."""
        reg = project_with_pathways["registry"]
        phen = project_with_pathways["phenotype"]

        sha_orig = reg.register(SIMPLE_GENE, "bridge_create")
        phen.promote("bridge_create", sha_orig)

        sha_new = reg.register("# new\n" + SIMPLE_GENE, "bridge_create")

        check_interactions("bridge_create", sha_new, orch_with_pathways)

        # Original dominant should be restored
        assert phen.get_dominant("bridge_create") == sha_orig

    def test_original_none_restored(self, orch_with_pathways, project_with_pathways):
        """If no original dominant, dominant is cleared after test."""
        reg = project_with_pathways["registry"]
        sha_new = reg.register(SIMPLE_GENE, "bridge_create")

        check_interactions("bridge_create", sha_new, orch_with_pathways)

        assert project_with_pathways["phenotype"].get_dominant("bridge_create") is None


class TestPromotionWithInteractions:
    def test_rollback_policy_blocks_promotion(self, orch_with_pathways, project_with_pathways):
        """With rollback policy, broken interactions block promotion."""
        reg = project_with_pathways["registry"]
        phen = project_with_pathways["phenotype"]

        # Set up: bridge_create dominant is a good gene
        sha_good = reg.register(SIMPLE_GENE, "bridge_create")
        phen.promote("bridge_create", sha_good)

        # Also need bridge_stp for the pathway
        sha_stp = reg.register(SIMPLE_GENE, "bridge_stp")
        phen.promote("bridge_stp", sha_stp)

        # Register a new allele that would be promoted but breaks the pathway
        sha_bad = reg.register(FAILING_GENE, "bridge_create")
        allele = reg.get(sha_bad)
        # Fake enough invocations for promotion
        allele.successful_invocations = 100
        allele.failed_invocations = 0
        allele.consecutive_failures = 0

        with patch.dict("os.environ", {"SG_INTERACTION_POLICY": "rollback"}):
            orch_with_pathways._check_promotion("bridge_create", sha_bad)

        # Should NOT be promoted because the pathway would break
        assert phen.get_dominant("bridge_create") == sha_good

    def test_mutate_policy_allows_promotion(self, orch_with_pathways, project_with_pathways):
        """With mutate policy, broken interactions don't block promotion."""
        reg = project_with_pathways["registry"]
        phen = project_with_pathways["phenotype"]

        sha_good = reg.register(SIMPLE_GENE, "bridge_create")
        phen.promote("bridge_create", sha_good)

        sha_stp = reg.register(SIMPLE_GENE, "bridge_stp")
        phen.promote("bridge_stp", sha_stp)

        sha_bad = reg.register(FAILING_GENE, "bridge_create")
        allele = reg.get(sha_bad)
        allele.successful_invocations = 100
        allele.failed_invocations = 0
        allele.consecutive_failures = 0

        with patch.dict("os.environ", {"SG_INTERACTION_POLICY": "mutate"}):
            orch_with_pathways._check_promotion("bridge_create", sha_bad)

        # Should be promoted despite interaction failure
        assert phen.get_dominant("bridge_create") == sha_bad
