"""Tests for lineage visualization, competition arena, and topology verify scheduling."""
import json
import shutil
import pytest
from pathlib import Path
from io import StringIO

from sg import arena
from sg.contracts import ContractStore
from sg.fusion import FusionTracker
from sg.kernel.mock import MockNetworkKernel
from sg.mutation import MockMutationEngine
from sg.orchestrator import Orchestrator
from sg.parser.types import TopologyContract, TopologyResource, VerifyStep
from sg.phenotype import PhenotypeMap
from sg.registry import Registry
from sg.topology import execute_topology


CONTRACTS_DIR = Path(__file__).parent.parent / "contracts"
GENES_DIR = Path(__file__).parent.parent / "genes"
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def project(tmp_path):
    shutil.copytree(FIXTURES_DIR, tmp_path / "fixtures")
    shutil.copytree(CONTRACTS_DIR, tmp_path / "contracts")

    contract_store = ContractStore.open(tmp_path / "contracts")
    registry = Registry.open(tmp_path / ".sg" / "registry")
    phenotype = PhenotypeMap()

    for locus in contract_store.known_loci():
        candidates = sorted(GENES_DIR.glob(f"{locus}_*.py"))
        if candidates:
            source = candidates[0].read_text()
            sha = registry.register(source, locus)
            phenotype.promote(locus, sha)
            allele = registry.get(sha)
            allele.state = "dominant"

    registry.save_index()
    phenotype.save(tmp_path / "phenotype.toml")
    return tmp_path


def _make_orch(project_root):
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


# --- Topology verify scheduling ---

class TestTopologyVerifyScheduling:
    def test_topology_schedules_verify(self, project):
        """execute_topology schedules verify diagnostics from topology contract."""
        orch = _make_orch(project)

        topo = TopologyContract(
            name="test_verify",
            does="test topology with verify",
            has=[
                TopologyResource(name="mgmt", resource_type="bridge", properties={}),
            ],
            verify=[
                VerifyStep(locus="check_connectivity", params={"bridge_name": "{bridge_name}"}),
            ],
            verify_within="30s",
        )

        execute_topology(topo, json.dumps({
            "bridge_name": "br0",
            "bridge_ifaces": ["eth0"],
        }), orch)

        assert orch.verify_scheduler.pending_count >= 1

    def test_topology_no_verify_no_extra_schedule(self, project):
        """Topology without verify block doesn't add topology-level verify.

        Note: the gene-level verify from bridge_create's own contract may
        still fire, so we check that topology adds nothing beyond that.
        """
        orch = _make_orch(project)

        # Run a bare gene to see how many gene-level verifies fire
        orch.execute_locus("bridge_create", json.dumps({
            "bridge_name": "br_baseline",
            "interfaces": ["eth0"],
        }))
        baseline_count = orch.verify_scheduler.pending_count

        # Now run topology without verify — should add no extra verifies
        orch2 = _make_orch(project)
        topo = TopologyContract(
            name="test_no_verify",
            does="test topology without verify",
            has=[
                TopologyResource(name="mgmt", resource_type="bridge", properties={}),
            ],
        )

        execute_topology(topo, json.dumps({
            "bridge_name": "br0",
            "bridge_ifaces": ["eth0"],
        }), orch2)

        # Should have same count as baseline (gene-level only, no topology-level)
        assert orch2.verify_scheduler.pending_count == baseline_count

    def test_production_server_schedules_verify(self, project):
        """Full production_server topology schedules its verify steps."""
        orch = _make_orch(project)

        outputs = orch.run_topology("production_server", json.dumps({
            "bridge_name": "br0",
            "bridge_ifaces": ["eth0"],
            "uplink": "eth1",
            "bond_name": "bond0",
            "bond_mode": "active-backup",
            "bond_members": ["eth2", "eth3"],
            "vlans": [100, 200],
        }))

        # production_server has 2 verify steps
        assert orch.verify_scheduler.pending_count >= 2

    def test_verify_not_scheduled_on_failure(self, project):
        """Verify is NOT scheduled when topology fails."""
        orch = _make_orch(project)

        # Remove bond_create allele and fixture to force failure
        orch.phenotype.loci.pop("bond_create", None)
        fixture = project / "fixtures" / "bond_create_fix.py"
        if fixture.exists():
            fixture.unlink()

        topo = TopologyContract(
            name="test_fail",
            does="test failing topology",
            has=[
                TopologyResource(name="storage", resource_type="bond",
                                 properties={"mode": "active-backup",
                                             "members": "{bond_members}"}),
            ],
            verify=[
                VerifyStep(locus="check_bond_state", params={"bond_name": "{bond_name}"}),
            ],
            verify_within="30s",
            on_failure="rollback all",
        )

        with pytest.raises(RuntimeError):
            execute_topology(topo, json.dumps({
                "bond_name": "bond0",
                "bond_members": ["eth2", "eth3"],
            }), orch)

        # Verify should NOT be scheduled when topology fails
        assert orch.verify_scheduler.pending_count == 0


# --- Lineage visualization ---

class TestLineage:
    def test_lineage_single_allele(self, project):
        """Single allele shows as root with no children."""
        orch = _make_orch(project)
        alleles = orch.registry.alleles_for_locus("bridge_create")
        assert len(alleles) >= 1
        # Root allele has no parent
        assert alleles[0].parent_sha is None

    def test_lineage_with_mutations(self, project):
        """Lineage chain via parent_sha is traceable."""
        registry = Registry.open(project / ".sg" / "registry")

        # Register a chain: root → child → grandchild
        root_src = "def execute(i): return '{\"success\": true}'"
        child_src = "def execute(i): return '{\"success\": true}' # v2"
        grandchild_src = "def execute(i): return '{\"success\": true}' # v3"

        root_sha = registry.register(root_src, "bridge_create", generation=0)
        child_sha = registry.register(child_src, "bridge_create",
                                      generation=1, parent_sha=root_sha)
        gc_sha = registry.register(grandchild_src, "bridge_create",
                                   generation=2, parent_sha=child_sha)

        # Verify chain
        child = registry.get(child_sha)
        assert child.parent_sha == root_sha
        assert child.generation == 1

        grandchild = registry.get(gc_sha)
        assert grandchild.parent_sha == child_sha
        assert grandchild.generation == 2

    def test_cmd_lineage_output(self, project, capsys):
        """cmd_lineage produces readable output."""
        from sg.cli import cmd_lineage
        import argparse

        # Register a child allele so we have a lineage
        registry = Registry.open(project / ".sg" / "registry")
        phenotype = PhenotypeMap.load(project / "phenotype.toml")
        parent_sha = phenotype.get_dominant("bridge_create")
        child_src = "def execute(i): return '{\"success\": true}' # mutant"
        child_sha = registry.register(child_src, "bridge_create",
                                      generation=1, parent_sha=parent_sha)
        registry.save_index()

        import os
        old = os.environ.get("SG_PROJECT_ROOT")
        os.environ["SG_PROJECT_ROOT"] = str(project)
        try:
            args = argparse.Namespace(locus="bridge_create")
            cmd_lineage(args)
        finally:
            if old is None:
                os.environ.pop("SG_PROJECT_ROOT", None)
            else:
                os.environ["SG_PROJECT_ROOT"] = old

        captured = capsys.readouterr()
        assert "Lineage: bridge_create" in captured.out
        assert "gen=0" in captured.out
        assert "gen=1" in captured.out
        assert parent_sha[:12] in captured.out
        assert child_sha[:12] in captured.out

    def test_cmd_lineage_no_alleles(self, project, capsys):
        """cmd_lineage with unknown locus says no alleles."""
        from sg.cli import cmd_lineage
        import argparse
        import os

        old = os.environ.get("SG_PROJECT_ROOT")
        os.environ["SG_PROJECT_ROOT"] = str(project)
        try:
            args = argparse.Namespace(locus="nonexistent_locus")
            cmd_lineage(args)
        finally:
            if old is None:
                os.environ.pop("SG_PROJECT_ROOT", None)
            else:
                os.environ["SG_PROJECT_ROOT"] = old

        captured = capsys.readouterr()
        assert "No alleles" in captured.out


# --- Allele competition arena ---

class TestCompete:
    def test_cmd_compete_runs_trials(self, project, capsys):
        """cmd_compete runs trial rounds and reports results."""
        from sg.cli import cmd_compete
        import argparse
        import os

        # Add a second allele so competition is possible
        registry = Registry.open(project / ".sg" / "registry")
        phenotype = PhenotypeMap.load(project / "phenotype.toml")
        alt_source = (GENES_DIR / "bridge_create_v1.py").read_text() + "\n# alt\n"
        alt_sha = registry.register(alt_source, "bridge_create",
                                    generation=1,
                                    parent_sha=phenotype.get_dominant("bridge_create"))
        alt = registry.get(alt_sha)
        alt.state = "recessive"
        phenotype.add_to_fallback("bridge_create", alt_sha)
        registry.save_index()
        phenotype.save(project / "phenotype.toml")

        old = os.environ.get("SG_PROJECT_ROOT")
        os.environ["SG_PROJECT_ROOT"] = str(project)
        try:
            args = argparse.Namespace(
                locus="bridge_create",
                input=json.dumps({
                    "bridge_name": "br0",
                    "interfaces": ["eth0"],
                }),
                rounds=5,
            )
            cmd_compete(args)
        finally:
            if old is None:
                os.environ.pop("SG_PROJECT_ROOT", None)
            else:
                os.environ["SG_PROJECT_ROOT"] = old

        captured = capsys.readouterr()
        assert "Competition: bridge_create" in captured.out
        assert "5 rounds" in captured.out

    def test_compete_with_multiple_alleles(self, project, capsys):
        """Competition with multiple alleles reports per-allele results."""
        from sg.cli import cmd_compete
        import argparse
        import os

        registry = Registry.open(project / ".sg" / "registry")
        phenotype = PhenotypeMap.load(project / "phenotype.toml")

        # Add a second allele (recessive)
        alt_source = (GENES_DIR / "bridge_create_v1.py").read_text()
        # Modify slightly to get different SHA
        alt_source_v2 = alt_source + "\n# variant 2\n"
        alt_sha = registry.register(alt_source_v2, "bridge_create",
                                    generation=1,
                                    parent_sha=phenotype.get_dominant("bridge_create"))
        alt = registry.get(alt_sha)
        alt.state = "recessive"
        phenotype.add_to_fallback("bridge_create", alt_sha)
        registry.save_index()
        phenotype.save(project / "phenotype.toml")

        old = os.environ.get("SG_PROJECT_ROOT")
        os.environ["SG_PROJECT_ROOT"] = str(project)
        try:
            args = argparse.Namespace(
                locus="bridge_create",
                input=json.dumps({
                    "bridge_name": "br0",
                    "interfaces": ["eth0"],
                }),
                rounds=5,
            )
            cmd_compete(args)
        finally:
            if old is None:
                os.environ.pop("SG_PROJECT_ROOT", None)
            else:
                os.environ["SG_PROJECT_ROOT"] = old

        captured = capsys.readouterr()
        assert "Competition: bridge_create" in captured.out
        assert "2 allele(s)" in captured.out

    def test_compete_no_alleles(self, project, capsys):
        """Competition for unknown locus reports no alleles."""
        from sg.cli import cmd_compete
        import argparse
        import os

        old = os.environ.get("SG_PROJECT_ROOT")
        os.environ["SG_PROJECT_ROOT"] = str(project)
        try:
            args = argparse.Namespace(
                locus="nonexistent_locus",
                input="{}",
                rounds=5,
            )
            cmd_compete(args)
        finally:
            if old is None:
                os.environ.pop("SG_PROJECT_ROOT", None)
            else:
                os.environ["SG_PROJECT_ROOT"] = old

        captured = capsys.readouterr()
        assert "No alleles" in captured.out

    def test_compete_promotion(self, project, capsys):
        """If a recessive allele accumulates enough invocations and fitness, it can be promoted."""
        from sg.cli import cmd_compete
        import argparse
        import os

        registry = Registry.open(project / ".sg" / "registry")
        phenotype = PhenotypeMap.load(project / "phenotype.toml")

        dominant_sha = phenotype.get_dominant("bridge_create")
        dominant = registry.get(dominant_sha)

        # Sabotage dominant's record to have low fitness
        dominant.successful_invocations = 10
        dominant.failed_invocations = 90

        # Add a strong recessive allele with pre-existing good record
        alt_source = (GENES_DIR / "bridge_create_v1.py").read_text()
        alt_source_v2 = alt_source + "\n# strong variant\n"
        alt_sha = registry.register(alt_source_v2, "bridge_create",
                                    generation=1, parent_sha=dominant_sha)
        alt = registry.get(alt_sha)
        alt.state = "recessive"
        alt.successful_invocations = 45  # will reach 50+ after 5 rounds
        alt.failed_invocations = 0
        phenotype.add_to_fallback("bridge_create", alt_sha)
        registry.save_index()
        phenotype.save(project / "phenotype.toml")

        old = os.environ.get("SG_PROJECT_ROOT")
        os.environ["SG_PROJECT_ROOT"] = str(project)
        try:
            args = argparse.Namespace(
                locus="bridge_create",
                input=json.dumps({
                    "bridge_name": "br0",
                    "interfaces": ["eth0"],
                }),
                rounds=10,
            )
            cmd_compete(args)
        finally:
            if old is None:
                os.environ.pop("SG_PROJECT_ROOT", None)
            else:
                os.environ["SG_PROJECT_ROOT"] = old

        captured = capsys.readouterr()
        # The strong variant should be promoted
        assert "Promoting" in captured.out

        # Verify in phenotype
        phenotype2 = PhenotypeMap.load(project / "phenotype.toml")
        assert phenotype2.get_dominant("bridge_create") == alt_sha
