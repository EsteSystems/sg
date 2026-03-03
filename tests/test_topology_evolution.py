"""Phase E.3: Topology evolution tests.

Verifies topology allele registry, arena, phenotype integration,
and orchestrator topology allele stack execution.
"""
from __future__ import annotations

import shutil

import pytest

import sg_network
from sg.contracts import ContractStore
from sg.fusion import FusionTracker
from sg_network import MockNetworkKernel
from sg.mutation import MockMutationEngine
from sg.orchestrator import Orchestrator
from sg.pathway_fitness import PathwayFitnessTracker
from sg.pathway_registry import PathwayRegistry
from sg.phenotype import PhenotypeMap
from sg.registry import Registry
from sg.topology import TopologyStep
from sg.topology_arena import (
    compute_topology_fitness,
    record_topology_success,
    record_topology_failure,
    should_promote_topology,
    should_demote_topology,
    set_topology_dominant,
    set_topology_recessive,
    set_topology_deprecated,
    TOPOLOGY_PROMOTION_ADVANTAGE,
    TOPOLOGY_PROMOTION_MIN_EXECUTIONS,
    TOPOLOGY_DEMOTION_CONSECUTIVE_FAILURES,
)
from sg.topology_registry import (
    TopologyStepSpec,
    TopologyAllele,
    TopologyRegistry,
    compute_topology_structure_sha,
    steps_from_decomposition,
)


CONTRACTS_DIR = sg_network.contracts_path()
GENES_DIR = sg_network.genes_path()
FIXTURES_DIR = sg_network.fixtures_path()


class TestTopologyStepSpec:
    """TopologyStepSpec serialization."""

    def test_to_dict_roundtrip(self):
        spec = TopologyStepSpec(
            resource_name="management",
            action="pathway",
            target="configure_bridge_with_stp",
            loop_target_count=0,
        )
        d = spec.to_dict()
        restored = TopologyStepSpec.from_dict(d)
        assert restored.resource_name == "management"
        assert restored.action == "pathway"
        assert restored.target == "configure_bridge_with_stp"

    def test_loop_count_preserved(self):
        spec = TopologyStepSpec(
            resource_name="vm_traffic",
            action="loop_gene",
            target="vlan_create",
            loop_target_count=5,
        )
        d = spec.to_dict()
        restored = TopologyStepSpec.from_dict(d)
        assert restored.loop_target_count == 5


class TestComputeTopologyStructureSha:
    """Structure SHA computation."""

    def test_deterministic(self):
        steps = [
            TopologyStepSpec("mgmt", "pathway", "pw_a"),
            TopologyStepSpec("storage", "gene", "bond_create"),
        ]
        sha1 = compute_topology_structure_sha(steps)
        sha2 = compute_topology_structure_sha(steps)
        assert sha1 == sha2

    def test_order_matters(self):
        steps_a = [
            TopologyStepSpec("mgmt", "pathway", "pw_a"),
            TopologyStepSpec("storage", "gene", "bond_create"),
        ]
        steps_b = [
            TopologyStepSpec("storage", "gene", "bond_create"),
            TopologyStepSpec("mgmt", "pathway", "pw_a"),
        ]
        assert compute_topology_structure_sha(steps_a) != compute_topology_structure_sha(steps_b)

    def test_ignores_loop_count(self):
        steps_a = [TopologyStepSpec("vms", "loop_gene", "vlan_create", loop_target_count=3)]
        steps_b = [TopologyStepSpec("vms", "loop_gene", "vlan_create", loop_target_count=10)]
        assert compute_topology_structure_sha(steps_a) == compute_topology_structure_sha(steps_b)

    def test_empty_steps(self):
        sha = compute_topology_structure_sha([])
        assert isinstance(sha, str) and len(sha) == 64


class TestTopologyAllele:
    """TopologyAllele serialization and properties."""

    def test_to_dict_roundtrip(self):
        allele = TopologyAllele(
            structure_sha="abc123",
            topology_name="production_server",
            steps=[TopologyStepSpec("mgmt", "pathway", "pw_a")],
            fitness=0.85,
            total_executions=100,
            successful_executions=85,
            consecutive_failures=0,
            state="dominant",
        )
        d = allele.to_dict()
        restored = TopologyAllele.from_dict(d)
        assert restored.structure_sha == "abc123"
        assert restored.topology_name == "production_server"
        assert len(restored.steps) == 1
        assert restored.fitness == 0.85
        assert restored.state == "dominant"

    def test_failed_executions_property(self):
        allele = TopologyAllele(
            structure_sha="x", topology_name="t",
            steps=[], total_executions=20, successful_executions=15,
        )
        assert allele.failed_executions == 5


class TestTopologyRegistry:
    """Registry operations."""

    def test_register_and_retrieve(self, tmp_path):
        reg = TopologyRegistry.open(tmp_path / "topo_reg")
        steps = [TopologyStepSpec("mgmt", "pathway", "pw_a")]
        sha = reg.register("production_server", steps)
        allele = reg.get(sha)
        assert allele is not None
        assert allele.topology_name == "production_server"
        assert len(allele.steps) == 1

    def test_register_idempotent(self, tmp_path):
        reg = TopologyRegistry.open(tmp_path / "topo_reg")
        steps = [TopologyStepSpec("mgmt", "pathway", "pw_a")]
        sha1 = reg.register("production_server", steps)
        sha2 = reg.register("production_server", steps)
        assert sha1 == sha2
        assert len(reg.alleles) == 1

    def test_get_for_topology_sorted(self, tmp_path):
        reg = TopologyRegistry.open(tmp_path / "topo_reg")
        steps_a = [TopologyStepSpec("mgmt", "pathway", "pw_a")]
        steps_b = [TopologyStepSpec("mgmt", "gene", "bridge_create")]
        sha_a = reg.register("topo", steps_a)
        sha_b = reg.register("topo", steps_b)
        reg.get(sha_a).fitness = 0.5
        reg.get(sha_b).fitness = 0.9
        alleles = reg.get_for_topology("topo")
        assert alleles[0].structure_sha == sha_b

    def test_get_for_topology_empty(self, tmp_path):
        reg = TopologyRegistry.open(tmp_path / "topo_reg")
        assert reg.get_for_topology("unknown") == []

    def test_save_load_index(self, tmp_path):
        reg = TopologyRegistry.open(tmp_path / "topo_reg")
        steps = [TopologyStepSpec("mgmt", "pathway", "pw_a")]
        sha = reg.register("production_server", steps)
        reg.save_index()

        reg2 = TopologyRegistry.open(tmp_path / "topo_reg")
        allele = reg2.get(sha)
        assert allele is not None
        assert allele.topology_name == "production_server"

    def test_register_with_parent(self, tmp_path):
        reg = TopologyRegistry.open(tmp_path / "topo_reg")
        steps_a = [TopologyStepSpec("mgmt", "pathway", "pw_a")]
        sha_a = reg.register("topo", steps_a)
        steps_b = [TopologyStepSpec("mgmt", "gene", "bridge_create")]
        sha_b = reg.register("topo", steps_b, parent_sha=sha_a, mutation_operator="reorder")
        allele = reg.get(sha_b)
        assert allele.parent_sha == sha_a
        assert allele.mutation_operator == "reorder"


class TestTopologyArena:
    """Arena fitness, promotion, demotion."""

    def test_zero_executions(self):
        allele = TopologyAllele(structure_sha="x", topology_name="t", steps=[])
        assert compute_topology_fitness(allele) == 0.0

    def test_with_executions(self):
        allele = TopologyAllele(
            structure_sha="x", topology_name="t", steps=[],
            total_executions=100, successful_executions=80,
        )
        assert compute_topology_fitness(allele) == pytest.approx(0.8)

    def test_min_denominator_30(self):
        allele = TopologyAllele(
            structure_sha="x", topology_name="t", steps=[],
            total_executions=10, successful_executions=10,
        )
        # 10/30 = 0.333...
        assert compute_topology_fitness(allele) == pytest.approx(10 / 30)

    def test_record_success_resets_failures(self):
        allele = TopologyAllele(
            structure_sha="x", topology_name="t", steps=[],
            consecutive_failures=3,
        )
        record_topology_success(allele)
        assert allele.total_executions == 1
        assert allele.successful_executions == 1
        assert allele.consecutive_failures == 0

    def test_record_failure_increments(self):
        allele = TopologyAllele(structure_sha="x", topology_name="t", steps=[])
        record_topology_failure(allele)
        assert allele.total_executions == 1
        assert allele.consecutive_failures == 1
        assert allele.successful_executions == 0

    def test_insufficient_executions_no_promote(self):
        candidate = TopologyAllele(
            structure_sha="x", topology_name="t", steps=[],
            total_executions=TOPOLOGY_PROMOTION_MIN_EXECUTIONS - 1,
            successful_executions=TOPOLOGY_PROMOTION_MIN_EXECUTIONS - 1,
        )
        assert not should_promote_topology(candidate, None)

    def test_promote_no_dominant(self):
        candidate = TopologyAllele(
            structure_sha="x", topology_name="t", steps=[],
            total_executions=TOPOLOGY_PROMOTION_MIN_EXECUTIONS,
            successful_executions=TOPOLOGY_PROMOTION_MIN_EXECUTIONS,
        )
        assert should_promote_topology(candidate, None)

    def test_promote_over_dominant(self):
        dominant = TopologyAllele(
            structure_sha="a", topology_name="t", steps=[],
            total_executions=600, successful_executions=360,  # fitness = 0.6
        )
        candidate = TopologyAllele(
            structure_sha="b", topology_name="t", steps=[],
            total_executions=600, successful_executions=480 + 1,  # fitness > 0.8 = 0.6 + 0.20
        )
        assert should_promote_topology(candidate, dominant)

    def test_insufficient_advantage(self):
        dominant = TopologyAllele(
            structure_sha="a", topology_name="t", steps=[],
            total_executions=600, successful_executions=360,  # fitness = 0.6
        )
        # Need >= 0.8 (0.6 + 0.20), but 474/600 = 0.79
        candidate = TopologyAllele(
            structure_sha="b", topology_name="t", steps=[],
            total_executions=600, successful_executions=474,
        )
        assert not should_promote_topology(candidate, dominant)

    def test_demote_at_8_failures(self):
        allele = TopologyAllele(
            structure_sha="x", topology_name="t", steps=[],
            consecutive_failures=TOPOLOGY_DEMOTION_CONSECUTIVE_FAILURES,
        )
        assert should_demote_topology(allele)

    def test_no_demote_at_7_failures(self):
        allele = TopologyAllele(
            structure_sha="x", topology_name="t", steps=[],
            consecutive_failures=TOPOLOGY_DEMOTION_CONSECUTIVE_FAILURES - 1,
        )
        assert not should_demote_topology(allele)

    def test_set_states(self):
        allele = TopologyAllele(structure_sha="x", topology_name="t", steps=[])
        set_topology_dominant(allele)
        assert allele.state == "dominant"
        set_topology_recessive(allele)
        assert allele.state == "recessive"
        set_topology_deprecated(allele)
        assert allele.state == "deprecated"


class TestStepsFromDecomposition:
    """Convert runtime TopologyStep to TopologyStepSpec."""

    def test_converts_topology_steps(self):
        runtime_steps = [
            TopologyStep(
                resource_name="management",
                action="pathway",
                target="configure_bridge_with_stp",
                input_json='{"bridge": "br0"}',
            ),
            TopologyStep(
                resource_name="vms",
                action="loop_gene",
                target="vlan_create",
                input_json="",
                loop_items=['{"id": 100}', '{"id": 200}'],
            ),
        ]
        specs = steps_from_decomposition(runtime_steps)
        assert len(specs) == 2
        assert specs[0].resource_name == "management"
        assert specs[0].action == "pathway"
        assert specs[0].target == "configure_bridge_with_stp"
        assert specs[1].loop_target_count == 2


class TestTopologyPhenotype:
    """Phenotype topology allele operations."""

    def test_promote_topology(self):
        pm = PhenotypeMap()
        pm.promote_topology("production_server", "sha1")
        assert pm.get_topology_dominant("production_server") == "sha1"
        assert pm.get_topology_stack("production_server") == ["sha1"]

        pm.promote_topology("production_server", "sha2")
        assert pm.get_topology_dominant("production_server") == "sha2"
        assert pm.get_topology_stack("production_server") == ["sha2", "sha1"]

    def test_fallback_topology(self):
        pm = PhenotypeMap()
        pm.add_topology_fallback("topo", "sha_fb")
        assert pm.get_topology_stack("topo") == ["sha_fb"]

    def test_save_load_topology_alleles(self, tmp_path):
        pm = PhenotypeMap()
        pm.promote_topology("production_server", "sha1")
        pm.promote_topology("production_server", "sha2")
        path = tmp_path / "phenotype.toml"
        pm.save(path)

        pm2 = PhenotypeMap.load(path)
        assert pm2.get_topology_dominant("production_server") == "sha2"
        assert pm2.get_topology_stack("production_server") == ["sha2", "sha1"]


class TestTopologyEvolutionIntegration:
    """Topology allele stack through orchestrator."""

    @pytest.fixture
    def full_project(self, tmp_path):
        shutil.copytree(FIXTURES_DIR, tmp_path / "fixtures")
        shutil.copytree(CONTRACTS_DIR, tmp_path / "contracts")
        (tmp_path / ".sg").mkdir(exist_ok=True)
        contract_store = ContractStore.open(tmp_path / "contracts")
        registry = Registry.open(tmp_path / ".sg" / "registry")
        phenotype = PhenotypeMap()
        for locus in contract_store.known_loci():
            candidates = sorted(GENES_DIR.glob(f"{locus}_*.py"))
            if candidates:
                source = candidates[0].read_text()
                sha = registry.register(source, locus)
                phenotype.promote(locus, sha)
                registry.get(sha).state = "dominant"
        registry.save_index()
        phenotype.save(tmp_path / "phenotype.toml")
        return tmp_path

    def _make_orchestrator(self, project_root):
        from sg.parser.types import GeneFamily
        contract_store = ContractStore.open(project_root / "contracts")
        for locus in contract_store.known_loci():
            gc = contract_store.get_gene(locus)
            if gc and gc.family == GeneFamily.CONFIGURATION and gc.verify_within:
                gc.verify_within = "0.01s"
        registry = Registry.open(project_root / ".sg" / "registry")
        phenotype = PhenotypeMap.load(project_root / "phenotype.toml")
        fusion_tracker = FusionTracker.open(project_root / "fusion_tracker.json")
        pft = PathwayFitnessTracker.open(project_root / "pathway_fitness.json")
        pr = PathwayRegistry.open(project_root / ".sg" / "pathway_registry")
        tr = TopologyRegistry.open(project_root / ".sg" / "topology_registry")
        mutation_engine = MockMutationEngine(project_root / "fixtures")
        kernel = MockNetworkKernel()
        return Orchestrator(
            registry=registry, phenotype=phenotype,
            mutation_engine=mutation_engine, fusion_tracker=fusion_tracker,
            kernel=kernel, contract_store=contract_store,
            project_root=project_root,
            pathway_fitness_tracker=pft, pathway_registry=pr,
            topology_registry=tr,
        )

    def _topology_input(self):
        import json
        return json.dumps({
            "bridge_name": "br0",
            "bridge_ifaces": ["eth0"],
            "uplink": "eth1",
            "bond_name": "bond0",
            "bond_mode": "active-backup",
            "bond_members": ["eth2", "eth3"],
            "vlans": [100, 200],
        })

    def test_topology_registers_allele_on_first_run(self, full_project):
        orch = self._make_orchestrator(full_project)
        topology_name = "production_server"

        orch.run_topology(topology_name, self._topology_input())

        # Allele should be registered and set as dominant
        stack = orch.phenotype.get_topology_stack(topology_name)
        assert len(stack) >= 1
        allele = orch.topology_registry.get(stack[0])
        assert allele is not None
        assert allele.state == "dominant"
        assert allele.total_executions == 1
        assert allele.successful_executions == 1

    def test_topology_records_success(self, full_project):
        orch = self._make_orchestrator(full_project)
        topology_name = "production_server"

        orch.run_topology(topology_name, self._topology_input())
        orch.run_topology(topology_name, self._topology_input())

        stack = orch.phenotype.get_topology_stack(topology_name)
        allele = orch.topology_registry.get(stack[0])
        assert allele.total_executions == 2
        assert allele.successful_executions == 2
