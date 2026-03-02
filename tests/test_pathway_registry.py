"""Tests for pathway allele registry (Phase 4)."""
import pytest
from pathlib import Path

from sg.pathway_registry import (
    StepSpec,
    PathwayAllele,
    PathwayRegistry,
    compute_structure_sha,
    steps_from_pathway,
)


class TestStepSpec:
    def test_to_dict_roundtrip(self):
        spec = StepSpec(
            step_type="locus", target="bridge_create",
            params={"name": "{bridge_name}"},
        )
        d = spec.to_dict()
        restored = StepSpec.from_dict(d)
        assert restored.step_type == "locus"
        assert restored.target == "bridge_create"
        assert restored.params == {"name": "{bridge_name}"}

    def test_omits_none(self):
        spec = StepSpec(step_type="locus", target="test")
        d = spec.to_dict()
        assert "loop_variable" not in d
        assert "condition_step_index" not in d
        assert "branches" not in d
        assert "params" not in d  # empty dict omitted


class TestComputeStructureSha:
    def test_deterministic(self):
        steps = [
            StepSpec(step_type="locus", target="a"),
            StepSpec(step_type="locus", target="b"),
        ]
        sha1 = compute_structure_sha(steps)
        sha2 = compute_structure_sha(steps)
        assert sha1 == sha2
        assert len(sha1) == 64  # SHA-256 hex

    def test_order_matters(self):
        steps_ab = [
            StepSpec(step_type="locus", target="a"),
            StepSpec(step_type="locus", target="b"),
        ]
        steps_ba = [
            StepSpec(step_type="locus", target="b"),
            StepSpec(step_type="locus", target="a"),
        ]
        assert compute_structure_sha(steps_ab) != compute_structure_sha(steps_ba)

    def test_ignores_params(self):
        steps1 = [StepSpec(step_type="locus", target="a", params={"x": "1"})]
        steps2 = [StepSpec(step_type="locus", target="a", params={"y": "2"})]
        steps3 = [StepSpec(step_type="locus", target="a")]
        sha1 = compute_structure_sha(steps1)
        sha2 = compute_structure_sha(steps2)
        sha3 = compute_structure_sha(steps3)
        assert sha1 == sha2 == sha3

    def test_conditional_branches_affect_sha(self):
        steps1 = [StepSpec(
            step_type="conditional", target="",
            condition_step_index=0, condition_field="status",
            branches={"ok": {"step_type": "locus", "target": "a"}},
        )]
        steps2 = [StepSpec(
            step_type="conditional", target="",
            condition_step_index=0, condition_field="status",
            branches={"ok": {"step_type": "locus", "target": "b"}},
        )]
        assert compute_structure_sha(steps1) != compute_structure_sha(steps2)

    def test_loop_structure_affects_sha(self):
        steps1 = [StepSpec(
            step_type="loop", target="body",
            loop_variable="item", loop_iterable="items",
        )]
        steps2 = [StepSpec(
            step_type="loop", target="body",
            loop_variable="elem", loop_iterable="elements",
        )]
        assert compute_structure_sha(steps1) != compute_structure_sha(steps2)


class TestComputeStructureShaBoundary:
    def test_sha_empty_steps(self):
        """Empty step list produces a consistent hash."""
        sha1 = compute_structure_sha([])
        sha2 = compute_structure_sha([])
        assert sha1 == sha2
        assert len(sha1) == 64

    def test_sha_nested_conditional(self):
        """Conditional inside branches affects SHA differently."""
        steps1 = [StepSpec(
            step_type="conditional", target="",
            condition_step_index=0, condition_field="mode",
            branches={
                "a": {"step_type": "conditional", "target": "",
                       "condition_field": "sub", "branches": {}},
            },
        )]
        steps2 = [StepSpec(
            step_type="conditional", target="",
            condition_step_index=0, condition_field="mode",
            branches={
                "a": {"step_type": "locus", "target": "x"},
            },
        )]
        assert compute_structure_sha(steps1) != compute_structure_sha(steps2)

    def test_allele_created_at_populated(self):
        """PathwayAllele.created_at is set automatically to current time."""
        import time
        before = time.time()
        allele = PathwayAllele(
            structure_sha="test", pathway_name="pw", steps=[],
        )
        after = time.time()
        assert before <= allele.created_at <= after


class TestPathwayAllele:
    def test_to_dict_roundtrip(self):
        steps = [StepSpec(step_type="locus", target="a")]
        allele = PathwayAllele(
            structure_sha="abc123",
            pathway_name="test_pw",
            steps=steps,
            fitness=0.8,
            total_executions=100,
            successful_executions=80,
            state="dominant",
            parent_sha="parent123",
            mutation_operator="step_reorder",
        )
        d = allele.to_dict()
        restored = PathwayAllele.from_dict(d)
        assert restored.structure_sha == "abc123"
        assert restored.pathway_name == "test_pw"
        assert len(restored.steps) == 1
        assert restored.fitness == 0.8
        assert restored.state == "dominant"
        assert restored.parent_sha == "parent123"
        assert restored.mutation_operator == "step_reorder"

    def test_failed_executions_property(self):
        allele = PathwayAllele(
            structure_sha="x", pathway_name="pw",
            steps=[], total_executions=10, successful_executions=7,
        )
        assert allele.failed_executions == 3


class TestPathwayRegistry:
    def test_register_and_retrieve(self, tmp_path):
        reg = PathwayRegistry.open(tmp_path / "pr")
        steps = [StepSpec(step_type="locus", target="a")]
        sha = reg.register("test_pw", steps)
        allele = reg.get(sha)
        assert allele is not None
        assert allele.pathway_name == "test_pw"
        assert len(allele.steps) == 1

    def test_register_idempotent(self, tmp_path):
        reg = PathwayRegistry.open(tmp_path / "pr")
        steps = [StepSpec(step_type="locus", target="a")]
        sha1 = reg.register("test_pw", steps)
        sha2 = reg.register("test_pw", steps)
        assert sha1 == sha2
        assert len(reg.get_for_pathway("test_pw")) == 1

    def test_get_for_pathway_sorted(self, tmp_path):
        reg = PathwayRegistry.open(tmp_path / "pr")
        steps_a = [StepSpec(step_type="locus", target="a")]
        steps_b = [StepSpec(step_type="locus", target="b")]
        sha_a = reg.register("pw", steps_a)
        sha_b = reg.register("pw", steps_b)
        # Set different fitness
        reg.get(sha_a).fitness = 0.5
        reg.get(sha_b).fitness = 0.9
        alleles = reg.get_for_pathway("pw")
        assert alleles[0].structure_sha == sha_b  # higher fitness first

    def test_get_for_pathway_empty(self, tmp_path):
        reg = PathwayRegistry.open(tmp_path / "pr")
        assert reg.get_for_pathway("nonexistent") == []

    def test_save_load_index(self, tmp_path):
        reg = PathwayRegistry.open(tmp_path / "pr")
        steps = [StepSpec(step_type="locus", target="a")]
        sha = reg.register("pw", steps, parent_sha="parent1", mutation_operator="reorder")
        reg.get(sha).fitness = 0.75
        reg.get(sha).total_executions = 50
        reg.save_index()

        reg2 = PathwayRegistry.open(tmp_path / "pr")
        allele = reg2.get(sha)
        assert allele is not None
        assert allele.pathway_name == "pw"
        assert allele.fitness == 0.75
        assert allele.total_executions == 50
        assert allele.parent_sha == "parent1"
        assert allele.mutation_operator == "reorder"

    def test_open_nonexistent(self, tmp_path):
        reg = PathwayRegistry.open(tmp_path / "new_dir" / "pr")
        assert len(reg.alleles) == 0

    def test_register_with_parent(self, tmp_path):
        reg = PathwayRegistry.open(tmp_path / "pr")
        steps = [StepSpec(step_type="locus", target="a")]
        sha = reg.register("pw", steps, parent_sha="p1", mutation_operator="insert")
        allele = reg.get(sha)
        assert allele.parent_sha == "p1"
        assert allele.mutation_operator == "insert"


class TestStepsFromPathway:
    def test_locus_step(self):
        from sg.pathway import PathwayStep, Pathway
        pathway = Pathway(name="test", steps=[
            PathwayStep(locus="bridge_create", input_transform=lambda x, y: x),
        ])
        specs = steps_from_pathway(pathway)
        assert len(specs) == 1
        assert specs[0].step_type == "locus"
        assert specs[0].target == "bridge_create"

    def test_composed_step(self):
        from sg.pathway import ComposedStep, Pathway
        pathway = Pathway(name="test", steps=[
            ComposedStep(pathway_name="sub_pw", input_transform=lambda x, y: x),
        ])
        specs = steps_from_pathway(pathway)
        assert len(specs) == 1
        assert specs[0].step_type == "composed"
        assert specs[0].target == "sub_pw"

    def test_loop_step(self):
        from sg.pathway import LoopStep, Pathway
        pathway = Pathway(name="test", steps=[
            LoopStep(variable="iface", iterable_field="interfaces",
                     body_locus="configure_iface"),
        ])
        specs = steps_from_pathway(pathway)
        assert len(specs) == 1
        assert specs[0].step_type == "loop"
        assert specs[0].target == "configure_iface"
        assert specs[0].loop_variable == "iface"
        assert specs[0].loop_iterable == "interfaces"

    def test_conditional_step(self):
        from sg.pathway import (
            PathwayStep, ConditionalExecStep, Pathway,
        )
        pathway = Pathway(name="test", steps=[
            ConditionalExecStep(
                condition_step_index=0,
                condition_field="mode",
                branches={
                    "fast": PathwayStep(locus="fast_path",
                                        input_transform=lambda x, y: x),
                    "slow": PathwayStep(locus="slow_path",
                                        input_transform=lambda x, y: x),
                },
            ),
        ])
        specs = steps_from_pathway(pathway)
        assert len(specs) == 1
        assert specs[0].step_type == "conditional"
        assert specs[0].condition_field == "mode"
        assert "fast" in specs[0].branches
        assert specs[0].branches["fast"]["target"] == "fast_path"
