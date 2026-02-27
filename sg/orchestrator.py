"""Core execution loop — the evolutionary engine.

execute → validate → score → fallback → mutate → register → retry
"""
from __future__ import annotations

from pathlib import Path

from sg import arena
from sg.contracts import validate_output, contract_info
from sg.fusion import FusionTracker
from sg.kernel.base import NetworkKernel
from sg.loader import load_gene, call_gene
from sg.mutation import MutationEngine, MutationContext
from sg.pathway import Pathway, execute_pathway, PATHWAYS
from sg.phenotype import PhenotypeMap
from sg.registry import Registry


MAX_MUTATION_RETRIES = 3


class Orchestrator:
    def __init__(
        self,
        registry: Registry,
        phenotype: PhenotypeMap,
        mutation_engine: MutationEngine,
        fusion_tracker: FusionTracker,
        kernel: NetworkKernel,
        project_root: Path,
    ):
        self.registry = registry
        self.phenotype = phenotype
        self.mutation_engine = mutation_engine
        self.fusion_tracker = fusion_tracker
        self.kernel = kernel
        self.project_root = project_root

    def execute_locus(self, locus: str, input_json: str) -> tuple[str, str] | None:
        """Execute a locus with the allele stack.

        Returns (output_json, used_sha) on success, or None if all exhausted.
        Triggers mutation on exhaustion.
        """
        stack = self.phenotype.get_stack(locus)
        last_error = ""

        for sha in stack:
            source = self.registry.load_source(sha)
            if source is None:
                continue

            allele = self.registry.get(sha)
            if allele is None:
                continue

            try:
                execute_fn = load_gene(source, self.kernel)
                result = call_gene(execute_fn, input_json)

                if not validate_output(locus, result):
                    raise RuntimeError(f"output validation failed for {locus}")

                arena.record_success(allele)
                self._check_promotion(locus, sha)
                print(f"  [{locus}] success via {sha[:12]} "
                      f"(fitness: {arena.compute_fitness(allele):.2f})")
                return (result, sha)

            except Exception as e:
                last_error = str(e)
                arena.record_failure(allele)
                self._check_demotion(locus, sha)
                print(f"  [{locus}] failed via {sha[:12]}: {e}")
                continue

        print(f"  [{locus}] all alleles exhausted, triggering mutation...")
        mutated = self._try_mutation(locus, input_json, last_error)
        if mutated is not None:
            return mutated

        return None

    def _try_mutation(
        self, locus: str, input_json: str, error: str
    ) -> tuple[str, str] | None:
        dominant_sha = self.phenotype.get_dominant(locus)
        gene_source = ""
        if dominant_sha:
            gene_source = self.registry.load_source(dominant_sha) or ""

        ctx = MutationContext(
            gene_source=gene_source,
            locus=locus,
            failing_input=input_json,
            error_message=error,
        )

        for attempt in range(MAX_MUTATION_RETRIES):
            try:
                new_source = self.mutation_engine.mutate(ctx)
            except Exception as e:
                print(f"  [mutation] generation failed (attempt {attempt + 1}): {e}")
                continue

            parent_sha = dominant_sha
            gen = 0
            if parent_sha:
                parent = self.registry.get(parent_sha)
                if parent:
                    gen = parent.generation + 1

            new_sha = self.registry.register(new_source, locus, gen, parent_sha)
            self.phenotype.add_to_fallback(locus, new_sha)

            try:
                execute_fn = load_gene(new_source, self.kernel)
                result = call_gene(execute_fn, input_json)

                if not validate_output(locus, result):
                    raise RuntimeError("output validation failed")

                allele = self.registry.get(new_sha)
                if allele:
                    arena.record_success(allele)
                print(f"  [mutation] mutant {new_sha[:12]} succeeded (attempt {attempt + 1})")
                return (result, new_sha)

            except Exception as e:
                allele = self.registry.get(new_sha)
                if allele:
                    arena.record_failure(allele)
                print(f"  [mutation] mutant {new_sha[:12]} failed: {e} (attempt {attempt + 1})")
                continue

        print(f"  [mutation] all {MAX_MUTATION_RETRIES} attempts failed for {locus}")
        return None

    def _check_promotion(self, locus: str, sha: str) -> None:
        allele = self.registry.get(sha)
        dominant_sha = self.phenotype.get_dominant(locus)
        dominant = self.registry.get(dominant_sha) if dominant_sha else None

        if allele and arena.should_promote(allele, dominant):
            arena.set_dominant(allele)
            if dominant:
                arena.set_recessive(dominant)
            self.phenotype.promote(locus, sha)
            print(f"  [arena] promoted {sha[:12]} to dominant for {locus}")

    def _check_demotion(self, locus: str, sha: str) -> None:
        allele = self.registry.get(sha)
        if allele and arena.should_demote(allele):
            arena.set_deprecated(allele)
            print(f"  [arena] demoted {sha[:12]} for {locus} (3 consecutive failures)")

    def run_pathway(self, pathway_name: str, input_json: str) -> list[str]:
        """Execute a named pathway."""
        pathway = PATHWAYS.get(pathway_name)
        if pathway is None:
            raise ValueError(f"unknown pathway: {pathway_name}")

        self.kernel.reset()
        print(f"Executing pathway: {pathway_name}")

        outputs = execute_pathway(
            pathway, input_json, self,
            self.fusion_tracker, self.registry,
            self.phenotype, self.mutation_engine,
            self.kernel,
        )

        print(f"Pathway '{pathway_name}' completed with {len(outputs)} output(s)")
        return outputs

    def save_state(self) -> None:
        self.registry.save_index()
        phenotype_path = self.project_root / "phenotype.toml"
        self.phenotype.save(phenotype_path)
        tracker_path = self.project_root / "fusion_tracker.json"
        self.fusion_tracker.save(tracker_path)
