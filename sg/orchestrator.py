"""Core execution loop — the evolutionary engine.

execute → validate → score → fallback → mutate → register → retry
"""
from __future__ import annotations

import json
from pathlib import Path

from sg import arena
from sg.contracts import ContractStore, validate_output
from sg.fitness import record_feedback
from sg.fusion import FusionTracker
from sg.kernel.base import NetworkKernel
from sg.loader import load_gene, call_gene
from sg.mutation import MutationEngine, MutationContext
from sg.parser.types import BlastRadius
from sg.pathway import Pathway, PathwayStep, execute_pathway, pathway_from_contract
from sg.phenotype import PhenotypeMap
from sg.registry import Registry
from sg.safety import Transaction, SafeKernel, requires_transaction, is_shadow_only, SHADOW_PROMOTION_THRESHOLD
from sg.verify import VerifyScheduler, parse_duration


MAX_MUTATION_RETRIES = 3


class Orchestrator:
    def __init__(
        self,
        registry: Registry,
        phenotype: PhenotypeMap,
        mutation_engine: MutationEngine,
        fusion_tracker: FusionTracker,
        kernel: NetworkKernel,
        contract_store: ContractStore,
        project_root: Path,
    ):
        self.registry = registry
        self.phenotype = phenotype
        self.mutation_engine = mutation_engine
        self.fusion_tracker = fusion_tracker
        self.kernel = kernel
        self.contract_store = contract_store
        self.project_root = project_root
        self.verify_scheduler = VerifyScheduler()
        self.feedback_timescale: str | None = None  # override feeds timescale (e.g. "resilience")

    def _get_risk(self, locus: str) -> BlastRadius:
        """Get the blast radius for a locus from its contract."""
        gene_contract = self.contract_store.get_gene(locus)
        if gene_contract is not None:
            return gene_contract.risk
        return BlastRadius.LOW  # default: wrap in transaction

    def execute_locus(self, locus: str, input_json: str) -> tuple[str, str] | None:
        """Execute a locus with the allele stack.

        Returns (output_json, used_sha) on success, or None if all exhausted.
        Triggers mutation on exhaustion. Wraps each allele attempt in a
        transaction — on failure, all kernel mutations are rolled back.
        """
        stack = self.phenotype.get_stack(locus)
        last_error = ""
        risk = self._get_risk(locus)
        use_txn = requires_transaction(risk)
        use_shadow = is_shadow_only(risk)

        for sha in stack:
            source = self.registry.load_source(sha)
            if source is None:
                continue

            allele = self.registry.get(sha)
            if allele is None:
                continue

            # Shadow mode: HIGH/CRITICAL risk alleles must accumulate shadow
            # successes against a mock kernel before live execution is allowed.
            if use_shadow and allele.shadow_successes < SHADOW_PROMOTION_THRESHOLD:
                result = self._try_shadow_execution(
                    locus, sha, source, allele, input_json
                )
                if result is not None:
                    return result
                continue

            txn = Transaction(locus, risk) if use_txn else None
            kernel = SafeKernel(self.kernel, txn) if txn else self.kernel

            try:
                execute_fn = load_gene(source, kernel)
                result = call_gene(execute_fn, input_json)

                if not validate_output(locus, result):
                    raise RuntimeError(f"output validation failed for {locus}")

                if txn:
                    txn.commit()
                arena.record_success(allele)
                self._process_diagnostic_feedback(locus, result)
                self._schedule_verify(locus, input_json)
                self._check_promotion(locus, sha)
                print(f"  [{locus}] success via {sha[:12]} "
                      f"(fitness: {arena.compute_fitness(allele):.2f})")
                return (result, sha)

            except Exception as e:
                if txn and txn.action_count > 0:
                    rolled = txn.rollback()
                    if rolled:
                        print(f"  [{locus}] rolled back {len(rolled)} action(s)")
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

        risk = self._get_risk(locus)
        use_txn = requires_transaction(risk)

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

            txn = Transaction(locus, risk) if use_txn else None
            kernel = SafeKernel(self.kernel, txn) if txn else self.kernel

            try:
                execute_fn = load_gene(new_source, kernel)
                result = call_gene(execute_fn, input_json)

                if not validate_output(locus, result):
                    raise RuntimeError("output validation failed")

                if txn:
                    txn.commit()
                allele = self.registry.get(new_sha)
                if allele:
                    arena.record_success(allele)
                print(f"  [mutation] mutant {new_sha[:12]} succeeded (attempt {attempt + 1})")
                return (result, new_sha)

            except Exception as e:
                if txn and txn.action_count > 0:
                    rolled = txn.rollback()
                    if rolled:
                        print(f"  [mutation] rolled back {len(rolled)} action(s)")
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

    def _process_diagnostic_feedback(self, locus: str, output_json: str) -> None:
        """If this locus is a diagnostic gene with feeds, feed results back."""
        gene_contract = self.contract_store.get_gene(locus)
        if gene_contract is None:
            return
        if not gene_contract.feeds:
            return

        # Parse the diagnostic output for health status
        try:
            data = json.loads(output_json)
        except (json.JSONDecodeError, TypeError):
            return

        healthy = data.get("healthy")
        if not isinstance(healthy, bool):
            return

        # Determine timescale from feeds declarations (or override)
        for feed in gene_contract.feeds:
            target_locus = feed.target_locus
            timescale = self.feedback_timescale or feed.timescale

            # Find the dominant allele at the target config locus
            dominant_sha = self.phenotype.get_dominant(target_locus)
            if dominant_sha is None:
                continue
            target_allele = self.registry.get(dominant_sha)
            if target_allele is None:
                continue

            record_feedback(target_allele, timescale, healthy, locus)
            fitness = arena.compute_fitness(target_allele)
            print(f"  [feedback] {locus} → {target_locus} "
                  f"({timescale}: {'healthy' if healthy else 'unhealthy'}, "
                  f"fitness: {fitness:.2f})")

    def _schedule_verify(self, locus: str, input_json: str) -> None:
        """Schedule verify diagnostics if the locus contract declares a verify block."""
        from sg.parser.types import GeneFamily

        gene_contract = self.contract_store.get_gene(locus)
        if gene_contract is None:
            return
        if gene_contract.family != GeneFamily.CONFIGURATION:
            return
        if not gene_contract.verify:
            return

        delay = 0.0
        if gene_contract.verify_within:
            try:
                delay = parse_duration(gene_contract.verify_within)
            except ValueError:
                return

        self.verify_scheduler.schedule(
            gene_contract.verify, delay, input_json, self
        )

    def _try_shadow_execution(
        self, locus: str, sha: str, source: str,
        allele: object, input_json: str,
    ) -> tuple[str, str] | None:
        """Run a gene against a mock kernel (shadow mode).

        Shadow results are returned to the caller but the real kernel
        is untouched. Accumulates shadow_successes on the allele; once
        the threshold is met, subsequent calls use the real kernel.
        """
        from sg.kernel.mock import MockNetworkKernel

        shadow_kernel = MockNetworkKernel()
        try:
            execute_fn = load_gene(source, shadow_kernel)
            result = call_gene(execute_fn, input_json)

            if not validate_output(locus, result):
                raise RuntimeError(f"shadow output validation failed for {locus}")

            allele.shadow_successes += 1
            remaining = SHADOW_PROMOTION_THRESHOLD - allele.shadow_successes
            if remaining > 0:
                print(f"  [{locus}] shadow success via {sha[:12]} "
                      f"({allele.shadow_successes}/{SHADOW_PROMOTION_THRESHOLD})")
            else:
                print(f"  [{locus}] shadow threshold met for {sha[:12]} "
                      f"— eligible for live execution")
            return (result, sha)

        except Exception as e:
            allele.shadow_successes = 0  # reset on failure
            print(f"  [{locus}] shadow failed via {sha[:12]}: {e}")
            return None

    def _check_demotion(self, locus: str, sha: str) -> None:
        allele = self.registry.get(sha)
        if allele and arena.should_demote(allele):
            arena.set_deprecated(allele)
            print(f"  [arena] demoted {sha[:12]} for {locus} (3 consecutive failures)")

    def run_pathway(self, pathway_name: str, input_json: str) -> list[str]:
        """Execute a named pathway.

        Handles on_failure strategy from the pathway contract:
        - "rollback all": clean up all tracked resources on failure
        - "report partial": let partial results stand
        """
        pathway_contract = self.contract_store.get_pathway(pathway_name)
        if pathway_contract is not None:
            pathway = pathway_from_contract(pathway_contract)
            on_failure = pathway_contract.on_failure
        else:
            raise ValueError(f"unknown pathway: {pathway_name}")

        print(f"Executing pathway: {pathway_name}")

        # Snapshot tracked resources before pathway starts
        resources_before = set(self.kernel.tracked_resources())

        try:
            outputs = execute_pathway(
                pathway, input_json, self,
                self.fusion_tracker, self.registry,
                self.phenotype, self.mutation_engine,
                self.kernel,
            )
        except RuntimeError:
            if on_failure == "rollback all":
                self._rollback_pathway_resources(resources_before)
            raise

        print(f"Pathway '{pathway_name}' completed with {len(outputs)} output(s)")
        return outputs

    def _rollback_pathway_resources(
        self, resources_before: set[tuple[str, str]]
    ) -> None:
        """Clean up resources created during a failed pathway."""
        current = set(self.kernel.tracked_resources())
        new_resources = current - resources_before
        if not new_resources:
            return

        print(f"  [safety] rolling back {len(new_resources)} resource(s)...")
        for resource_type, name in new_resources:
            try:
                if resource_type == "bridge":
                    self.kernel.delete_bridge(name)
                elif resource_type == "bond":
                    self.kernel.delete_bond(name)
                elif resource_type == "vlan":
                    parts = name.split(".", 1)
                    if len(parts) == 2:
                        self.kernel.delete_vlan(parts[0], int(parts[1]))
                self.kernel.untrack_resource(resource_type, name)
                print(f"  [safety] cleaned up {resource_type} '{name}'")
            except Exception as e:
                print(f"  [safety] cleanup failed for {resource_type} '{name}': {e}")

    def run_topology(self, topology_name: str, input_json: str) -> list[str]:
        """Execute a named topology by decomposing into pathway/gene calls."""
        from sg.topology import execute_topology

        topology = self.contract_store.get_topology(topology_name)
        if topology is None:
            raise ValueError(f"unknown topology: {topology_name}")

        print(f"Deploying topology: {topology_name}")
        outputs = execute_topology(topology, input_json, self)
        print(f"Topology '{topology_name}' deployed ({len(outputs)} output(s))")
        return outputs

    def save_state(self) -> None:
        self.registry.save_index()
        phenotype_path = self.project_root / "phenotype.toml"
        self.phenotype.save(phenotype_path)
        tracker_path = self.project_root / "fusion_tracker.json"
        self.fusion_tracker.save(tracker_path)
