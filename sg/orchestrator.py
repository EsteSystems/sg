"""Core execution loop — the evolutionary engine.

execute → validate → score → fallback → mutate → register → retry
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from sg import arena
from sg.audit import AuditLog
from sg.filelock import atomic_write_text, file_lock, file_lock_shared
from sg.log import get_logger, correlation_scope

logger = get_logger("orchestrator")
from sg.contracts import ContractStore, validate_output
from sg.fitness import record_feedback
from sg.fusion import FusionTracker
from sg.kernel.base import Kernel
from sg.loader import load_gene, call_gene
from sg.mutation import MutationEngine, MutationContext
from sg.parser.types import BlastRadius
from sg.pathway import Pathway, PathwayStep, execute_pathway, pathway_from_contract
from sg.phenotype import PhenotypeMap
from sg.registry import Registry
from sg.safety import Transaction, SafeKernel, requires_transaction, is_shadow_only, SHADOW_PROMOTION_THRESHOLD
from sg.contract_evolution import ContractEvolution
from sg.decomposition import DecompositionDetector
from sg.locus_discovery import CrossLocusFailureAnalyzer
from sg.pathway_fitness import PathwayFitnessTracker
from sg.pathway_registry import PathwayRegistry, steps_from_pathway, compute_structure_sha
from sg import pathway_arena
from sg.pathway_mutation import PathwayMutationThrottle
from sg.regression import RegressionDetector
from sg.verify import VerifyScheduler, parse_duration


MAX_MUTATION_RETRIES = 3


class Orchestrator:
    def __init__(
        self,
        registry: Registry,
        phenotype: PhenotypeMap,
        mutation_engine: MutationEngine,
        fusion_tracker: FusionTracker,
        kernel: Kernel,
        contract_store: ContractStore,
        project_root: Path,
        audit_log: AuditLog | None = None,
        pathway_fitness_tracker: PathwayFitnessTracker | None = None,
        pathway_registry: PathwayRegistry | None = None,
        topology_registry=None,
        meta_param_tracker=None,
        event_bus=None,
    ):
        self.registry = registry
        self.phenotype = phenotype
        self.mutation_engine = mutation_engine
        self.fusion_tracker = fusion_tracker
        self.kernel = kernel
        self.contract_store = contract_store
        self.project_root = project_root
        self.audit_log = audit_log
        self._event_bus = event_bus
        self.pathway_fitness_tracker = pathway_fitness_tracker
        self.pathway_registry = pathway_registry
        self.topology_registry = topology_registry
        self._meta_param_tracker = meta_param_tracker
        self.verify_scheduler = VerifyScheduler()
        self._regression_path = project_root / ".sg" / "regression.json"
        self.regression_detector = RegressionDetector.open(self._regression_path)
        self._decomposition_path = project_root / ".sg" / "decomposition.json"
        self.decomposition_detector = DecompositionDetector.open(self._decomposition_path)
        self.feedback_timescale: str | None = None  # override feeds timescale (e.g. "resilience")
        self._current_pathway_context: str | None = None
        self._current_pathway_structure_hash: str = ""
        self._pathway_structure_histories: dict[str, list[str]] = {}
        self._pathway_mutation_throttle = self._load_pathway_mutation_throttle()
        from sg.stabilization import StabilizationTracker
        self._stabilization_tracker = StabilizationTracker.open(
            project_root / ".sg" / "stabilization.json"
        )
        from sg.failure_discovery import FailureDiscovery
        self._failure_discovery = FailureDiscovery.open(
            project_root / ".sg" / "failure_discovery.json"
        )
        self._contract_evolution = ContractEvolution.open(
            project_root / ".sg" / "contract_evolution.json"
        )
        self._cross_locus_analyzer = CrossLocusFailureAnalyzer.open(
            project_root / ".sg" / "locus_discovery.json"
        )

    def _audit(self, event: str, locus: str = "", sha: str = "",
                **details) -> None:
        """Record an audit entry if an audit log is configured."""
        if self.audit_log is not None:
            self.audit_log.record(event, locus=locus, sha=sha, **details)

    def _publish(self, event) -> None:
        """Publish an event to the event bus if configured."""
        if self._event_bus is not None:
            self._event_bus.publish(event)

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
        logger.debug("execute_locus %s", locus, extra={"locus": locus})
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

                if not validate_output(locus, result, self.contract_store):
                    raise RuntimeError(f"output validation failed for {locus}")

                if txn:
                    txn.commit()
                arena.record_success(allele)
                self._process_diagnostic_feedback(locus, result)
                self._schedule_verify(locus, input_json)
                self._check_promotion(locus, sha)
                self._check_regression(locus, sha, input_json)
                gene_contract = self.contract_store.get_gene(locus)
                self._contract_evolution.record_output(
                    locus, result, gene_contract)
                logger.info("success via %s (fitness: %.2f)",
                            sha[:12], arena.compute_fitness(allele),
                            extra={"locus": locus, "sha": sha[:12]})
                return (result, sha)

            except Exception as e:
                if txn and txn.action_count > 0:
                    rolled = txn.rollback()
                    if rolled:
                        logger.info("rolled back %d action(s)",
                                    len(rolled), extra={"locus": locus})
                last_error = str(e)
                self.decomposition_detector.record_error(locus, sha, last_error)
                self._cross_locus_analyzer.record_error(locus, last_error)
                gene_contract = self.contract_store.get_gene(locus)
                known_fails = gene_contract.fails_when if gene_contract else []
                proposal = self._failure_discovery.record_error(
                    locus, sha, last_error, known_fails,
                )
                if proposal:
                    self._audit("failure_mode_discovered", locus=locus,
                                pattern=proposal.pattern,
                                count=proposal.occurrence_count)
                arena.record_failure(allele)
                self._check_demotion(locus, sha)
                logger.warning("failed via %s: %s", sha[:12], e,
                               extra={"locus": locus, "sha": sha[:12]})
                continue

        logger.info("all alleles exhausted, triggering mutation...",
                    extra={"locus": locus})

        # Try decomposition if error diversity warrants it
        if not self.decomposition_detector.is_decomposed(locus):
            signal = self.decomposition_detector.analyze(locus)
            if signal is not None:
                decomposed = self._try_decomposition(locus, input_json, signal)
                if decomposed is not None:
                    return decomposed

        mutated = self._try_mutation(locus, input_json, last_error)
        if mutated is not None:
            return mutated

        return None

    def _build_kernel_state(self) -> str | None:
        """Serialize tracked kernel resources for mutation context."""
        resources = self.kernel.tracked_resources()
        if not resources:
            return None
        lines = [f"  {rtype}: {name}" for rtype, name in resources]
        return "Tracked resources:\n" + "\n".join(lines)

    def _build_prior_mutations(self, locus: str) -> list[str]:
        """Summarize recent alleles at a locus for mutation context."""
        alleles = self.registry.alleles_for_locus(locus)
        summaries = []
        for allele in alleles[:3]:
            source = self.registry.load_source(allele.sha256)
            first_line = ""
            if source:
                for line in source.splitlines():
                    stripped = line.strip()
                    if stripped and not stripped.startswith("#"):
                        first_line = stripped[:80]
                        break
            fitness = arena.compute_fitness(allele)
            summaries.append(
                f"gen{allele.generation} ({allele.sha256[:8]}): "
                f"fitness={fitness:.2f}, "
                f"failures={allele.consecutive_failures}, "
                f"approach: {first_line}"
            )
        return summaries

    def _build_sibling_summaries(
        self, locus: str, exclude_sha: str | None = None,
    ) -> list[str]:
        """Brief descriptions of other alleles at a locus."""
        alleles = self.registry.alleles_for_locus(locus)
        summaries = []
        for allele in alleles[:5]:
            if allele.sha256 == exclude_sha:
                continue
            fitness = arena.compute_fitness(allele)
            summaries.append(
                f"{allele.sha256[:8]}: state={allele.state}, "
                f"fitness={fitness:.2f}, "
                f"invocations={allele.total_invocations}"
            )
        return summaries

    def _try_mutation(
        self, locus: str, input_json: str, error: str
    ) -> tuple[str, str] | None:
        """Attempt to generate a working mutant allele.

        Wrapped in an outer try/except for graceful degradation — if the
        mutation subsystem is completely unavailable (e.g. LLM API down),
        the system continues with existing alleles.
        """
        try:
            return self._try_mutation_inner(locus, input_json, error)
        except Exception as e:
            logger.warning("mutation subsystem unavailable for %s: %s",
                           locus, e, extra={"locus": locus})
            return None

    def _try_mutation_inner(
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
            kernel_state=self._build_kernel_state(),
            prior_mutations=self._build_prior_mutations(locus),
            pathway_context=self._current_pathway_context,
            sibling_summaries=self._build_sibling_summaries(
                locus, exclude_sha=dominant_sha),
        )

        risk = self._get_risk(locus)
        use_txn = requires_transaction(risk)

        # Generate batch of diverse candidates
        candidates: list[str] = []
        try:
            candidates = self.mutation_engine.mutate_batch(
                ctx, count=MAX_MUTATION_RETRIES)
        except Exception as e:
            logger.warning("batch mutation generation failed: %s", e,
                           extra={"locus": locus})

        # Fallback: sequential single mutations if batch returned empty
        if not candidates:
            for attempt in range(MAX_MUTATION_RETRIES):
                try:
                    candidates.append(self.mutation_engine.mutate(ctx))
                except Exception as e:
                    logger.warning("mutation generation failed (attempt %d): %s",
                                   attempt + 1, e, extra={"locus": locus})

        # Test all candidates, collect passing ones
        first_passing: tuple[str, str] | None = None

        for i, new_source in enumerate(candidates):
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

                if not validate_output(locus, result, self.contract_store):
                    raise RuntimeError("output validation failed")

                if txn:
                    txn.commit()
                allele = self.registry.get(new_sha)
                if allele:
                    arena.record_success(allele)
                logger.info("mutant %s passed (candidate %d/%d)",
                            new_sha[:12], i + 1, len(candidates),
                            extra={"locus": locus, "sha": new_sha[:12]})

                if first_passing is None:
                    first_passing = (result, new_sha)
                self._contract_evolution.record_mutation_success(locus)

            except Exception as e:
                if txn and txn.action_count > 0:
                    rolled = txn.rollback()
                    if rolled:
                        logger.info("mutation rolled back %d action(s)",
                                    len(rolled), extra={"locus": locus})
                allele = self.registry.get(new_sha)
                if allele:
                    arena.record_failure(allele)
                self._contract_evolution.record_mutation_failure(locus, str(e))
                logger.warning("mutant %s failed: %s (candidate %d/%d)",
                               new_sha[:12], e, i + 1, len(candidates),
                               extra={"locus": locus, "sha": new_sha[:12]})

        if first_passing is None:
            logger.warning("all %d mutation candidates failed for %s",
                           len(candidates), locus, extra={"locus": locus})
            return None

        result, sha = first_passing
        self._audit("mutation_success", locus=locus, sha=sha,
                    candidates_tested=len(candidates))
        from sg.events import mutation_generated
        self._publish(mutation_generated(locus, sha, len(candidates)))
        return first_passing

    def _try_decomposition(
        self, locus: str, input_json: str, signal: 'DecompositionSignal',
    ) -> tuple[str, str] | None:
        """Attempt to decompose a gene into a pathway of sub-genes."""
        try:
            return self._try_decomposition_inner(locus, input_json, signal)
        except Exception as e:
            logger.warning("decomposition failed for %s: %s, falling back",
                           locus, e, extra={"locus": locus})
            return None

    def _try_decomposition_inner(
        self, locus: str, input_json: str, signal: 'DecompositionSignal',
    ) -> tuple[str, str] | None:
        from sg.decomposition import DecompositionResult

        dominant_sha = self.phenotype.get_dominant(locus)
        gene_source = ""
        if dominant_sha:
            gene_source = self.registry.load_source(dominant_sha) or ""

        contract_source = ""
        gene_contract = self.contract_store.get_gene(locus)
        if gene_contract and gene_contract.does:
            contract_source = gene_contract.does

        result: DecompositionResult = self.mutation_engine.decompose(
            locus, gene_source, signal.error_clusters,
            contract_source, signal.recommended_split_count,
        )

        # Register sub-gene contracts and seed alleles
        contracts_dir = self.project_root / "contracts"
        sub_loci = []
        for i, (contract_src, seed_src) in enumerate(
            zip(result.sub_gene_contract_sources, result.sub_gene_seed_sources)
        ):
            sub_locus = f"{locus}_sub{i + 1}"
            sub_loci.append(sub_locus)
            contract_path = contracts_dir / "genes" / f"{sub_locus}.sg"
            self.contract_store.register_contract(contract_src, contract_path)
            sha = self.registry.register(seed_src, sub_locus, 0, None)
            self.phenotype.add_to_fallback(sub_locus, sha)
            self.phenotype.promote(sub_locus, sha)

        # Register pathway contract
        pathway_name = f"{locus}_decomposed"
        pathway_path = contracts_dir / "pathways" / f"{pathway_name}.sg"
        self.contract_store.register_contract(result.pathway_contract_source, pathway_path)

        # Record decomposition state
        self.decomposition_detector.record_decomposition(locus, pathway_name, sub_loci)

        self._audit("decomposition", locus=locus,
                    pathway=pathway_name, sub_loci=sub_loci,
                    cluster_count=len(signal.error_clusters))

        logger.info("decomposed %s into pathway '%s' with %d sub-genes",
                    locus, pathway_name, len(sub_loci),
                    extra={"locus": locus, "event": "decomposition"})

        # Try executing the new pathway
        try:
            outputs = self.run_pathway(pathway_name, input_json)
            if outputs:
                return (outputs[-1], f"decomposed:{pathway_name}")
        except Exception as e:
            logger.warning("decomposed pathway %s failed on first run: %s",
                           pathway_name, e, extra={"locus": locus})

        return None

    def _check_promotion(self, locus: str, sha: str) -> None:
        allele = self.registry.get(sha)
        dominant_sha = self.phenotype.get_dominant(locus)
        dominant = self.registry.get(dominant_sha) if dominant_sha else None

        if allele and arena.should_promote(allele, dominant):
            # Test cross-locus interactions before promoting
            interaction_failures = self._test_promotion_interactions(locus, sha)
            if interaction_failures:
                policy = os.environ.get("SG_INTERACTION_POLICY", "rollback")
                if policy == "rollback":
                    logger.warning(
                        "promotion blocked for %s: %d interaction failure(s)",
                        sha[:12], len(interaction_failures),
                        extra={"locus": locus, "sha": sha[:12]})
                    self._audit("promotion_blocked", locus=locus, sha=sha,
                                failures=[f.pathway_name
                                          for f in interaction_failures])
                    return

            arena.set_dominant(allele)
            if dominant:
                arena.set_recessive(dominant)
            self.phenotype.promote(locus, sha)
            logger.info("promoted %s to dominant for %s", sha[:12], locus,
                        extra={"locus": locus, "sha": sha[:12], "event": "promotion"})
            self._audit("promotion", locus=locus, sha=sha,
                        fitness=arena.compute_fitness(allele))
            from sg.events import allele_promoted
            self._publish(allele_promoted(locus, sha, arena.compute_fitness(allele)))
            if self._meta_param_tracker is not None:
                self._meta_param_tracker.record_snapshot(
                    entity_name=locus, entity_type="gene",
                    outcome_fitness=arena.compute_fitness(allele),
                    allele_sha=sha, allele_survived=True,
                )

    def _test_promotion_interactions(self, locus: str, sha: str) -> list:
        """Test an allele against cross-locus pathways before promotion."""
        try:
            from sg.interactions import check_interactions
            return check_interactions(locus, sha, self)
        except Exception as e:
            logger.warning("interaction testing failed: %s", e,
                           extra={"locus": locus})
            return []

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

        # Record for feeds correlation analysis
        self._contract_evolution.record_diagnostic_output(locus, data)

        # Determine timescale from feeds declarations (or override)
        for feed in gene_contract.feeds:
            target_locus = feed.target_locus
            timescale = self.feedback_timescale or feed.timescale

            # Ensure correlation pair exists for feeds discovery
            self._contract_evolution.ensure_correlation_pair(locus, target_locus)

            # Find the dominant allele at the target config locus
            dominant_sha = self.phenotype.get_dominant(target_locus)
            if dominant_sha is None:
                continue
            target_allele = self.registry.get(dominant_sha)
            if target_allele is None:
                continue

            record_feedback(target_allele, timescale, healthy, locus,
                           structure_hash=self._current_pathway_structure_hash)
            fitness = arena.compute_fitness(target_allele)
            self._contract_evolution.record_config_fitness(target_locus, fitness)
            logger.info("feedback %s -> %s (%s: %s, fitness: %.2f)",
                        locus, target_locus, timescale,
                        "healthy" if healthy else "unhealthy", fitness,
                        extra={"locus": locus})

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
        shadow_kernel = self.kernel.create_shadow()
        try:
            execute_fn = load_gene(source, shadow_kernel)
            result = call_gene(execute_fn, input_json)

            if not validate_output(locus, result, self.contract_store):
                raise RuntimeError(f"shadow output validation failed for {locus}")

            allele.shadow_successes += 1
            remaining = SHADOW_PROMOTION_THRESHOLD - allele.shadow_successes
            if remaining > 0:
                logger.info("shadow success via %s (%d/%d)",
                            sha[:12], allele.shadow_successes,
                            SHADOW_PROMOTION_THRESHOLD,
                            extra={"locus": locus, "sha": sha[:12]})
            else:
                logger.info("shadow threshold met for %s — eligible for "
                            "live execution", sha[:12],
                            extra={"locus": locus, "sha": sha[:12]})
            return (result, sha)

        except Exception as e:
            allele.shadow_successes = 0  # reset on failure
            logger.warning("shadow failed via %s: %s", sha[:12], e,
                           extra={"locus": locus, "sha": sha[:12]})
            return None

    def _check_demotion(self, locus: str, sha: str) -> None:
        allele = self.registry.get(sha)
        if allele and arena.should_demote(allele):
            arena.set_deprecated(allele)
            logger.info("demoted %s for %s (3 consecutive failures)",
                        sha[:12], locus,
                        extra={"locus": locus, "sha": sha[:12], "event": "demotion"})
            self._audit("demotion", locus=locus, sha=sha)
            from sg.events import allele_demoted
            self._publish(allele_demoted(locus, sha))
            if self._meta_param_tracker is not None:
                self._meta_param_tracker.record_snapshot(
                    entity_name=locus, entity_type="gene",
                    outcome_fitness=arena.compute_fitness(allele),
                    allele_sha=sha, allele_survived=False,
                )

    def _check_regression(self, locus: str, sha: str, input_json: str) -> None:
        """Check for fitness regression and respond proactively."""
        allele = self.registry.get(sha)
        if allele is None:
            return
        severity = self.regression_detector.record(allele)
        self.regression_detector.save(self._regression_path)
        if severity == "severe":
            arena.set_deprecated(allele)
            logger.warning("severe regression for %s, demoting", sha[:12],
                           extra={"locus": locus, "sha": sha[:12],
                                  "event": "regression"})
            self._audit("regression_severe", locus=locus, sha=sha)
        elif severity == "mild":
            logger.info("mild regression for %s, proactive mutation queued",
                        sha[:12], extra={"locus": locus, "sha": sha[:12],
                                         "event": "regression"})
            self._audit("regression_mild", locus=locus, sha=sha)
            self._queue_proactive_mutation(locus, input_json)

    def _queue_proactive_mutation(self, locus: str, input_json: str) -> None:
        """Generate a competing allele without waiting for total failure."""
        dominant_sha = self.phenotype.get_dominant(locus)
        if not dominant_sha:
            return

        if hasattr(self.mutation_engine, '_contract_prompt'):
            contract_prompt = self.mutation_engine._contract_prompt(locus)
        else:
            contract_prompt = f"Locus: {locus}"

        try:
            sources = self.mutation_engine.generate(locus, contract_prompt, 1)
            for source in sources:
                parent = self.registry.get(dominant_sha)
                gen = (parent.generation + 1) if parent else 1
                sha = self.registry.register(source, locus, gen, dominant_sha)
                self.phenotype.add_to_fallback(locus, sha)
                allele = self.registry.get(sha)
                if allele:
                    allele.state = "recessive"
                logger.info("proactive mutant registered: %s", sha[:12],
                            extra={"locus": locus, "sha": sha[:12]})
        except Exception as e:
            logger.warning("proactive mutation failed: %s", e,
                           extra={"locus": locus})

    def run_pathway(self, pathway_name: str, input_json: str) -> list[str]:
        """Execute a named pathway.

        Handles on_failure strategy from the pathway contract:
        - "rollback all": clean up all tracked resources on failure
        - "report partial": let partial results stand
        """
        with correlation_scope() as cid:
            return self._run_pathway_inner(pathway_name, input_json)

    def _run_pathway_inner(self, pathway_name: str, input_json: str) -> list[str]:
        pathway_contract = self.contract_store.get_pathway(pathway_name)
        if pathway_contract is not None:
            default_pathway = pathway_from_contract(pathway_contract)
            on_failure = pathway_contract.on_failure
        else:
            raise ValueError(f"unknown pathway: {pathway_name}")

        # Register structural allele if pathway registry is active
        if self.pathway_registry is not None:
            steps = steps_from_pathway(default_pathway)
            sha = self.pathway_registry.register(pathway_name, steps)
            if not self.phenotype.get_pathway_stack(pathway_name):
                self.phenotype.add_pathway_fallback(pathway_name, sha)
                self.phenotype.promote_pathway(pathway_name, sha)
                pw_allele = self.pathway_registry.get(sha)
                if pw_allele:
                    pw_allele.state = "dominant"

        # Use pathway allele stack if registry active
        allele_stack = (
            self.phenotype.get_pathway_stack(pathway_name)
            if self.pathway_registry else []
        )
        if allele_stack and self.pathway_registry:
            return self._execute_pathway_allele_stack(
                pathway_name, input_json, allele_stack,
                on_failure, default_pathway,
            )
        return self._execute_single_pathway(
            default_pathway, pathway_name, input_json, on_failure,
        )

    def _execute_single_pathway(
        self,
        pathway: Pathway,
        pathway_name: str,
        input_json: str,
        on_failure: str,
    ) -> list[str]:
        """Execute a single pathway structure."""
        logger.info("Executing pathway: %s", pathway_name,
                    extra={"pathway": pathway_name})

        resources_before = set(self.kernel.tracked_resources())

        try:
            outputs = execute_pathway(
                pathway, input_json, self,
                self.fusion_tracker, self.registry,
                self.phenotype, self.mutation_engine,
                self.kernel,
                pathway_fitness_tracker=self.pathway_fitness_tracker,
            )
        except Exception:
            if on_failure == "rollback all":
                self._rollback_pathway_resources(resources_before)
            raise

        logger.info("Pathway '%s' completed with %d output(s)",
                    pathway_name, len(outputs), extra={"pathway": pathway_name})
        return outputs

    def _execute_pathway_allele_stack(
        self,
        pathway_name: str,
        input_json: str,
        allele_stack: list[str],
        on_failure: str,
        default_pathway: Pathway,
    ) -> list[str]:
        """Try each pathway allele in the stack, recording success/failure."""
        last_error = ""
        for sha in allele_stack:
            pw_allele = self.pathway_registry.get(sha)
            if pw_allele is None or pw_allele.state == "deprecated":
                continue

            pathway = self._pathway_from_allele(pw_allele, default_pathway)

            try:
                self._current_pathway_structure_hash = pw_allele.structure_sha
                outputs = self._execute_single_pathway(
                    pathway, pathway_name, input_json, on_failure,
                )
                pathway_arena.record_pathway_success(pw_allele)
                self._record_stabilization_fitness(pathway_name)
                self._check_pathway_promotion(pathway_name, sha)
                return outputs
            except Exception as e:
                last_error = str(e)
                pathway_arena.record_pathway_failure(pw_allele)
                self._check_pathway_demotion(pathway_name, sha)
                logger.warning(
                    "pathway allele %s failed for '%s': %s",
                    sha[:12], pathway_name, e,
                    extra={"pathway": pathway_name, "sha": sha[:12]},
                )
                continue
            finally:
                self._current_pathway_structure_hash = ""

        # Attempt pathway mutation before giving up
        self._try_pathway_mutation(pathway_name, default_pathway)

        raise RuntimeError(
            f"pathway '{pathway_name}' failed: all pathway alleles exhausted"
        )

    def _pathway_from_allele(
        self, pw_allele: 'PathwayAllele', default_pathway: Pathway,
    ) -> Pathway:
        """Convert a PathwayAllele back to a runtime Pathway.

        For the contract-derived allele, returns the default directly.
        For mutated alleles, reconstructs from StepSpec list.
        """
        default_steps = steps_from_pathway(default_pathway)
        default_sha = compute_structure_sha(default_steps)
        if pw_allele.structure_sha == default_sha:
            return default_pathway
        return self._pathway_from_stepspecs(
            pw_allele.pathway_name, pw_allele.steps, default_pathway,
        )

    def _pathway_from_stepspecs(
        self,
        pathway_name: str,
        specs: list,
        default_pathway: Pathway,
    ) -> Pathway:
        """Reconstruct a runtime Pathway from a StepSpec list.

        Uses the default pathway's step transforms as a lookup table.
        For steps not present in the default, generates a param-based
        or passthrough transform.
        """
        from sg.pathway import (
            PathwayStep as RTPathwayStep, ComposedStep, LoopStep,
            ConditionalExecStep, _make_reference_transform,
            _make_passthrough_transform,
        )

        # Build lookup: target -> input_transform from default pathway
        default_transforms: dict[str, object] = {}
        for step in default_pathway.steps:
            if isinstance(step, RTPathwayStep):
                default_transforms[step.locus] = step.input_transform
            elif isinstance(step, ComposedStep):
                default_transforms[step.pathway_name] = step.input_transform

        steps = []
        for spec in specs:
            if spec.step_type == "locus":
                transform = default_transforms.get(spec.target)
                if transform is None:
                    transform = (
                        _make_reference_transform(spec.params)
                        if spec.params
                        else _make_passthrough_transform()
                    )
                steps.append(RTPathwayStep(
                    locus=spec.target, input_transform=transform,
                ))
            elif spec.step_type == "composed":
                transform = default_transforms.get(spec.target)
                if transform is None:
                    transform = (
                        _make_reference_transform(spec.params)
                        if spec.params
                        else _make_passthrough_transform()
                    )
                steps.append(ComposedStep(
                    pathway_name=spec.target, input_transform=transform,
                ))
            elif spec.step_type == "loop":
                steps.append(LoopStep(
                    variable=spec.loop_variable or "",
                    iterable_field=spec.loop_iterable or "",
                    body_locus=spec.target,
                    body_params=spec.params,
                ))
            elif spec.step_type == "conditional":
                branches: dict[str, RTPathwayStep | ComposedStep] = {}
                if spec.branches:
                    for value, branch_info in spec.branches.items():
                        bt = branch_info.get("step_type", "locus")
                        bt_target = branch_info.get("target", "")
                        if bt == "composed":
                            branches[value] = ComposedStep(
                                pathway_name=bt_target,
                                input_transform=_make_passthrough_transform(),
                            )
                        else:
                            branches[value] = RTPathwayStep(
                                locus=bt_target,
                                input_transform=_make_passthrough_transform(),
                            )
                steps.append(ConditionalExecStep(
                    condition_step_index=spec.condition_step_index or 0,
                    condition_field=spec.condition_field or "",
                    branches=branches,
                ))

        return Pathway(name=pathway_name, steps=steps)

    def _check_pathway_promotion(self, pathway_name: str, sha: str) -> None:
        if self.pathway_registry is None:
            return
        allele = self.pathway_registry.get(sha)
        dominant_sha = self.phenotype.get_pathway_dominant(pathway_name)
        dominant = self.pathway_registry.get(dominant_sha) if dominant_sha else None

        if allele and pathway_arena.should_promote_pathway(allele, dominant):
            old_structure_hash = dominant.structure_sha if dominant else ""
            pathway_arena.set_pathway_dominant(allele)
            if dominant:
                pathway_arena.set_pathway_recessive(dominant)
            self.phenotype.promote_pathway(pathway_name, sha)
            if old_structure_hash:
                self._tag_gene_fitness_records(pathway_name, old_structure_hash)
                # Track structure history for progressive fitness decay
                history = self._pathway_structure_histories.get(pathway_name, [])
                history.insert(0, old_structure_hash)
                self._pathway_structure_histories[pathway_name] = history[:5]
            self._stabilization_tracker.start_stabilization(
                pathway_name, sha, self._pathway_loci(pathway_name),
            )
            self._pathway_mutation_throttle.reset_cooldown(pathway_name)
            logger.info("promoted pathway allele %s for %s",
                        sha[:12], pathway_name,
                        extra={"pathway": pathway_name, "sha": sha[:12],
                               "event": "pathway_promotion"})
            self._audit("pathway_promotion", locus=pathway_name, sha=sha,
                        fitness=pathway_arena.compute_pathway_fitness(allele))
            from sg.events import pathway_promoted
            self._publish(pathway_promoted(
                pathway_name, sha, pathway_arena.compute_pathway_fitness(allele)))
            if self._meta_param_tracker is not None:
                self._meta_param_tracker.record_snapshot(
                    entity_name=pathway_name, entity_type="pathway",
                    outcome_fitness=pathway_arena.compute_pathway_fitness(allele),
                    allele_sha=sha, allele_survived=True,
                )

    def _check_pathway_demotion(self, pathway_name: str, sha: str) -> None:
        if self.pathway_registry is None:
            return
        allele = self.pathway_registry.get(sha)
        if allele and pathway_arena.should_demote_pathway(allele):
            pathway_arena.set_pathway_deprecated(allele)
            logger.info("demoted pathway allele %s for %s (5 consecutive failures)",
                        sha[:12], pathway_name,
                        extra={"pathway": pathway_name, "sha": sha[:12],
                               "event": "pathway_demotion"})
            self._audit("pathway_demotion", locus=pathway_name, sha=sha)
            if self._meta_param_tracker is not None:
                self._meta_param_tracker.record_snapshot(
                    entity_name=pathway_name, entity_type="pathway",
                    outcome_fitness=pathway_arena.compute_pathway_fitness(allele),
                    allele_sha=sha, allele_survived=False,
                )

    def _pathway_loci(self, pathway_name: str) -> list[str]:
        """Extract gene locus names from a pathway contract's steps."""
        from sg.parser.types import PathwayStep as ASTPathwayStep
        contract = self.contract_store.get_pathway(pathway_name)
        if contract is None:
            return []
        return [s.locus for s in contract.steps
                if isinstance(s, ASTPathwayStep) and not s.is_pathway_ref]

    def _tag_gene_fitness_records(
        self, pathway_name: str, old_structure_hash: str,
    ) -> None:
        """After pathway promotion, tag untagged gene fitness records with old hash."""
        for locus in self._pathway_loci(pathway_name):
            dom_sha = self.phenotype.get_dominant(locus)
            if not dom_sha:
                continue
            allele = self.registry.get(dom_sha)
            if allele is None:
                continue
            for rec in allele.fitness_records:
                if not rec.get("structure_hash"):
                    rec["structure_hash"] = old_structure_hash

    def _record_stabilization_fitness(self, pathway_name: str) -> None:
        """Record current gene fitness for stabilization tracking."""
        if not self._stabilization_tracker.is_stabilizing(pathway_name):
            return
        for locus in self._pathway_loci(pathway_name):
            dom_sha = self.phenotype.get_dominant(locus)
            if not dom_sha:
                continue
            allele = self.registry.get(dom_sha)
            if allele is None:
                continue
            fitness = arena.compute_fitness(allele)
            self._stabilization_tracker.record_gene_fitness(
                pathway_name, locus, fitness,
            )

    def _consider_pathway_revert(self, pathway_name: str) -> None:
        """Consider reverting a pathway promotion if stabilization timed out."""
        if self.pathway_registry is None:
            return
        current_sha = self.phenotype.get_pathway_dominant(pathway_name)
        if current_sha is None:
            return
        current = self.pathway_registry.get(current_sha)
        if current is None or current.parent_sha is None:
            return
        parent = self.pathway_registry.get(current.parent_sha)
        if parent is None:
            return
        current_fitness = pathway_arena.compute_pathway_fitness(current)
        parent_fitness = pathway_arena.compute_pathway_fitness(parent)
        if current_fitness < parent_fitness:
            pathway_arena.set_pathway_recessive(current)
            pathway_arena.set_pathway_dominant(parent)
            self.phenotype.promote_pathway(pathway_name, parent.structure_sha)
            logger.warning(
                "reverted pathway '%s' to parent %s after stabilization timeout",
                pathway_name, parent.structure_sha[:12],
            )
            self._audit("pathway_revert", locus=pathway_name,
                        sha=parent.structure_sha, reason="stabilization_timeout")

    def _load_pathway_mutation_throttle(self) -> PathwayMutationThrottle:
        """Load pathway mutation throttle from disk or create fresh."""
        throttle_path = self.project_root / ".sg" / "pathway_mutation_throttle.json"
        if throttle_path.exists():
            try:
                with file_lock_shared(throttle_path):
                    data = json.loads(throttle_path.read_text())
                throttle = PathwayMutationThrottle.from_dict(data)
            except json.JSONDecodeError:
                logger.warning("pathway mutation throttle corrupted, starting fresh")
                throttle = PathwayMutationThrottle()
        else:
            throttle = PathwayMutationThrottle()
        cooldown_env = os.environ.get("SG_PATHWAY_MUTATION_COOLDOWN")
        if cooldown_env is not None:
            throttle.cooldown_seconds = float(cooldown_env) * 3600
        return throttle

    def _try_pathway_mutation(
        self, pathway_name: str, default_pathway: Pathway,
    ) -> str | None:
        """Attempt structural mutation on a pathway. Returns new SHA if registered."""
        if self.pathway_registry is None:
            return None
        if self.pathway_fitness_tracker is None:
            return None
        if self._stabilization_tracker.is_stabilizing(pathway_name):
            status = self._stabilization_tracker.check_stabilization(pathway_name)
            if status == "stabilizing":
                logger.info("pathway mutation blocked: stabilization in progress for '%s'",
                            pathway_name)
                return None
            if status == "timed_out":
                self._consider_pathway_revert(pathway_name)
        if not self._pathway_mutation_throttle.can_mutate(pathway_name):
            logger.info("pathway mutation throttled for '%s'", pathway_name)
            return None
        if not self._is_structural_problem(pathway_name):
            return None

        try:
            ctx = self._build_pathway_mutation_context(pathway_name)
            if ctx is None:
                return None

            from sg.pathway_mutation import select_operator, default_operators
            operators = default_operators(mutation_engine=self.mutation_engine)
            result = select_operator(ctx, operators)
            if result is None:
                return None

            dominant_sha = self.phenotype.get_pathway_dominant(pathway_name)
            new_sha = self.pathway_registry.register(
                pathway_name, result.new_steps,
                parent_sha=dominant_sha,
                mutation_operator=result.operator_name,
            )
            self.phenotype.add_pathway_fallback(pathway_name, new_sha)
            self._pathway_mutation_throttle.record_mutation(pathway_name)
            self._audit("pathway_mutation", locus=pathway_name, sha=new_sha,
                        operator=result.operator_name, rationale=result.rationale)
            logger.info(
                "registered pathway mutation for '%s': %s (operator: %s)",
                pathway_name, new_sha[:12], result.operator_name,
                extra={"pathway": pathway_name, "sha": new_sha[:12],
                       "event": "pathway_mutation"},
            )
            return new_sha
        except Exception:
            logger.warning(
                "pathway mutation failed for '%s', skipping",
                pathway_name, exc_info=True,
            )
            return None

    def _is_structural_problem(self, pathway_name: str) -> bool:
        """Detect structural problem: gene fitness high but pathway fitness low."""
        pathway_fitness = self.pathway_fitness_tracker.compute_fitness(pathway_name)
        if pathway_fitness > 0.7:
            return False

        pathway_contract = self.contract_store.get_pathway(pathway_name)
        if pathway_contract is None:
            return False

        from sg.parser.types import PathwayStep as ASTPathwayStep
        gene_fitnesses = []
        for step in pathway_contract.steps:
            if isinstance(step, ASTPathwayStep) and not step.is_pathway_ref:
                dominant_sha = self.phenotype.get_dominant(step.locus)
                if dominant_sha:
                    allele = self.registry.get(dominant_sha)
                    if allele:
                        gene_fitnesses.append(arena.compute_fitness(allele))

        if not gene_fitnesses:
            return False

        avg_gene_fitness = sum(gene_fitnesses) / len(gene_fitnesses)
        return avg_gene_fitness > 0.7 and pathway_fitness < 0.5

    def _build_pathway_mutation_context(self, pathway_name: str):
        """Assemble the full mutation context for a pathway."""
        from sg.pathway_mutation import PathwayMutationContext

        pathway_contract = self.contract_store.get_pathway(pathway_name)
        dominant_sha = self.phenotype.get_pathway_dominant(pathway_name)
        if dominant_sha is None:
            return None
        pw_allele = self.pathway_registry.get(dominant_sha)
        if pw_allele is None:
            return None

        per_step_fitness: dict[str, float] = {}
        gene_fitness_map: dict[str, float] = {}
        for locus in self.contract_store.known_loci():
            dom = self.phenotype.get_dominant(locus)
            if dom:
                allele = self.registry.get(dom)
                if allele:
                    fitness = arena.compute_fitness(allele)
                    gene_fitness_map[locus] = fitness
                    per_step_fitness[locus] = fitness

        return PathwayMutationContext(
            pathway_name=pathway_name,
            current_steps=pw_allele.steps,
            pathway_fitness=self.pathway_fitness_tracker.compute_fitness(pathway_name),
            per_step_fitness=per_step_fitness,
            timing_anomalies=self.pathway_fitness_tracker.get_timing_anomalies(pathway_name),
            failure_distribution=self.pathway_fitness_tracker.get_failure_distribution(pathway_name),
            input_clusters=self.pathway_fitness_tracker.get_input_clusters(pathway_name),
            available_loci=self.contract_store.known_loci(),
            available_pathways=self.contract_store.known_pathways(),
            gene_fitness_map=gene_fitness_map,
            contract=pathway_contract,
            contract_store=self.contract_store,
        )

    def _rollback_pathway_resources(
        self, resources_before: set[tuple[str, str]]
    ) -> None:
        """Clean up resources created during a failed pathway."""
        current = set(self.kernel.tracked_resources())
        new_resources = current - resources_before
        if not new_resources:
            return

        logger.info("rolling back %d resource(s)...", len(new_resources))
        for resource_type, name in new_resources:
            try:
                self.kernel.delete_resource(resource_type, name)
                logger.info("cleaned up %s '%s'", resource_type, name)
            except Exception as e:
                logger.error("cleanup failed for %s '%s': %s",
                             resource_type, name, e)

    def run_topology(self, topology_name: str, input_json: str) -> list[str]:
        """Execute a named topology with allele stack support."""
        from sg.topology import decompose, execute_topology

        topology = self.contract_store.get_topology(topology_name)
        if topology is None:
            raise ValueError(f"unknown topology: {topology_name}")

        resource_mappers = self.kernel.resource_mappers()

        # Register structural allele if topology registry is active
        if self.topology_registry is not None:
            from sg.topology_registry import steps_from_decomposition
            raw_steps = decompose(topology, input_json, resource_mappers)
            step_specs = steps_from_decomposition(raw_steps)
            sha = self.topology_registry.register(topology_name, step_specs)
            if not self.phenotype.get_topology_stack(topology_name):
                self.phenotype.add_topology_fallback(topology_name, sha)
                self.phenotype.promote_topology(topology_name, sha)
                topo_allele = self.topology_registry.get(sha)
                if topo_allele:
                    topo_allele.state = "dominant"

        # Use topology allele stack if registry active
        allele_stack = (
            self.phenotype.get_topology_stack(topology_name)
            if self.topology_registry else []
        )
        if allele_stack and self.topology_registry:
            return self._execute_topology_allele_stack(
                topology_name, input_json, allele_stack, topology,
            )

        logger.info("Deploying topology: %s", topology_name)
        outputs = execute_topology(topology, input_json, self, resource_mappers)
        logger.info("Topology '%s' deployed (%d output(s))",
                    topology_name, len(outputs))
        return outputs

    def _execute_topology_allele_stack(
        self,
        topology_name: str,
        input_json: str,
        allele_stack: list[str],
        topology,
    ) -> list[str]:
        """Try each topology allele in the stack, recording success/failure."""
        from sg.topology import execute_topology
        from sg import topology_arena

        last_error = ""
        for sha in allele_stack:
            topo_allele = self.topology_registry.get(sha)
            if topo_allele is None or topo_allele.state == "deprecated":
                continue
            try:
                outputs = execute_topology(
                    topology, input_json, self, self.kernel.resource_mappers(),
                )
                topology_arena.record_topology_success(topo_allele)
                self._check_topology_promotion(topology_name, sha)
                logger.info("Topology '%s' deployed via allele %s (%d output(s))",
                            topology_name, sha[:12], len(outputs))
                return outputs
            except Exception as e:
                last_error = str(e)
                topology_arena.record_topology_failure(topo_allele)
                self._check_topology_demotion(topology_name, sha)
                logger.warning(
                    "topology allele %s failed for '%s': %s",
                    sha[:12], topology_name, e,
                    extra={"topology": topology_name, "sha": sha[:12]},
                )
                continue
        raise RuntimeError(
            f"topology '{topology_name}' failed: all topology alleles exhausted"
            + (f" (last error: {last_error})" if last_error else "")
        )

    def _check_topology_promotion(self, topology_name: str, sha: str) -> None:
        if self.topology_registry is None:
            return
        from sg import topology_arena
        allele = self.topology_registry.get(sha)
        dominant_sha = self.phenotype.get_topology_dominant(topology_name)
        dominant = (
            self.topology_registry.get(dominant_sha)
            if dominant_sha else None
        )
        if allele and topology_arena.should_promote_topology(allele, dominant):
            topology_arena.set_topology_dominant(allele)
            if dominant:
                topology_arena.set_topology_recessive(dominant)
            self.phenotype.promote_topology(topology_name, sha)
            logger.info("promoted topology allele %s for %s",
                        sha[:12], topology_name,
                        extra={"topology": topology_name, "sha": sha[:12],
                               "event": "topology_promotion"})
            self._audit("topology_promotion", locus=topology_name, sha=sha,
                        fitness=topology_arena.compute_topology_fitness(allele))
            if self._meta_param_tracker is not None:
                self._meta_param_tracker.record_snapshot(
                    entity_name=topology_name, entity_type="topology",
                    outcome_fitness=topology_arena.compute_topology_fitness(allele),
                    allele_sha=sha, allele_survived=True,
                )

    def _check_topology_demotion(self, topology_name: str, sha: str) -> None:
        if self.topology_registry is None:
            return
        from sg import topology_arena
        allele = self.topology_registry.get(sha)
        if allele and topology_arena.should_demote_topology(allele):
            topology_arena.set_topology_deprecated(allele)
            logger.info("demoted topology allele %s for %s",
                        sha[:12], topology_name,
                        extra={"topology": topology_name, "sha": sha[:12],
                               "event": "topology_demotion"})
            self._audit("topology_demotion", locus=topology_name, sha=sha)
            if self._meta_param_tracker is not None:
                self._meta_param_tracker.record_snapshot(
                    entity_name=topology_name, entity_type="topology",
                    outcome_fitness=topology_arena.compute_topology_fitness(allele),
                    allele_sha=sha, allele_survived=False,
                )

    def save_state(self) -> None:
        try:
            self.registry.save_index()
        except Exception:
            logger.error("save_state: failed to save registry", exc_info=True)
        try:
            phenotype_path = self.project_root / "phenotype.toml"
            self.phenotype.save(phenotype_path)
        except Exception:
            logger.error("save_state: failed to save phenotype", exc_info=True)
        try:
            tracker_path = self.project_root / "fusion_tracker.json"
            self.fusion_tracker.save(tracker_path)
        except Exception:
            logger.error("save_state: failed to save fusion tracker", exc_info=True)
        if self.pathway_fitness_tracker is not None:
            try:
                fitness_path = self.project_root / "pathway_fitness.json"
                self.pathway_fitness_tracker.save(fitness_path)
            except Exception:
                logger.error("save_state: failed to save pathway fitness", exc_info=True)
        if self.pathway_registry is not None:
            try:
                self.pathway_registry.save_index()
            except Exception:
                logger.error("save_state: failed to save pathway registry", exc_info=True)
        try:
            throttle_path = self.project_root / ".sg" / "pathway_mutation_throttle.json"
            with file_lock(throttle_path):
                atomic_write_text(throttle_path, json.dumps(
                    self._pathway_mutation_throttle.to_dict(), indent=2,
                ))
        except Exception:
            logger.error("save_state: failed to save pathway mutation throttle", exc_info=True)
        try:
            self.decomposition_detector.save(self._decomposition_path)
        except Exception:
            logger.error("save_state: failed to save decomposition detector", exc_info=True)
        try:
            self._stabilization_tracker.save(
                self.project_root / ".sg" / "stabilization.json"
            )
        except Exception:
            logger.error("save_state: failed to save stabilization tracker", exc_info=True)
        try:
            self._failure_discovery.save(
                self.project_root / ".sg" / "failure_discovery.json"
            )
        except Exception:
            logger.error("save_state: failed to save failure discovery", exc_info=True)
        try:
            self._contract_evolution.save(
                self.project_root / ".sg" / "contract_evolution.json"
            )
        except Exception:
            logger.error("save_state: failed to save contract evolution", exc_info=True)
        try:
            self._cross_locus_analyzer.save(
                self.project_root / ".sg" / "locus_discovery.json"
            )
        except Exception:
            logger.error("save_state: failed to save locus discovery", exc_info=True)
        if self.topology_registry is not None:
            try:
                self.topology_registry.save_index()
            except Exception:
                logger.error("save_state: failed to save topology registry", exc_info=True)
        if self._meta_param_tracker is not None:
            try:
                self._meta_param_tracker.save(
                    self.project_root / ".sg" / "meta_params.json"
                )
            except Exception:
                logger.error("save_state: failed to save meta params", exc_info=True)
