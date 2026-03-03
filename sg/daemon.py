"""Daemon mode — continuous evolutionary loop.

Runs the orchestrator on a tick interval, performing health checks,
processing verify queues, running auto-tune cycles, and saving state.
"""
from __future__ import annotations

import signal
import time
from dataclasses import dataclass, field
from typing import Callable

from sg.events import tick_complete
from sg.log import get_logger

logger = get_logger("daemon")


@dataclass
class DaemonConfig:
    """Configuration for the daemon loop."""
    tick_interval: float = 60.0
    health_check_interval: int = 5
    auto_tune_interval: int = 100
    max_ticks: int | None = None
    workload: Callable | None = field(default=None, repr=False)


class Daemon:
    """Runs the orchestrator in a continuous loop."""

    def __init__(
        self,
        orchestrator,
        event_bus=None,
        config: DaemonConfig | None = None,
        metrics_collector=None,
    ) -> None:
        self.orchestrator = orchestrator
        self.event_bus = event_bus
        self.config = config or DaemonConfig()
        self.metrics_collector = metrics_collector
        self._running = False
        self._tick_count = 0
        self._tuner = None
        self._safety = None

    @property
    def tick_count(self) -> int:
        return self._tick_count

    def start_dashboard(self, host: str = "127.0.0.1", port: int = 8420) -> None:
        """Start the web dashboard in a background thread with shared metrics."""
        try:
            import sg.dashboard as dash
            import uvicorn
            import threading
        except ImportError:
            logger.warning("dashboard extras not installed, skipping --with-dashboard")
            return
        dash._project_root = self.orchestrator.project_root
        dash._metrics_collector = self.metrics_collector

        def _serve():
            uvicorn.run(dash.app, host=host, port=port, log_level="warning")

        t = threading.Thread(target=_serve, daemon=True, name="sg-dashboard")
        t.start()
        logger.info("dashboard started at http://%s:%d (in-process)", host, port)

    def start(self) -> None:
        """Start the daemon loop.  Blocks until stop() or max_ticks."""
        self._running = True
        self._install_signal_handlers()
        logger.info("daemon started (interval=%.1fs, max_ticks=%s)",
                    self.config.tick_interval, self.config.max_ticks)
        try:
            while self._running:
                if (self.config.max_ticks is not None
                        and self._tick_count >= self.config.max_ticks):
                    break
                self._tick()
                if self._running and (
                    self.config.max_ticks is None
                    or self._tick_count < self.config.max_ticks
                ):
                    time.sleep(self.config.tick_interval)
        finally:
            self.orchestrator.save_state()
            logger.info("daemon stopped after %d ticks", self._tick_count)

    def stop(self) -> None:
        """Signal the daemon to stop after the current tick."""
        self._running = False

    def _tick(self) -> None:
        """Execute one daemon tick."""
        start = time.monotonic()
        self._tick_count += 1
        logger.debug("tick %d", self._tick_count)

        try:
            # 1. Health checks at intervals
            if self._tick_count % self.config.health_check_interval == 0:
                self._run_health_checks()

            # 2. Run workload (if configured)
            if self.config.workload is not None:
                try:
                    self.config.workload(self.orchestrator, self._tick_count)
                except Exception:
                    logger.debug("workload failed on tick %d",
                                 self._tick_count, exc_info=True)

            # 3. Process verify queue
            self.orchestrator.verify_scheduler.process_ready(self.orchestrator)

            # 4. Auto-tune at intervals
            if self._tick_count % self.config.auto_tune_interval == 0:
                self._run_auto_tune()

            # 5. Save state
            self.orchestrator.save_state()

        except Exception:
            logger.exception("tick %d failed", self._tick_count)

        duration_ms = (time.monotonic() - start) * 1000
        if self.event_bus is not None:
            self.event_bus.publish(tick_complete(self._tick_count, duration_ms))

        # Persist metrics snapshot for cross-process consumers (dashboard)
        if self.metrics_collector is not None:
            try:
                metrics_path = self.orchestrator.project_root / ".sg" / "metrics.json"
                self.metrics_collector.save(metrics_path)
            except Exception:
                logger.debug("failed to save metrics snapshot", exc_info=True)

    def _run_health_checks(self) -> None:
        """Periodic health tasks: run diagnostics, update gauges, record speciation."""
        logger.debug("running health checks (tick %d)", self._tick_count)

        # Run diagnostic pathways with synthetic inputs
        store = getattr(self.orchestrator, "contract_store", None)
        if store and hasattr(self.orchestrator, "run_pathway"):
            try:
                from sg.parser.types import PathwayContract, PathwayStep
                for name, pathway in store.items():
                    if not isinstance(pathway, PathwayContract):
                        continue
                    steps = getattr(pathway, "steps", [])
                    if not steps:
                        continue
                    first = steps[0]
                    if isinstance(first, PathwayStep):
                        gene = store.get(first.locus)
                        if gene and getattr(gene, "family", None) == "diagnostic":
                            try:
                                self.orchestrator.run_pathway(name, "{}")
                            except Exception:
                                logger.debug("health check pathway '%s' failed",
                                             name, exc_info=True)
            except Exception:
                logger.debug("diagnostic pathway scan failed", exc_info=True)

        # Update metric gauges
        if self.metrics_collector is not None:
            self._update_gauges()

        # Record speciation snapshot periodically
        if (self._tick_count % 50 == 0
                and hasattr(self.orchestrator, '_speciation_tracker')):
            try:
                import os
                organism_id = os.environ.get("SG_ORGANISM_ID", "default")
                self.orchestrator._speciation_tracker.record_snapshot(
                    organism_id,
                    self.orchestrator.phenotype,
                    self.orchestrator.registry,
                    meta_param_tracker=self.orchestrator._meta_param_tracker,
                )
            except Exception:
                logger.debug("speciation snapshot failed", exc_info=True)

    def _update_gauges(self) -> None:
        """Compute and set sg_avg_fitness and sg_active_loci gauges."""
        phenotype = getattr(self.orchestrator, "phenotype", None)
        registry = getattr(self.orchestrator, "registry", None)
        if not phenotype or not registry:
            return
        try:
            from sg.arena import compute_fitness
            fitnesses = []
            active = 0
            for locus in phenotype.loci():
                dom_sha = phenotype.get_dominant(locus)
                if dom_sha:
                    active += 1
                    allele = registry.get(dom_sha)
                    if allele:
                        params = self.orchestrator._get_params(locus)
                        fitnesses.append(compute_fitness(allele, params=params))
            self.metrics_collector.sg_active_loci.set(active)
            if fitnesses:
                self.metrics_collector.sg_avg_fitness.set(
                    sum(fitnesses) / len(fitnesses))
        except Exception:
            logger.debug("gauge update failed", exc_info=True)

    def _run_auto_tune(self) -> None:
        """Run adaptive parameter tuning and safety analysis."""
        logger.debug("running auto-tune (tick %d)", self._tick_count)

        # Adaptive parameter tuning (persistent tuner across ticks)
        if self.orchestrator._meta_param_tracker is not None:
            try:
                if self._tuner is None:
                    from sg.adaptation import AdaptiveParamTuner
                    self._tuner = AdaptiveParamTuner(
                        self.orchestrator._meta_param_tracker)
                recs = self._tuner.auto_tune()
                if recs:
                    logger.info("auto-tune applied %d recommendation(s)", len(recs))
                    for rec in recs:
                        logger.info("  %s.%s: %.4f -> %.4f (%s)",
                                    rec.entity_name, rec.param_name,
                                    rec.current_value, rec.recommended_value,
                                    rec.reason)
            except Exception:
                logger.exception("auto-tune failed")

        # Adaptive safety analysis (advisory only — logged, not auto-applied)
        if self.orchestrator.audit_log is not None:
            try:
                if self._safety is None:
                    from sg.adaptation import AdaptiveSafety
                    self._safety = AdaptiveSafety(self.orchestrator.audit_log)
                adjustments = self._safety.analyze(self.orchestrator.contract_store)
                if adjustments:
                    logger.info("safety analysis: %d recommendation(s)", len(adjustments))
                    for adj in adjustments:
                        logger.info("  %s: %s -> %s (%s)",
                                    adj.locus, adj.current_risk,
                                    adj.recommended_risk, adj.reason)
            except Exception:
                logger.exception("safety analysis failed")

    def _install_signal_handlers(self) -> None:
        """Install SIGTERM/SIGINT handlers for graceful shutdown."""
        def _handler(signum, frame):
            logger.info("received signal %d, stopping...", signum)
            self.stop()

        try:
            signal.signal(signal.SIGTERM, _handler)
            signal.signal(signal.SIGINT, _handler)
        except (OSError, ValueError):
            # Can't set signal handlers in non-main thread
            pass
