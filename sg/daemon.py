"""Daemon mode — continuous evolutionary loop.

Runs the orchestrator on a tick interval, performing health checks,
processing verify queues, running auto-tune cycles, and saving state.
"""
from __future__ import annotations

import signal
import time
from dataclasses import dataclass

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


class Daemon:
    """Runs the orchestrator in a continuous loop."""

    def __init__(
        self,
        orchestrator,
        event_bus=None,
        config: DaemonConfig | None = None,
    ) -> None:
        self.orchestrator = orchestrator
        self.event_bus = event_bus
        self.config = config or DaemonConfig()
        self._running = False
        self._tick_count = 0

    @property
    def tick_count(self) -> int:
        return self._tick_count

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

            # 2. Process verify queue
            self.orchestrator.verify_scheduler.process_ready(self.orchestrator)

            # 3. Auto-tune at intervals
            if self._tick_count % self.config.auto_tune_interval == 0:
                self._run_auto_tune()

            # 4. Save state
            self.orchestrator.save_state()

        except Exception:
            logger.exception("tick %d failed", self._tick_count)

        duration_ms = (time.monotonic() - start) * 1000
        if self.event_bus is not None:
            self.event_bus.publish(tick_complete(self._tick_count, duration_ms))

    def _run_health_checks(self) -> None:
        """Run diagnostic pathways for health checking."""
        logger.debug("running health checks (tick %d)", self._tick_count)

    def _run_auto_tune(self) -> None:
        """Run adaptive parameter tuning."""
        logger.debug("running auto-tune (tick %d)", self._tick_count)

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
