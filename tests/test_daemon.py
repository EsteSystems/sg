"""Tests for daemon mode."""
from __future__ import annotations

import pytest

from sg.daemon import Daemon, DaemonConfig
from sg.events import EventBus


class FakeVerifyScheduler:
    def __init__(self):
        self.process_count = 0

    def process_ready(self, orchestrator):
        self.process_count += 1


class FakeOrchestrator:
    def __init__(self):
        self.save_count = 0
        self.verify_scheduler = FakeVerifyScheduler()
        self._meta_param_tracker = None
        self.audit_log = None
        self.contract_store = None

    def save_state(self):
        self.save_count += 1


class TestDaemon:
    def test_tick_count(self):
        orch = FakeOrchestrator()
        config = DaemonConfig(tick_interval=0.0, max_ticks=3)
        daemon = Daemon(orch, config=config)
        daemon.start()
        assert daemon.tick_count == 3

    def test_state_saved_each_tick(self):
        orch = FakeOrchestrator()
        config = DaemonConfig(tick_interval=0.0, max_ticks=5)
        daemon = Daemon(orch, config=config)
        daemon.start()
        # save_state called each tick + once at finally
        assert orch.save_count == 6

    def test_verify_processed_each_tick(self):
        orch = FakeOrchestrator()
        config = DaemonConfig(tick_interval=0.0, max_ticks=3)
        daemon = Daemon(orch, config=config)
        daemon.start()
        assert orch.verify_scheduler.process_count == 3

    def test_events_published(self):
        orch = FakeOrchestrator()
        bus = EventBus()
        received = []
        bus.subscribe("tick_complete", lambda e: received.append(e))
        config = DaemonConfig(tick_interval=0.0, max_ticks=3)
        daemon = Daemon(orch, event_bus=bus, config=config)
        daemon.start()
        assert len(received) == 3
        assert received[0].details["tick_number"] == 1
        assert received[2].details["tick_number"] == 3

    def test_graceful_stop(self):
        orch = FakeOrchestrator()
        config = DaemonConfig(tick_interval=0.0, max_ticks=100)
        daemon = Daemon(orch, config=config)

        # Stop after 2 ticks by monkey-patching
        original_tick = daemon._tick

        def counting_tick():
            original_tick()
            if daemon.tick_count >= 2:
                daemon.stop()

        daemon._tick = counting_tick
        daemon.start()
        assert daemon.tick_count == 2

    def test_health_check_interval(self):
        orch = FakeOrchestrator()
        config = DaemonConfig(
            tick_interval=0.0, max_ticks=10, health_check_interval=3,
        )
        health_check_ticks = []

        daemon = Daemon(orch, config=config)
        original = daemon._run_health_checks

        def tracking_health():
            health_check_ticks.append(daemon.tick_count)
            original()

        daemon._run_health_checks = tracking_health
        daemon.start()
        # Health checks at ticks 3, 6, 9
        assert health_check_ticks == [3, 6, 9]

    def test_auto_tune_calls_at_interval(self):
        """Auto-tune runs at the configured interval."""
        orch = FakeOrchestrator()
        config = DaemonConfig(
            tick_interval=0.0, max_ticks=4, auto_tune_interval=2,
        )
        auto_tune_ticks = []
        daemon = Daemon(orch, config=config)
        original = daemon._run_auto_tune

        def tracking_tune():
            auto_tune_ticks.append(daemon.tick_count)
            original()

        daemon._run_auto_tune = tracking_tune
        daemon.start()
        assert auto_tune_ticks == [2, 4]

    def test_auto_tune_skipped_without_tracker(self):
        """Auto-tune completes without error when meta_param_tracker is None."""
        orch = FakeOrchestrator()
        orch._meta_param_tracker = None
        config = DaemonConfig(tick_interval=0.0, max_ticks=2, auto_tune_interval=1)
        daemon = Daemon(orch, config=config)
        daemon.start()  # Should not raise
        assert daemon.tick_count == 2

    def test_metrics_collector_counts_ticks(self):
        """MetricsCollector attached to event bus counts tick events."""
        from sg.metrics import MetricsCollector
        orch = FakeOrchestrator()
        bus = EventBus()
        collector = MetricsCollector()
        collector.attach(bus)
        config = DaemonConfig(tick_interval=0.0, max_ticks=3)
        daemon = Daemon(orch, event_bus=bus, config=config,
                        metrics_collector=collector)
        daemon.start()
        assert collector.sg_daemon_ticks_total.value == 3
