"""Tests for Prometheus-style metrics export."""
from __future__ import annotations

from sg.metrics import Counter, Gauge, MetricsCollector
from sg.events import (
    EventBus, allele_promoted, allele_demoted, mutation_generated,
    pathway_promoted, pathway_failed, tick_complete,
)


class TestCounter:
    def test_starts_at_zero(self):
        c = Counter("test_counter")
        assert c.value == 0.0

    def test_inc_default(self):
        c = Counter("test_counter")
        c.inc()
        assert c.value == 1.0

    def test_inc_amount(self):
        c = Counter("test_counter")
        c.inc(5.0)
        assert c.value == 5.0

    def test_multiple_inc(self):
        c = Counter("test_counter")
        c.inc()
        c.inc(3.0)
        assert c.value == 4.0


class TestGauge:
    def test_starts_at_zero(self):
        g = Gauge("test_gauge")
        assert g.value == 0.0

    def test_set(self):
        g = Gauge("test_gauge")
        g.set(42.0)
        assert g.value == 42.0

    def test_set_overwrites(self):
        g = Gauge("test_gauge")
        g.set(10.0)
        g.set(20.0)
        assert g.value == 20.0


class TestMetricsCollector:
    def test_attach_and_count_events(self):
        bus = EventBus()
        mc = MetricsCollector()
        mc.attach(bus)

        bus.publish(allele_promoted("l", "s", 0.9))
        bus.publish(allele_promoted("l2", "s2", 0.8))
        bus.publish(allele_demoted("l", "s"))
        bus.publish(mutation_generated("l", "s", 3))
        bus.publish(pathway_promoted("p", "s", 0.7))
        bus.publish(pathway_failed("p", "err"))
        bus.publish(tick_complete(1, 42.5))

        assert mc.sg_promotions_total.value == 2.0
        assert mc.sg_demotions_total.value == 1.0
        assert mc.sg_mutations_total.value == 1.0
        assert mc.sg_pathway_executions_total.value == 1.0
        assert mc.sg_pathway_failures_total.value == 1.0
        assert mc.sg_daemon_ticks_total.value == 1.0
        assert mc.sg_last_tick_duration_ms.value == 42.5

    def test_export_prometheus_format(self):
        mc = MetricsCollector()
        mc.sg_promotions_total.inc(5)
        mc.sg_avg_fitness.set(0.85)

        output = mc.export()
        assert "# TYPE sg_promotions_total counter" in output
        assert "sg_promotions_total 5.0" in output
        assert "# TYPE sg_avg_fitness gauge" in output
        assert "sg_avg_fitness 0.85" in output

    def test_export_contains_help(self):
        mc = MetricsCollector()
        output = mc.export()
        assert "# HELP sg_promotions_total" in output
        assert "# HELP sg_avg_fitness" in output

    def test_export_all_metrics_present(self):
        mc = MetricsCollector()
        output = mc.export()
        expected_names = [
            "sg_promotions_total", "sg_demotions_total",
            "sg_mutations_total", "sg_pathway_executions_total",
            "sg_pathway_failures_total", "sg_daemon_ticks_total",
            "sg_avg_fitness", "sg_active_loci", "sg_last_tick_duration_ms",
        ]
        for name in expected_names:
            assert name in output, f"Missing metric: {name}"

    def test_tick_updates_duration(self):
        bus = EventBus()
        mc = MetricsCollector()
        mc.attach(bus)

        bus.publish(tick_complete(1, 100.0))
        assert mc.sg_last_tick_duration_ms.value == 100.0
        bus.publish(tick_complete(2, 50.0))
        assert mc.sg_last_tick_duration_ms.value == 50.0

    def test_save_and_load_round_trip(self, tmp_path):
        """Metrics survive save/load cycle."""
        mc = MetricsCollector()
        mc.sg_promotions_total.inc(3)
        mc.sg_avg_fitness.set(0.75)
        mc.sg_daemon_ticks_total.inc(10)

        path = tmp_path / "metrics.json"
        mc.save(path)

        loaded = MetricsCollector.load(path)
        assert loaded is not None
        assert loaded.sg_promotions_total.value == 3.0
        assert loaded.sg_avg_fitness.value == 0.75
        assert loaded.sg_daemon_ticks_total.value == 10.0

    def test_load_missing_returns_none(self, tmp_path):
        """Loading from a nonexistent path returns None."""
        assert MetricsCollector.load(tmp_path / "nope.json") is None

    def test_load_corrupt_returns_none(self, tmp_path):
        """Loading from a corrupt file returns None."""
        bad = tmp_path / "bad.json"
        bad.write_text("not json at all")
        assert MetricsCollector.load(bad) is None
