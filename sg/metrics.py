"""Prometheus-style metrics export.

Provides Counter and Gauge primitives, a MetricsCollector that subscribes
to the event bus, and an export() method producing Prometheus exposition format.
"""
from __future__ import annotations

from sg.log import get_logger

logger = get_logger("metrics")


class Counter:
    """Monotonically increasing counter."""

    def __init__(self, name: str, help_text: str = "") -> None:
        self.name = name
        self.help_text = help_text
        self._value: float = 0.0

    def inc(self, amount: float = 1.0) -> None:
        self._value += amount

    @property
    def value(self) -> float:
        return self._value


class Gauge:
    """Point-in-time value."""

    def __init__(self, name: str, help_text: str = "") -> None:
        self.name = name
        self.help_text = help_text
        self._value: float = 0.0

    def set(self, value: float) -> None:
        self._value = value

    @property
    def value(self) -> float:
        return self._value


class MetricsCollector:
    """Collects SG runtime metrics and exports in Prometheus format."""

    def __init__(self) -> None:
        self.sg_promotions_total = Counter(
            "sg_promotions_total", "Total gene promotions")
        self.sg_demotions_total = Counter(
            "sg_demotions_total", "Total gene demotions")
        self.sg_mutations_total = Counter(
            "sg_mutations_total", "Total mutations generated")
        self.sg_pathway_executions_total = Counter(
            "sg_pathway_executions_total", "Total pathway executions")
        self.sg_pathway_failures_total = Counter(
            "sg_pathway_failures_total", "Total pathway failures")
        self.sg_daemon_ticks_total = Counter(
            "sg_daemon_ticks_total", "Total daemon ticks")
        self.sg_avg_fitness = Gauge(
            "sg_avg_fitness", "Average fitness across dominant alleles")
        self.sg_active_loci = Gauge(
            "sg_active_loci", "Number of active loci")
        self.sg_last_tick_duration_ms = Gauge(
            "sg_last_tick_duration_ms", "Duration of last daemon tick in ms")

    def attach(self, event_bus) -> None:
        """Subscribe to all events on the bus."""
        event_bus.subscribe_all(self._handle_event)

    def _handle_event(self, event) -> None:
        t = event.event_type
        if t == "allele_promoted":
            self.sg_promotions_total.inc()
        elif t == "allele_demoted":
            self.sg_demotions_total.inc()
        elif t == "mutation_generated":
            self.sg_mutations_total.inc()
        elif t == "pathway_promoted":
            self.sg_pathway_executions_total.inc()
        elif t == "pathway_failed":
            self.sg_pathway_failures_total.inc()
        elif t == "tick_complete":
            self.sg_daemon_ticks_total.inc()
            duration = event.details.get("duration_ms", 0.0)
            self.sg_last_tick_duration_ms.set(duration)

    def export(self) -> str:
        """Export metrics in Prometheus exposition format."""
        lines: list[str] = []
        for attr_name in dir(self):
            obj = getattr(self, attr_name)
            if isinstance(obj, (Counter, Gauge)):
                kind = "counter" if isinstance(obj, Counter) else "gauge"
                if obj.help_text:
                    lines.append(f"# HELP {obj.name} {obj.help_text}")
                lines.append(f"# TYPE {obj.name} {kind}")
                lines.append(f"{obj.name} {obj.value}")
        return "\n".join(lines) + "\n"
