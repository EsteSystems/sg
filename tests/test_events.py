"""Tests for the event bus (E.5.2)."""
from sg.events import (
    Event, EventBus, MAX_EVENT_HISTORY,
    allele_promoted, allele_demoted, pathway_promoted,
    pathway_failed, mutation_generated, stabilization_complete,
    fitness_feedback, tick_complete,
)


class TestEvent:
    def test_to_dict(self):
        e = allele_promoted("bridge_create", "abc123", 0.9)
        d = e.to_dict()
        assert d["event_type"] == "allele_promoted"
        assert d["details"]["locus"] == "bridge_create"
        assert d["details"]["sha"] == "abc123"
        assert d["details"]["fitness"] == 0.9
        assert "timestamp" in d

    def test_factory_functions(self):
        assert allele_promoted("l", "s", 0.5).event_type == "allele_promoted"
        assert allele_demoted("l", "s").event_type == "allele_demoted"
        assert pathway_promoted("p", "s", 0.5).event_type == "pathway_promoted"
        assert pathway_failed("p", "err").event_type == "pathway_failed"
        assert mutation_generated("l", "s", 3).event_type == "mutation_generated"
        assert stabilization_complete("p").event_type == "stabilization_complete"
        assert fitness_feedback("d", "c", "convergence", True).event_type == "fitness_feedback"
        assert tick_complete(1, 50.0).event_type == "tick_complete"


class TestEventBus:
    def test_subscribe_and_publish(self):
        bus = EventBus()
        received = []
        bus.subscribe("allele_promoted", lambda e: received.append(e))
        bus.publish(allele_promoted("l", "s", 0.9))
        assert len(received) == 1
        assert received[0].details["locus"] == "l"

    def test_topic_filtering(self):
        bus = EventBus()
        received = []
        bus.subscribe("allele_promoted", lambda e: received.append(e))
        bus.publish(allele_demoted("l", "s"))
        assert len(received) == 0

    def test_global_handler_receives_all(self):
        bus = EventBus()
        received = []
        bus.subscribe_all(lambda e: received.append(e))
        bus.publish(allele_promoted("l", "s", 0.9))
        bus.publish(allele_demoted("l", "s"))
        assert len(received) == 2

    def test_handler_error_does_not_block(self):
        bus = EventBus()
        received = []

        def bad_handler(e):
            raise RuntimeError("oops")

        bus.subscribe("allele_promoted", bad_handler)
        bus.subscribe("allele_promoted", lambda e: received.append(e))
        bus.publish(allele_promoted("l", "s", 0.9))
        # Second handler still called despite first raising
        assert len(received) == 1

    def test_global_handler_error_does_not_block(self):
        bus = EventBus()
        received = []
        bus.subscribe_all(lambda e: (_ for _ in ()).throw(RuntimeError("fail")))
        bus.subscribe("allele_promoted", lambda e: received.append(e))
        bus.publish(allele_promoted("l", "s", 0.9))
        assert len(received) == 1

    def test_event_history_bounded(self):
        bus = EventBus()
        for i in range(MAX_EVENT_HISTORY + 100):
            bus.publish(tick_complete(i, 1.0))
        assert len(bus._history) == MAX_EVENT_HISTORY

    def test_recent_returns_latest(self):
        bus = EventBus()
        for i in range(10):
            bus.publish(tick_complete(i, 1.0))
        recent = bus.recent(count=3)
        assert len(recent) == 3
        assert recent[0].details["tick_number"] == 7
        assert recent[2].details["tick_number"] == 9

    def test_recent_filtered_by_type(self):
        bus = EventBus()
        bus.publish(allele_promoted("l", "s", 0.9))
        bus.publish(allele_demoted("l", "s"))
        bus.publish(allele_promoted("l2", "s2", 0.8))
        recent = bus.recent(count=10, event_type="allele_promoted")
        assert len(recent) == 2
        assert all(e.event_type == "allele_promoted" for e in recent)

    def test_multiple_handlers_same_type(self):
        bus = EventBus()
        a, b = [], []
        bus.subscribe("x", lambda e: a.append(e))
        bus.subscribe("x", lambda e: b.append(e))
        bus.publish(Event("x"))
        assert len(a) == 1
        assert len(b) == 1
