"""Pub/sub event bus for orchestrator lifecycle events.

Enables decoupled listeners for metrics, daemon scheduling, contract
evolution, and dashboard streaming.  Events are simple dataclasses.
Subscribers register callables.  Delivery is synchronous and in-process.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from sg.log import get_logger

logger = get_logger("events")

MAX_EVENT_HISTORY = 1000


@dataclass
class Event:
    """Base event type."""
    event_type: str
    timestamp: float = field(default_factory=time.time)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "details": self.details,
        }


# --- Factory functions for typed events ---

def allele_promoted(locus: str, sha: str, fitness: float) -> Event:
    return Event("allele_promoted", details={
        "locus": locus, "sha": sha, "fitness": fitness,
    })


def allele_demoted(locus: str, sha: str) -> Event:
    return Event("allele_demoted", details={"locus": locus, "sha": sha})


def pathway_promoted(pathway: str, sha: str, fitness: float) -> Event:
    return Event("pathway_promoted", details={
        "pathway": pathway, "sha": sha, "fitness": fitness,
    })


def pathway_failed(pathway: str, error: str) -> Event:
    return Event("pathway_failed", details={
        "pathway": pathway, "error": error,
    })


def mutation_generated(locus: str, sha: str, candidates: int) -> Event:
    return Event("mutation_generated", details={
        "locus": locus, "sha": sha, "candidates": candidates,
    })


def stabilization_complete(pathway: str) -> Event:
    return Event("stabilization_complete", details={"pathway": pathway})


def fitness_feedback(
    source_locus: str, target_locus: str, timescale: str, healthy: bool,
) -> Event:
    return Event("fitness_feedback", details={
        "source_locus": source_locus, "target_locus": target_locus,
        "timescale": timescale, "healthy": healthy,
    })


def tick_complete(tick_number: int, duration_ms: float) -> Event:
    return Event("tick_complete", details={
        "tick_number": tick_number, "duration_ms": duration_ms,
    })


EventHandler = Callable[[Event], None]


class EventBus:
    """In-process synchronous event bus with topic filtering."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandler]] = {}
        self._global_handlers: list[EventHandler] = []
        self._history: list[Event] = []

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """Subscribe to a specific event type."""
        self._handlers.setdefault(event_type, []).append(handler)

    def subscribe_all(self, handler: EventHandler) -> None:
        """Subscribe to all events."""
        self._global_handlers.append(handler)

    def publish(self, event: Event) -> None:
        """Publish an event to all matching subscribers.

        Handler errors are logged but do not block other handlers.
        """
        self._history.append(event)
        if len(self._history) > MAX_EVENT_HISTORY:
            self._history = self._history[-MAX_EVENT_HISTORY:]

        for handler in self._global_handlers:
            try:
                handler(event)
            except Exception:
                logger.exception("global event handler failed for %s",
                                 event.event_type)

        for handler in self._handlers.get(event.event_type, []):
            try:
                handler(event)
            except Exception:
                logger.exception("event handler failed for %s",
                                 event.event_type)

    def recent(
        self, count: int = 50, event_type: str | None = None,
    ) -> list[Event]:
        """Return recent events, optionally filtered by type."""
        if event_type is not None:
            filtered = [e for e in self._history if e.event_type == event_type]
            return filtered[-count:]
        return self._history[-count:]
