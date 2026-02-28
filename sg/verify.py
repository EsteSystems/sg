"""Verify block auto-scheduling — fire diagnostics after config gene success.

After a configuration gene succeeds, its contract's verify block lists
diagnostic genes to run (with a delay). The VerifyScheduler fires these
on background threads. The diagnostics' existing `feeds` mechanism then
routes results into convergence fitness automatically.
"""
from __future__ import annotations

import json
import re
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sg.orchestrator import Orchestrator
    from sg.parser.types import VerifyStep


def parse_duration(s: str) -> float:
    """Convert a duration string to seconds.

    Supports: "30s", "5m", "1h".
    """
    s = s.strip()
    if s.endswith("s"):
        return float(s[:-1])
    elif s.endswith("m"):
        return float(s[:-1]) * 60
    elif s.endswith("h"):
        return float(s[:-1]) * 3600
    raise ValueError(f"unrecognized duration: {s!r}")


def resolve_verify_params(params: dict[str, str], input_json: str) -> str:
    """Resolve {reference} syntax against the config gene's input.

    {"interface": "{bridge_name}"} + '{"bridge_name": "br0"}'
    → '{"interface": "br0"}'
    """
    data = json.loads(input_json)
    result = {}
    for param_name, param_value in params.items():
        match = re.fullmatch(r"\{(\w+)\}", param_value)
        if match:
            key = match.group(1)
            if key in data:
                result[param_name] = data[key]
        else:
            result[param_name] = param_value
    return json.dumps(result)


class VerifyScheduler:
    """Schedule and run verify diagnostics after config gene success.

    Uses threading.Timer for delayed execution. All timers must be
    joined via wait() before the process exits and state is saved.

    Thread safety: timers fire after run_pathway returns and the main
    thread is blocked in wait(), so only timer threads are active.
    Acceptable for single-shot CLI invocations.
    """

    def __init__(self) -> None:
        self._timers: list[threading.Timer] = []
        self._lock = threading.Lock()

    def schedule(
        self,
        steps: list[VerifyStep],
        delay: float,
        input_json: str,
        orchestrator: Orchestrator,
    ) -> None:
        """Schedule verify steps to fire after delay seconds."""
        for step in steps:
            resolved = resolve_verify_params(step.params, input_json)
            timer = threading.Timer(
                delay,
                self._run_step,
                args=(step.locus, resolved, orchestrator),
            )
            timer.daemon = True
            with self._lock:
                self._timers.append(timer)
            timer.start()

    def _run_step(
        self, locus: str, resolved_input: str, orchestrator: Orchestrator
    ) -> None:
        """Execute a single verify diagnostic."""
        print(f"  [verify] running scheduled diagnostic: {locus}")
        try:
            result = orchestrator.execute_locus(locus, resolved_input)
            if result is None:
                print(f"  [verify] diagnostic {locus} returned no result")
        except Exception as e:
            print(f"  [verify] diagnostic {locus} failed: {e}")

    def wait(self, timeout: float = 60.0) -> None:
        """Block until all pending verify timers have completed."""
        with self._lock:
            timers = list(self._timers)
        for timer in timers:
            timer.join(timeout=timeout)

    @property
    def pending_count(self) -> int:
        """Number of scheduled timers (for testing)."""
        with self._lock:
            return len(self._timers)
