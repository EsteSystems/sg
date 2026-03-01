"""Safety mechanisms — transactions, blast radius, rollback.

Wraps configuration gene execution in undo-log transactions.
Classifies loci by blast radius (none -> low -> medium -> high -> critical)
and applies graded safety policies: shadow mode, canary, convergence
checks, resilience requirements.

SafeKernel is a generic proxy that intercepts @mutating kernel operations
and records undo actions. On gene failure, the transaction rolls back all
changes made by that allele attempt.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from sg.kernel.base import Kernel
from sg.parser.types import BlastRadius


@dataclass
class UndoAction:
    """A recorded undo action — reverses one kernel mutation."""
    description: str
    fn: Callable[[], None]


class Transaction:
    """Undo-log transaction wrapping a single gene execution.

    Records undo actions as mutating kernel operations succeed.
    On rollback, replays them in reverse order to restore prior state.
    """

    def __init__(self, locus: str, risk: BlastRadius) -> None:
        self.locus = locus
        self.risk = risk
        self.undo_log: list[UndoAction] = []
        self.committed = False
        self.rolled_back = False

    def record(self, description: str, undo_fn: Callable[[], None]) -> None:
        self.undo_log.append(UndoAction(description=description, fn=undo_fn))

    def rollback(self) -> list[str]:
        """Roll back all recorded actions in reverse order.

        Returns list of descriptions of actions that were rolled back.
        Errors during rollback are printed but do not halt the process.
        """
        rolled_back: list[str] = []
        for action in reversed(self.undo_log):
            try:
                action.fn()
                rolled_back.append(action.description)
            except Exception as e:
                print(f"  [rollback] failed: {action.description}: {e}")
        self.undo_log.clear()
        self.rolled_back = True
        return rolled_back

    def commit(self) -> None:
        """Commit the transaction — discard undo log."""
        self.committed = True
        self.undo_log.clear()

    @property
    def action_count(self) -> int:
        return len(self.undo_log)


def _get_undo_spec(kernel_type: type, method_name: str):
    """Walk the MRO to find @mutating metadata for a method."""
    for cls in kernel_type.__mro__:
        method = cls.__dict__.get(method_name)
        if method is not None and hasattr(method, '_sg_undo_spec'):
            return method._sg_undo_spec
    return None


class SafeKernel(Kernel):
    """Generic proxy that wraps @mutating kernel methods with undo recording.

    Delegates all calls to the inner kernel. Methods decorated with @mutating
    on the inner kernel's class hierarchy are intercepted: a pre-state snapshot
    is taken (if declared), the method executes, and an undo action is recorded
    in the transaction. Non-mutating methods pass through unchanged.

    Works with any domain kernel — no domain-specific code here.
    """

    def __init__(self, inner: Kernel, transaction: Transaction) -> None:
        self._inner = inner
        self._txn = transaction

    # --- Kernel abstract methods (direct delegation) ---

    def reset(self) -> None:
        self._inner.reset()

    def track_resource(self, resource_type: str, name: str) -> None:
        self._inner.track_resource(resource_type, name)

    def untrack_resource(self, resource_type: str, name: str) -> None:
        self._inner.untrack_resource(resource_type, name)

    def tracked_resources(self) -> list[tuple[str, str]]:
        return self._inner.tracked_resources()

    # --- Kernel concrete methods (delegate to preserve domain overrides) ---

    def delete_resource(self, resource_type: str, name: str) -> None:
        self._inner.delete_resource(resource_type, name)

    def describe_operations(self) -> list[str]:
        return self._inner.describe_operations()

    def mutation_prompt_context(self) -> str:
        return self._inner.mutation_prompt_context()

    def domain_name(self) -> str:
        return self._inner.domain_name()

    # --- Domain-specific methods (auto-wrapped via __getattr__) ---

    def __getattr__(self, name: str):
        attr = getattr(self._inner, name)
        if callable(attr):
            spec = _get_undo_spec(type(self._inner), name)
            if spec is not None:
                return self._make_wrapper(name, attr, spec)
        return attr

    def _make_wrapper(self, name: str, bound_method, spec):
        inner = self._inner
        txn = self._txn

        def wrapper(*args, **kwargs):
            snap = None
            if spec.snapshot:
                snap = spec.snapshot(inner, *args, **kwargs)
            result = bound_method(*args, **kwargs)
            txn.record(
                f"undo {name}",
                lambda s=snap, a=args, k=kwargs: spec.undo(inner, s, *a, **k),
            )
            return result

        return wrapper


def requires_transaction(risk: BlastRadius) -> bool:
    """Return True if a blast radius level requires transaction wrapping."""
    return risk in (BlastRadius.LOW, BlastRadius.MEDIUM,
                    BlastRadius.HIGH, BlastRadius.CRITICAL)


SHADOW_PROMOTION_THRESHOLD = 3


def is_shadow_only(risk: BlastRadius) -> bool:
    """Return True if blast radius requires shadow execution before live."""
    return risk in (BlastRadius.HIGH, BlastRadius.CRITICAL)
