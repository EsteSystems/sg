"""Kernel ABC — domain-agnostic base interface.

Kernel: the domain-agnostic interface that the evolutionary engine depends on.

Domain plugins implement concrete subclasses of Kernel. The engine never
imports domain-specific kernel classes — it only depends on this ABC.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable


class Kernel(ABC):
    """Domain-agnostic kernel interface injected into gene namespaces as `gene_sdk`.

    Every domain kernel must extend this class. The evolutionary engine
    interacts with kernels only through this base interface; domain-specific
    methods are called by genes directly via the gene_sdk binding.
    """

    # --- State management ---

    @abstractmethod
    def reset(self) -> None:
        """Clear all state."""
        ...

    # --- Resource tracking ---

    @abstractmethod
    def track_resource(self, resource_type: str, name: str) -> None:
        """Track a created resource for cleanup."""
        ...

    @abstractmethod
    def untrack_resource(self, resource_type: str, name: str) -> None:
        """Untrack a resource."""
        ...

    @abstractmethod
    def tracked_resources(self) -> list[tuple[str, str]]:
        """Return all tracked resources as (type, name) pairs."""
        ...

    def delete_resource(self, resource_type: str, name: str) -> None:
        """Delete a tracked resource by type and name.

        Domain kernels override this to dispatch to type-specific delete
        operations. Default implementation just untracks the resource.
        """
        self.untrack_resource(resource_type, name)

    # --- Self-description (used by mutation engine for dynamic prompts) ---

    def describe_operations(self) -> list[str]:
        """Return human-readable list of available gene_sdk operations.

        Used by the mutation engine to build domain-aware LLM prompts.
        Override in domain kernels to list all operations genes can call.
        """
        return []

    def mutation_prompt_context(self) -> str:
        """Domain-specific context injected into LLM mutation prompts.

        Override to provide domain conventions, examples, and constraints
        that help the LLM generate better mutations.
        """
        return ""

    def domain_name(self) -> str:
        """Short identifier for this kernel's domain (e.g., 'network', 'data')."""
        return "generic"

    def resource_mappers(self) -> dict:
        """Return topology resource type → mapper function mapping.

        Each mapper is a callable(resource, data) -> TopologyStep.
        Domain kernels override this to register their resource types
        for topology decomposition.
        """
        return {}

    def create_shadow(self) -> Kernel:
        """Create a shadow (mock) kernel for safe testing.

        Shadow kernels simulate the domain without real side-effects.
        Used by the safety layer to validate HIGH/CRITICAL risk alleles
        before live execution.

        Domain kernels must override this to return an appropriate mock.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support shadow execution"
        )


# --- @mutating decorator for SafeKernel auto-wrapping ---


@dataclass
class UndoSpec:
    """Undo specification attached to @mutating kernel methods."""
    undo: Callable  # (kernel, snapshot, *args, **kwargs) -> None
    snapshot: Callable | None = None  # (kernel, *args, **kwargs) -> Any


def mutating(*, undo: Callable, snapshot: Callable | None = None):
    """Decorator declaring a kernel method as state-mutating with undo.

    SafeKernel uses this metadata to automatically wrap calls with
    transaction recording. Undecorated methods pass through unwrapped.

    Args:
        undo: Callable(kernel, snapshot, *args, **kwargs) reversing the mutation.
        snapshot: Optional callable(kernel, *args, **kwargs) capturing pre-state.
    """
    def decorator(method):
        method._sg_undo_spec = UndoSpec(undo=undo, snapshot=snapshot)
        return method
    return decorator


