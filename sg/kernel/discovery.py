"""Kernel plugin discovery via entry points.

Discovers kernels registered under the ``sg.kernels`` entry-point group.
Built-in kernels (stub, mock, production) are registered in pyproject.toml.
Third-party packages register additional kernels via the same group.

Usage::

    from sg.kernel.discovery import discover_kernels, load_kernel

    available = discover_kernels()   # {"stub": <EntryPoint>, ...}
    kernel = load_kernel("stub")     # StubKernel instance
"""
from __future__ import annotations

import importlib.metadata
from typing import Dict, List, Type

from sg.kernel.base import Kernel

_ENTRY_POINT_GROUP = "sg.kernels"


class KernelNotFoundError(Exception):
    """Raised when a requested kernel name is not registered."""


class KernelLoadError(Exception):
    """Raised when a kernel entry point fails to load."""


def _get_entry_points(group: str):
    """Get entry points for a group, Python 3.9+ compatible."""
    eps = importlib.metadata.entry_points()
    if hasattr(eps, "select"):
        # Python 3.10+
        return eps.select(group=group)
    # Python 3.9: returns dict[str, list[EntryPoint]]
    return eps.get(group, [])


def discover_kernels() -> Dict[str, importlib.metadata.EntryPoint]:
    """Return a mapping of kernel name -> EntryPoint for all registered kernels."""
    result: Dict[str, importlib.metadata.EntryPoint] = {}
    for ep in _get_entry_points(_ENTRY_POINT_GROUP):
        result[ep.name] = ep
    return result


def list_kernel_names() -> List[str]:
    """Return sorted list of all registered kernel names."""
    return sorted(discover_kernels().keys())


def load_kernel_class(name: str) -> Type[Kernel]:
    """Load and return the kernel class for the given name.

    Raises:
        KernelNotFoundError: If no kernel is registered with that name.
        KernelLoadError: If the entry point fails to load.
    """
    kernels = discover_kernels()
    if name not in kernels:
        available = ", ".join(sorted(kernels.keys())) or "(none)"
        raise KernelNotFoundError(
            f"unknown kernel '{name}'. Available: {available}"
        )

    ep = kernels[name]
    try:
        cls = ep.load()
    except Exception as e:
        raise KernelLoadError(
            f"failed to load kernel '{name}' from {ep.value}: {e}"
        ) from e

    if not (isinstance(cls, type) and issubclass(cls, Kernel)):
        raise KernelLoadError(
            f"kernel '{name}' ({ep.value}) is not a Kernel subclass"
        )
    return cls


def load_kernel(name: str, **kwargs) -> Kernel:
    """Load and instantiate a kernel by name."""
    cls = load_kernel_class(name)
    return cls(**kwargs)
