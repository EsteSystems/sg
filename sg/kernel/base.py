"""NetworkKernel ABC — abstract kernel interface for gene execution.

Production: NM D-Bus, ip link, sysfs, tcpdump.
Development: in-memory simulation with configurable failures.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class NetworkKernel(ABC):
    """Abstract kernel interface injected into gene namespaces as `gene_sdk`."""

    @abstractmethod
    def reset(self) -> None:
        """Clear all state."""
        ...

    @abstractmethod
    def create_bridge(self, name: str, interfaces: list[str]) -> dict:
        """Create a network bridge. Returns bridge state as dict."""
        ...

    @abstractmethod
    def set_stp(self, bridge_name: str, enabled: bool, forward_delay: int) -> dict:
        """Configure STP on an existing bridge. Returns updated bridge state."""
        ...

    @abstractmethod
    def get_bridge(self, name: str) -> dict | None:
        """Get bridge state, or None if it doesn't exist."""
        ...
