"""MockNetworkKernel — in-memory simulation for development/testing.

Supports configurable failure injection, bridge/STP management,
and state inspection.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from sg.kernel.base import NetworkKernel


@dataclass
class BridgeState:
    name: str
    interfaces: list[str]
    stp_enabled: bool = False
    forward_delay: int = 15


class MockNetworkKernel(NetworkKernel):
    """In-memory network simulation. Injected into genes as `gene_sdk`."""

    def __init__(self) -> None:
        self._bridges: dict[str, BridgeState] = {}
        self._injected_failures: dict[str, str] = {}

    def reset(self) -> None:
        self._bridges.clear()
        self._injected_failures.clear()

    def inject_failure(self, operation: str, message: str) -> None:
        """Inject a one-shot failure for the given operation."""
        self._injected_failures[operation] = message

    def _check_failure(self, operation: str) -> None:
        msg = self._injected_failures.pop(operation, None)
        if msg is not None:
            raise RuntimeError(msg)

    def create_bridge(self, name: str, interfaces: list[str]) -> dict:
        self._check_failure("create_bridge")
        if not name:
            raise ValueError("bridge name cannot be empty")
        if name in self._bridges:
            raise ValueError(f"bridge '{name}' already exists")
        bridge = BridgeState(name=name, interfaces=list(interfaces))
        self._bridges[name] = bridge
        return self._bridge_dict(bridge)

    def set_stp(self, bridge_name: str, enabled: bool, forward_delay: int) -> dict:
        self._check_failure("set_stp")
        bridge = self._bridges.get(bridge_name)
        if bridge is None:
            raise ValueError(f"bridge '{bridge_name}' does not exist")
        if forward_delay < 1 or forward_delay > 30:
            raise ValueError(f"forward_delay must be 1-30, got {forward_delay}")
        bridge.stp_enabled = enabled
        bridge.forward_delay = forward_delay
        return self._bridge_dict(bridge)

    def get_bridge(self, name: str) -> dict | None:
        bridge = self._bridges.get(name)
        if bridge is None:
            return None
        return self._bridge_dict(bridge)

    @staticmethod
    def _bridge_dict(bridge: BridgeState) -> dict:
        return {
            "name": bridge.name,
            "interfaces": bridge.interfaces,
            "stp_enabled": bridge.stp_enabled,
            "forward_delay": bridge.forward_delay,
        }
