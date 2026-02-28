"""NetworkKernel ABC — abstract kernel interface for gene execution.

Production: NM D-Bus, ip link, sysfs, tcpdump.
Development: in-memory simulation with configurable failures.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class NetworkKernel(ABC):
    """Abstract kernel interface injected into gene namespaces as `gene_sdk`."""

    # --- State management ---

    @abstractmethod
    def reset(self) -> None:
        """Clear all state."""
        ...

    # --- Bridge operations ---

    @abstractmethod
    def create_bridge(self, name: str, interfaces: list[str]) -> dict:
        """Create a network bridge. Returns bridge state as dict."""
        ...

    @abstractmethod
    def delete_bridge(self, name: str) -> None:
        """Delete a bridge and detach all interfaces."""
        ...

    @abstractmethod
    def attach_interface(self, bridge: str, interface: str) -> None:
        """Attach an interface to an existing bridge."""
        ...

    @abstractmethod
    def detach_interface(self, bridge: str, interface: str) -> None:
        """Detach an interface from a bridge."""
        ...

    @abstractmethod
    def get_bridge(self, name: str) -> dict | None:
        """Get bridge state, or None if it doesn't exist."""
        ...

    # --- STP operations ---

    @abstractmethod
    def set_stp(self, bridge_name: str, enabled: bool, forward_delay: int) -> dict:
        """Configure STP on an existing bridge. Returns updated bridge state."""
        ...

    @abstractmethod
    def get_stp_state(self, bridge: str) -> dict:
        """Get STP state for a bridge."""
        ...

    # --- MAC operations ---

    @abstractmethod
    def get_device_mac(self, device: str) -> str:
        """Get the MAC address of a device."""
        ...

    @abstractmethod
    def set_device_mac(self, device: str, mac: str) -> None:
        """Set the MAC address of a device."""
        ...

    @abstractmethod
    def send_gratuitous_arp(self, interface: str, mac: str) -> None:
        """Send a gratuitous ARP to announce a MAC address change."""
        ...

    # --- Bond operations ---

    @abstractmethod
    def create_bond(self, name: str, mode: str, members: list[str]) -> dict:
        """Create a network bond. Returns bond state as dict."""
        ...

    @abstractmethod
    def delete_bond(self, name: str) -> None:
        """Delete a bond."""
        ...

    @abstractmethod
    def get_bond(self, name: str) -> dict | None:
        """Get bond state, or None if it doesn't exist."""
        ...

    # --- VLAN operations ---

    @abstractmethod
    def create_vlan(self, parent: str, vlan_id: int) -> dict:
        """Create a VLAN on a parent interface. Returns VLAN state as dict."""
        ...

    @abstractmethod
    def delete_vlan(self, parent: str, vlan_id: int) -> None:
        """Delete a VLAN."""
        ...

    @abstractmethod
    def get_vlan(self, parent: str, vlan_id: int) -> dict | None:
        """Get VLAN state, or None if it doesn't exist."""
        ...

    # --- Diagnostic reads ---

    @abstractmethod
    def read_fdb(self, bridge: str) -> list[dict]:
        """Read the forwarding database of a bridge."""
        ...

    @abstractmethod
    def get_interface_state(self, interface: str) -> dict:
        """Get interface state (carrier, operstate, mac, etc.)."""
        ...

    @abstractmethod
    def get_arp_table(self) -> list[dict]:
        """Read the ARP table."""
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
