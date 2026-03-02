"""NetworkKernel — abstract kernel interface for Linux networking operations.

Production: NM D-Bus, ip link, sysfs, tcpdump.
Development: in-memory simulation with configurable failures.
"""
from __future__ import annotations

from abc import abstractmethod

from sg.kernel.base import Kernel, mutating


class NetworkKernel(Kernel):
    """Abstract kernel interface for Linux networking operations.

    Production: NM D-Bus, ip link, sysfs, tcpdump.
    Development: in-memory simulation with configurable failures.
    """

    # --- Self-description ---

    def resource_mappers(self) -> dict:
        """Return network topology resource type -> mapper mappings."""
        from sg_network.mappers import NETWORK_RESOURCE_MAPPERS
        return NETWORK_RESOURCE_MAPPERS

    def delete_resource(self, resource_type: str, name: str) -> None:
        """Delete a network resource by type."""
        if resource_type == "bridge":
            self.delete_bridge(name)
        elif resource_type == "bond":
            self.delete_bond(name)
        elif resource_type == "vlan":
            parts = name.split(".", 1)
            if len(parts) == 2:
                self.delete_vlan(parts[0], int(parts[1]))
        self.untrack_resource(resource_type, name)

    def describe_operations(self) -> list[str]:
        return [
            "create_bridge(name: str, interfaces: list[str]) -> dict",
            "delete_bridge(name: str) -> None",
            "attach_interface(bridge: str, interface: str) -> None",
            "detach_interface(bridge: str, interface: str) -> None",
            "get_bridge(name: str) -> dict | None",
            "set_stp(bridge_name: str, enabled: bool, forward_delay: int) -> dict",
            "get_stp_state(bridge: str) -> dict",
            "get_device_mac(device: str) -> str",
            "set_device_mac(device: str, mac: str) -> None",
            "send_gratuitous_arp(interface: str, mac: str) -> None",
            "create_bond(name: str, mode: str, members: list[str]) -> dict",
            "delete_bond(name: str) -> None",
            "get_bond(name: str) -> dict | None",
            "create_vlan(parent: str, vlan_id: int) -> dict",
            "delete_vlan(parent: str, vlan_id: int) -> None",
            "get_vlan(parent: str, vlan_id: int) -> dict | None",
            "read_fdb(bridge: str) -> list[dict]",
            "get_interface_state(interface: str) -> dict",
            "get_arp_table() -> list[dict]",
        ]

    def mutation_prompt_context(self) -> str:
        return (
            "This gene operates on Linux network configuration. "
            "gene_sdk is a NetworkKernel providing bridge, bond, VLAN, STP, "
            "MAC, and diagnostic operations. Bridges group interfaces. "
            "VLANs segment traffic. STP prevents loops. Bonds aggregate links."
        )

    def domain_name(self) -> str:
        return "network"

    # --- Bridge operations ---

    @mutating(
        undo=lambda k, snap, name, interfaces: k.delete_bridge(name),
    )
    @abstractmethod
    def create_bridge(self, name: str, interfaces: list[str]) -> dict:
        """Create a network bridge. Returns bridge state as dict."""
        ...

    @mutating(
        snapshot=lambda k, name: k.get_bridge(name),
        undo=lambda k, snap, name: (
            k.create_bridge(snap["name"], snap["interfaces"]) if snap else None
        ),
    )
    @abstractmethod
    def delete_bridge(self, name: str) -> None:
        """Delete a bridge and detach all interfaces."""
        ...

    @mutating(
        undo=lambda k, snap, bridge, interface: k.detach_interface(bridge, interface),
    )
    @abstractmethod
    def attach_interface(self, bridge: str, interface: str) -> None:
        """Attach an interface to an existing bridge."""
        ...

    @mutating(
        undo=lambda k, snap, bridge, interface: k.attach_interface(bridge, interface),
    )
    @abstractmethod
    def detach_interface(self, bridge: str, interface: str) -> None:
        """Detach an interface from a bridge."""
        ...

    @abstractmethod
    def get_bridge(self, name: str) -> dict | None:
        """Get bridge state, or None if it doesn't exist."""
        ...

    # --- STP operations ---

    @mutating(
        snapshot=lambda k, bridge_name, enabled, forward_delay: k.get_stp_state(bridge_name),
        undo=lambda k, snap, bridge_name, enabled, forward_delay: (
            k.set_stp(bridge_name, snap["enabled"], snap["forward_delay"])
        ),
    )
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

    @mutating(
        snapshot=lambda k, device, mac: k.get_device_mac(device),
        undo=lambda k, snap, device, mac: k.set_device_mac(device, snap),
    )
    @abstractmethod
    def set_device_mac(self, device: str, mac: str) -> None:
        """Set the MAC address of a device."""
        ...

    @abstractmethod
    def send_gratuitous_arp(self, interface: str, mac: str) -> None:
        """Send a gratuitous ARP to announce a MAC address change."""
        ...

    # --- Bond operations ---

    @mutating(
        undo=lambda k, snap, name, mode, members: k.delete_bond(name),
    )
    @abstractmethod
    def create_bond(self, name: str, mode: str, members: list[str]) -> dict:
        """Create a network bond. Returns bond state as dict."""
        ...

    @mutating(
        snapshot=lambda k, name: k.get_bond(name),
        undo=lambda k, snap, name: (
            k.create_bond(snap["name"], snap["mode"], snap["members"]) if snap else None
        ),
    )
    @abstractmethod
    def delete_bond(self, name: str) -> None:
        """Delete a bond."""
        ...

    @abstractmethod
    def get_bond(self, name: str) -> dict | None:
        """Get bond state, or None if it doesn't exist."""
        ...

    # --- VLAN operations ---

    @mutating(
        undo=lambda k, snap, parent, vlan_id: k.delete_vlan(parent, vlan_id),
    )
    @abstractmethod
    def create_vlan(self, parent: str, vlan_id: int) -> dict:
        """Create a VLAN on a parent interface. Returns VLAN state as dict."""
        ...

    @mutating(
        snapshot=lambda k, parent, vlan_id: k.get_vlan(parent, vlan_id),
        undo=lambda k, snap, parent, vlan_id: (
            k.create_vlan(parent, vlan_id) if snap else None
        ),
    )
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
