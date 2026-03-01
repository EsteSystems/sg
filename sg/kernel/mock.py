"""MockNetworkKernel — in-memory simulation for development/testing.

Supports configurable failure injection, bridge/STP/bond/VLAN management,
FDB simulation, MAC operations, and anomaly injection.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from sg.kernel.base import NetworkKernel


@dataclass
class BridgeState:
    name: str
    interfaces: list[str]
    stp_enabled: bool = False
    forward_delay: int = 15


@dataclass
class BondState:
    name: str
    mode: str
    members: list[str]
    active: bool = True


@dataclass
class VlanState:
    parent: str
    vlan_id: int
    name: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = f"{self.parent}.{self.vlan_id}"


@dataclass
class InterfaceState:
    name: str
    mac: str = ""
    carrier: bool = True
    operstate: str = "up"
    master: str = ""


@dataclass
class FdbEntry:
    mac: str
    port: str
    vlan: int = 0
    is_local: bool = False
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class ArpEntry:
    ip: str
    mac: str
    device: str


class MockNetworkKernel(NetworkKernel):
    """In-memory network simulation. Injected into genes as `gene_sdk`."""

    def create_shadow(self) -> MockNetworkKernel:
        """Return a fresh mock kernel for shadow execution."""
        return MockNetworkKernel()

    def __init__(self) -> None:
        self._bridges: dict[str, BridgeState] = {}
        self._bonds: dict[str, BondState] = {}
        self._vlans: dict[str, VlanState] = {}  # keyed by "parent.vlan_id"
        self._interfaces: dict[str, InterfaceState] = {}
        self._fdb: dict[str, list[FdbEntry]] = {}  # keyed by bridge name
        self._arp_table: list[ArpEntry] = []
        self._tracked: list[tuple[str, str]] = []
        self._injected_failures: dict[str, str] = {}
        self._fail_at: int | None = None
        self._mutation_count: int = 0
        self._gratuitous_arps: list[dict] = []

    def reset(self) -> None:
        self._bridges.clear()
        self._bonds.clear()
        self._vlans.clear()
        self._interfaces.clear()
        self._fdb.clear()
        self._arp_table.clear()
        self._tracked.clear()
        self._injected_failures.clear()
        self._fail_at = None
        self._mutation_count = 0
        self._gratuitous_arps.clear()

    # --- Failure injection ---

    def inject_failure(self, operation: str, message: str) -> None:
        """Inject a one-shot failure for the given operation."""
        self._injected_failures[operation] = message

    def set_fail_at(self, n: int) -> None:
        """Fail on the Nth mutating operation (1-indexed)."""
        self._fail_at = n
        self._mutation_count = 0

    def inject_mac_flapping(self, bridge: str, mac: str, ports: list[str]) -> None:
        """Inject MAC flapping: the same MAC oscillates between ports.

        Creates alternating FDB entries so read_fdb returns multiple
        ports for the same MAC address.
        """
        if bridge not in self._bridges:
            raise ValueError(f"bridge '{bridge}' does not exist")
        if bridge not in self._fdb:
            self._fdb[bridge] = []

        now = time.time()
        for i, port in enumerate(ports):
            self._fdb[bridge].append(FdbEntry(
                mac=mac,
                port=port,
                timestamp=now + i * 0.001,
            ))

    def inject_link_failure(self, interface: str) -> None:
        """Simulate a link going down."""
        iface = self._interfaces.get(interface)
        if iface is None:
            self._interfaces[interface] = InterfaceState(
                name=interface, carrier=False, operstate="down"
            )
        else:
            iface.carrier = False
            iface.operstate = "down"

    def _check_failure(self, operation: str) -> None:
        msg = self._injected_failures.pop(operation, None)
        if msg is not None:
            raise RuntimeError(msg)

    def _check_mutation_count(self) -> None:
        if self._fail_at is not None:
            self._mutation_count += 1
            if self._mutation_count >= self._fail_at:
                self._fail_at = None
                raise RuntimeError(f"simulated failure at mutation #{self._mutation_count}")

    def _ensure_interface(self, name: str) -> InterfaceState:
        if name not in self._interfaces:
            mac = self._generate_mac(name)
            self._interfaces[name] = InterfaceState(name=name, mac=mac)
        return self._interfaces[name]

    @staticmethod
    def _generate_mac(seed: str) -> str:
        h = hash(seed) & 0xFFFFFFFFFFFF
        octets = [(h >> (i * 8)) & 0xFF for i in range(6)]
        octets[0] = (octets[0] & 0xFE) | 0x02  # locally administered, unicast
        return ":".join(f"{b:02x}" for b in octets)

    # --- Bridge operations ---

    def create_bridge(self, name: str, interfaces: list[str]) -> dict:
        self._check_failure("create_bridge")
        self._check_mutation_count()
        if not name:
            raise ValueError("bridge name cannot be empty")
        if name in self._bridges:
            raise ValueError(f"bridge '{name}' already exists")
        bridge = BridgeState(name=name, interfaces=list(interfaces))
        self._bridges[name] = bridge
        self._fdb[name] = []
        self._ensure_interface(name)
        for iface in interfaces:
            state = self._ensure_interface(iface)
            state.master = name
        return self._bridge_dict(bridge)

    def delete_bridge(self, name: str) -> None:
        self._check_failure("delete_bridge")
        self._check_mutation_count()
        bridge = self._bridges.pop(name, None)
        if bridge is None:
            raise ValueError(f"bridge '{name}' does not exist")
        for iface in bridge.interfaces:
            state = self._interfaces.get(iface)
            if state:
                state.master = ""
        self._fdb.pop(name, None)
        self._interfaces.pop(name, None)

    def attach_interface(self, bridge: str, interface: str) -> None:
        self._check_failure("attach_interface")
        self._check_mutation_count()
        br = self._bridges.get(bridge)
        if br is None:
            raise ValueError(f"bridge '{bridge}' does not exist")
        if interface in br.interfaces:
            raise ValueError(f"interface '{interface}' already attached to '{bridge}'")
        br.interfaces.append(interface)
        state = self._ensure_interface(interface)
        state.master = bridge

    def detach_interface(self, bridge: str, interface: str) -> None:
        self._check_failure("detach_interface")
        self._check_mutation_count()
        br = self._bridges.get(bridge)
        if br is None:
            raise ValueError(f"bridge '{bridge}' does not exist")
        if interface not in br.interfaces:
            raise ValueError(f"interface '{interface}' not attached to '{bridge}'")
        br.interfaces.remove(interface)
        state = self._interfaces.get(interface)
        if state:
            state.master = ""

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

    # --- STP operations ---

    def set_stp(self, bridge_name: str, enabled: bool, forward_delay: int) -> dict:
        self._check_failure("set_stp")
        self._check_mutation_count()
        bridge = self._bridges.get(bridge_name)
        if bridge is None:
            raise ValueError(f"bridge '{bridge_name}' does not exist")
        if forward_delay < 1 or forward_delay > 30:
            raise ValueError(f"forward_delay must be 1-30, got {forward_delay}")
        bridge.stp_enabled = enabled
        bridge.forward_delay = forward_delay
        return self._bridge_dict(bridge)

    def get_stp_state(self, bridge: str) -> dict:
        self._check_failure("get_stp_state")
        br = self._bridges.get(bridge)
        if br is None:
            raise ValueError(f"bridge '{bridge}' does not exist")
        return {
            "bridge": bridge,
            "enabled": br.stp_enabled,
            "forward_delay": br.forward_delay,
            "root_id": bridge,
            "bridge_id": bridge,
            "topology_change": False,
        }

    # --- MAC operations ---

    def get_device_mac(self, device: str) -> str:
        self._check_failure("get_device_mac")
        iface = self._interfaces.get(device)
        if iface is None:
            raise ValueError(f"device '{device}' does not exist")
        return iface.mac

    def set_device_mac(self, device: str, mac: str) -> None:
        self._check_failure("set_device_mac")
        self._check_mutation_count()
        iface = self._interfaces.get(device)
        if iface is None:
            raise ValueError(f"device '{device}' does not exist")
        iface.mac = mac

    def send_gratuitous_arp(self, interface: str, mac: str) -> None:
        self._check_failure("send_gratuitous_arp")
        self._gratuitous_arps.append({"interface": interface, "mac": mac})

    # --- Bond operations ---

    def create_bond(self, name: str, mode: str, members: list[str]) -> dict:
        self._check_failure("create_bond")
        self._check_mutation_count()
        if not name:
            raise ValueError("bond name cannot be empty")
        if name in self._bonds:
            raise ValueError(f"bond '{name}' already exists")
        bond = BondState(name=name, mode=mode, members=list(members))
        self._bonds[name] = bond
        self._ensure_interface(name)
        for member in members:
            state = self._ensure_interface(member)
            state.master = name
        return self._bond_dict(bond)

    def delete_bond(self, name: str) -> None:
        self._check_failure("delete_bond")
        self._check_mutation_count()
        bond = self._bonds.pop(name, None)
        if bond is None:
            raise ValueError(f"bond '{name}' does not exist")
        for member in bond.members:
            state = self._interfaces.get(member)
            if state:
                state.master = ""
        self._interfaces.pop(name, None)

    def get_bond(self, name: str) -> dict | None:
        bond = self._bonds.get(name)
        if bond is None:
            return None
        return self._bond_dict(bond)

    @staticmethod
    def _bond_dict(bond: BondState) -> dict:
        return {
            "name": bond.name,
            "mode": bond.mode,
            "members": bond.members,
            "active": bond.active,
        }

    # --- VLAN operations ---

    def create_vlan(self, parent: str, vlan_id: int) -> dict:
        self._check_failure("create_vlan")
        self._check_mutation_count()
        key = f"{parent}.{vlan_id}"
        if key in self._vlans:
            raise ValueError(f"VLAN {vlan_id} already exists on '{parent}'")
        if vlan_id < 1 or vlan_id > 4094:
            raise ValueError(f"VLAN ID must be 1-4094, got {vlan_id}")
        vlan = VlanState(parent=parent, vlan_id=vlan_id)
        self._vlans[key] = vlan
        self._ensure_interface(key)
        return self._vlan_dict(vlan)

    def delete_vlan(self, parent: str, vlan_id: int) -> None:
        self._check_failure("delete_vlan")
        self._check_mutation_count()
        key = f"{parent}.{vlan_id}"
        vlan = self._vlans.pop(key, None)
        if vlan is None:
            raise ValueError(f"VLAN {vlan_id} does not exist on '{parent}'")
        self._interfaces.pop(key, None)

    def get_vlan(self, parent: str, vlan_id: int) -> dict | None:
        key = f"{parent}.{vlan_id}"
        vlan = self._vlans.get(key)
        if vlan is None:
            return None
        return self._vlan_dict(vlan)

    @staticmethod
    def _vlan_dict(vlan: VlanState) -> dict:
        return {
            "name": vlan.name,
            "parent": vlan.parent,
            "vlan_id": vlan.vlan_id,
        }

    # --- Diagnostic reads ---

    def read_fdb(self, bridge: str) -> list[dict]:
        self._check_failure("read_fdb")
        if bridge not in self._bridges:
            raise ValueError(f"bridge '{bridge}' does not exist")
        entries = self._fdb.get(bridge, [])
        return [
            {
                "mac": e.mac,
                "port": e.port,
                "vlan": e.vlan,
                "is_local": e.is_local,
            }
            for e in entries
        ]

    def add_fdb_entry(self, bridge: str, mac: str, port: str,
                      vlan: int = 0, is_local: bool = False) -> None:
        """Directly add an FDB entry (for test setup)."""
        if bridge not in self._bridges:
            raise ValueError(f"bridge '{bridge}' does not exist")
        if bridge not in self._fdb:
            self._fdb[bridge] = []
        self._fdb[bridge].append(FdbEntry(
            mac=mac, port=port, vlan=vlan, is_local=is_local
        ))

    def get_interface_state(self, interface: str) -> dict:
        self._check_failure("get_interface_state")
        iface = self._interfaces.get(interface)
        if iface is None:
            raise ValueError(f"interface '{interface}' does not exist")
        return {
            "name": iface.name,
            "mac": iface.mac,
            "carrier": iface.carrier,
            "operstate": iface.operstate,
            "master": iface.master,
        }

    def get_arp_table(self) -> list[dict]:
        self._check_failure("get_arp_table")
        return [
            {"ip": e.ip, "mac": e.mac, "device": e.device}
            for e in self._arp_table
        ]

    def add_arp_entry(self, ip: str, mac: str, device: str) -> None:
        """Directly add an ARP entry (for test setup)."""
        self._arp_table.append(ArpEntry(ip=ip, mac=mac, device=device))

    # --- Resource tracking ---

    def track_resource(self, resource_type: str, name: str) -> None:
        pair = (resource_type, name)
        if pair not in self._tracked:
            self._tracked.append(pair)

    def untrack_resource(self, resource_type: str, name: str) -> None:
        pair = (resource_type, name)
        if pair in self._tracked:
            self._tracked.remove(pair)

    def tracked_resources(self) -> list[tuple[str, str]]:
        return list(self._tracked)
