"""ProductionNetworkKernel — real network operations via ip/bridge/sysfs.

Executes actual network configuration on a Linux host. All test resources
are prefixed with 'sg-test-' for safety. The management interface (eth0)
is never modified.

Requires: Linux, ip, bridge commands, sysfs, sudo access.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from sg.kernel.base import NetworkKernel


SAFETY_PREFIX = "sg-test-"
PROTECTED_INTERFACES = {"eth0", "lo"}


class ProductionNetworkKernel(NetworkKernel):
    """Real network kernel using ip/bridge/sysfs commands."""

    def __init__(self, use_sudo: bool = True, dry_run: bool = False) -> None:
        self._use_sudo = use_sudo
        self._dry_run = dry_run
        self._tracked: list[tuple[str, str]] = []

    def _run(self, cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
        """Run a command, optionally with sudo."""
        if self._use_sudo:
            cmd = ["sudo"] + cmd
        if self._dry_run:
            print(f"  [dry-run] {' '.join(cmd)}")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return subprocess.run(
            cmd, capture_output=True, text=True, check=check, timeout=30,
        )

    @staticmethod
    def _safe_name(name: str) -> str:
        """Ensure a resource name has the safety prefix."""
        if not name.startswith(SAFETY_PREFIX):
            raise ValueError(
                f"resource name '{name}' must start with '{SAFETY_PREFIX}' "
                f"for safety"
            )
        return name

    @staticmethod
    def _check_protected(interface: str) -> None:
        """Refuse to modify protected interfaces."""
        if interface in PROTECTED_INTERFACES:
            raise ValueError(
                f"cannot modify protected interface '{interface}'"
            )

    # --- State management ---

    def reset(self) -> None:
        """Clean up all tracked resources."""
        for rtype, name in reversed(self._tracked):
            try:
                self._run(["ip", "link", "del", name], check=False)
            except Exception:
                pass
        self._tracked.clear()

    # --- Bridge operations ---

    def create_bridge(self, name: str, interfaces: list[str]) -> dict:
        name = self._safe_name(name)
        for iface in interfaces:
            self._check_protected(iface)

        self._run(["ip", "link", "add", name, "type", "bridge"])
        self._run(["ip", "link", "set", name, "up"])
        self.track_resource("bridge", name)

        for iface in interfaces:
            self._ensure_interface_exists(iface)
            self._run(["ip", "link", "set", iface, "master", name])
            self._run(["ip", "link", "set", iface, "up"])

        return self.get_bridge(name)

    def delete_bridge(self, name: str) -> None:
        name = self._safe_name(name)
        bridge = self.get_bridge(name)
        if bridge:
            for iface in bridge.get("interfaces", []):
                self._run(["ip", "link", "set", iface, "nomaster"], check=False)
        self._run(["ip", "link", "del", name])
        self.untrack_resource("bridge", name)

    def attach_interface(self, bridge: str, interface: str) -> None:
        bridge = self._safe_name(bridge)
        self._check_protected(interface)
        self._ensure_interface_exists(interface)
        self._run(["ip", "link", "set", interface, "master", bridge])
        self._run(["ip", "link", "set", interface, "up"])

    def detach_interface(self, bridge: str, interface: str) -> None:
        bridge = self._safe_name(bridge)
        self._check_protected(interface)
        self._run(["ip", "link", "set", interface, "nomaster"])

    def get_bridge(self, name: str) -> dict | None:
        result = self._run(["ip", "-j", "link", "show", name], check=False)
        if result.returncode != 0:
            return None
        try:
            data = json.loads(result.stdout)
        except (json.JSONDecodeError, TypeError):
            return None
        if not data:
            return None

        interfaces = self._get_bridge_interfaces(name)

        stp_path = Path(f"/sys/class/net/{name}/bridge/stp_state")
        stp_enabled = False
        if stp_path.exists():
            stp_enabled = stp_path.read_text().strip() != "0"

        fd_path = Path(f"/sys/class/net/{name}/bridge/forward_delay")
        forward_delay = 15
        if fd_path.exists():
            # sysfs stores forward_delay in jiffies (centiseconds)
            raw = int(fd_path.read_text().strip())
            forward_delay = raw // 100 if raw >= 100 else raw

        return {
            "name": name,
            "interfaces": interfaces,
            "stp_enabled": stp_enabled,
            "forward_delay": forward_delay,
        }

    def _get_bridge_interfaces(self, bridge_name: str) -> list[str]:
        result = self._run(
            ["ip", "-j", "link", "show", "master", bridge_name],
            check=False,
        )
        if result.returncode != 0:
            return []
        try:
            data = json.loads(result.stdout)
        except (json.JSONDecodeError, TypeError):
            return []
        return [iface["ifname"] for iface in data if "ifname" in iface]

    def _ensure_interface_exists(self, name: str) -> None:
        """For test interfaces (sg-test-*), auto-create veth pairs."""
        result = self._run(["ip", "link", "show", name], check=False)
        if result.returncode != 0 and name.startswith(SAFETY_PREFIX):
            peer = name + "-peer"
            self._run(["ip", "link", "add", name, "type", "veth",
                        "peer", "name", peer])
            self._run(["ip", "link", "set", name, "up"])
            self._run(["ip", "link", "set", peer, "up"])
            self.track_resource("veth", name)

    # --- STP operations ---

    def set_stp(self, bridge_name: str, enabled: bool, forward_delay: int) -> dict:
        bridge_name = self._safe_name(bridge_name)
        stp_val = "1" if enabled else "0"
        self._run(["ip", "link", "set", bridge_name, "type", "bridge",
                    "stp_state", stp_val,
                    "forward_delay", str(forward_delay * 100)])
        return self.get_bridge(bridge_name)

    def get_stp_state(self, bridge: str) -> dict:
        bridge_info = self.get_bridge(bridge)
        if bridge_info is None:
            raise ValueError(f"bridge '{bridge}' does not exist")

        result = {
            "bridge": bridge,
            "enabled": bridge_info["stp_enabled"],
            "forward_delay": bridge_info["forward_delay"],
            "root_id": bridge,
            "bridge_id": bridge,
            "topology_change": False,
        }

        root_path = Path(f"/sys/class/net/{bridge}/bridge/root_id")
        if root_path.exists():
            result["root_id"] = root_path.read_text().strip()
        bridge_id_path = Path(f"/sys/class/net/{bridge}/bridge/bridge_id")
        if bridge_id_path.exists():
            result["bridge_id"] = bridge_id_path.read_text().strip()
        tc_path = Path(f"/sys/class/net/{bridge}/bridge/topology_change")
        if tc_path.exists():
            result["topology_change"] = tc_path.read_text().strip() != "0"

        return result

    # --- MAC operations ---

    def get_device_mac(self, device: str) -> str:
        addr_path = Path(f"/sys/class/net/{device}/address")
        if addr_path.exists():
            return addr_path.read_text().strip()
        result = self._run(["ip", "-j", "link", "show", device])
        data = json.loads(result.stdout)
        if data and "address" in data[0]:
            return data[0]["address"]
        raise ValueError(f"device '{device}' does not exist")

    def set_device_mac(self, device: str, mac: str) -> None:
        self._check_protected(device)
        self._run(["ip", "link", "set", device, "down"])
        self._run(["ip", "link", "set", device, "address", mac])
        self._run(["ip", "link", "set", device, "up"])

    def send_gratuitous_arp(self, interface: str, mac: str) -> None:
        self._run(
            ["arping", "-c", "1", "-U", "-I", interface, "0.0.0.0"],
            check=False,
        )

    # --- Bond operations ---

    def create_bond(self, name: str, mode: str, members: list[str]) -> dict:
        name = self._safe_name(name)
        for member in members:
            self._check_protected(member)

        mode_map = {
            "802.3ad": "4", "balance-rr": "0", "active-backup": "1",
            "balance-xor": "2", "broadcast": "3",
            "balance-tlb": "5", "balance-alb": "6",
        }
        kernel_mode = mode_map.get(mode, mode)

        self._run(["ip", "link", "add", name, "type", "bond",
                    "mode", kernel_mode])
        self._run(["ip", "link", "set", name, "up"])
        self.track_resource("bond", name)

        for member in members:
            self._ensure_interface_exists(member)
            self._run(["ip", "link", "set", member, "down"])
            self._run(["ip", "link", "set", member, "master", name])
            self._run(["ip", "link", "set", member, "up"])

        return self.get_bond(name)

    def delete_bond(self, name: str) -> None:
        name = self._safe_name(name)
        bond = self.get_bond(name)
        if bond:
            for member in bond.get("members", []):
                self._run(["ip", "link", "set", member, "nomaster"], check=False)
        self._run(["ip", "link", "del", name])
        self.untrack_resource("bond", name)

    def get_bond(self, name: str) -> dict | None:
        result = self._run(["ip", "-j", "link", "show", name], check=False)
        if result.returncode != 0:
            return None
        try:
            data = json.loads(result.stdout)
        except (json.JSONDecodeError, TypeError):
            return None
        if not data:
            return None

        members = self._get_bond_members(name)
        mode = self._get_bond_mode(name)

        return {
            "name": name,
            "mode": mode,
            "members": members,
            "active": data[0].get("operstate", "unknown").lower() == "up",
        }

    def _get_bond_members(self, bond_name: str) -> list[str]:
        slaves_path = Path(f"/sys/class/net/{bond_name}/bonding/slaves")
        if slaves_path.exists():
            content = slaves_path.read_text().strip()
            return content.split() if content else []
        return []

    def _get_bond_mode(self, bond_name: str) -> str:
        mode_path = Path(f"/sys/class/net/{bond_name}/bonding/mode")
        if mode_path.exists():
            return mode_path.read_text().strip().split()[0]
        return "unknown"

    # --- VLAN operations ---

    def create_vlan(self, parent: str, vlan_id: int) -> dict:
        if not parent.startswith(SAFETY_PREFIX):
            raise ValueError(
                f"VLAN parent '{parent}' must start with '{SAFETY_PREFIX}'"
            )
        if vlan_id < 1 or vlan_id > 4094:
            raise ValueError(f"VLAN ID must be 1-4094, got {vlan_id}")

        vlan_name = f"{parent}.{vlan_id}"
        self._run(["ip", "link", "add", "link", parent, "name", vlan_name,
                    "type", "vlan", "id", str(vlan_id)])
        self._run(["ip", "link", "set", vlan_name, "up"])
        self.track_resource("vlan", vlan_name)

        return self.get_vlan(parent, vlan_id)

    def delete_vlan(self, parent: str, vlan_id: int) -> None:
        vlan_name = f"{parent}.{vlan_id}"
        self._run(["ip", "link", "del", vlan_name])
        self.untrack_resource("vlan", vlan_name)

    def get_vlan(self, parent: str, vlan_id: int) -> dict | None:
        vlan_name = f"{parent}.{vlan_id}"
        result = self._run(["ip", "-j", "link", "show", vlan_name], check=False)
        if result.returncode != 0:
            return None
        try:
            data = json.loads(result.stdout)
        except (json.JSONDecodeError, TypeError):
            return None
        if not data:
            return None
        return {
            "name": vlan_name,
            "parent": parent,
            "vlan_id": vlan_id,
        }

    # --- Diagnostic reads ---

    def read_fdb(self, bridge: str) -> list[dict]:
        result = self._run(["bridge", "-j", "fdb", "show", "br", bridge],
                           check=False)
        if result.returncode != 0:
            return []
        try:
            data = json.loads(result.stdout)
        except (json.JSONDecodeError, TypeError):
            return []
        return [
            {
                "mac": entry.get("mac", ""),
                "port": entry.get("ifname", ""),
                "vlan": entry.get("vlan", 0),
                "is_local": "self" in entry.get("flags", []),
            }
            for entry in data
        ]

    def get_interface_state(self, interface: str) -> dict:
        result = self._run(["ip", "-j", "link", "show", interface])
        data = json.loads(result.stdout)
        if not data:
            raise ValueError(f"interface '{interface}' does not exist")
        iface = data[0]

        carrier_path = Path(f"/sys/class/net/{interface}/carrier")
        carrier = True
        if carrier_path.exists():
            try:
                carrier = carrier_path.read_text().strip() == "1"
            except OSError:
                carrier = False

        return {
            "name": iface.get("ifname", interface),
            "mac": iface.get("address", ""),
            "carrier": carrier,
            "operstate": iface.get("operstate", "unknown"),
            "master": iface.get("master", ""),
        }

    def get_arp_table(self) -> list[dict]:
        result = self._run(["ip", "-j", "neigh", "show"], check=False)
        if result.returncode != 0:
            return []
        try:
            data = json.loads(result.stdout)
        except (json.JSONDecodeError, TypeError):
            return []
        return [
            {
                "ip": entry.get("dst", ""),
                "mac": entry.get("lladdr", ""),
                "device": entry.get("dev", ""),
            }
            for entry in data
            if entry.get("lladdr")
        ]

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

    # --- Utility ---

    def create_veth_pair(self, name1: str, name2: str) -> None:
        """Create a veth pair for testing."""
        self._safe_name(name1)
        self._safe_name(name2)
        self._run(["ip", "link", "add", name1, "type", "veth",
                    "peer", "name", name2])
        self._run(["ip", "link", "set", name1, "up"])
        self._run(["ip", "link", "set", name2, "up"])
        self.track_resource("veth", name1)

    def cleanup_all_test_resources(self) -> None:
        """Remove ALL sg-test-* interfaces from the system."""
        result = self._run(["ip", "-j", "link", "show"], check=False)
        if result.returncode != 0:
            return
        try:
            data = json.loads(result.stdout)
        except (json.JSONDecodeError, TypeError):
            return
        for iface in data:
            name = iface.get("ifname", "")
            if name.startswith(SAFETY_PREFIX):
                self._run(["ip", "link", "del", name], check=False)
        self._tracked.clear()
