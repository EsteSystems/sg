"""Safety mechanisms — transactions, blast radius, rollback.

Wraps configuration gene execution in undo-log transactions.
Classifies loci by blast radius (none → low → medium → high → critical)
and applies graded safety policies: shadow mode, canary, convergence
checks, resilience requirements.

SafeKernel is a proxy that intercepts mutating kernel operations and
records undo actions. On gene failure, the transaction rolls back all
changes made by that allele attempt.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from sg.kernel.base import NetworkKernel
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


class SafeKernel(NetworkKernel):
    """Proxy kernel that records undo actions in a transaction.

    Wraps a real NetworkKernel, delegating all calls. Mutating operations
    record their inverse in the transaction's undo log. Read-only
    operations pass through directly.
    """

    def __init__(self, inner: NetworkKernel, transaction: Transaction) -> None:
        self.inner = inner
        self.txn = transaction

    # --- State management ---

    def reset(self) -> None:
        self.inner.reset()

    # --- Bridge operations ---

    def create_bridge(self, name: str, interfaces: list[str]) -> dict:
        result = self.inner.create_bridge(name, interfaces)
        self.txn.record(
            f"delete bridge '{name}'",
            lambda n=name: self.inner.delete_bridge(n),
        )
        return result

    def delete_bridge(self, name: str) -> None:
        # Snapshot state for undo
        bridge = self.inner.get_bridge(name)
        self.inner.delete_bridge(name)
        if bridge:
            self.txn.record(
                f"recreate bridge '{name}'",
                lambda b=bridge: self.inner.create_bridge(
                    b["name"], b["interfaces"]
                ),
            )

    def attach_interface(self, bridge: str, interface: str) -> None:
        self.inner.attach_interface(bridge, interface)
        self.txn.record(
            f"detach '{interface}' from '{bridge}'",
            lambda br=bridge, iface=interface: self.inner.detach_interface(br, iface),
        )

    def detach_interface(self, bridge: str, interface: str) -> None:
        self.inner.detach_interface(bridge, interface)
        self.txn.record(
            f"reattach '{interface}' to '{bridge}'",
            lambda br=bridge, iface=interface: self.inner.attach_interface(br, iface),
        )

    def get_bridge(self, name: str) -> dict | None:
        return self.inner.get_bridge(name)

    # --- STP operations ---

    def set_stp(self, bridge_name: str, enabled: bool, forward_delay: int) -> dict:
        old = self.inner.get_stp_state(bridge_name)
        result = self.inner.set_stp(bridge_name, enabled, forward_delay)
        self.txn.record(
            f"restore STP on '{bridge_name}'",
            lambda br=bridge_name, e=old["enabled"], fd=old["forward_delay"]: (
                self.inner.set_stp(br, e, fd)
            ),
        )
        return result

    def get_stp_state(self, bridge: str) -> dict:
        return self.inner.get_stp_state(bridge)

    # --- MAC operations ---

    def get_device_mac(self, device: str) -> str:
        return self.inner.get_device_mac(device)

    def set_device_mac(self, device: str, mac: str) -> None:
        old_mac = self.inner.get_device_mac(device)
        self.inner.set_device_mac(device, mac)
        self.txn.record(
            f"restore MAC on '{device}'",
            lambda d=device, m=old_mac: self.inner.set_device_mac(d, m),
        )

    def send_gratuitous_arp(self, interface: str, mac: str) -> None:
        # Gratuitous ARP cannot be un-sent — record but no meaningful undo
        self.inner.send_gratuitous_arp(interface, mac)

    # --- Bond operations ---

    def create_bond(self, name: str, mode: str, members: list[str]) -> dict:
        result = self.inner.create_bond(name, mode, members)
        self.txn.record(
            f"delete bond '{name}'",
            lambda n=name: self.inner.delete_bond(n),
        )
        return result

    def delete_bond(self, name: str) -> None:
        bond = self.inner.get_bond(name)
        self.inner.delete_bond(name)
        if bond:
            self.txn.record(
                f"recreate bond '{name}'",
                lambda b=bond: self.inner.create_bond(
                    b["name"], b["mode"], b["members"]
                ),
            )

    def get_bond(self, name: str) -> dict | None:
        return self.inner.get_bond(name)

    # --- VLAN operations ---

    def create_vlan(self, parent: str, vlan_id: int) -> dict:
        result = self.inner.create_vlan(parent, vlan_id)
        self.txn.record(
            f"delete VLAN {vlan_id} on '{parent}'",
            lambda p=parent, v=vlan_id: self.inner.delete_vlan(p, v),
        )
        return result

    def delete_vlan(self, parent: str, vlan_id: int) -> None:
        vlan = self.inner.get_vlan(parent, vlan_id)
        self.inner.delete_vlan(parent, vlan_id)
        if vlan:
            self.txn.record(
                f"recreate VLAN {vlan_id} on '{parent}'",
                lambda p=parent, v=vlan_id: self.inner.create_vlan(p, v),
            )

    def get_vlan(self, parent: str, vlan_id: int) -> dict | None:
        return self.inner.get_vlan(parent, vlan_id)

    # --- Diagnostic reads (pass-through, no undo needed) ---

    def read_fdb(self, bridge: str) -> list[dict]:
        return self.inner.read_fdb(bridge)

    def get_interface_state(self, interface: str) -> dict:
        return self.inner.get_interface_state(interface)

    def get_arp_table(self) -> list[dict]:
        return self.inner.get_arp_table()

    # --- Resource tracking (pass-through) ---

    def track_resource(self, resource_type: str, name: str) -> None:
        self.inner.track_resource(resource_type, name)

    def untrack_resource(self, resource_type: str, name: str) -> None:
        self.inner.untrack_resource(resource_type, name)

    def tracked_resources(self) -> list[tuple[str, str]]:
        return self.inner.tracked_resources()


def requires_transaction(risk: BlastRadius) -> bool:
    """Return True if a blast radius level requires transaction wrapping."""
    return risk in (BlastRadius.LOW, BlastRadius.MEDIUM,
                    BlastRadius.HIGH, BlastRadius.CRITICAL)


def is_shadow_only(risk: BlastRadius) -> bool:
    """Return True if blast radius requires shadow mode (stub — always False for now)."""
    # Future: HIGH and CRITICAL could default to shadow mode
    return False
