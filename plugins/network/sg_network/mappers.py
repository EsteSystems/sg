"""Network-specific topology resource mappers.

Maps network resource types (bridge, bond, vlan_bridges) to topology
execution steps. Used by NetworkKernel.resource_mappers() to provide
domain-specific topology decomposition.
"""
from __future__ import annotations

import json

from sg.parser.types import TopologyResource
from sg.topology import TopologyStep, _resolve_value


def map_bridge(resource: TopologyResource, data: dict) -> TopologyStep:
    """Map a bridge resource to a pathway or gene call."""
    props = resource.properties

    if "uplink" in props:
        # Full management bridge with uplink — use provision_management_bridge
        input_data = {
            "bridge_name": data.get("bridge_name", resource.name),
            "interfaces": data.get("bridge_ifaces", data.get("interfaces", [])),
            "uplink": _resolve_value(props["uplink"], data),
            "stp_enabled": True,
            "forward_delay": data.get("forward_delay", 15),
        }
        return TopologyStep(
            resource_name=resource.name,
            action="pathway",
            target="provision_management_bridge",
            input_json=json.dumps(input_data),
        )

    if "stp" in props:
        # Bridge with STP but no uplink
        input_data = {
            "bridge_name": data.get("bridge_name", resource.name),
            "interfaces": data.get("bridge_ifaces", data.get("interfaces", [])),
            "stp_enabled": True,
            "forward_delay": data.get("forward_delay", 15),
        }
        return TopologyStep(
            resource_name=resource.name,
            action="pathway",
            target="configure_bridge_with_stp",
            input_json=json.dumps(input_data),
        )

    # Bare bridge
    input_data = {
        "bridge_name": data.get("bridge_name", resource.name),
        "interfaces": data.get("bridge_ifaces", data.get("interfaces", [])),
    }
    return TopologyStep(
        resource_name=resource.name,
        action="gene",
        target="bridge_create",
        input_json=json.dumps(input_data),
    )


def map_bond(resource: TopologyResource, data: dict) -> TopologyStep:
    """Map a bond resource to a gene call."""
    props = resource.properties
    input_data = {
        "bond_name": data.get("bond_name", resource.name),
        "mode": _resolve_value(props.get("mode", "active-backup"), data),
        "members": _resolve_value(props.get("members", "[]"), data),
    }
    return TopologyStep(
        resource_name=resource.name,
        action="gene",
        target="bond_create",
        input_json=json.dumps(input_data),
    )


def map_vlan_bridges(resource: TopologyResource, data: dict) -> TopologyStep:
    """Map a vlan_bridges resource to a loop gene call."""
    props = resource.properties
    vlans = _resolve_value(props.get("vlans", "[]"), data)
    if isinstance(vlans, str):
        vlans = json.loads(vlans)

    # Resolve the trunk reference — use the bond name from data
    trunk_ref = props.get("trunk", "")
    parent = data.get("bond_name", trunk_ref)

    loop_items = []
    for vlan_id in vlans:
        loop_items.append(json.dumps({
            "parent": parent,
            "vlan_id": vlan_id,
        }))

    return TopologyStep(
        resource_name=resource.name,
        action="loop_gene",
        target="vlan_create",
        input_json="{}",
        loop_items=loop_items,
    )


NETWORK_RESOURCE_MAPPERS = {
    "bridge": map_bridge,
    "bond": map_bond,
    "vlan_bridges": map_vlan_bridges,
}
