"""Topology execution — decompose resource declarations into pathway/gene calls.

A topology declares *what* resources should exist. The engine figures out
*how* to create them by mapping resource types to pathways/genes, resolving
inter-resource dependencies via topological sort, and executing in order.

Composition hierarchy: topology → pathway → locus → allele.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field

from sg.parser.types import TopologyContract, TopologyResource


@dataclass
class TopologyStep:
    """One resolved execution step produced by decomposition."""
    resource_name: str      # from has block (e.g. "management")
    action: str             # "pathway" | "gene" | "loop_gene"
    target: str             # pathway/gene name to execute
    input_json: str         # resolved input JSON for this step
    loop_items: list = field(default_factory=list)  # for loop_gene: per-item inputs


def _resolve_value(value: str, data: dict) -> object:
    """Resolve a {reference} from topology input, or return literal."""
    match = re.fullmatch(r"\{(\w+)\}", value)
    if match:
        key = match.group(1)
        return data.get(key, value)
    return value


def _build_dependency_graph(
    resources: list[TopologyResource],
) -> dict[str, list[str]]:
    """Build a dependency graph from inter-resource references.

    A property value that matches another resource's name creates a dependency.
    E.g. resource vm_traffic with property `trunk storage` depends on resource storage.
    """
    resource_names = {r.name for r in resources}
    # adjacency: depends_on[A] = [B] means A depends on B (B must run first)
    depends_on: dict[str, list[str]] = defaultdict(list)

    for resource in resources:
        for prop_value in resource.properties.values():
            # Check if the value is a bare resource name reference
            if prop_value in resource_names and prop_value != resource.name:
                depends_on[resource.name].append(prop_value)

    return dict(depends_on)


def _topological_sort(
    resources: list[TopologyResource],
    depends_on: dict[str, list[str]],
) -> list[TopologyResource]:
    """Kahn's algorithm — returns resources in dependency order."""
    name_to_resource = {r.name: r for r in resources}
    in_degree: dict[str, int] = {r.name: 0 for r in resources}
    reverse_deps: dict[str, list[str]] = defaultdict(list)

    for node, deps in depends_on.items():
        in_degree[node] = len(deps)
        for dep in deps:
            reverse_deps[dep].append(node)

    queue = deque(
        r.name for r in resources if in_degree[r.name] == 0
    )
    ordered: list[TopologyResource] = []

    while queue:
        name = queue.popleft()
        ordered.append(name_to_resource[name])
        for dependent in reverse_deps.get(name, []):
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    if len(ordered) != len(resources):
        seen = {r.name for r in ordered}
        missing = [r.name for r in resources if r.name not in seen]
        raise ValueError(f"circular dependency among resources: {missing}")

    return ordered


def _map_bridge(resource: TopologyResource, data: dict) -> TopologyStep:
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


def _map_bond(resource: TopologyResource, data: dict) -> TopologyStep:
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


def _map_vlan_bridges(resource: TopologyResource, data: dict) -> TopologyStep:
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


# Resource type → mapper function
_RESOURCE_MAPPERS = {
    "bridge": _map_bridge,
    "bond": _map_bond,
    "vlan_bridges": _map_vlan_bridges,
}


def decompose(
    topology: TopologyContract, input_json: str,
) -> list[TopologyStep]:
    """Decompose a topology into ordered execution steps.

    1. Resolve {reference} values from input
    2. Build dependency graph from inter-resource references
    3. Topological sort
    4. Map each resource to a pathway/gene call
    """
    data = json.loads(input_json)
    depends_on = _build_dependency_graph(topology.has)
    ordered = _topological_sort(topology.has, depends_on)

    steps = []
    for resource in ordered:
        mapper = _RESOURCE_MAPPERS.get(resource.resource_type)
        if mapper is None:
            raise ValueError(
                f"unknown resource type '{resource.resource_type}' "
                f"for resource '{resource.name}'"
            )
        steps.append(mapper(resource, data))

    return steps


def execute_topology(
    topology: TopologyContract,
    input_json: str,
    orchestrator: object,
) -> list[str]:
    """Execute a topology by decomposing and running each step.

    Respects on_failure policy:
    - "preserve what works": catch errors per-resource, continue
    - "rollback all": re-raise on first failure (let caller handle)

    Schedules verify diagnostics after successful deployment.
    """
    steps = decompose(topology, input_json)
    outputs: list[str] = []
    errors: list[str] = []
    preserve = topology.on_failure == "preserve what works"

    for step in steps:
        print(f"  [topology] {step.resource_name}: "
              f"{step.action} → {step.target}")
        try:
            if step.action == "pathway":
                step_outputs = orchestrator.run_pathway(
                    step.target, step.input_json
                )
                outputs.extend(step_outputs)

            elif step.action == "gene":
                result = orchestrator.execute_locus(
                    step.target, step.input_json
                )
                if result is None:
                    raise RuntimeError(
                        f"all alleles exhausted for {step.target}"
                    )
                outputs.append(result[0])

            elif step.action == "loop_gene":
                for item_input in step.loop_items:
                    result = orchestrator.execute_locus(
                        step.target, item_input
                    )
                    if result is None:
                        raise RuntimeError(
                            f"all alleles exhausted for {step.target}"
                        )
                    outputs.append(result[0])

        except Exception as e:
            msg = f"resource '{step.resource_name}' failed: {e}"
            print(f"  [topology] {msg}")
            if preserve:
                errors.append(msg)
            else:
                raise RuntimeError(msg) from e

    if errors:
        raise RuntimeError(
            f"topology partially failed ({len(errors)} error(s)): "
            + "; ".join(errors)
        )

    # Schedule verify diagnostics if topology declares them
    _schedule_topology_verify(topology, input_json, orchestrator)

    return outputs


def _schedule_topology_verify(
    topology: TopologyContract,
    input_json: str,
    orchestrator: object,
) -> None:
    """Schedule verify diagnostics from the topology contract."""
    if not topology.verify:
        return

    from sg.verify import parse_duration

    delay = 0.0
    if topology.verify_within:
        try:
            delay = parse_duration(topology.verify_within)
        except ValueError:
            return

    orchestrator.verify_scheduler.schedule(
        topology.verify, delay, input_json, orchestrator
    )
