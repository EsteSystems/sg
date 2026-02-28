"""Fusion fixture: health_check_bridge fused gene."""
import json
from collections import defaultdict

def execute(input_json: str) -> str:
    data = json.loads(input_json)
    bridge_name = data.get("bridge_name", "")

    try:
        bridge = gene_sdk.get_bridge(bridge_name)
        if bridge is None:
            return json.dumps({"success": False, "error": f"bridge '{bridge_name}' not found"})

        # Connectivity check
        bridge_state = gene_sdk.get_interface_state(bridge_name)
        bridge_up = bridge_state["carrier"]
        ports = bridge["interfaces"]
        ports_down = [p for p in ports if not gene_sdk.get_interface_state(p)["carrier"]]

        # MAC stability check
        fdb = gene_sdk.read_fdb(bridge_name)
        mac_ports = defaultdict(set)
        for entry in fdb:
            mac_ports[entry["mac"]].add(entry["port"])
        flapping_macs = [m for m, p in mac_ports.items() if len(p) > 1]

        # FDB analysis
        total_entries = len(fdb)
        local_entries = sum(1 for e in fdb if e.get("is_local"))

        healthy = bridge_up and not ports_down and not flapping_macs

        return json.dumps({
            "success": True,
            "healthy": healthy,
            "bridge_up": bridge_up,
            "ports_down": ports_down,
            "flapping_macs": flapping_macs,
            "fdb_entries": total_entries,
        })
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})
