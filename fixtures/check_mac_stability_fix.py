"""Mutation fixture: check_mac_stability fix."""
import json
from collections import defaultdict

def execute(input_json: str) -> str:
    data = json.loads(input_json)
    bridge_name = data.get("bridge_name", "")
    try:
        fdb = gene_sdk.read_fdb(bridge_name)
        mac_ports = defaultdict(set)
        for entry in fdb:
            mac_ports[entry["mac"]].add(entry["port"])
        flapping = [m for m, p in mac_ports.items() if len(p) > 1]
        return json.dumps({"success": True, "healthy": not flapping,
                           "total_macs": len(mac_ports), "flapping_macs": flapping})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})
