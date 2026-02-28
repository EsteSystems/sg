"""Seed gene: check MAC address stability on a bridge."""
import json
from collections import defaultdict

def execute(input_json: str) -> str:
    data = json.loads(input_json)

    bridge_name = data.get("bridge_name")
    if not bridge_name:
        return json.dumps({"success": False, "error": "missing bridge_name"})

    try:
        fdb = gene_sdk.read_fdb(bridge_name)

        # Group MACs by address to find those on multiple ports
        mac_ports = defaultdict(set)
        for entry in fdb:
            mac_ports[entry["mac"]].add(entry["port"])

        flapping_macs = [mac for mac, ports in mac_ports.items() if len(ports) > 1]
        total_macs = len(mac_ports)
        healthy = len(flapping_macs) == 0

        return json.dumps({
            "success": True,
            "healthy": healthy,
            "total_macs": total_macs,
            "flapping_macs": flapping_macs,
        })
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})
