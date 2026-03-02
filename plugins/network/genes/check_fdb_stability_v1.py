"""Seed gene: analyze FDB health of a bridge."""
import json
from collections import defaultdict

def execute(input_json: str) -> str:
    data = json.loads(input_json)

    bridge_name = data.get("bridge_name")
    if not bridge_name:
        return json.dumps({"success": False, "error": "missing bridge_name"})

    try:
        fdb = gene_sdk.read_fdb(bridge_name)

        total_entries = len(fdb)
        local_entries = sum(1 for e in fdb if e.get("is_local", False))
        dynamic_entries = total_entries - local_entries

        anomalies = []

        # Check for duplicate MACs across ports
        mac_ports = defaultdict(set)
        for entry in fdb:
            mac_ports[entry["mac"]].add(entry["port"])
        duplicates = [mac for mac, ports in mac_ports.items() if len(ports) > 1]
        if duplicates:
            anomalies.append(f"duplicate MACs across ports: {', '.join(duplicates)}")

        # Check for excessive FDB size
        if total_entries > 1000:
            anomalies.append(f"excessive FDB size: {total_entries} entries")

        healthy = len(anomalies) == 0

        return json.dumps({
            "success": True,
            "healthy": healthy,
            "total_entries": total_entries,
            "local_entries": local_entries,
            "dynamic_entries": dynamic_entries,
            "anomalies": anomalies,
        })
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})
