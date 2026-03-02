"""Mutation fixture: check_fdb_stability fix."""
import json
from collections import defaultdict

def execute(input_json: str) -> str:
    data = json.loads(input_json)
    bridge_name = data.get("bridge_name", "")
    try:
        fdb = gene_sdk.read_fdb(bridge_name)
        total = len(fdb)
        local = sum(1 for e in fdb if e.get("is_local"))
        dynamic = total - local
        anomalies = []
        mac_ports = defaultdict(set)
        for entry in fdb:
            mac_ports[entry["mac"]].add(entry["port"])
        dupes = [m for m, p in mac_ports.items() if len(p) > 1]
        if dupes:
            anomalies.append(f"duplicate MACs: {', '.join(dupes)}")
        return json.dumps({"success": True, "healthy": not anomalies,
                           "total_entries": total, "local_entries": local,
                           "dynamic_entries": dynamic, "anomalies": anomalies})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})
