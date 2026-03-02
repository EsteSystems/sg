"""Mutation fixture: check_connectivity fix."""
import json

def execute(input_json: str) -> str:
    data = json.loads(input_json)
    bridge_name = data.get("bridge_name", "")
    try:
        bridge = gene_sdk.get_bridge(bridge_name)
        if bridge is None:
            return json.dumps({"success": False, "error": f"bridge '{bridge_name}' not found"})
        bridge_state = gene_sdk.get_interface_state(bridge_name)
        bridge_up = bridge_state["carrier"]
        ports = bridge["interfaces"]
        ports_down = []
        for port in ports:
            state = gene_sdk.get_interface_state(port)
            if not state["carrier"]:
                ports_down.append(port)
        return json.dumps({"success": True, "healthy": bridge_up and not ports_down,
                           "bridge_up": bridge_up, "ports": ports, "ports_down": ports_down})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})
