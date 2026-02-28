"""Seed gene: check bridge connectivity."""
import json

def execute(input_json: str) -> str:
    data = json.loads(input_json)

    bridge_name = data.get("bridge_name")
    if not bridge_name:
        return json.dumps({"success": False, "error": "missing bridge_name"})

    try:
        bridge = gene_sdk.get_bridge(bridge_name)
        if bridge is None:
            return json.dumps({"success": False, "error": f"bridge '{bridge_name}' does not exist"})

        bridge_state = gene_sdk.get_interface_state(bridge_name)
        bridge_up = bridge_state["carrier"] and bridge_state["operstate"] == "up"

        ports = bridge["interfaces"]
        ports_down = []
        for port in ports:
            state = gene_sdk.get_interface_state(port)
            if not state["carrier"]:
                ports_down.append(port)

        healthy = bridge_up and len(ports_down) == 0

        return json.dumps({
            "success": True,
            "healthy": healthy,
            "bridge_up": bridge_up,
            "ports": ports,
            "ports_down": ports_down,
        })
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})
