"""Seed gene: attach an uplink interface to a bridge."""
import json

def execute(input_json: str) -> str:
    data = json.loads(input_json)

    bridge_name = data.get("bridge_name")
    if not bridge_name:
        return json.dumps({"success": False, "error": "missing bridge_name"})

    uplink = data.get("uplink")
    if not uplink:
        return json.dumps({"success": False, "error": "missing uplink"})

    try:
        gene_sdk.attach_interface(bridge_name, uplink)
        return json.dumps({"success": True})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})
