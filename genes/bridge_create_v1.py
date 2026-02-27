"""Seed gene: create a network bridge."""
import json

def execute(input_json: str) -> str:
    data = json.loads(input_json)

    bridge_name = data.get("bridge_name")
    if not bridge_name or not isinstance(bridge_name, str):
        return json.dumps({"success": False, "error": "missing or invalid bridge_name"})

    interfaces = data.get("interfaces")
    if not isinstance(interfaces, list):
        return json.dumps({"success": False, "error": "missing or invalid interfaces"})

    try:
        result = gene_sdk.create_bridge(bridge_name, interfaces)
        return json.dumps({"success": True, "bridge": result})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})
