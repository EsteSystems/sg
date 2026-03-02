"""Fused gene: bridge_create + bridge_stp in a single execution.

Combines both steps without intermediate JSON serialization.
"""
import json

def execute(input_json: str) -> str:
    data = json.loads(input_json)

    bridge_name = data.get("bridge_name")
    if not bridge_name or not isinstance(bridge_name, str):
        return json.dumps({"success": False, "error": "missing or invalid bridge_name"})

    interfaces = data.get("interfaces", [])
    if not isinstance(interfaces, list):
        interfaces = []

    stp_enabled = data.get("stp_enabled", True)
    if not isinstance(stp_enabled, bool):
        stp_enabled = True

    forward_delay = data.get("forward_delay", 15)
    if not isinstance(forward_delay, int):
        forward_delay = 15

    try:
        gene_sdk.create_bridge(bridge_name, interfaces)
        result = gene_sdk.set_stp(bridge_name, stp_enabled, forward_delay)
        return json.dumps({"success": True, "bridge": result})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})
