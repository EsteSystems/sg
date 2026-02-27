"""Seed gene: configure STP on a bridge."""
import json

def execute(input_json: str) -> str:
    data = json.loads(input_json)

    bridge_name = data.get("bridge_name")
    if not bridge_name or not isinstance(bridge_name, str):
        return json.dumps({"success": False, "error": "missing or invalid bridge_name"})

    stp_enabled = data.get("stp_enabled")
    if not isinstance(stp_enabled, bool):
        return json.dumps({"success": False, "error": "missing or invalid stp_enabled"})

    forward_delay = data.get("forward_delay")
    if not isinstance(forward_delay, int):
        return json.dumps({"success": False, "error": "missing or invalid forward_delay"})

    try:
        result = gene_sdk.set_stp(bridge_name, stp_enabled, forward_delay)
        return json.dumps({"success": True, "bridge": result})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})
