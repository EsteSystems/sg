"""Seed gene: check link state of an interface."""
import json

def execute(input_json: str) -> str:
    data = json.loads(input_json)

    interface = data.get("interface")
    if not interface:
        return json.dumps({"success": False, "error": "missing interface"})

    try:
        state = gene_sdk.get_interface_state(interface)
        healthy = state["carrier"] and state["operstate"] == "up"

        result = {
            "success": True,
            "healthy": healthy,
            "carrier": state["carrier"],
            "operstate": state["operstate"],
            "mac": state["mac"],
        }
        if state.get("master"):
            result["master"] = state["master"]

        return json.dumps(result)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})
