"""Mutant: bridge_create with improved robustness.

Handles missing interfaces by defaulting to empty list.
"""
import json

def execute(input_json: str) -> str:
    data = json.loads(input_json)

    bridge_name = data.get("bridge_name")
    if not bridge_name or not isinstance(bridge_name, str):
        return json.dumps({"success": False, "error": "missing or invalid bridge_name"})

    # Improvement: default to empty interfaces instead of rejecting
    interfaces = data.get("interfaces", [])
    if not isinstance(interfaces, list):
        interfaces = []

    try:
        result = gene_sdk.create_bridge(bridge_name, interfaces)
        gene_sdk.track_resource("bridge", bridge_name)
        return json.dumps({
            "success": True,
            "resources_created": [bridge_name],
        })
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})
