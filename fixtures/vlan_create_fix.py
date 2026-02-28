"""Mutation fixture: vlan_create fix."""
import json

def execute(input_json: str) -> str:
    data = json.loads(input_json)
    parent = data.get("parent", "")
    vlan_id = data.get("vlan_id", 1)
    try:
        result = gene_sdk.create_vlan(parent, vlan_id)
        return json.dumps({"success": True, "vlan_name": result["name"]})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})
