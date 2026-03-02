"""Seed gene: create a VLAN interface."""
import json

def execute(input_json: str) -> str:
    data = json.loads(input_json)

    parent = data.get("parent")
    if not parent:
        return json.dumps({"success": False, "error": "missing parent"})

    vlan_id = data.get("vlan_id")
    if not isinstance(vlan_id, int):
        return json.dumps({"success": False, "error": "missing or invalid vlan_id"})

    try:
        result = gene_sdk.create_vlan(parent, vlan_id)
        return json.dumps({
            "success": True,
            "vlan_name": result["name"],
        })
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})
