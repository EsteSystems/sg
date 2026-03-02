"""Seed gene: create a network bond."""
import json

def execute(input_json: str) -> str:
    data = json.loads(input_json)

    bond_name = data.get("bond_name")
    if not bond_name:
        return json.dumps({"success": False, "error": "missing bond_name"})

    mode = data.get("mode")
    if not mode:
        return json.dumps({"success": False, "error": "missing mode"})

    members = data.get("members")
    if not isinstance(members, list):
        return json.dumps({"success": False, "error": "missing or invalid members"})

    try:
        result = gene_sdk.create_bond(bond_name, mode, members)
        return json.dumps({"success": True, "bond": result["name"]})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})
