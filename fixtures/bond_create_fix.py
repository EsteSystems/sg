"""Mutation fixture: bond_create fix."""
import json

def execute(input_json: str) -> str:
    data = json.loads(input_json)
    bond_name = data.get("bond_name", "")
    mode = data.get("mode", "active-backup")
    members = data.get("members", [])
    try:
        result = gene_sdk.create_bond(bond_name, mode, members)
        return json.dumps({"success": True, "bond": result["name"]})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})
