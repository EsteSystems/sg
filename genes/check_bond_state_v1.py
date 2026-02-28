"""Seed gene: check bond health."""
import json

def execute(input_json: str) -> str:
    data = json.loads(input_json)

    bond_name = data.get("bond_name")
    if not bond_name:
        return json.dumps({"success": False, "error": "missing bond_name"})

    try:
        bond = gene_sdk.get_bond(bond_name)
        if bond is None:
            return json.dumps({"success": False, "error": f"bond '{bond_name}' does not exist"})

        members = bond["members"]
        members_down = []
        for member in members:
            state = gene_sdk.get_interface_state(member)
            if not state["carrier"]:
                members_down.append(member)

        healthy = bond["active"] and len(members_down) == 0

        return json.dumps({
            "success": True,
            "healthy": healthy,
            "active": bond["active"],
            "members": members,
            "members_down": members_down,
        })
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})
