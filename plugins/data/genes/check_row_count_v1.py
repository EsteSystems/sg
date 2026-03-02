"""Seed gene: check row count of a database table."""
import json

def execute(input_json: str) -> str:
    data = json.loads(input_json)

    connection = data.get("connection")
    if not connection:
        return json.dumps({"success": False, "error": "missing connection"})

    table = data.get("table")
    if not table:
        return json.dumps({"success": False, "error": "missing table"})

    try:
        count = gene_sdk.row_count(connection, table)
        healthy = count > 0
        return json.dumps({
            "success": True,
            "healthy": healthy,
            "row_count": count,
        })
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})
