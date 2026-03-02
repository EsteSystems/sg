"""Mutant: check_nulls with configurable threshold."""
import json

def execute(input_json: str) -> str:
    data = json.loads(input_json)

    connection = data.get("connection")
    if not connection:
        return json.dumps({"success": False, "error": "missing connection"})

    table = data.get("table")
    if not table:
        return json.dumps({"success": False, "error": "missing table"})

    column = data.get("column")
    if not column:
        return json.dumps({"success": False, "error": "missing column"})

    try:
        result = gene_sdk.check_nulls(connection, table, column)
        null_ratio = result["null_ratio"]
        healthy = null_ratio < 0.1

        return json.dumps({
            "success": True,
            "healthy": healthy,
            "null_count": result["null_count"],
            "total_rows": result["total_rows"],
            "null_ratio": null_ratio,
        })
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})
