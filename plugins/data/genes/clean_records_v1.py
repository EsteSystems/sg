"""Seed gene: clean records by removing duplicates and null rows."""
import json


def execute(input_json: str) -> str:
    data = json.loads(input_json)

    connection = data.get("connection")
    if not connection:
        return json.dumps({"success": False, "error": "missing connection"})

    table = data.get("table")
    if not table:
        return json.dumps({"success": False, "error": "missing table"})

    rules = data.get("rules", {})

    try:
        result = gene_sdk.clean_records(connection, table, rules)
        gene_sdk.track_resource("table_clean", f"{connection}.{table}")
        return json.dumps({
            "success": True,
            "cleaned_count": result["cleaned_count"],
            "dropped_count": result["dropped_count"],
        })
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})
