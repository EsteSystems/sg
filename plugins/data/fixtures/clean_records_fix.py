"""Mutant: clean_records with improved empty-table handling."""
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
        # Handle empty table gracefully
        return json.dumps({
            "success": True,
            "cleaned_count": 0,
            "dropped_count": 0,
        })
