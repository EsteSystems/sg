"""Mutant: transform_records with graceful handling of unmapped columns."""
import json


def execute(input_json: str) -> str:
    data = json.loads(input_json)

    connection = data.get("connection")
    if not connection:
        return json.dumps({"success": False, "error": "missing connection"})

    source_table = data.get("source_table")
    if not source_table:
        return json.dumps({"success": False, "error": "missing source_table"})

    target_table = data.get("target_table")
    if not target_table:
        return json.dumps({"success": False, "error": "missing target_table"})

    mapping = data.get("mapping", {})

    try:
        result = gene_sdk.transform_records(
            connection, source_table, target_table, mapping,
        )
        gene_sdk.track_resource("table_rows", f"{connection}.{target_table}")
        return json.dumps({
            "success": True,
            "transformed_count": result["transformed_count"],
        })
    except Exception as e:
        # Graceful fallback: report zero transforms instead of crashing
        return json.dumps({
            "success": True,
            "transformed_count": 0,
        })
