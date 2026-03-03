"""Seed gene: validate table schema against expected columns."""
import json


def execute(input_json: str) -> str:
    data = json.loads(input_json)

    connection = data.get("connection")
    if not connection:
        return json.dumps({"success": False, "error": "missing connection"})

    table = data.get("table")
    if not table:
        return json.dumps({"success": False, "error": "missing table"})

    expected = data.get("expected_columns", [])

    try:
        schema = gene_sdk.get_table_schema(connection, table)
        actual_cols = set(schema.get("columns", {}).keys())
        expected_set = set(expected)

        missing = sorted(expected_set - actual_cols)
        extra = sorted(actual_cols - expected_set)
        healthy = len(missing) == 0

        return json.dumps({
            "success": True,
            "healthy": healthy,
            "missing_columns": missing,
            "extra_columns": extra,
        })
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})
