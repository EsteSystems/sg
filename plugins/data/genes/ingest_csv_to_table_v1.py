"""Seed gene: fetch CSV from URL and load into a database table."""
import json

def execute(input_json: str) -> str:
    data = json.loads(input_json)

    url = data.get("url")
    if not url or not isinstance(url, str):
        return json.dumps({"success": False, "rows_written": 0, "error": "missing or invalid url"})

    connection = data.get("connection")
    if not connection or not isinstance(connection, str):
        return json.dumps({"success": False, "rows_written": 0, "error": "missing or invalid connection"})

    table = data.get("table")
    if not table or not isinstance(table, str):
        return json.dumps({"success": False, "rows_written": 0, "error": "missing or invalid table"})

    try:
        response = gene_sdk.http_get(url)
        records = response.get("records", [])
        rows_written = gene_sdk.write_records(connection, table, records)
        gene_sdk.track_resource("table_rows", f"{connection}.{table}")
        return json.dumps({
            "success": True,
            "rows_written": rows_written,
        })
    except Exception as e:
        return json.dumps({"success": False, "rows_written": 0, "error": str(e)})
