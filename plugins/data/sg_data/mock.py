"""MockDataKernel — in-memory simulation for development/testing.

Provides in-memory tables, mock HTTP responses, and data quality checking.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from sg_data.kernel import DataKernel


@dataclass
class TableState:
    """In-memory table with schema and data."""
    name: str
    columns: dict[str, str]  # column_name -> type
    rows: list[dict] = field(default_factory=list)


class MockDataKernel(DataKernel):
    """In-memory data pipeline simulation. Injected into genes as `gene_sdk`."""

    def __init__(self) -> None:
        self._tables: dict[str, dict[str, TableState]] = {}  # connection -> table_name -> state
        self._http_responses: dict[str, dict] = {}  # url -> response
        self._tracked: list[tuple[str, str]] = []
        self._injected_failures: dict[str, str] = {}

    def create_shadow(self) -> MockDataKernel:
        """Return a fresh mock kernel for shadow execution."""
        return MockDataKernel()

    def reset(self) -> None:
        self._tables.clear()
        self._http_responses.clear()
        self._tracked.clear()
        self._injected_failures.clear()

    # --- Test setup helpers ---

    def add_table(self, connection: str, table: str, columns: dict[str, str],
                  rows: list[dict] | None = None) -> None:
        """Create a table for testing."""
        if connection not in self._tables:
            self._tables[connection] = {}
        self._tables[connection][table] = TableState(
            name=table, columns=columns, rows=list(rows or []),
        )

    def add_http_response(self, url: str, response: dict) -> None:
        """Register a mock HTTP response."""
        self._http_responses[url] = response

    def inject_failure(self, operation: str, message: str) -> None:
        """Inject a one-shot failure for the given operation."""
        self._injected_failures[operation] = message

    def _check_failure(self, operation: str) -> None:
        msg = self._injected_failures.pop(operation, None)
        if msg is not None:
            raise RuntimeError(msg)

    # --- HTTP operations ---

    def http_get(self, url: str, headers: dict | None = None) -> dict:
        self._check_failure("http_get")
        if url not in self._http_responses:
            raise ValueError(f"no mock response for URL: {url}")
        return self._http_responses[url]

    # --- Database operations ---

    def _get_table(self, connection: str, table: str) -> TableState:
        conn = self._tables.get(connection, {})
        tbl = conn.get(table)
        if tbl is None:
            raise ValueError(f"table '{table}' does not exist in '{connection}'")
        return tbl

    def write_records(self, connection: str, table: str, records: list[dict]) -> int:
        self._check_failure("write_records")
        tbl = self._get_table(connection, table)
        tbl.rows.extend(records)
        self.track_resource("table_rows", f"{connection}.{table}")
        return len(records)

    def query_db(self, connection: str, sql: str) -> list[dict]:
        self._check_failure("query_db")
        # Simple mock: return all rows from the first table mentioned
        conn = self._tables.get(connection, {})
        for tbl in conn.values():
            return list(tbl.rows)
        return []

    def get_table_schema(self, connection: str, table: str) -> dict:
        self._check_failure("get_table_schema")
        tbl = self._get_table(connection, table)
        return {"table": table, "columns": dict(tbl.columns)}

    def row_count(self, connection: str, table: str) -> int:
        self._check_failure("row_count")
        tbl = self._get_table(connection, table)
        return len(tbl.rows)

    def check_nulls(self, connection: str, table: str, column: str) -> dict:
        self._check_failure("check_nulls")
        tbl = self._get_table(connection, table)
        total = len(tbl.rows)
        null_count = sum(1 for row in tbl.rows if row.get(column) is None)
        return {
            "column": column,
            "total_rows": total,
            "null_count": null_count,
            "null_ratio": null_count / total if total > 0 else 0.0,
        }

    def delete_records(self, connection: str, table: str, count: int) -> None:
        self._check_failure("delete_records")
        tbl = self._get_table(connection, table)
        tbl.rows = tbl.rows[:-count] if count <= len(tbl.rows) else []

    # --- Resource tracking ---

    def track_resource(self, resource_type: str, name: str) -> None:
        pair = (resource_type, name)
        if pair not in self._tracked:
            self._tracked.append(pair)

    def untrack_resource(self, resource_type: str, name: str) -> None:
        pair = (resource_type, name)
        if pair in self._tracked:
            self._tracked.remove(pair)

    def tracked_resources(self) -> list[tuple[str, str]]:
        return list(self._tracked)
