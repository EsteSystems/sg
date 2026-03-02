"""DataKernel — abstract kernel interface for data pipeline operations.

Provides HTTP fetch, database query, record writing, schema inspection,
null checking, and row counting operations.
"""
from __future__ import annotations

from abc import abstractmethod

from sg.kernel.base import Kernel, mutating


class DataKernel(Kernel):
    """Abstract kernel interface for data pipeline operations.

    Genes use gene_sdk (a DataKernel instance) to ingest, transform,
    validate, and load data.
    """

    # --- HTTP operations ---

    @abstractmethod
    def http_get(self, url: str, headers: dict | None = None) -> dict:
        """Fetch data from a URL. Returns parsed response."""
        ...

    # --- Database operations ---

    @mutating(
        undo=lambda k, snap, connection, table, records: (
            k.delete_records(connection, table, len(records))
        ),
    )
    @abstractmethod
    def write_records(self, connection: str, table: str, records: list[dict]) -> int:
        """Write records to a table. Returns number of rows written."""
        ...

    @abstractmethod
    def query_db(self, connection: str, sql: str) -> list[dict]:
        """Execute a SQL query and return results as list of dicts."""
        ...

    @abstractmethod
    def get_table_schema(self, connection: str, table: str) -> dict:
        """Get table schema. Returns dict with column names and types."""
        ...

    @abstractmethod
    def row_count(self, connection: str, table: str) -> int:
        """Get the number of rows in a table."""
        ...

    @abstractmethod
    def check_nulls(self, connection: str, table: str, column: str) -> dict:
        """Check for null values in a column. Returns null count and total."""
        ...

    @abstractmethod
    def delete_records(self, connection: str, table: str, count: int) -> None:
        """Delete the last N records from a table (for undo)."""
        ...

    # --- Self-description ---

    def describe_operations(self) -> list[str]:
        return [
            "http_get(url: str, headers: dict | None) -> dict",
            "write_records(connection: str, table: str, records: list[dict]) -> int",
            "query_db(connection: str, sql: str) -> list[dict]",
            "get_table_schema(connection: str, table: str) -> dict",
            "row_count(connection: str, table: str) -> int",
            "check_nulls(connection: str, table: str, column: str) -> dict",
            "delete_records(connection: str, table: str, count: int) -> None",
        ]

    def mutation_prompt_context(self) -> str:
        return (
            "This gene operates on data pipelines. gene_sdk is a DataKernel "
            "providing HTTP fetch, database query, record writing, schema "
            "inspection, and data quality operations. Genes ingest, transform, "
            "validate, and load data."
        )

    def domain_name(self) -> str:
        return "data"
