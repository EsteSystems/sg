"""ProductionDataKernel — real database and HTTP operations.

Uses SQLite by default (zero-config), PostgreSQL via DSN override.
HTTP ingestion via urllib (stdlib). No heavy dependencies required.

Configuration:
    SG_DATA_DSN: Database connection string (default: sqlite:///sg_data.db)
    SG_PROTECTED_TABLES: Comma-separated table names that refuse writes
    SG_HTTP_TIMEOUT: HTTP timeout in seconds (default: 30)
"""
from __future__ import annotations

import csv
import io
import json
import os
import sqlite3
import urllib.request
import urllib.error
from pathlib import Path

from sg_data.kernel import DataKernel

_DEFAULT_DSN = "sqlite:///sg_data.db"
_DEFAULT_HTTP_TIMEOUT = 30


def _parse_protected_tables() -> set[str]:
    raw = os.environ.get("SG_PROTECTED_TABLES", "")
    return {t.strip() for t in raw.split(",") if t.strip()}


class ProductionDataKernel(DataKernel):
    """Real data pipeline kernel backed by SQLite and HTTP.

    SQLite is used because it requires zero infrastructure, runs on any
    OS, and is sufficient for single-daemon operation. The connection
    string can be overridden via SG_DATA_DSN for PostgreSQL if needed.
    """

    def __init__(
        self,
        dsn: str | None = None,
        dry_run: bool = False,
    ) -> None:
        self._dsn = dsn or os.environ.get("SG_DATA_DSN", _DEFAULT_DSN)
        self._dry_run = dry_run
        self._tracked: list[tuple[str, str]] = []
        self._protected = _parse_protected_tables()
        self._http_timeout = int(os.environ.get("SG_HTTP_TIMEOUT", _DEFAULT_HTTP_TIMEOUT))

        self._db_path = self._dsn
        if self._db_path.startswith("sqlite:///"):
            self._db_path = self._db_path[len("sqlite:///"):]

    def create_shadow(self) -> DataKernel:
        from sg_data.mock import MockDataKernel
        return MockDataKernel()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _check_protected(self, table: str) -> None:
        if table in self._protected:
            raise PermissionError(
                f"table '{table}' is protected (SG_PROTECTED_TABLES)"
            )

    def _ensure_table(self, connection: str, table: str, columns: dict[str, str]) -> None:
        """Create table if it doesn't exist."""
        col_defs = ", ".join(f'"{c}" TEXT' for c in columns)
        sql = f'CREATE TABLE IF NOT EXISTS "{table}" ({col_defs})'
        with self._connect() as conn:
            conn.execute(sql)

    # --- HTTP operations ---

    def http_get(self, url: str, headers: dict | None = None) -> dict:
        req = urllib.request.Request(url, headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=self._http_timeout) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.URLError as e:
            raise ConnectionError(f"HTTP GET failed: {e}") from e

        content_type = resp.headers.get("Content-Type", "")
        if "json" in content_type or body.strip().startswith(("{", "[")):
            return json.loads(body)

        reader = csv.DictReader(io.StringIO(body))
        records = list(reader)
        return {"records": records}

    # --- Database operations ---

    def write_records(self, connection: str, table: str, records: list[dict]) -> int:
        self._check_protected(table)
        if not records:
            return 0
        if self._dry_run:
            return len(records)

        columns = list(records[0].keys())
        placeholders = ", ".join("?" for _ in columns)
        col_names = ", ".join(f'"{c}"' for c in columns)
        sql = f'INSERT INTO "{table}" ({col_names}) VALUES ({placeholders})'

        col_defs = ", ".join(f'"{c}" TEXT' for c in columns)
        create_sql = f'CREATE TABLE IF NOT EXISTS "{table}" ({col_defs})'

        with self._connect() as conn:
            conn.execute(create_sql)
            rows = [tuple(r.get(c) for c in columns) for r in records]
            conn.executemany(sql, rows)

        self.track_resource("table_rows", f"{connection}.{table}")
        return len(records)

    def query_db(self, connection: str, sql: str) -> list[dict]:
        with self._connect() as conn:
            cursor = conn.execute(sql)
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def get_table_schema(self, connection: str, table: str) -> dict:
        with self._connect() as conn:
            cursor = conn.execute(f'PRAGMA table_info("{table}")')
            rows = cursor.fetchall()
        if not rows:
            raise ValueError(f"table '{table}' does not exist")
        columns = {row[1]: row[2] for row in rows}
        return {"table": table, "columns": columns}

    def row_count(self, connection: str, table: str) -> int:
        with self._connect() as conn:
            cursor = conn.execute(f'SELECT COUNT(*) FROM "{table}"')
            return cursor.fetchone()[0]

    def check_nulls(self, connection: str, table: str, column: str) -> dict:
        with self._connect() as conn:
            total = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            null_count = conn.execute(
                f'SELECT COUNT(*) FROM "{table}" WHERE "{column}" IS NULL'
            ).fetchone()[0]
        return {
            "column": column,
            "total_rows": total,
            "null_count": null_count,
            "null_ratio": null_count / total if total > 0 else 0.0,
        }

    def delete_records(self, connection: str, table: str, count: int) -> None:
        self._check_protected(table)
        if self._dry_run:
            return
        with self._connect() as conn:
            conn.execute(
                f'DELETE FROM "{table}" WHERE rowid IN '
                f'(SELECT rowid FROM "{table}" ORDER BY rowid DESC LIMIT ?)',
                (count,),
            )

    def clean_records(self, connection: str, table: str, rules: dict) -> dict:
        self._check_protected(table)
        if self._dry_run:
            return {"cleaned_count": 0, "dropped_count": 0}

        with self._connect() as conn:
            original = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]

            drop_nulls = rules.get("drop_nulls", [])
            for col in drop_nulls:
                conn.execute(f'DELETE FROM "{table}" WHERE "{col}" IS NULL')

            dedup_columns = rules.get("dedup_columns", [])
            if dedup_columns:
                cols = ", ".join(f'"{c}"' for c in dedup_columns)
                conn.execute(
                    f'DELETE FROM "{table}" WHERE rowid NOT IN '
                    f'(SELECT MIN(rowid) FROM "{table}" GROUP BY {cols})'
                )

            fill_defaults = rules.get("fill_defaults", {})
            for col, default in fill_defaults.items():
                conn.execute(
                    f'UPDATE "{table}" SET "{col}" = ? WHERE "{col}" IS NULL',
                    (default,),
                )

            remaining = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]

        dropped = original - remaining
        self.track_resource("table_clean", f"{connection}.{table}")
        return {"cleaned_count": remaining, "dropped_count": dropped}

    def transform_records(self, connection: str, source_table: str,
                          target_table: str, mapping: dict) -> dict:
        self._check_protected(target_table)
        if self._dry_run:
            return {"transformed_count": 0}

        src_cols = ", ".join(f'"{c}"' for c in mapping.keys())
        tgt_cols = ", ".join(f'"{c}"' for c in mapping.values())

        tgt_col_defs = ", ".join(f'"{c}" TEXT' for c in mapping.values())
        create_sql = f'CREATE TABLE IF NOT EXISTS "{target_table}" ({tgt_col_defs})'

        with self._connect() as conn:
            conn.execute(create_sql)
            cursor = conn.execute(f'SELECT {src_cols} FROM "{source_table}"')
            rows = cursor.fetchall()
            if rows:
                placeholders = ", ".join("?" for _ in mapping.values())
                conn.executemany(
                    f'INSERT INTO "{target_table}" ({tgt_cols}) VALUES ({placeholders})',
                    rows,
                )

        self.track_resource("table_rows", f"{connection}.{target_table}")
        return {"transformed_count": len(rows)}

    # --- State management ---

    def reset(self) -> None:
        self._tracked.clear()

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
