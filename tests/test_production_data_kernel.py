"""Tests for ProductionDataKernel — real SQLite operations."""
import json
import os
import pytest

sg_data = pytest.importorskip("sg_data", reason="sg_data plugin not installed")

from sg_data.production import ProductionDataKernel


@pytest.fixture
def kernel(tmp_path):
    db_path = tmp_path / "test.db"
    k = ProductionDataKernel(dsn=f"sqlite:///{db_path}")
    return k


@pytest.fixture
def kernel_with_table(kernel):
    kernel.write_records("test", "events", [
        {"id": "1", "name": "alpha", "value": "10.0"},
        {"id": "2", "name": "beta", "value": "20.0"},
        {"id": "3", "name": None, "value": "30.0"},
    ])
    return kernel


class TestWriteAndQuery:
    def test_write_records(self, kernel):
        count = kernel.write_records("conn", "users", [
            {"name": "alice", "age": "30"},
            {"name": "bob", "age": "25"},
        ])
        assert count == 2

    def test_query_db(self, kernel_with_table):
        rows = kernel_with_table.query_db("test", 'SELECT * FROM "events"')
        assert len(rows) == 3
        assert rows[0]["name"] == "alpha"

    def test_write_empty(self, kernel):
        assert kernel.write_records("conn", "t", []) == 0


class TestSchema:
    def test_get_table_schema(self, kernel_with_table):
        schema = kernel_with_table.get_table_schema("test", "events")
        assert schema["table"] == "events"
        assert "id" in schema["columns"]
        assert "name" in schema["columns"]

    def test_schema_missing_table(self, kernel):
        with pytest.raises(ValueError, match="does not exist"):
            kernel.get_table_schema("test", "nonexistent")


class TestRowCount:
    def test_row_count(self, kernel_with_table):
        assert kernel_with_table.row_count("test", "events") == 3


class TestCheckNulls:
    def test_check_nulls(self, kernel_with_table):
        result = kernel_with_table.check_nulls("test", "events", "name")
        assert result["total_rows"] == 3
        assert result["null_count"] == 1
        assert result["null_ratio"] == pytest.approx(1 / 3)


class TestDeleteRecords:
    def test_delete_last_records(self, kernel_with_table):
        kernel_with_table.delete_records("test", "events", 1)
        assert kernel_with_table.row_count("test", "events") == 2


class TestCleanRecords:
    def test_drop_nulls(self, kernel_with_table):
        result = kernel_with_table.clean_records("test", "events", {
            "drop_nulls": ["name"],
        })
        assert result["dropped_count"] == 1
        assert result["cleaned_count"] == 2

    def test_dedup(self, kernel):
        kernel.write_records("c", "t", [
            {"x": "1", "y": "a"},
            {"x": "1", "y": "a"},
            {"x": "2", "y": "b"},
        ])
        result = kernel.clean_records("c", "t", {
            "dedup_columns": ["x", "y"],
        })
        assert result["dropped_count"] == 1
        assert result["cleaned_count"] == 2

    def test_fill_defaults(self, kernel_with_table):
        result = kernel_with_table.clean_records("test", "events", {
            "fill_defaults": {"name": "unknown"},
        })
        assert result["dropped_count"] == 0
        nulls = kernel_with_table.check_nulls("test", "events", "name")
        assert nulls["null_count"] == 0


class TestTransformRecords:
    def test_transform(self, kernel_with_table):
        result = kernel_with_table.transform_records(
            "test", "events", "output", {"id": "event_id", "name": "event_name"},
        )
        assert result["transformed_count"] == 3
        rows = kernel_with_table.query_db("test", 'SELECT * FROM "output"')
        assert len(rows) == 3
        assert "event_id" in rows[0]


class TestProtectedTables:
    def test_protected_table_blocks_write(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SG_PROTECTED_TABLES", "secret,users")
        db_path = tmp_path / "test.db"
        k = ProductionDataKernel(dsn=f"sqlite:///{db_path}")
        with pytest.raises(PermissionError, match="protected"):
            k.write_records("c", "secret", [{"x": "1"}])

    def test_protected_table_blocks_clean(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SG_PROTECTED_TABLES", "events")
        db_path = tmp_path / "test.db"
        k = ProductionDataKernel(dsn=f"sqlite:///{db_path}")
        with pytest.raises(PermissionError):
            k.clean_records("c", "events", {})


class TestDryRun:
    def test_dry_run_write_returns_count(self, tmp_path):
        db_path = tmp_path / "test.db"
        k = ProductionDataKernel(dsn=f"sqlite:///{db_path}", dry_run=True)
        count = k.write_records("c", "t", [{"x": "1"}, {"x": "2"}])
        assert count == 2

    def test_dry_run_clean_returns_zero(self, tmp_path):
        db_path = tmp_path / "test.db"
        k = ProductionDataKernel(dsn=f"sqlite:///{db_path}", dry_run=True)
        result = k.clean_records("c", "t", {"drop_nulls": ["x"]})
        assert result["cleaned_count"] == 0


class TestShadow:
    def test_create_shadow_returns_mock(self, kernel):
        shadow = kernel.create_shadow()
        from sg_data.mock import MockDataKernel
        assert isinstance(shadow, MockDataKernel)


class TestResourceTracking:
    def test_track_and_list(self, kernel):
        kernel.track_resource("table", "events")
        assert ("table", "events") in kernel.tracked_resources()
        kernel.untrack_resource("table", "events")
        assert ("table", "events") not in kernel.tracked_resources()
