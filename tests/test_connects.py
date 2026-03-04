"""Tests for the connects section and default parameter values in contracts."""
import json
import pytest

from sg.parser.lexer import tokenize, TokenType
from sg.parser.parser import parse_sg
from sg.parser.types import GeneContract, ConnectsDef, PathwayContract


GENE_WITH_CONNECTS = """\
gene ingest_csv_to_table for data
  is configuration
  risk low

  does:
    Fetch CSV data from a URL and load it into a database table.

  takes:
    url         string  "URL to fetch CSV data from"
    connection  string  "Database connection name"
    table       string  "Target table name"

  gives:
    success        bool    "Whether the ingest completed"
    rows_written   int     "Number of rows written"

  connects:
    url         https   "CSV dataset endpoint"
    connection  sqlite  "Warehouse database"

  fails when:
    - URL unreachable -> success=false
"""

GENE_NO_CONNECTS = """\
gene pure_transform for data
  is configuration
  risk none

  does:
    A purely internal gene with no external interfaces.

  takes:
    data  string  "Input data blob"

  gives:
    success  bool  "Whether transform completed"
"""

GENE_MULTI_PROTOCOL = """\
gene etl_pipeline for data
  is configuration
  risk medium

  does:
    Extract from HTTP API, load into PostgreSQL.

  takes:
    api_url     string  "Source API endpoint"
    pg_conn     string  "PostgreSQL connection string"
    s3_bucket   string  "S3 bucket for staging"

  gives:
    success  bool  "Whether ETL completed"

  connects:
    api_url     https       "REST API source"
    pg_conn     postgresql  "Target database"
    s3_bucket   s3          "Staging area"
"""

PATHWAY_WITH_DEFAULTS = """\
pathway ingest_and_validate for data
  risk low

  does:
    Ingest CSV data from a URL into a database table.

  takes:
    url         string  "URL to fetch CSV data from"         default="https://example.com/data.csv"
    connection  string  "Database connection name"            default="warehouse"
    table       string  "Target table name"                  default="test_table"
    column      string  "Column to check for nulls"          default="id"

  steps:
    1. ingest_csv_to_table
         url = {url}
         connection = {connection}
         table = {table}

  on failure:
    rollback all
"""

PATHWAY_NO_DEFAULTS = """\
pathway simple_path for data
  risk low

  does:
    A pathway with no default values.

  takes:
    input_data  string  "Input data"

  steps:
    1. some_gene
         data = {input_data}
"""


class TestConnectsLexer:
    def test_connects_keyword_tokenized(self):
        tokens = tokenize("  connects:")
        keywords = [t for t in tokens if t.type == TokenType.KEYWORD]
        assert any(t.value == "connects" for t in keywords)

    def test_protocol_names_are_identifiers(self):
        tokens = tokenize("    url  https  \"description\"")
        identifiers = [t.value for t in tokens if t.type == TokenType.IDENTIFIER]
        assert "url" in identifiers
        assert "https" in identifiers


class TestConnectsParser:
    def test_parse_connects_section(self):
        result = parse_sg(GENE_WITH_CONNECTS)
        assert isinstance(result, GeneContract)
        assert len(result.connects) == 2

    def test_connects_https_interface(self):
        result = parse_sg(GENE_WITH_CONNECTS)
        url_conn = next(c for c in result.connects if c.param == "url")
        assert url_conn.interface == "https"
        assert url_conn.description == "CSV dataset endpoint"

    def test_connects_sqlite_interface(self):
        result = parse_sg(GENE_WITH_CONNECTS)
        db_conn = next(c for c in result.connects if c.param == "connection")
        assert db_conn.interface == "sqlite"
        assert db_conn.description == "Warehouse database"

    def test_no_connects_is_empty_list(self):
        result = parse_sg(GENE_NO_CONNECTS)
        assert isinstance(result, GeneContract)
        assert result.connects == []

    def test_multi_protocol_connects(self):
        result = parse_sg(GENE_MULTI_PROTOCOL)
        assert len(result.connects) == 3
        interfaces = {c.param: c.interface for c in result.connects}
        assert interfaces == {
            "api_url": "https",
            "pg_conn": "postgresql",
            "s3_bucket": "s3",
        }

    def test_connects_preserves_other_sections(self):
        result = parse_sg(GENE_WITH_CONNECTS)
        assert result.name == "ingest_csv_to_table"
        assert result.domain == "data"
        assert len(result.takes) == 3
        assert len(result.gives) == 2
        assert len(result.fails_when) == 1


class TestPathwayDefaults:
    def test_parse_defaults(self):
        result = parse_sg(PATHWAY_WITH_DEFAULTS)
        assert isinstance(result, PathwayContract)
        defaults = {f.name: f.default for f in result.takes if f.default}
        assert defaults == {
            "url": "https://example.com/data.csv",
            "connection": "warehouse",
            "table": "test_table",
            "column": "id",
        }

    def test_no_defaults(self):
        result = parse_sg(PATHWAY_NO_DEFAULTS)
        assert isinstance(result, PathwayContract)
        assert all(f.default is None for f in result.takes)

    def test_defaults_coexist_with_description(self):
        result = parse_sg(PATHWAY_WITH_DEFAULTS)
        url_field = next(f for f in result.takes if f.name == "url")
        assert url_field.description == "URL to fetch CSV data from"
        assert url_field.default == "https://example.com/data.csv"

    def test_fields_with_defaults_remain_required(self):
        """Fields with defaults are still structurally required by the contract."""
        result = parse_sg(PATHWAY_WITH_DEFAULTS)
        for f in result.takes:
            assert f.default is not None
