"""Tests for the pipeline .sg contract parsing and runtime."""
import json
import pytest

from sg.parser.lexer import tokenize, TokenType
from sg.parser.parser import parse_sg, ParseError
from sg.parser.types import (
    PipelineContract, SourceDecl, SinkDecl, SourceType, SinkType,
)
from sg.pipeline import build_pipeline_input
from sg.contracts import ContractStore


SIMPLE_PIPELINE = """\
pipeline ingest_airtravel for data

  does:
    Fetch the FAU airtravel CSV dataset and load it into the
    warehouse SQLite database.

  source http_csv
    url "https://example.com/airtravel.csv"

  sink sqlite
    connection "warehouse"
    table "airtravel"

  through ingest_and_validate

  bind:
    column = "Month"
"""

MINIMAL_PIPELINE = """\
pipeline minimal_pipe

  does:
    A minimal pipeline with no bind.

  source http_csv
    url "https://example.com/data.csv"

  sink sqlite
    connection "db"
    table "records"

  through my_pathway
"""

PIPELINE_WITH_SCHEMA = """\
pipeline typed_pipe for data

  does:
    Pipeline with source schema declaration.

  source http_csv
    url "https://example.com/typed.csv"
    schema:
      month   string  "Calendar month"
      value   int     "Numeric value"

  sink sqlite
    connection "warehouse"
    table "typed_data"

  through ingest_and_validate

  bind:
    column = "month"
"""

PIPELINE_WITH_VERIFY = """\
pipeline verified_pipe for data

  does:
    Pipeline with verification.

  source http_csv
    url "https://example.com/v.csv"

  sink sqlite
    connection "db"
    table "vdata"

  through ingest_and_validate

  bind:
    column = "id"

  verify:
    check_row_count connection="db" table="vdata"
    within 30s
"""


class TestPipelineLexer:
    def test_pipeline_keyword_tokenized(self):
        tokens = tokenize("pipeline my_pipe")
        keywords = [t for t in tokens if t.type == TokenType.KEYWORD]
        assert any(t.value == "pipeline" for t in keywords)

    def test_source_sink_keywords(self):
        tokens = tokenize("  source http_csv\n  sink sqlite\n  through my_path\n  bind:")
        keywords = [t.value for t in tokens if t.type == TokenType.KEYWORD]
        assert "source" in keywords
        assert "sink" in keywords
        assert "through" in keywords
        assert "bind" in keywords


class TestPipelineParser:
    def test_parse_simple_pipeline(self):
        result = parse_sg(SIMPLE_PIPELINE)
        assert isinstance(result, PipelineContract)
        assert result.name == "ingest_airtravel"
        assert result.domain == "data"
        assert "airtravel" in result.does

    def test_source_parsed(self):
        result = parse_sg(SIMPLE_PIPELINE)
        assert result.source is not None
        assert result.source.source_type == SourceType.HTTP_CSV
        assert result.source.properties["url"] == "https://example.com/airtravel.csv"

    def test_sink_parsed(self):
        result = parse_sg(SIMPLE_PIPELINE)
        assert result.sink is not None
        assert result.sink.sink_type == SinkType.SQLITE
        assert result.sink.properties["connection"] == "warehouse"
        assert result.sink.properties["table"] == "airtravel"

    def test_through_parsed(self):
        result = parse_sg(SIMPLE_PIPELINE)
        assert result.through == "ingest_and_validate"

    def test_bind_parsed(self):
        result = parse_sg(SIMPLE_PIPELINE)
        assert result.bind == {"column": "Month"}

    def test_minimal_pipeline(self):
        result = parse_sg(MINIMAL_PIPELINE)
        assert isinstance(result, PipelineContract)
        assert result.name == "minimal_pipe"
        assert result.domain is None
        assert result.through == "my_pathway"
        assert result.bind == {}

    def test_pipeline_with_schema(self):
        result = parse_sg(PIPELINE_WITH_SCHEMA)
        assert isinstance(result, PipelineContract)
        assert result.source is not None
        assert len(result.source.schema_fields) == 2
        assert result.source.schema_fields[0].name == "month"
        assert result.source.schema_fields[0].type == "string"
        assert result.source.schema_fields[1].name == "value"
        assert result.source.schema_fields[1].type == "int"

    def test_pipeline_with_verify(self):
        result = parse_sg(PIPELINE_WITH_VERIFY)
        assert isinstance(result, PipelineContract)
        assert len(result.verify) == 1
        assert result.verify[0].locus == "check_row_count"
        assert result.verify_within == "30s"


class TestBuildPipelineInput:
    def test_simple_input_construction(self):
        pipeline = parse_sg(SIMPLE_PIPELINE)
        result = build_pipeline_input(pipeline)
        assert result["url"] == "https://example.com/airtravel.csv"
        assert result["connection"] == "warehouse"
        assert result["table"] == "airtravel"
        assert result["column"] == "Month"

    def test_minimal_input_construction(self):
        pipeline = parse_sg(MINIMAL_PIPELINE)
        result = build_pipeline_input(pipeline)
        assert result["url"] == "https://example.com/data.csv"
        assert result["connection"] == "db"
        assert result["table"] == "records"

    def test_bind_overrides_source_sink(self):
        src = """\
pipeline override_test
  does:
    Test bind override.
  source http_csv
    url "https://example.com/data.csv"
  sink sqlite
    connection "db"
    table "original"
  through my_pathway
  bind:
    table = "overridden"
"""
        pipeline = parse_sg(src)
        result = build_pipeline_input(pipeline)
        assert result["table"] == "overridden"

    def test_input_is_valid_json(self):
        pipeline = parse_sg(SIMPLE_PIPELINE)
        result = build_pipeline_input(pipeline)
        serialized = json.dumps(result)
        roundtripped = json.loads(serialized)
        assert roundtripped == result


class TestContractStoreWithPipelines:
    def test_load_pipeline_contract(self):
        store = ContractStore()
        from pathlib import Path
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sg", delete=False) as f:
            f.write(SIMPLE_PIPELINE)
            f.flush()
            store.load_file(Path(f.name))

        assert "ingest_airtravel" in store.known_pipelines()
        p = store.get_pipeline("ingest_airtravel")
        assert p is not None
        assert p.through == "ingest_and_validate"
