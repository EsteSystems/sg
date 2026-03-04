"""Pipeline runtime — binds sources and sinks to pathways.

A pipeline connects a data source to a pathway to a data sink,
auto-constructing the input JSON from source + sink + bind declarations.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from sg.contracts import ContractStore
from sg.parser.types import PipelineContract, SourceType, SinkType


@dataclass
class PipelineResult:
    """Result of a pipeline execution."""
    pipeline_name: str
    pathway_name: str
    success: bool
    outputs: list[str] = field(default_factory=list)
    error: str | None = None
    input_json: str = ""


def build_pipeline_input(pipeline: PipelineContract) -> dict:
    """Construct the input JSON dict from source + sink + bind declarations.

    Maps source/sink properties into the flat key-value structure that
    pathway steps expect, then overlays explicit bind values.
    """
    result: dict[str, str] = {}

    if pipeline.source:
        src = pipeline.source
        if src.source_type in (SourceType.HTTP_CSV, SourceType.HTTP_JSON):
            if "url" in src.properties:
                result["url"] = src.properties["url"]
            if "format" in src.properties:
                result["format"] = src.properties["format"]
        elif src.source_type == SourceType.SQLITE:
            if "connection" in src.properties:
                result["connection"] = src.properties["connection"]
            if "table" in src.properties:
                result["source_table"] = src.properties["table"]
            if "query" in src.properties:
                result["query"] = src.properties["query"]
        elif src.source_type == SourceType.FILE_CSV:
            if "path" in src.properties:
                result["source_path"] = src.properties["path"]
        elif src.source_type == SourceType.POSTGRES:
            for k, v in src.properties.items():
                result[f"source_{k}"] = v

        for k, v in src.properties.items():
            if k not in result and f"source_{k}" not in result:
                result[k] = v

    if pipeline.sink:
        snk = pipeline.sink
        if snk.sink_type == SinkType.SQLITE:
            if "connection" in snk.properties:
                result["connection"] = snk.properties["connection"]
            if "table" in snk.properties:
                result["table"] = snk.properties["table"]
        elif snk.sink_type == SinkType.FILE_CSV:
            if "path" in snk.properties:
                result["dest_path"] = snk.properties["path"]
        elif snk.sink_type == SinkType.POSTGRES:
            for k, v in snk.properties.items():
                result[f"dest_{k}"] = v

        for k, v in snk.properties.items():
            if k not in result and f"dest_{k}" not in result:
                result[k] = v

    for k, v in pipeline.bind.items():
        if v.startswith("source.") or v.startswith("sink."):
            parts = v.split(".", 1)
            section = parts[0]
            prop_key = parts[1] if len(parts) > 1 else ""
            if section == "source" and pipeline.source:
                result[k] = pipeline.source.properties.get(prop_key, v)
            elif section == "sink" and pipeline.sink:
                result[k] = pipeline.sink.properties.get(prop_key, v)
            else:
                result[k] = v
        else:
            result[k] = v

    return result


def run_pipeline(
    pipeline: PipelineContract,
    orchestrator,
    input_override: dict | None = None,
) -> PipelineResult:
    """Execute a pipeline: build input from source/sink/bind, run the pathway.

    Args:
        pipeline: The parsed pipeline contract.
        orchestrator: An Orchestrator instance.
        input_override: Optional dict to merge over the auto-constructed input.

    Returns:
        PipelineResult with outputs and status.
    """
    if not pipeline.through:
        return PipelineResult(
            pipeline_name=pipeline.name,
            pathway_name="",
            success=False,
            error="pipeline has no 'through' pathway",
        )

    input_dict = build_pipeline_input(pipeline)
    if input_override:
        input_dict.update(input_override)

    input_json = json.dumps(input_dict)

    try:
        outputs = orchestrator.run_pathway(pipeline.through, input_json)
        all_ok = True
        for out in outputs:
            try:
                parsed = json.loads(out)
                if isinstance(parsed, dict) and not parsed.get("success", True):
                    all_ok = False
            except (json.JSONDecodeError, TypeError):
                pass

        return PipelineResult(
            pipeline_name=pipeline.name,
            pathway_name=pipeline.through,
            success=all_ok,
            outputs=outputs,
            input_json=input_json,
        )
    except Exception as e:
        return PipelineResult(
            pipeline_name=pipeline.name,
            pathway_name=pipeline.through,
            success=False,
            error=str(e),
            input_json=input_json,
        )


def discover_source_schema(pipeline: PipelineContract, kernel) -> list[dict]:
    """Fetch the source and discover its schema (column names and types).

    Returns a list of {name, type, sample} dicts.
    """
    if pipeline.source is None:
        return []

    src = pipeline.source
    fields: list[dict] = []

    if src.source_type in (SourceType.HTTP_CSV, SourceType.HTTP_JSON):
        url = src.properties.get("url")
        if not url:
            return []
        try:
            data = kernel.http_get(url)
            records = data.get("records", [])
            if records:
                first = records[0]
                for col_name, sample_val in first.items():
                    inferred_type = "string"
                    if sample_val is not None:
                        try:
                            float(sample_val)
                            inferred_type = "float" if "." in str(sample_val) else "int"
                        except (ValueError, TypeError):
                            pass
                    fields.append({
                        "name": col_name.strip(),
                        "type": inferred_type,
                        "sample": str(sample_val)[:50] if sample_val else "",
                    })
        except Exception:
            pass

    elif src.source_type == SinkType.SQLITE:
        connection = src.properties.get("connection")
        table = src.properties.get("table")
        if connection and table:
            try:
                schema = kernel.table_schema(connection, table)
                for col in schema:
                    fields.append({
                        "name": col.get("name", ""),
                        "type": col.get("type", "string"),
                        "sample": "",
                    })
            except Exception:
                pass

    return fields
