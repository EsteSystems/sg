"""Mutation engines — mock (fixtures) and Claude (API).

When all alleles are exhausted, the orchestrator triggers mutation:
an LLM generates a new allele from the contract, failing source,
error context, and environmental state.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from sg.contracts import contract_info


@dataclass
class MutationContext:
    gene_source: str
    locus: str
    failing_input: str
    error_message: str


class MutationEngine(ABC):
    @abstractmethod
    def mutate(self, ctx: MutationContext) -> str:
        """Generate a mutated gene source. Returns Python source code."""
        ...

    def generate_fused(self, pathway_name: str, gene_sources: list[str],
                       loci: list[str]) -> str:
        """Generate a fused gene combining multiple genes into one."""
        raise NotImplementedError("this engine does not support fusion generation")


class MockMutationEngine(MutationEngine):
    """Loads fixture files as mutation results. For development/testing."""

    def __init__(self, fixtures_dir: Path):
        self.fixtures_dir = fixtures_dir

    def mutate(self, ctx: MutationContext) -> str:
        fixture_path = self.fixtures_dir / f"{ctx.locus}_fix.py"
        if not fixture_path.exists():
            raise FileNotFoundError(f"no fixture at {fixture_path}")
        return fixture_path.read_text()

    def generate_fused(self, pathway_name: str, gene_sources: list[str],
                       loci: list[str]) -> str:
        fixture_path = self.fixtures_dir / f"{pathway_name}_fused.py"
        if not fixture_path.exists():
            raise FileNotFoundError(f"no fusion fixture at {fixture_path}")
        return fixture_path.read_text()


class ClaudeMutationEngine(MutationEngine):
    """Calls the Anthropic API to generate mutations."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-5-20250929"):
        self.api_key = api_key
        self.model = model

    def _call_api(self, prompt: str) -> str:
        import httpx

        resp = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": self.model,
                "max_tokens": 4096,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=60.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["content"][0]["text"]

    def _extract_python(self, text: str) -> str:
        match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()

    def mutate(self, ctx: MutationContext) -> str:
        info = contract_info(ctx.locus)
        prompt = f"""You are a gene mutation engine for the Software Genomics runtime.

A gene is a Python function that takes a JSON string and returns a JSON string.
The gene has access to `gene_sdk` in its namespace (a NetworkKernel instance).

## Contract
Locus: {ctx.locus}
Description: {info.description}
Input schema: {info.input_schema}
Output schema: {info.output_schema}

## Current gene source (failing):
```python
{ctx.gene_source}
```

## Failure context:
Input: {ctx.failing_input}
Error: {ctx.error_message}

## Task
Write a fixed version of this gene. The gene must:
1. Define an `execute(input_json: str) -> str` function
2. Use `gene_sdk` for kernel operations (create_bridge, set_stp, etc.)
3. Return valid JSON with at least a "success" boolean field
4. Handle the error case described above

Return ONLY the Python source code in a ```python``` block."""
        text = self._call_api(prompt)
        return self._extract_python(text)

    def generate_fused(self, pathway_name: str, gene_sources: list[str],
                       loci: list[str]) -> str:
        steps_desc = []
        for i, (source, locus) in enumerate(zip(gene_sources, loci)):
            info = contract_info(locus)
            steps_desc.append(f"""### Step {i + 1}: {locus}
Description: {info.description}
```python
{source}
```""")

        prompt = f"""You are a gene fusion engine for the Software Genomics runtime.

A fused gene combines multiple pathway steps into a single optimized gene.
The gene has access to `gene_sdk` in its namespace (a NetworkKernel instance).

## Pathway: {pathway_name}

{chr(10).join(steps_desc)}

## Task
Write a single fused gene that performs all steps in sequence, optimizing
away intermediate JSON serialization where possible. The gene must:
1. Define an `execute(input_json: str) -> str` function
2. Accept the full pathway input (all fields from all steps)
3. Use `gene_sdk` for all kernel operations
4. Return valid JSON with "success": true on success

Return ONLY the Python source code in a ```python``` block."""
        text = self._call_api(prompt)
        return self._extract_python(text)
