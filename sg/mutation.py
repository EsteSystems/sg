"""Mutation engines — mock (fixtures) and LLM (Claude, OpenAI, DeepSeek).

When all alleles are exhausted, the orchestrator triggers mutation:
an LLM generates a new allele from the contract, failing source,
error context, and environmental state.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sg.contracts import ContractStore


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

    def generate(self, locus: str, contract_prompt: str,
                 count: int = 1) -> list[str]:
        """Proactively generate competing implementations from a contract.

        Returns a list of Python source code strings (one per variant).
        """
        raise NotImplementedError("this engine does not support proactive generation")

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

    def generate(self, locus: str, contract_prompt: str,
                 count: int = 1) -> list[str]:
        fixture_path = self.fixtures_dir / f"{locus}_fix.py"
        if not fixture_path.exists():
            raise FileNotFoundError(f"no generation fixture at {fixture_path}")
        return [fixture_path.read_text()]

    def generate_fused(self, pathway_name: str, gene_sources: list[str],
                       loci: list[str]) -> str:
        fixture_path = self.fixtures_dir / f"{pathway_name}_fused.py"
        if not fixture_path.exists():
            raise FileNotFoundError(f"no fusion fixture at {fixture_path}")
        return fixture_path.read_text()


class LLMMutationEngine(MutationEngine):
    """Base class for LLM-backed mutation engines.

    Handles prompt building, Python extraction, and multi-variant parsing.
    Subclasses only need to implement _call_api().
    """

    def __init__(self, contract_store: ContractStore):
        self.contract_store = contract_store

    @abstractmethod
    def _call_api(self, prompt: str) -> str:
        """Send a prompt to the LLM and return the response text."""
        ...

    @property
    def provider_name(self) -> str:
        return "llm"

    def _extract_python(self, text: str) -> str:
        match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()

    def _contract_prompt(self, locus: str) -> str:
        """Build contract context for a mutation prompt."""
        gene = self.contract_store.get_gene(locus)
        if gene:
            sections = [f"Locus: {locus}"]
            if gene.does:
                sections.append(f"Description:\n{gene.does}")
            if gene.takes:
                fields = "\n".join(
                    f"  {f.name}: {f.type}" + (f" (optional)" if f.optional else "")
                    for f in gene.takes
                )
                sections.append(f"Input fields:\n{fields}")
            if gene.gives:
                fields = "\n".join(
                    f"  {f.name}: {f.type}" + (f" (optional)" if f.optional else "")
                    for f in gene.gives
                )
                sections.append(f"Output fields:\n{fields}")
            if gene.before:
                sections.append("Preconditions:\n" + "\n".join(f"  - {c}" for c in gene.before))
            if gene.after:
                sections.append("Postconditions:\n" + "\n".join(f"  - {c}" for c in gene.after))
            if gene.fails_when:
                sections.append("Failure modes:\n" + "\n".join(f"  - {c}" for c in gene.fails_when))
            return "\n\n".join(sections)
        # Fallback if no .sg contract
        try:
            info = self.contract_store.contract_info(locus)
            return f"Locus: {locus}\nDescription: {info.description}\nInput: {info.input_schema}\nOutput: {info.output_schema}"
        except ValueError:
            return f"Locus: {locus}"

    def mutate(self, ctx: MutationContext) -> str:
        contract_ctx = self._contract_prompt(ctx.locus)
        prompt = f"""You are a gene mutation engine for the Software Genomics runtime.

A gene is a Python function that takes a JSON string and returns a JSON string.
The gene has access to `gene_sdk` in its namespace (a NetworkKernel instance).

## Contract
{contract_ctx}

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

    def generate(self, locus: str, contract_prompt: str,
                 count: int = 1) -> list[str]:
        if count == 1:
            prompt = f"""You are a gene generation engine for the Software Genomics runtime.

A gene is a Python function that takes a JSON string and returns a JSON string.
The gene has access to `gene_sdk` in its namespace (a NetworkKernel instance).

## Contract
{contract_prompt}

## Task
Write a Python implementation of this gene. The gene must:
1. Define an `execute(input_json: str) -> str` function
2. Use `gene_sdk` for kernel operations (create_bridge, set_stp, get_device_mac, etc.)
3. Parse input with `json.loads(input_json)`
4. Return valid JSON (via `json.dumps()`) with at least a "success" boolean field
5. Handle all failure modes described in the contract

Return ONLY the Python source code in a ```python``` block."""
        else:
            prompt = f"""You are a gene generation engine for the Software Genomics runtime.

A gene is a Python function that takes a JSON string and returns a JSON string.
The gene has access to `gene_sdk` in its namespace (a NetworkKernel instance).

## Contract
{contract_prompt}

## Task
Write {count} DIFFERENT implementations of this gene, each using a different approach
or strategy. Each implementation must:
1. Define an `execute(input_json: str) -> str` function
2. Use `gene_sdk` for kernel operations (create_bridge, set_stp, get_device_mac, etc.)
3. Parse input with `json.loads(input_json)`
4. Return valid JSON (via `json.dumps()`) with at least a "success" boolean field
5. Handle all failure modes described in the contract

Separate each implementation with a line containing only: ---VARIANT---

Return ONLY Python source code in ```python``` blocks, separated by ---VARIANT---."""

        text = self._call_api(prompt)

        if count == 1:
            return [self._extract_python(text)]

        # Split on variant separator
        variants = []
        for chunk in re.split(r"---VARIANT---", text):
            chunk = chunk.strip()
            if not chunk:
                continue
            variants.append(self._extract_python(chunk))
        return variants if variants else [self._extract_python(text)]

    def generate_fused(self, pathway_name: str, gene_sources: list[str],
                       loci: list[str]) -> str:
        steps_desc = []
        for i, (source, locus) in enumerate(zip(gene_sources, loci)):
            contract_ctx = self._contract_prompt(locus)
            steps_desc.append(f"### Step {i + 1}: {locus}\n{contract_ctx}\n```python\n{source}\n```")

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


class ClaudeMutationEngine(LLMMutationEngine):
    """Calls the Anthropic API to generate mutations."""

    def __init__(self, api_key: str, contract_store: ContractStore,
                 model: str = "claude-sonnet-4-5-20250929"):
        super().__init__(contract_store)
        self.api_key = api_key
        self.model = model

    @property
    def provider_name(self) -> str:
        return "claude"

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


class OpenAIMutationEngine(LLMMutationEngine):
    """Calls the OpenAI-compatible API to generate mutations.

    Works with OpenAI, Azure OpenAI, and any OpenAI-compatible endpoint.
    """

    DEFAULT_BASE_URL = "https://api.openai.com/v1"
    DEFAULT_MODEL = "gpt-4o"

    def __init__(self, api_key: str, contract_store: ContractStore,
                 model: str | None = None, base_url: str | None = None):
        super().__init__(contract_store)
        self.api_key = api_key
        self.model = model or self.DEFAULT_MODEL
        self.base_url = base_url or self.DEFAULT_BASE_URL

    @property
    def provider_name(self) -> str:
        return "openai"

    def _call_api(self, prompt: str) -> str:
        import httpx

        resp = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
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
        return data["choices"][0]["message"]["content"]


class DeepSeekMutationEngine(OpenAIMutationEngine):
    """Calls the DeepSeek API to generate mutations.

    DeepSeek uses the OpenAI-compatible API format with a different base URL.
    """

    DEFAULT_BASE_URL = "https://api.deepseek.com"
    DEFAULT_MODEL = "deepseek-chat"

    @property
    def provider_name(self) -> str:
        return "deepseek"
