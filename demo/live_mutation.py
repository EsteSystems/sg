#!/usr/bin/env python3
"""Live mutation demo: inject failure, watch an LLM fix it.

Demonstrates the full evolutionary loop:
  broken gene → all alleles exhausted → LLM mutation → fix → success

Requires one of:
  ANTHROPIC_API_KEY  (uses Claude)
  OPENAI_API_KEY     (uses ChatGPT)
  DEEPSEEK_API_KEY   (uses DeepSeek)

Usage:
  python demo/live_mutation.py
  python demo/live_mutation.py --model gpt-4-turbo
  python demo/live_mutation.py --locus bridge_stp
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sg import arena
from sg.contracts import ContractStore
from sg.fusion import FusionTracker
from sg_network import MockNetworkKernel
from sg.mutation import (
    LLMMutationEngine, ClaudeMutationEngine,
    OpenAIMutationEngine, DeepSeekMutationEngine,
)
from sg.orchestrator import Orchestrator
from sg.phenotype import PhenotypeMap
from sg.registry import Registry


# --- Deliberately broken genes ---

BROKEN_GENES = {
    "bridge_create": '''\
"""Broken bridge_create gene — intentional failure for demo."""
import json

def execute(input_json):
    data = json.loads(input_json)
    # BUG: tries to access non-existent field, then calls wrong API
    bridge_name = data["name"]  # wrong key — should be "bridge_name"
    gene_sdk.create_bridge(bridge_name, [])
    return json.dumps({"success": True})
''',
    "bridge_stp": '''\
"""Broken bridge_stp gene — intentional failure for demo."""
import json

def execute(input_json):
    data = json.loads(input_json)
    # BUG: passes string instead of bool/int to set_stp
    gene_sdk.set_stp(data["bridge_name"], "yes", "fast")
    return json.dumps({"success": True})
''',
    "bond_create": '''\
"""Broken bond_create gene — intentional failure for demo."""
import json

def execute(input_json):
    data = json.loads(input_json)
    # BUG: wrong field names
    gene_sdk.create_bond(data["name"], data["type"], data["ports"])
    return json.dumps({"success": True})
''',
}

# Default inputs for each locus
DEFAULT_INPUTS = {
    "bridge_create": {
        "bridge_name": "br0",
        "interfaces": ["eth0", "eth1"],
    },
    "bridge_stp": {
        "bridge_name": "br0",
        "stp_enabled": True,
        "forward_delay": 15,
    },
    "bond_create": {
        "bond_name": "bond0",
        "mode": "active-backup",
        "members": ["eth2", "eth3"],
    },
}


def detect_engine(
    contract_store: ContractStore, model: str | None = None,
) -> LLMMutationEngine:
    """Auto-detect available LLM engine from environment."""
    for env_var, cls, name in [
        ("ANTHROPIC_API_KEY", ClaudeMutationEngine, "Claude"),
        ("OPENAI_API_KEY", OpenAIMutationEngine, "ChatGPT"),
        ("DEEPSEEK_API_KEY", DeepSeekMutationEngine, "DeepSeek"),
    ]:
        api_key = os.environ.get(env_var)
        if api_key:
            kwargs = {"model": model} if model else {}
            engine = cls(api_key, contract_store, **kwargs)
            print(f"Using {name} ({engine.model})")
            return engine

    print("error: no API key found. Set one of:", file=sys.stderr)
    print("  ANTHROPIC_API_KEY  (Claude)", file=sys.stderr)
    print("  OPENAI_API_KEY     (ChatGPT)", file=sys.stderr)
    print("  DEEPSEEK_API_KEY   (DeepSeek)", file=sys.stderr)
    sys.exit(1)


def setup_project(tmp_dir: Path, locus: str) -> Path:
    """Set up a minimal project in tmp_dir with a broken gene."""
    # Copy contracts
    import sg_network
    shutil.copytree(sg_network.contracts_path(), tmp_dir / "contracts")

    # Create empty fixtures dir (no mock fallbacks — forces real mutation)
    (tmp_dir / "fixtures").mkdir()

    # Initialize registry and phenotype with the broken gene
    contract_store = ContractStore.open(tmp_dir / "contracts")
    registry = Registry.open(tmp_dir / ".sg" / "registry")
    phenotype = PhenotypeMap()

    if locus not in BROKEN_GENES:
        print(f"error: no broken gene defined for locus '{locus}'", file=sys.stderr)
        print(f"available: {', '.join(BROKEN_GENES.keys())}", file=sys.stderr)
        sys.exit(1)

    # Register the broken gene as dominant
    broken_source = BROKEN_GENES[locus]
    sha = registry.register(broken_source, locus)
    phenotype.promote(locus, sha)
    allele = registry.get(sha)
    allele.state = "dominant"

    # For bridge_stp, we also need a working bridge_create to set up the bridge
    if locus == "bridge_stp":
        good_source = (sg_network.genes_path() / "bridge_create_v1.py").read_text()
        sha2 = registry.register(good_source, "bridge_create")
        phenotype.promote("bridge_create", sha2)
        allele2 = registry.get(sha2)
        allele2.state = "dominant"

    registry.save_index()
    phenotype.save(tmp_dir / "phenotype.toml")
    return tmp_dir


def run_demo(locus: str, model: str | None = None) -> None:
    """Run the live mutation demo."""
    print("=" * 60)
    print("  Software Genome — Live Mutation Demo")
    print("=" * 60)
    print()

    with tempfile.TemporaryDirectory(prefix="sg-demo-") as tmp:
        tmp_dir = Path(tmp)
        setup_project(tmp_dir, locus)

        contract_store = ContractStore.open(tmp_dir / "contracts")
        registry = Registry.open(tmp_dir / ".sg" / "registry")
        phenotype = PhenotypeMap.load(tmp_dir / "phenotype.toml")
        fusion_tracker = FusionTracker.open(tmp_dir / "fusion_tracker.json")
        kernel = MockNetworkKernel()
        mutation_engine = detect_engine(contract_store, model)

        orch = Orchestrator(
            registry=registry,
            phenotype=phenotype,
            mutation_engine=mutation_engine,
            fusion_tracker=fusion_tracker,
            kernel=kernel,
            contract_store=contract_store,
            project_root=tmp_dir,
        )

        # For bridge_stp, create the bridge first
        if locus == "bridge_stp":
            print("--- Pre-requisite: creating bridge ---")
            orch.execute_locus("bridge_create", json.dumps(
                DEFAULT_INPUTS["bridge_create"]
            ))
            print()

        input_json = json.dumps(DEFAULT_INPUTS[locus])
        print(f"--- Executing locus: {locus} ---")
        print(f"Input: {input_json}")
        print()

        # Show the broken gene
        broken_sha = phenotype.get_dominant(locus)
        broken_source = registry.load_source(broken_sha)
        print("--- Broken gene (dominant) ---")
        print(broken_source)

        # Execute — broken gene fails, triggers LLM mutation
        print("--- Execution begins ---")
        result = orch.execute_locus(locus, input_json)
        print()

        if result is not None:
            output_json, used_sha = result
            print("--- Result ---")
            print(f"Output: {json.dumps(json.loads(output_json), indent=2)}")
            print()

            # Show the mutant
            if used_sha != broken_sha:
                mutant_source = registry.load_source(used_sha)
                mutant = registry.get(used_sha)
                parent = registry.get(broken_sha)

                print("--- LLM-generated mutant ---")
                print(mutant_source)

                print("--- Evolutionary lineage ---")
                print(f"  Original (broken):")
                print(f"    SHA:        {broken_sha[:16]}...")
                print(f"    Generation: {parent.generation}")
                print(f"    State:      {parent.state}")
                print(f"    Fitness:    {arena.compute_fitness(parent):.3f}")
                print()
                print(f"  Mutant (fixed by {mutation_engine.provider_name}):")
                print(f"    SHA:        {used_sha[:16]}...")
                print(f"    Generation: {mutant.generation}")
                print(f"    State:      {mutant.state}")
                print(f"    Parent:     {mutant.parent_sha[:16]}...")
                print(f"    Fitness:    {arena.compute_fitness(mutant):.3f}")
            else:
                print("(original gene succeeded — no mutation needed)")
        else:
            print("--- All mutation attempts failed ---")
            print("The LLM could not generate a working fix.")

    print()
    print("=" * 60)
    print("  Demo complete.")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Live mutation demo: watch an LLM fix a broken gene"
    )
    parser.add_argument(
        "--locus", default="bridge_create",
        choices=list(BROKEN_GENES.keys()),
        help="which locus to break (default: bridge_create)",
    )
    parser.add_argument(
        "--model", default=None,
        help="override default model (e.g. gpt-4-turbo, deepseek-coder)",
    )
    args = parser.parse_args()
    run_demo(args.locus, args.model)


if __name__ == "__main__":
    main()
