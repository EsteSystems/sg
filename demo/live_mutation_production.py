#!/usr/bin/env python3
"""Live mutation demo targeting the PRODUCTION kernel.

Demonstrates the full evolutionary loop on a real Linux host:
  broken gene → all alleles exhausted → LLM mutation → real kernel fix → success

All resources use sg-test- prefix for safety.
The production kernel enforces this prefix and protects eth0/lo.

Requires:
  - Linux host with ip/bridge commands (e.g. 192.168.139.206)
  - sudo access for network commands
  - One of: ANTHROPIC_API_KEY, OPENAI_API_KEY, DEEPSEEK_API_KEY

Usage:
  python demo/live_mutation_production.py
  python demo/live_mutation_production.py --dry-run
  python demo/live_mutation_production.py --locus bridge_create --model gpt-4-turbo
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sg import arena
from sg.contracts import ContractStore
from sg.fusion import FusionTracker
from sg.kernel.production import ProductionNetworkKernel
from sg.mutation import (
    LLMMutationEngine, ClaudeMutationEngine,
    OpenAIMutationEngine, DeepSeekMutationEngine,
)
from sg.orchestrator import Orchestrator
from sg.phenotype import PhenotypeMap
from sg.registry import Registry


# Broken genes using sg-test- prefixed names
BROKEN_GENES = {
    "bridge_create": '''\
"""Broken bridge_create gene for production demo."""
import json

def execute(input_json):
    data = json.loads(input_json)
    # BUG: wrong key name
    bridge_name = data["name"]  # should be "bridge_name"
    gene_sdk.create_bridge(bridge_name, [])
    return json.dumps({"success": True})
''',
}

DEFAULT_INPUTS = {
    "bridge_create": {
        "bridge_name": "sg-test-br0",
        "interfaces": [],
    },
}


def detect_engine(contract_store, model=None):
    """Auto-detect available LLM engine."""
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

    print("error: no API key found.", file=sys.stderr)
    sys.exit(1)


def run_demo(locus: str, model: str | None = None, dry_run: bool = False):
    """Run the production kernel live mutation demo."""
    print("=" * 60)
    print("  Software Genome — Production Kernel Mutation Demo")
    print("=" * 60)
    print()

    if dry_run:
        print("[DRY RUN] Commands will be logged but not executed\n")

    kernel = ProductionNetworkKernel(use_sudo=True, dry_run=dry_run)

    with tempfile.TemporaryDirectory(prefix="sg-prod-demo-") as tmp:
        tmp_dir = Path(tmp)
        shutil.copytree(PROJECT_ROOT / "contracts", tmp_dir / "contracts")
        (tmp_dir / "fixtures").mkdir()

        contract_store = ContractStore.open(tmp_dir / "contracts")
        registry = Registry.open(tmp_dir / ".sg" / "registry")
        phenotype = PhenotypeMap()

        broken_source = BROKEN_GENES[locus]
        sha = registry.register(broken_source, locus)
        phenotype.promote(locus, sha)
        allele = registry.get(sha)
        allele.state = "dominant"

        registry.save_index()
        phenotype.save(tmp_dir / "phenotype.toml")

        fusion_tracker = FusionTracker.open(tmp_dir / "fusion_tracker.json")
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

        input_json = json.dumps(DEFAULT_INPUTS[locus])
        print(f"--- Executing locus: {locus} ---")
        print(f"Input: {input_json}")
        print()

        try:
            result = orch.execute_locus(locus, input_json)

            if result is not None:
                output_json, used_sha = result
                print(f"\n--- Result ---")
                print(f"Output: {json.dumps(json.loads(output_json), indent=2)}")

                broken_sha = sha
                if used_sha != broken_sha:
                    mutant_source = registry.load_source(used_sha)
                    mutant = registry.get(used_sha)
                    print(f"\n--- LLM-generated mutant ---")
                    print(mutant_source)
                    print(f"\n--- Lineage ---")
                    print(f"  Broken:  {broken_sha[:16]}...")
                    print(f"  Mutant:  {used_sha[:16]}... (gen {mutant.generation})")
                    print(f"  Fitness: {arena.compute_fitness(mutant):.3f}")
            else:
                print("\n--- All mutation attempts failed ---")
        finally:
            # Always clean up test resources
            print("\n--- Cleanup ---")
            kernel.reset()

    print()
    print("=" * 60)
    print("  Demo complete.")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Production kernel live mutation demo"
    )
    parser.add_argument("--locus", default="bridge_create",
                        choices=list(BROKEN_GENES.keys()))
    parser.add_argument("--model", default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="log commands without executing them")
    args = parser.parse_args()
    run_demo(args.locus, args.model, args.dry_run)


if __name__ == "__main__":
    main()
