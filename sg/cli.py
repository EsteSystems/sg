"""CLI: init, run, status."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from sg import arena
from sg.contracts import ContractStore
from sg.fusion import FusionTracker
from sg.kernel.mock import MockNetworkKernel
from sg.mutation import MockMutationEngine, MutationEngine
from sg.orchestrator import Orchestrator
from sg.phenotype import PhenotypeMap
from sg.registry import Registry


def get_project_root() -> Path:
    return Path(os.environ.get("SG_PROJECT_ROOT", ".")).resolve()


def load_contract_store(root: Path) -> ContractStore:
    """Load contracts from the project's contracts directory."""
    return ContractStore.open(root / "contracts")


def make_mutation_engine(
    args: argparse.Namespace, project_root: Path, contract_store: ContractStore
) -> MutationEngine:
    engine = getattr(args, "mutation_engine", "auto")

    if engine == "mock":
        return MockMutationEngine(project_root / "fixtures")

    if engine == "claude":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("error: --mutation-engine=claude requires ANTHROPIC_API_KEY", file=sys.stderr)
            sys.exit(1)
        from sg.mutation import ClaudeMutationEngine
        return ClaudeMutationEngine(api_key, contract_store)

    # auto: try Claude, fall back to mock
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        from sg.mutation import ClaudeMutationEngine
        return ClaudeMutationEngine(api_key, contract_store)
    return MockMutationEngine(project_root / "fixtures")


def make_orchestrator(args: argparse.Namespace) -> Orchestrator:
    root = get_project_root()
    contract_store = load_contract_store(root)
    registry = Registry.open(root / ".sg" / "registry")
    phenotype = PhenotypeMap.load(root / "phenotype.toml")
    fusion_tracker = FusionTracker.open(root / "fusion_tracker.json")
    mutation_engine = make_mutation_engine(args, root, contract_store)
    kernel = MockNetworkKernel()

    return Orchestrator(
        registry=registry,
        phenotype=phenotype,
        mutation_engine=mutation_engine,
        fusion_tracker=fusion_tracker,
        kernel=kernel,
        contract_store=contract_store,
        project_root=root,
    )


def _discover_seed_genes(genes_dir: Path, known_loci: list[str]) -> dict[str, Path]:
    """Auto-discover seed genes by matching files to known loci.

    Looks for files named {locus}_*.py in the genes directory.
    """
    seeds: dict[str, Path] = {}
    if not genes_dir.exists():
        return seeds
    for locus in known_loci:
        candidates = sorted(genes_dir.glob(f"{locus}_*.py"))
        if candidates:
            seeds[locus] = candidates[0]
    return seeds


def cmd_init(args: argparse.Namespace) -> None:
    """Register seed genes, create phenotype.toml."""
    root = get_project_root()
    genes_dir = root / "genes"
    contract_store = load_contract_store(root)

    if not genes_dir.exists():
        print(f"error: genes directory not found at {genes_dir}", file=sys.stderr)
        sys.exit(1)

    registry = Registry.open(root / ".sg" / "registry")
    phenotype = PhenotypeMap()

    seeds = _discover_seed_genes(genes_dir, contract_store.known_loci())
    if not seeds:
        print("warning: no seed genes found matching known loci")

    for locus, gene_path in seeds.items():
        source = gene_path.read_text()
        sha = registry.register(source, locus)
        phenotype.promote(locus, sha)

        allele = registry.get(sha)
        if allele:
            allele.state = "dominant"

        print(f"  registered {locus} → {sha[:12]}")

    registry.save_index()
    phenotype.save(root / "phenotype.toml")
    print("Genome initialized.")


def cmd_run(args: argparse.Namespace) -> None:
    """Execute a pathway."""
    orch = make_orchestrator(args)

    try:
        outputs = orch.run_pathway(args.pathway, args.input)
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        orch.save_state()

    for i, output in enumerate(outputs):
        if len(outputs) > 1:
            print(f"--- Step {i + 1} ---")
        try:
            parsed = json.loads(output)
            print(json.dumps(parsed, indent=2))
        except json.JSONDecodeError:
            print(output)


def cmd_status(args: argparse.Namespace) -> None:
    """Show genome state."""
    root = get_project_root()
    contract_store = load_contract_store(root)
    registry = Registry.open(root / ".sg" / "registry")
    phenotype = PhenotypeMap.load(root / "phenotype.toml")
    fusion_tracker = FusionTracker.open(root / "fusion_tracker.json")

    print("=== Software Genome Status ===\n")

    for locus in contract_store.known_loci():
        alleles = registry.alleles_for_locus(locus)
        dominant_sha = phenotype.get_dominant(locus)
        print(f"Locus: {locus}")
        print(f"  Dominant: {dominant_sha[:12] if dominant_sha else 'none'}")
        print(f"  Alleles ({len(alleles)}):")
        for a in alleles:
            fitness = arena.compute_fitness(a)
            marker = " *" if a.sha256 == dominant_sha else ""
            print(f"    {a.sha256[:12]}  fitness={fitness:.3f}  "
                  f"invocations={a.total_invocations}  "
                  f"state={a.state}{marker}")
        print()

    for name in contract_store.known_pathways():
        fusion_config = phenotype.get_fused(name)
        track = fusion_tracker.get_track(name)
        print(f"Pathway: {name}")
        if fusion_config and fusion_config.fused_sha:
            print(f"  Fused: {fusion_config.fused_sha[:12]}")
        else:
            print(f"  Fused: no")
        if track:
            print(f"  Reinforcement: {track.reinforcement_count}/{10}")
            print(f"  Total: {track.total_successes} successes, {track.total_failures} failures")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(prog="sg", description="Software Genome runtime")
    parser.add_argument("--mutation-engine", default="auto",
                        choices=["auto", "mock", "claude"],
                        help="mutation engine to use")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init", help="initialize genome from seed genes")

    run_parser = subparsers.add_parser("run", help="execute a pathway")
    run_parser.add_argument("pathway", help="pathway name")
    run_parser.add_argument("--input", required=True, help="input JSON string")

    subparsers.add_parser("status", help="show genome status")

    args = parser.parse_args()

    if args.command == "init":
        cmd_init(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "status":
        cmd_status(args)
