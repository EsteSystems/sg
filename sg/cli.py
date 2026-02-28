"""CLI: init, run, status, generate, watch."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
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
        orch.verify_scheduler.wait()
        orch.save_state()

    for i, output in enumerate(outputs):
        if len(outputs) > 1:
            print(f"--- Step {i + 1} ---")
        try:
            parsed = json.loads(output)
            print(json.dumps(parsed, indent=2))
        except json.JSONDecodeError:
            print(output)


def cmd_generate(args: argparse.Namespace) -> None:
    """Proactively generate competing alleles from contracts."""
    root = get_project_root()
    contract_store = load_contract_store(root)
    registry = Registry.open(root / ".sg" / "registry")
    phenotype = PhenotypeMap.load(root / "phenotype.toml")
    mutation_engine = make_mutation_engine(args, root, contract_store)
    count = getattr(args, "count", 1)

    if getattr(args, "all", False):
        # Generate for every locus that has a dominant allele
        targets = [
            locus for locus in contract_store.known_loci()
            if phenotype.get_dominant(locus) is not None
        ]
    else:
        targets = [args.locus]

    if not targets:
        print("no loci with registered alleles found")
        return

    total_generated = 0
    for locus in targets:
        gene_contract = contract_store.get_gene(locus)
        if gene_contract is None:
            print(f"  skipping {locus}: no gene contract")
            continue

        # Build contract prompt for the engine
        try:
            info = contract_store.contract_info(locus)
        except ValueError:
            print(f"  skipping {locus}: no contract info")
            continue

        # Use ClaudeMutationEngine._contract_prompt if available, else build minimal
        if hasattr(mutation_engine, '_contract_prompt'):
            contract_prompt = mutation_engine._contract_prompt(locus)
        else:
            contract_prompt = f"Locus: {locus}\nDescription: {info.description}"

        dominant_sha = phenotype.get_dominant(locus)
        parent_gen = 0
        if dominant_sha:
            parent = registry.get(dominant_sha)
            if parent:
                parent_gen = parent.generation

        print(f"Generating {count} variant(s) for {locus}...")
        try:
            sources = mutation_engine.generate(locus, contract_prompt, count)
        except Exception as e:
            print(f"  error: {e}")
            continue

        for source in sources:
            sha = registry.register(
                source, locus,
                generation=parent_gen + 1,
                parent_sha=dominant_sha,
            )
            allele = registry.get(sha)
            if allele:
                allele.state = "recessive"
            phenotype.add_to_fallback(locus, sha)
            print(f"  registered {sha[:12]} (recessive)")
            total_generated += 1

    registry.save_index()
    phenotype.save(root / "phenotype.toml")
    print(f"Generated {total_generated} variant(s) across {len(targets)} locus/loci.")


def cmd_watch(args: argparse.Namespace) -> None:
    """Periodically run a diagnostic pathway for resilience fitness."""
    orch = make_orchestrator(args)
    orch.feedback_timescale = "resilience"

    interval = getattr(args, "interval", 300.0)
    max_count = getattr(args, "count", 0)
    iteration = 0

    print(f"Watching '{args.pathway}' every {interval}s "
          f"({'infinite' if max_count == 0 else max_count} iterations)")
    print(f"Feedback timescale: resilience")
    print()

    try:
        while max_count == 0 or iteration < max_count:
            iteration += 1
            print(f"=== Watch iteration {iteration} ===")
            try:
                outputs = orch.run_pathway(args.pathway, args.input)
            except Exception as e:
                print(f"  error: {e}")

            orch.verify_scheduler.wait()
            orch.save_state()

            if max_count == 0 or iteration < max_count:
                time.sleep(interval)

            # Reload state from disk to pick up changes from other processes
            orch.registry.load_index()
            orch.phenotype = PhenotypeMap.load(orch.project_root / "phenotype.toml")

    except KeyboardInterrupt:
        print("\nWatch interrupted.")
        orch.save_state()


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

    gen_parser = subparsers.add_parser("generate", help="proactively generate competing alleles")
    gen_parser.add_argument("locus", nargs="?", help="locus to generate for")
    gen_parser.add_argument("--all", action="store_true", help="generate for all loci with registered alleles")
    gen_parser.add_argument("--count", type=int, default=1, help="number of variants to generate (default: 1)")

    watch_parser = subparsers.add_parser("watch", help="periodically run diagnostics for resilience fitness")
    watch_parser.add_argument("pathway", help="diagnostic pathway to run")
    watch_parser.add_argument("--input", required=True, help="input JSON string")
    watch_parser.add_argument("--interval", type=float, default=300.0, help="seconds between runs (default: 300)")
    watch_parser.add_argument("--count", type=int, default=0, help="number of iterations (0 = infinite, default: 0)")

    subparsers.add_parser("status", help="show genome status")

    args = parser.parse_args()

    if args.command == "init":
        cmd_init(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "generate":
        if not getattr(args, "all", False) and not args.locus:
            gen_parser.error("either provide a locus name or use --all")
        cmd_generate(args)
    elif args.command == "watch":
        cmd_watch(args)
    elif args.command == "status":
        cmd_status(args)
