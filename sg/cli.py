"""CLI: init, run, status, generate, watch, lineage, compete, deploy, dashboard, evolve, share, pull."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from sg import __version__, arena
from sg.contracts import ContractStore, validate_output
from sg.fusion import FusionTracker
from sg.kernel.mock import MockNetworkKernel
from sg.mutation import MockMutationEngine, MutationEngine
from sg.orchestrator import Orchestrator
from sg.phenotype import PhenotypeMap
from sg.registry import Registry

MUTATION_ENGINE_CHOICES = ["auto", "mock", "claude", "openai", "deepseek"]


def get_project_root() -> Path:
    return Path(os.environ.get("SG_PROJECT_ROOT", ".")).resolve()


def load_contract_store(root: Path) -> ContractStore:
    """Load contracts from the project's contracts directory."""
    return ContractStore.open(root / "contracts")


def make_mutation_engine(
    args: argparse.Namespace, project_root: Path, contract_store: ContractStore
) -> MutationEngine:
    engine = getattr(args, "mutation_engine", "auto")
    model = getattr(args, "model", None)

    if engine == "mock":
        return MockMutationEngine(project_root / "fixtures")

    if engine == "claude":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("error: --mutation-engine=claude requires ANTHROPIC_API_KEY", file=sys.stderr)
            sys.exit(1)
        from sg.mutation import ClaudeMutationEngine
        return ClaudeMutationEngine(api_key, contract_store, **({"model": model} if model else {}))

    if engine == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("error: --mutation-engine=openai requires OPENAI_API_KEY", file=sys.stderr)
            sys.exit(1)
        from sg.mutation import OpenAIMutationEngine
        return OpenAIMutationEngine(api_key, contract_store, **({"model": model} if model else {}))

    if engine == "deepseek":
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            print("error: --mutation-engine=deepseek requires DEEPSEEK_API_KEY", file=sys.stderr)
            sys.exit(1)
        from sg.mutation import DeepSeekMutationEngine
        return DeepSeekMutationEngine(api_key, contract_store, **({"model": model} if model else {}))

    # auto: try Claude, OpenAI, DeepSeek, fall back to mock
    for env_var, engine_cls_name in [
        ("ANTHROPIC_API_KEY", "ClaudeMutationEngine"),
        ("OPENAI_API_KEY", "OpenAIMutationEngine"),
        ("DEEPSEEK_API_KEY", "DeepSeekMutationEngine"),
    ]:
        api_key = os.environ.get(env_var)
        if api_key:
            import sg.mutation as mut
            engine_cls = getattr(mut, engine_cls_name)
            return engine_cls(api_key, contract_store, **({"model": model} if model else {}))

    return MockMutationEngine(project_root / "fixtures")


def make_kernel(args: argparse.Namespace):
    """Create the appropriate kernel based on --kernel flag."""
    kernel_type = getattr(args, "kernel", "mock")
    if kernel_type == "production":
        from sg.kernel.production import ProductionNetworkKernel
        return ProductionNetworkKernel()
    return MockNetworkKernel()


def make_orchestrator(args: argparse.Namespace) -> Orchestrator:
    root = get_project_root()
    contract_store = load_contract_store(root)
    registry = Registry.open(root / ".sg" / "registry")
    phenotype = PhenotypeMap.load(root / "phenotype.toml")
    fusion_tracker = FusionTracker.open(root / "fusion_tracker.json")
    mutation_engine = make_mutation_engine(args, root, contract_store)
    kernel = make_kernel(args)

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


def _inject_broken_genes(orch: Orchestrator) -> None:
    """Replace dominant alleles with broken versions to force mutation."""
    broken = (
        'import json\n'
        'def execute(input_json):\n'
        '    raise RuntimeError("force-mutate: intentional failure")\n'
    )
    for locus in list(orch.phenotype.loci.keys()):
        sha = orch.registry.register(broken, locus, generation=0)
        orch.phenotype.promote(locus, sha)
        allele = orch.registry.get(sha)
        if allele:
            allele.state = "dominant"
    print("Force-mutate: replaced all dominant alleles with broken versions")


def cmd_run(args: argparse.Namespace) -> None:
    """Execute a pathway."""
    orch = make_orchestrator(args)

    if getattr(args, "force_mutate", False):
        _inject_broken_genes(orch)

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


def cmd_deploy(args: argparse.Namespace) -> None:
    """Deploy a topology."""
    orch = make_orchestrator(args)

    try:
        outputs = orch.run_topology(args.topology, args.input)
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        orch.verify_scheduler.wait()
        orch.save_state()

    for i, output in enumerate(outputs):
        if len(outputs) > 1:
            print(f"--- Output {i + 1} ---")
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


def cmd_lineage(args: argparse.Namespace) -> None:
    """Show mutation ancestry for a locus."""
    root = get_project_root()
    registry = Registry.open(root / ".sg" / "registry")
    phenotype = PhenotypeMap.load(root / "phenotype.toml")

    locus = args.locus
    alleles = registry.alleles_for_locus(locus)
    if not alleles:
        print(f"No alleles registered for locus '{locus}'")
        return

    dominant_sha = phenotype.get_dominant(locus)

    # Build parent→children map
    children: dict[str | None, list] = {}
    for a in alleles:
        children.setdefault(a.parent_sha, []).append(a)

    # Find roots (alleles with no parent or whose parent is from another locus)
    roots = [a for a in alleles if a.parent_sha is None or a.parent_sha not in registry.alleles]

    print(f"=== Lineage: {locus} ({len(alleles)} allele(s)) ===\n")

    def _print_allele(a, indent=0):
        fitness = arena.compute_fitness(a)
        marker = " <-- dominant" if a.sha256 == dominant_sha else ""
        prefix = "  " * indent + ("├── " if indent > 0 else "")
        print(f"{prefix}{a.sha256[:12]}  gen={a.generation}  "
              f"fitness={fitness:.3f}  "
              f"{a.successful_invocations}ok/{a.failed_invocations}fail  "
              f"state={a.state}{marker}")
        for child in children.get(a.sha256, []):
            _print_allele(child, indent + 1)

    for root_allele in roots:
        _print_allele(root_allele)
    print()


def cmd_compete(args: argparse.Namespace) -> None:
    """Run allele competition trials for a locus."""
    root = get_project_root()
    contract_store = load_contract_store(root)
    registry = Registry.open(root / ".sg" / "registry")
    phenotype = PhenotypeMap.load(root / "phenotype.toml")
    kernel = MockNetworkKernel()

    locus = args.locus
    input_json = args.input
    rounds = getattr(args, "rounds", 10)

    alleles = registry.alleles_for_locus(locus)
    if not alleles:
        print(f"No alleles registered for locus '{locus}'")
        return

    dominant_sha = phenotype.get_dominant(locus)
    if dominant_sha is None:
        print(f"No dominant allele for locus '{locus}'")
        return

    # Filter to alleles that aren't deprecated
    candidates = [a for a in alleles if a.state != "deprecated"]
    if len(candidates) < 2:
        print(f"Only {len(candidates)} non-deprecated allele(s) — nothing to compete")
        return

    print(f"=== Competition: {locus} ({rounds} rounds, {len(candidates)} allele(s)) ===\n")

    # Run each allele through the rounds
    results: dict[str, dict] = {}
    for a in candidates:
        source = registry.load_source(a.sha256)
        if source is None:
            continue

        successes = 0
        failures = 0
        for _ in range(rounds):
            trial_kernel = MockNetworkKernel()
            try:
                from sg.loader import load_gene, call_gene
                execute_fn = load_gene(source, trial_kernel)
                result = call_gene(execute_fn, input_json)
                if validate_output(locus, result):
                    successes += 1
                    arena.record_success(a)
                else:
                    failures += 1
                    arena.record_failure(a)
            except Exception:
                failures += 1
                arena.record_failure(a)

        trial_fitness = successes / max(rounds, 1)
        results[a.sha256] = {
            "allele": a,
            "successes": successes,
            "failures": failures,
            "trial_fitness": trial_fitness,
        }

        marker = " *" if a.sha256 == dominant_sha else ""
        print(f"  {a.sha256[:12]}  {successes}/{rounds} passed  "
              f"trial_fitness={trial_fitness:.3f}  "
              f"overall_fitness={arena.compute_fitness(a):.3f}  "
              f"state={a.state}{marker}")

    # Check for promotions
    dominant = registry.get(dominant_sha)
    promoted = False
    if dominant:
        dominant_result = results.get(dominant_sha)
        for sha, r in results.items():
            if sha == dominant_sha:
                continue
            if arena.should_promote(r["allele"], dominant):
                print(f"\n  Promoting {sha[:12]} over {dominant_sha[:12]}!")
                arena.set_dominant(r["allele"])
                arena.set_recessive(dominant)
                phenotype.promote(locus, sha)
                promoted = True
                break

    if not promoted:
        print(f"\n  No promotion triggered — dominant {dominant_sha[:12]} holds")

    registry.save_index()
    phenotype.save(root / "phenotype.toml")
    print()


def cmd_completions(args: argparse.Namespace) -> None:
    """Print shell completion script."""
    subcommands = ("init run deploy generate watch status lineage compete "
                   "dashboard evolve share pull completions")
    engines = "auto mock claude openai deepseek"
    kernels = "mock production"

    if args.shell == "bash":
        print(f'''_sg_completions() {{
    local cur=${{COMP_WORDS[COMP_CWORD]}}
    local prev=${{COMP_WORDS[COMP_CWORD-1]}}
    if [ $COMP_CWORD -eq 1 ]; then
        COMPREPLY=($(compgen -W "{subcommands}" -- "$cur"))
    elif [ "$prev" = "--mutation-engine" ]; then
        COMPREPLY=($(compgen -W "{engines}" -- "$cur"))
    elif [ "$prev" = "--kernel" ]; then
        COMPREPLY=($(compgen -W "{kernels}" -- "$cur"))
    fi
}}
complete -F _sg_completions sg''')
    elif args.shell == "zsh":
        print(f'''#compdef sg
_sg() {{
    _arguments '1:command:({subcommands})' \\
               '--mutation-engine[engine]:engine:({engines})' \\
               '--kernel[kernel]:kernel:({kernels})'
}}
_sg "$@"''')
    elif args.shell == "fish":
        for cmd in subcommands.split():
            print(f"complete -c sg -n '__fish_use_subcommand' -a '{cmd}'")


def cmd_dashboard(args: argparse.Namespace) -> None:
    """Start the web dashboard."""
    try:
        from sg.dashboard import run_dashboard
    except ImportError:
        print("error: install dashboard extras: pip install 'sg[dashboard]'",
              file=sys.stderr)
        sys.exit(1)
    root = get_project_root()
    run_dashboard(root, host=args.host, port=args.port)


def cmd_evolve(args: argparse.Namespace) -> None:
    """Generate a new contract via LLM."""
    root = get_project_root()
    contract_store = load_contract_store(root)
    mutation_engine = make_mutation_engine(args, root, contract_store)

    existing = contract_store.known_loci()
    try:
        source = mutation_engine.generate_contract(
            args.family, args.context, existing
        )
    except (NotImplementedError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

    # Validate by parsing
    from sg.parser.parser import parse_sg
    try:
        contract = parse_sg(source)
    except Exception as e:
        print(f"error: generated contract failed to parse: {e}", file=sys.stderr)
        print("--- Raw output ---")
        print(source)
        sys.exit(1)

    out_path = root / "contracts" / "genes" / f"{contract.name}.sg"
    contract_store.register_contract(source, out_path)
    print(f"Generated contract: {contract.name}")
    print(f"  Written to: {out_path}")
    print(f"  Family: {contract.family.value}")
    print(f"  Risk: {contract.risk.value}")


def cmd_share(args: argparse.Namespace) -> None:
    """Push successful alleles to peers."""
    from sg.federation import load_peers, export_allele, push_allele

    root = get_project_root()
    registry = Registry.open(root / ".sg" / "registry")
    phenotype = PhenotypeMap.load(root / "phenotype.toml")

    locus = args.locus
    dominant_sha = phenotype.get_dominant(locus)
    if dominant_sha is None:
        print(f"No dominant allele for locus '{locus}'")
        return

    allele_data = export_allele(registry, dominant_sha)
    if allele_data is None:
        print(f"Could not export allele {dominant_sha[:12]}")
        return

    if args.peer:
        from sg.federation import PeerConfig
        peers = [PeerConfig(url=args.peer)]
    else:
        peers = load_peers(root / "peers.json")

    if not peers:
        print("No peers configured. Create peers.json or use --peer URL")
        return

    for peer in peers:
        ok = push_allele(peer, allele_data)
        status = "ok" if ok else "failed"
        print(f"  {peer.url}: {status}")


def cmd_pull(args: argparse.Namespace) -> None:
    """Fetch alleles from peers."""
    from sg.federation import load_peers, import_allele, pull_alleles

    root = get_project_root()
    registry = Registry.open(root / ".sg" / "registry")
    phenotype = PhenotypeMap.load(root / "phenotype.toml")

    locus = args.locus

    if args.peer:
        from sg.federation import PeerConfig
        peers = [PeerConfig(url=args.peer)]
    else:
        peers = load_peers(root / "peers.json")

    if not peers:
        print("No peers configured. Create peers.json or use --peer URL")
        return

    imported = 0
    for peer in peers:
        alleles = pull_alleles(peer, locus)
        for data in alleles:
            sha = import_allele(registry, data)
            phenotype.add_to_fallback(locus, sha)
            allele = registry.get(sha)
            if allele:
                allele.state = "recessive"
            print(f"  imported {sha[:12]} from {peer.url}")
            imported += 1

    if imported:
        registry.save_index()
        phenotype.save(root / "phenotype.toml")
    print(f"Imported {imported} allele(s) for {locus}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="sg", description="Software Genome runtime")
    parser.add_argument("--version", action="version", version=f"sg {__version__}")
    parser.add_argument("--mutation-engine", default="auto",
                        choices=MUTATION_ENGINE_CHOICES,
                        help="mutation engine to use (auto|mock|claude|openai|deepseek)")
    parser.add_argument("--model", default=None,
                        help="override default model for the mutation engine")
    parser.add_argument("--kernel", default="mock", choices=["mock", "production"],
                        help="kernel to use (mock|production)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init", help="initialize genome from seed genes")

    run_parser = subparsers.add_parser("run", help="execute a pathway")
    run_parser.add_argument("pathway", help="pathway name")
    run_parser.add_argument("--input", required=True, help="input JSON string")
    run_parser.add_argument("--force-mutate", action="store_true",
                            help="replace all dominant alleles with broken versions to force mutation")

    gen_parser = subparsers.add_parser("generate", help="proactively generate competing alleles")
    gen_parser.add_argument("locus", nargs="?", help="locus to generate for")
    gen_parser.add_argument("--all", action="store_true", help="generate for all loci with registered alleles")
    gen_parser.add_argument("--count", type=int, default=1, help="number of variants to generate (default: 1)")

    watch_parser = subparsers.add_parser("watch", help="periodically run diagnostics for resilience fitness")
    watch_parser.add_argument("pathway", help="diagnostic pathway to run")
    watch_parser.add_argument("--input", required=True, help="input JSON string")
    watch_parser.add_argument("--interval", type=float, default=300.0, help="seconds between runs (default: 300)")
    watch_parser.add_argument("--count", type=int, default=0, help="number of iterations (0 = infinite, default: 0)")

    deploy_parser = subparsers.add_parser("deploy", help="deploy a topology")
    deploy_parser.add_argument("topology", help="topology name")
    deploy_parser.add_argument("--input", required=True, help="input JSON string")

    subparsers.add_parser("status", help="show genome status")

    lineage_parser = subparsers.add_parser("lineage", help="show mutation ancestry for a locus")
    lineage_parser.add_argument("locus", help="locus to show lineage for")

    compete_parser = subparsers.add_parser("compete", help="run allele competition trials")
    compete_parser.add_argument("locus", help="locus to compete")
    compete_parser.add_argument("--input", required=True, help="test input JSON string")
    compete_parser.add_argument("--rounds", type=int, default=10, help="number of trial rounds (default: 10)")

    dash_parser = subparsers.add_parser("dashboard", help="start web dashboard")
    dash_parser.add_argument("--port", type=int, default=8420, help="port (default: 8420)")
    dash_parser.add_argument("--host", default="127.0.0.1", help="host (default: 127.0.0.1)")

    evolve_parser = subparsers.add_parser("evolve", help="generate a new contract via LLM")
    evolve_parser.add_argument("--family", default="diagnostic",
                               choices=["configuration", "diagnostic"],
                               help="gene family (default: diagnostic)")
    evolve_parser.add_argument("--context", required=True,
                               help="description of what the new gene should do")

    share_parser = subparsers.add_parser("share", help="push successful alleles to peers")
    share_parser.add_argument("locus", help="locus to share")
    share_parser.add_argument("--peer", help="specific peer URL (default: all peers)")

    pull_parser = subparsers.add_parser("pull", help="fetch alleles from peers")
    pull_parser.add_argument("locus", help="locus to pull")
    pull_parser.add_argument("--peer", help="specific peer URL (default: all peers)")

    comp_parser = subparsers.add_parser("completions", help="generate shell completions")
    comp_parser.add_argument("shell", choices=["bash", "zsh", "fish"],
                             help="shell type")

    args = parser.parse_args()

    if args.command == "init":
        cmd_init(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "deploy":
        cmd_deploy(args)
    elif args.command == "generate":
        if not getattr(args, "all", False) and not args.locus:
            gen_parser.error("either provide a locus name or use --all")
        cmd_generate(args)
    elif args.command == "watch":
        cmd_watch(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "lineage":
        cmd_lineage(args)
    elif args.command == "compete":
        cmd_compete(args)
    elif args.command == "dashboard":
        cmd_dashboard(args)
    elif args.command == "evolve":
        cmd_evolve(args)
    elif args.command == "share":
        cmd_share(args)
    elif args.command == "pull":
        cmd_pull(args)
    elif args.command == "completions":
        cmd_completions(args)
