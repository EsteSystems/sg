# Software Genomics — Project Context for Claude Code

## What This Project Is

Python implementation of the Software Genomics paradigm — an evolutionary software runtime. Genes are pure Python functions, loaded via `exec()`, scored by temporal fitness, and replaced by LLM-generated mutations when they fail. Contracts are defined in the `.sg` format and authored by domain experts.

See `docs/SOFTWARE-GENOMICS.md` for the full paradigm whitepaper.

## Project Structure

```
sg/                               # core runtime
├── cli.py                        # CLI: init, run, status
├── orchestrator.py               # evolutionary loop
├── registry.py                   # SHA-256 CAS + JSON index
├── phenotype.py                  # TOML phenotype map + fusion state
├── arena.py                      # temporal fitness: immediate + convergence + resilience
├── loader.py                     # exec()-based gene loading
├── mutation.py                   # mock + Claude mutation engines
├── pathway.py                    # pathway execution (fusion-aware, composed, conditional)
├── fusion.py                     # reinforcement tracking, fuse/decompose
├── contracts.py                  # contract loading, locus types, validation
├── safety.py                     # transactions, blast radius, rollback
├── parser/                       # .sg format parser
│   ├── types.py                  # AST node types (GeneContract, PathwayContract, etc.)
│   ├── lexer.py                  # tokenization
│   └── parser.py                 # AST construction
└── kernel/                       # gene_sdk — kernel interface
    ├── base.py                   # NetworkKernel ABC
    ├── mock.py                   # MockNetworkKernel (dev/test)
    └── production.py             # ProductionNetworkKernel (NM D-Bus, ip link)

contracts/                        # .sg contract definitions
├── genes/                        # gene contracts (bridge_create.sg, etc.)
├── pathways/                     # pathway contracts
└── topologies/                   # topology contracts
```

## Conventions

- Genes are `execute(input_json: str) -> str` functions
- All gene I/O is JSON strings. `gene_sdk` is injected into the gene's namespace at load time.
- Content addressing: alleles identified by SHA-256 of source code
- Temporal fitness: immediate (30%) + convergence (50%) + resilience (20%)
- Promotion: fitness > dominant + 0.1, invocations >= 50
- Demotion: 3 consecutive failures
- Fusion: 10 consecutive pathway successes with identical allele composition
- Two gene families: configuration (effectors) and diagnostic (sensors)
- Contracts are `.sg` files with verb-based sections: does, takes, gives, before, after, fails when, verify, feeds

## Running

```bash
pip install -e ".[dev]"
pytest
sg init
sg run <pathway> --input '{...}'
sg status
```

## Key Design Decisions

- **`.sg` contract format**: purpose-built for this paradigm. Verb-based sections (`does`, `takes`, `gives`). Domain experts author contracts, not developers.
- **Two gene families**: configuration genes act, diagnostic genes observe. Diagnostics produce the fitness signal for configuration genes.
- **Temporal fitness**: immediate (t=0) + convergence (t=30s) + resilience (t=hours). Retroactive decay when convergence/resilience fail.
- **Composition hierarchy**: gene → pathway → topology → intent. Pathways compose via `->`, iterate via `for`, bind conditionally via `when`.
- **Safety**: transactions with undo-log, blast radius classification (none → critical), shadow mode → canary → recessive → dominant allele lifecycle.
