# Software Genomics — Project Context for Claude Code

## What This Project Is

Python implementation of the Software Genomics paradigm — an evolutionary software runtime. Genes are pure Python functions, loaded via `exec()`, scored by temporal fitness, and replaced by LLM-generated mutations when they fail. Contracts are defined in the `.sg` format and authored by domain experts.

See `docs/SOFTWARE-GENOMICS.md` for the full paradigm whitepaper.

## Project Structure

```
sg/                               # core runtime
├── cli.py                        # CLI: init, run, status, lineage, compete
├── orchestrator.py               # evolutionary loop + allele stack traversal
├── contracts.py                  # contract loading, locus types, validation
├── parser/                       # .sg format parser
│   ├── types.py                  # AST node types (GeneContract, PathwayContract, etc.)
│   ├── lexer.py                  # tokenization
│   └── parser.py                 # AST construction
├── kernel/                       # domain-agnostic kernel interface
│   ├── base.py                   # Kernel ABC (abstract, domain-agnostic)
│   ├── discovery.py              # plugin kernel discovery
│   └── stub.py                   # stub kernel for testing
│
│  ── Gene lifecycle ──
├── registry.py                   # SHA-256 CAS + JSON allele index
├── phenotype.py                  # TOML phenotype map (gene/pathway/topology alleles + fusion)
├── loader.py                     # exec()-based gene loading
├── mutation.py                   # mock + LLM mutation engines
├── mutation_cache.py             # LLM mutation result caching
├── arena.py                      # gene fitness: immediate + convergence + resilience
├── fitness.py                    # per-allele fitness history (bounded)
│
│  ── Pathway evolution ──
├── pathway.py                    # pathway execution (fusion-aware, composed, conditional)
├── pathway_registry.py           # pathway allele versioning + structure SHAs
├── pathway_arena.py              # pathway fitness scoring
├── pathway_fitness.py            # pathway execution timing + failure patterns
├── pathway_mutation.py           # structural mutation operators (reorder, insert, delete)
├── fusion.py                     # reinforcement tracking → fuse pathway to single gene
├── stabilization.py              # post-mutation stabilization tracking
│
│  ── Topology evolution ──
├── topology.py                   # topology composition (resources → pathways/genes)
├── topology_registry.py          # topology allele versioning + structure SHAs
├── topology_arena.py             # topology fitness (most conservative thresholds)
│
│  ── Error detection & learning ──
├── decomposition.py              # auto gene splitting on diverse error clusters
├── regression.py                 # fitness regression detection → proactive mutation
├── failure_discovery.py          # novel error pattern discovery + mutation proposals
├── probe.py                      # diagnostic probing of loci
├── interactions.py               # cross-locus interaction testing before promotion
│
│  ── Safety & verification ──
├── safety.py                     # transactions, blast radius, rollback
├── conformance.py                # output validation against contract specs
├── sandbox.py                    # isolated Python execution context
├── verify.py                     # timer-based verify scheduling (convergence/resilience)
│
├── topology_mutation.py           # topology mutation operators (reorder, action swap, eliminate, LLM)
│
│  ── Contract & meta evolution ──
├── contract_evolution.py         # contract tightening, relaxation, feeds discovery
├── locus_discovery.py            # cross-locus failure analysis → new locus proposals
├── adaptation.py                 # adaptive param tuning + adaptive safety policies
├── speciation.py                 # phenotype divergence tracking across organisms
│
│  ── Continuous operation ──
├── daemon.py                     # tick-based daemon loop (sg daemon)
├── events.py                     # in-process event bus (pub/sub)
├── metrics.py                    # Prometheus-style metrics collector + export
│
│  ── Persistence & ops ──
├── filelock.py                   # file locking (shared/exclusive) + atomic writes
├── audit.py                      # append-only JSONL event log
├── meta_params.py                # evolutionary parameter tracking + snapshots
├── log.py                        # structured logging
├── scaffold.py                   # project initialization
├── dashboard.py                  # status dashboard
├── snapshot.py                   # project state snapshots
├── diff.py                       # state diffing
│
│  ── Multi-organism ──
├── federation.py                 # peer observation sharing
├── pool.py                       # gene pool configuration
└── pool_server.py                # gene pool HTTP server

plugins/                          # domain-specific plugins
├── data/                         # data pipeline domain (active)
│   ├── sg_data/                  # installable package
│   │   ├── kernel.py             # DataKernel ABC
│   │   └── mock.py               # MockDataKernel (dev/test)
│   ├── contracts/                # .sg contracts (genes/, pathways/)
│   ├── genes/                    # seed gene implementations
│   └── fixtures/                 # mutation fix fixtures
└── network/                      # network infrastructure domain (shelved)
    └── ...                       # validated the architecture; not active
```

## Conventions

- Genes are `execute(input_json: str) -> str` functions
- All gene I/O is JSON strings. `gene_sdk` is injected into the gene's namespace at load time.
- Content addressing: alleles identified by SHA-256 of source code
- Temporal fitness: immediate (30%) + convergence (50%) + resilience (20%)
- Composition hierarchy: gene → pathway → topology → intent (each level has progressively more conservative thresholds)
- Gene promotion: fitness > dominant + 0.1, invocations >= 50
- Gene demotion: 3 consecutive failures → deprecated
- Pathway alleles: versioned by structure SHA (normalized step specs). Promotion advantage 0.15, min 200 executions
- Topology alleles: versioned by decomposition strategy SHA. Promotion advantage 0.20, min 500 executions
- Fusion: 10 consecutive pathway successes with identical allele composition → fuse to single gene
- Two gene families: configuration (effectors) and diagnostic (sensors)
- Decomposition: 10+ errors with 3+ distinct clusters → auto-split gene into sub-gene pathway
- Interaction detection: cross-locus testing before promotion (configurable policy: warn/rollback/mutate)
- Batch mutation: multiple diverse candidates per LLM mutation attempt
- Contracts are `.sg` files with verb-based sections: does, takes, gives, before, after, fails when, verify, feeds

## Running

```bash
pip install -e ".[dev]"
pip install -e plugins/data
pytest
sg init --kernel data-mock
sg run ingest_and_validate --input '{...}'
sg status
```

## Key Design Decisions

- **`.sg` contract format**: purpose-built for this paradigm. Verb-based sections (`does`, `takes`, `gives`). Domain experts author contracts, not developers.
- **Two gene families**: configuration genes act, diagnostic genes observe. Diagnostics produce the fitness signal for configuration genes.
- **Temporal fitness**: immediate (t=0) + convergence (t=30s) + resilience (t=hours). Retroactive decay when convergence/resilience fail.
- **Composition hierarchy**: gene → pathway → topology → intent. Pathways compose via `->`, iterate via `for`, bind conditionally via `when`.
- **Safety**: transactions with undo-log, blast radius classification (none → critical), shadow mode → canary → recessive → dominant allele lifecycle.
- **Domain-agnostic core**: kernel is an abstract interface. Domain-specific logic lives in plugins. The active plugin is `data` (data pipelines). The `network` plugin validated the architecture but is shelved.
- **Operational hardening**: atomic writes (temp file + rename), corrupted state recovery (try/except on all loads), shared read locks, fault-isolated save_state.
