# sg — Software Genomics Runtime

An evolutionary software runtime where genes (pure functions) compete at typed contract slots, scored by fitness from real execution, and replaced by LLM-generated mutations when they fail. The system treats runtime errors as evolutionary signal, not bugs.

See [SOFTWARE-GENOMICS.md](SOFTWARE-GENOMICS.md) for the full paradigm whitepaper.

## Concepts

- **Gene** — pure function. Takes JSON in, returns JSON out. No knowledge of the system it lives in.
- **Locus** — typed slot defined by a `.sg` contract. Input schema, output schema, behavioral description. The contract never mutates.
- **Allele** — one implementation at a locus. Identified by SHA-256 of source. Ranked: dominant → recessive → deprecated.
- **Pathway** — declared sequence of loci. Late-binding — references loci, not alleles.
- **Topology** — declarative desired state. The genome figures out which pathways to run.
- **Phenotype** — deployment-specific expression. Same genome, different hardware/environment.
- **Organism** — the running system: orchestrator + phenotype + registry + fusion tracker.

## Project Structure

```
sg/
├── SOFTWARE-GENOMICS.md              # paradigm whitepaper
├── README.md
├── pyproject.toml
│
├── sg/                               # core runtime
│   ├── __init__.py
│   ├── __main__.py                   # entry point
│   ├── cli.py                        # CLI: init, run, status
│   ├── orchestrator.py               # evolutionary loop
│   ├── registry.py                   # SHA-256 CAS + JSON index
│   ├── phenotype.py                  # TOML phenotype map + fusion state
│   ├── arena.py                      # fitness scoring, promotion/demotion
│   ├── loader.py                     # exec()-based gene loading
│   ├── mutation.py                   # mock + Claude mutation engines
│   ├── pathway.py                    # pathway execution (fusion-aware)
│   ├── fusion.py                     # reinforcement tracking, fuse/decompose
│   ├── contracts.py                  # locus types, validation, contract loading
│   ├── safety.py                     # transactions, blast radius, rollback
│   │
│   ├── parser/                       # .sg format parser
│   │   ├── __init__.py
│   │   ├── lexer.py                  # tokenization
│   │   ├── parser.py                 # AST construction
│   │   └── types.py                  # AST node types
│   │
│   └── kernel/                       # gene_sdk — kernel interface
│       ├── __init__.py
│       ├── base.py                   # NetworkKernel ABC
│       ├── mock.py                   # MockNetworkKernel (dev/test)
│       └── production.py             # ProductionNetworkKernel (NM D-Bus, ip link)
│
├── contracts/                        # .sg contract definitions
│   ├── genes/                        # gene contracts
│   │   ├── bridge_create.sg
│   │   ├── bridge_uplink.sg
│   │   ├── mac_preserve.sg
│   │   ├── bond_create.sg
│   │   ├── vlan_create.sg
│   │   ├── check_connectivity.sg
│   │   ├── check_mac_stability.sg
│   │   └── check_fdb_stability.sg
│   ├── pathways/                     # pathway contracts
│   │   ├── provision_management_bridge.sg
│   │   ├── provision_bond.sg
│   │   └── health_check_bridge.sg
│   └── topologies/                   # topology contracts
│       └── production_server.sg
│
├── genes/                            # seed gene source files
├── fixtures/mutations/               # mock mutation fixtures
└── tests/
    ├── test_parser.py                # .sg format parser tests
    ├── test_contracts.py             # contract loading + validation
    ├── test_registry.py              # CAS registry
    ├── test_arena.py                 # fitness scoring
    ├── test_orchestrator.py          # evolutionary loop
    ├── test_pathway.py               # pathway execution
    ├── test_fusion.py                # fusion/decomposition
    ├── test_kernel.py                # mock kernel
    └── test_safety.py                # transactions, rollback
```

## The `.sg` Contract Format

Contracts are the human-machine interface. They define what genes do, what they take, what they give back, and how to verify they worked. Domain experts author them — no Python required.

```
gene bridge_create
  is configuration
  risk low

  does:
    Create a Linux bridge with the given interfaces.

  takes:
    bridge_name  string    "Name for the bridge"
    interfaces   string[]  "Physical interfaces to attach"
    stp_enabled  bool = false  "Enable STP"

  gives:
    success            bool      "Whether the bridge was created"
    resources_created  string[]  "NM connections created, for rollback"
    error              string?   "Error message on failure"

  before:
    - Bridge with this name does not already exist

  after:
    - Bridge exists and is in UP state
    - All interfaces are attached as bridge ports

  fails when:
    - bridge already exists -> success=false
    - interface not found -> success=false

  verify:
    check_link_state device={bridge_name}
    within 30s
```

## The Evolutionary Loop

```
execute → validate → score → fallback → mutate → register → promote
```

1. Gene executes against input
2. Output validated against contract
3. Fitness scored (immediate + convergence + resilience)
4. On failure: try next allele in fallback stack
5. All alleles exhausted: LLM generates a mutant
6. Mutant registered in genome (SHA-256, lineage)
7. Accumulates fitness → promoted to dominant

## Installation

```bash
pip install -e .          # install in development mode
pip install -e ".[dev]"   # with test dependencies
pip install -e ".[claude]" # with Claude mutation engine
```

## Usage

```bash
sg init                    # parse contracts, register seed genes, create phenotype
sg run <pathway> --input '{...}'   # execute a pathway
sg status                  # show genome state: alleles, fitness, phenotype
```

Use `--mutation-engine=mock` for development (fixture-based, no API key).
Use `--mutation-engine=claude` with `ANTHROPIC_API_KEY` for LLM mutations.

## Development

```bash
pytest                     # run all tests
pytest tests/test_parser.py  # test .sg parser only
pytest -v                  # verbose output
```

## License

Copyright Este Systems. All rights reserved.
