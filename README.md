# sg — Software Genomics Runtime

An evolutionary software runtime where genes (pure functions) compete at typed contract slots, scored by fitness from real execution, and replaced by LLM-generated mutations when they fail. The system treats runtime errors as evolutionary signal, not bugs.

See [SOFTWARE-GENOMICS.md](SOFTWARE-GENOMICS.md) for the full paradigm whitepaper.

## Concepts

- **Gene** — pure function. Takes JSON in, returns JSON out. Sandboxed: no filesystem access, restricted imports.
- **Locus** — typed slot defined by a `.sg` contract. Input schema, output schema, behavioral description. The contract never mutates.
- **Allele** — one implementation at a locus. Identified by SHA-256 of source. Ranked: dominant → recessive → deprecated.
- **Pathway** — declared sequence of loci. Late-binding — references loci, not alleles. Supports loops (`for`) and conditionals (`when`).
- **Topology** — declarative desired state. The genome figures out which pathways to run.
- **Fusion** — reinforced pathways (10 consecutive successes) consolidate into a single optimized gene. Decomposes back on failure.
- **Phenotype** — deployment-specific expression. Same genome, different hardware/environment.
- **Organism** — the running system: orchestrator + phenotype + registry + fusion tracker.

## The Evolutionary Loop

```
execute → validate → score → fallback → mutate → register → promote
```

1. Gene executes against input (sandboxed `exec()`)
2. Output validated against contract
3. Fitness scored (immediate 30% + convergence 50% + resilience 20%)
4. On failure: try next allele in fallback stack, rollback kernel changes
5. All alleles exhausted: LLM generates a mutant (Claude, OpenAI, or DeepSeek)
6. Mutant registered in genome (SHA-256, lineage tracking)
7. Accumulates fitness → promoted to dominant (fitness > current + 0.1, >= 50 invocations)

## Installation

```bash
pip install -e .                    # core runtime
pip install -e ".[dev]"             # + test dependencies
pip install -e ".[dashboard]"       # + web dashboard (FastAPI + uvicorn)
pip install -e ".[federation]"      # + multi-organism federation (httpx)
pip install -e ".[claude]"          # + Claude mutation engine
pip install -e ".[openai]"          # + OpenAI mutation engine
pip install -e ".[deepseek]"        # + DeepSeek mutation engine
pip install -e ".[all]"             # everything
```

## Quick Start

```bash
sg init                                   # register seed genes, create phenotype
sg run configure_bridge_with_stp \
  --input '{"bridge_name":"br0","interfaces":["eth0","eth1"],"stp_enabled":true,"forward_delay":15}'
sg status                                 # show genome state
sg dashboard                              # start web dashboard at :8420
```

## CLI Reference

| Command | Description |
|---------|-------------|
| `sg init` | Parse contracts, register seed genes, create phenotype |
| `sg run <pathway> --input '{...}'` | Execute a pathway |
| `sg deploy <topology> --input '{...}'` | Deploy a topology |
| `sg generate [locus] [--all] [--count N]` | Proactively generate competing alleles via LLM |
| `sg watch <pathway> --input '{...}'` | Periodically run diagnostics for resilience fitness |
| `sg compete <locus> --input '{...}'` | Run allele competition trials |
| `sg status` | Show genome state: alleles, fitness, phenotype |
| `sg lineage <locus>` | Show mutation ancestry tree |
| `sg test [locus] [-v]` | Run contract conformance tests |
| `sg evolve --family <fam> --context '...'` | Generate a new contract via LLM |
| `sg dashboard [--port 8420] [--host 0.0.0.0]` | Start web dashboard |
| `sg share <locus> [--peer URL]` | Push alleles to federation peers |
| `sg pull <locus> [--peer URL]` | Fetch alleles from federation peers |
| `sg snapshot [--name N] [--description D]` | Create genome snapshot |
| `sg snapshots` | List all snapshots |
| `sg rollback <name>` | Restore genome from snapshot |
| `sg diff [--snapshot N] [--a A --b B]` | Compare genome states |
| `sg completions <bash\|zsh\|fish>` | Generate shell completions |

### Global Flags

| Flag | Description |
|------|-------------|
| `--mutation-engine auto\|mock\|claude\|openai\|deepseek` | Mutation engine (default: auto) |
| `--model <name>` | Override default model for mutation engine |
| `--kernel mock\|production` | Kernel type (default: mock) |
| `--version` | Show version |

## Web Dashboard

```bash
pip install -e ".[dashboard]"
sg dashboard --host 0.0.0.0 --port 8420
```

Endpoints:

| Route | Returns |
|-------|---------|
| `GET /` | Interactive HTML dashboard |
| `GET /api/status` | Genome counts and average fitness |
| `GET /api/loci` | All loci with dominant allele and fitness |
| `GET /api/locus/{name}` | Alleles, contract details, lineage |
| `GET /api/pathways` | Pathway fusion state and reinforcement counts |
| `GET /api/allele/{sha}/source` | Gene source code |
| `GET /api/lineage/{sha}` | Mutation ancestry chain |
| `GET /api/regression` | Fitness regression history |
| `GET /api/events` | SSE stream for live updates |

## Multi-Organism Federation

Organisms share successful alleles over HTTP. Each organism makes independent promotion decisions.

```bash
# Configure peers
cat > peers.json << 'EOF'
{"peers": [
  {"url": "http://peer1:8420", "name": "server1", "secret": "shared-key"},
  {"url": "http://peer2:8420", "name": "server2"}
]}
EOF

# Share and pull alleles
sg share bridge_create                    # push dominant to all peers
sg pull bridge_create                     # fetch from all peers
sg share bridge_create --peer http://...  # push to specific peer
```

Federation includes:
- SHA-256 integrity verification on import
- Optional HMAC-SHA256 peer authentication via shared secrets
- Distributed fitness: 70% local + 30% peer observations

## The `.sg` Contract Format

Contracts are the human-machine interface. Domain experts author them — no Python required.

```
gene bridge_create
  is configuration
  risk low

  does:
    Create a Linux bridge with the given interfaces.

  takes:
    bridge_name  string    "Name for the bridge"
    interfaces   string[]  "Physical interfaces to attach"

  gives:
    success  bool      "Whether the bridge was created"
    error    string?   "Error message on failure"

  before:
    - Bridge with this name does not already exist

  after:
    - Bridge exists and is in UP state
    - All interfaces are attached as bridge ports

  fails when:
    - bridge already exists -> success=false

  verify:
    check_connectivity bridge_name={bridge_name}
    within 30s
```

## Safety

- **Sandboxing**: Genes run in restricted `exec()` — no `open()`, `eval()`, `os`, `subprocess`. Only safe imports (json, math, re, collections, etc.)
- **Transactions**: Configuration genes wrapped in rollback transactions. Kernel changes undone on failure.
- **Blast radius**: Contracts declare risk levels (none, low, medium, high, critical). High-risk genes require shadow execution against mock kernel before live.
- **Protected interfaces**: Production kernel enforces `sg-test-*` prefix on all resources. Management interfaces (eth0, lo, + `SG_PROTECTED_INTERFACES` env var) cannot be modified.
- **Snapshots**: `sg snapshot` captures full genome state. `sg rollback` restores it.
- **Regression detection**: Automatic fitness regression detection with proactive mutation on mild regression, auto-demotion on severe regression.

## Configuration

| Environment Variable | Description |
|---------------------|-------------|
| `SG_PROJECT_ROOT` | Project root directory (default: `.`) |
| `ANTHROPIC_API_KEY` | Claude API key for mutations |
| `OPENAI_API_KEY` | OpenAI API key for mutations |
| `DEEPSEEK_API_KEY` | DeepSeek API key for mutations |
| `SG_PROTECTED_INTERFACES` | Comma-separated extra protected interfaces |

## Project Structure

```
sg/
├── sg/                               # core runtime
│   ├── cli.py                        # CLI entry point
│   ├── orchestrator.py               # evolutionary loop
│   ├── registry.py                   # SHA-256 content-addressed store
│   ├── phenotype.py                  # TOML phenotype map
│   ├── arena.py                      # fitness scoring, promotion/demotion
│   ├── loader.py                     # sandboxed gene loading
│   ├── sandbox.py                    # exec() restrictions, timeout
│   ├── mutation.py                   # mock + LLM mutation engines
│   ├── pathway.py                    # pathway execution (fusion-aware)
│   ├── fusion.py                     # reinforcement tracking
│   ├── contracts.py                  # contract loading, validation
│   ├── conformance.py                # contract conformance testing
│   ├── safety.py                     # transactions, blast radius
│   ├── regression.py                 # fitness regression detection
│   ├── federation.py                 # multi-organism allele sharing
│   ├── dashboard.py                  # web dashboard (FastAPI)
│   ├── snapshot.py                   # genome snapshots/rollback
│   ├── diff.py                       # phenotype diffing
│   ├── parser/                       # .sg format parser
│   └── kernel/                       # NetworkKernel implementations
│       ├── base.py                   # ABC
│       ├── mock.py                   # development/test
│       └── production.py             # real network operations
├── contracts/                        # .sg contract definitions
│   ├── genes/                        # gene contracts (11)
│   ├── pathways/                     # pathway contracts (5)
│   └── topologies/                   # topology contracts (1)
├── genes/                            # seed gene source files (11)
├── fixtures/                         # mock mutation fixtures
├── demo/                             # demos and examples
└── tests/                            # 476+ tests
```

## Development

```bash
pip install -e ".[dev,dashboard,federation]"
python3 -m pytest tests/ -x -q        # run all tests
python3 -m pytest tests/ -v            # verbose output
sg test                                # run contract conformance
sg test bridge_create -v               # test specific locus
```

## License

Copyright Este Systems. All rights reserved.
