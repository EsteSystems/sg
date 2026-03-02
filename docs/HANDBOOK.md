# Software Genome Handbook

A comprehensive guide to using, developing, and operating the Software Genome runtime.

For the paradigm whitepaper and design philosophy, see [SOFTWARE-GENOMICS.md](SOFTWARE-GENOMICS.md).

---

## 1. Getting Started

### Prerequisites

- Python 3.10+
- pip
- Linux for production kernel operations (macOS for development with mock kernel)

### Installation

```bash
pip install -e .                       # core runtime
pip install -e ".[dev]"                # + pytest
pip install -e ".[dashboard]"          # + FastAPI, uvicorn
pip install -e ".[federation]"         # + httpx
pip install -e ".[claude]"             # + anthropic SDK
pip install -e ".[openai]"             # + openai SDK
pip install -e ".[deepseek]"           # + openai SDK (DeepSeek-compatible)
pip install -e ".[all]"                # everything
```

### First Run

```bash
sg init                                # parse contracts, register seed genes, create phenotype
sg status                              # show genome state
sg run configure_bridge_with_stp \
  --input '{"bridge_name":"br0","interfaces":["eth0","eth1"],"stp_enabled":true,"forward_delay":15}'
```

`sg init` does three things:
1. Parses all `.sg` contracts in `contracts/`
2. Registers seed genes from `genes/` into the content-addressed registry
3. Creates `phenotype.toml` mapping each locus to its dominant allele

### Project Layout

```
sg/                               # core runtime
├── cli.py                        # CLI entry point
├── orchestrator.py               # evolutionary loop
├── registry.py                   # SHA-256 content-addressed store
├── phenotype.py                  # TOML phenotype map
├── arena.py                      # fitness scoring, promotion/demotion
├── loader.py                     # sandboxed gene loading
├── sandbox.py                    # exec() restrictions, timeout
├── mutation.py                   # mock + LLM mutation engines
├── pathway.py                    # pathway execution (fusion-aware)
├── fusion.py                     # reinforcement tracking
├── contracts.py                  # contract loading, validation
├── conformance.py                # contract conformance testing
├── safety.py                     # transactions, blast radius
├── regression.py                 # fitness regression detection
├── federation.py                 # multi-organism allele sharing
├── dashboard.py                  # web dashboard (FastAPI)
├── snapshot.py                   # genome snapshots/rollback
├── diff.py                       # phenotype diffing
├── parser/                       # .sg format parser
│   ├── types.py                  # AST node types
│   ├── lexer.py                  # tokenization
│   └── parser.py                 # AST construction
└── kernel/                       # gene_sdk implementations
    ├── base.py                   # NetworkKernel ABC
    ├── mock.py                   # development/test kernel
    └── production.py             # real network operations

contracts/                        # .sg contract definitions
├── genes/                        # gene contracts (11)
├── pathways/                     # pathway contracts (5)
└── topologies/                   # topology contracts (1)

genes/                            # seed gene source files (11)
fixtures/                         # mock mutation fixtures
vim/sg/                           # Neovim syntax highlighting plugin
```

---

## 2. Core Concepts

### Gene

A pure function. Takes JSON in, returns JSON out. Every gene implements:

```python
def execute(input_json: str) -> str:
    data = json.loads(input_json)
    # ... do work using gene_sdk ...
    return json.dumps({"success": True, ...})
```

Genes are sandboxed — no filesystem access, restricted imports, timeout enforcement. The gene has access to `gene_sdk`, a kernel interface injected into its namespace at load time.

### Locus

A typed slot defined by a `.sg` contract. Input schema, output schema, behavioral description. The contract never mutates — it's the fixed question that implementations answer.

### Allele

One implementation at a locus. Identified by SHA-256 of its source code. Alleles have three states:

- **Dominant** — currently active, handling live traffic, being scored
- **Recessive** — present in the genome, available as fallback or for re-promotion
- **Deprecated** — no longer expressed, candidate for removal

### Pathway

A declared sequence of loci producing a composite behavior. Late-binding — references loci, not alleles. The system resolves which allele to use at runtime based on the phenotype map.

### Topology

Declarative desired state. The domain expert describes what should exist (`has` blocks), not how to create it. The system figures out which pathways to run.

### Phenotype

Deployment-specific expression. A TOML file mapping each locus to its dominant allele and fallback stack. Same genome, different phenotype = same gene pool, different expression.

### Organism

The running system: orchestrator + phenotype + registry + fusion tracker. The application is what emerges when the orchestrator expresses a phenotype from the genome.

---

## 3. Writing Contracts

Contracts are the human-machine interface. Domain experts author them — no Python required. The `.sg` format uses verb-based section names that communicate the role of each section in the evolutionary loop.

### Gene Contracts

```
gene bridge_create
  is configuration
  risk low

  does:
    Create a Linux bridge with the given interfaces. The bridge is the
    fundamental L2 primitive — it connects physical NICs, VLAN interfaces,
    and VM virtual NICs into a single broadcast domain.

  takes:
    bridge_name  string    "Name for the bridge"
    interfaces   string[]  "Physical interfaces to attach as ports"

  gives:
    success            bool      "Whether the bridge was created"
    resources_created  string[]  "NM connections created, for rollback"
    error              string?   "Error message on failure"

  before:
    - Bridge with this name does not already exist
    - All specified interfaces exist and are not enslaved

  after:
    - Bridge exists and is in UP state
    - All specified interfaces are attached as bridge ports

  fails when:
    - bridge already exists -> success=false
    - interface not found -> success=false

  verify:
    check_connectivity bridge_name={bridge_name}
    within 30s
```

#### Section Reference

| Section | Purpose | Required |
|---------|---------|----------|
| `is` | Gene family: `configuration` or `diagnostic` | Yes |
| `risk` | Blast radius: `none`, `low`, `medium`, `high`, `critical` | No (default: `none`) |
| `does:` | Free-form prose — injected into mutation prompts | Yes |
| `takes:` | Input parameters with types | Yes |
| `gives:` | Output parameters with types | Yes |
| `types:` | Inline type definitions for custom types | No |
| `before:` | Preconditions (bullet list) | No |
| `after:` | Postconditions (bullet list) | No |
| `fails when:` | Known failure modes (bullet list) | No |
| `unhealthy when:` | Unhealthy states — diagnostic genes only | No |
| `verify:` | Diagnostic loci to run after execution | No |
| `within` | Convergence window for verification | No |
| `feeds:` | Fitness feedback targets — diagnostic genes only | No |

#### Field Types

| Type | Description | Example |
|------|-------------|---------|
| `string` | Text value | `bridge_name string` |
| `bool` | Boolean | `stp_enabled bool` |
| `int` | Integer | `forward_delay int` |
| `float` | Floating point | `churn_rate float` |
| `string[]` | Array of strings | `interfaces string[]` |
| `int[]` | Array of integers | `vlans int[]` |
| `type?` | Optional (may be absent) | `error string?` |
| `type = val` | Default value (implies optional) | `stp_enabled bool = false` |

#### Two Gene Families

**Configuration genes** (`is configuration`) act on the environment — they create, modify, or delete resources. They have blast radius, transactions, and rollback.

**Diagnostic genes** (`is diagnostic`) observe the environment — they read state and report health. They have `risk none`, use `unhealthy when` instead of `fails when`, and declare `feeds` to provide fitness feedback to configuration genes.

```
gene check_mac_stability
  is diagnostic
  risk none

  does:
    Check MAC address stability on a bridge.

  takes:
    bridge_name  string  "Bridge to check"

  gives:
    healthy     bool    "True if MAC is stable"
    mac         string  "Current MAC address"

  unhealthy when:
    - MAC address has changed since last check

  feeds:
    bridge_create convergence
    mac_preserve  convergence
```

The `feeds` section declares the fitness feedback loop. This diagnostic's output feeds the convergence fitness (50% weight) of the listed configuration genes.

### Pathway Contracts

Pathways compose genes into multi-step operations.

```
pathway deploy_server_network
  risk medium

  does:
    Full server network deployment.

  takes:
    bridge_name    string    "Management bridge name"
    interfaces     string[]  "Physical interfaces for bridge"
    bond_name      string    "Bond interface name"
    bond_mode      string    "Bond mode"
    bond_members   string[]  "Bond member interfaces"
    vlans          int[]     "VLAN IDs to provision"

  steps:
    1. -> provision_management_bridge
         bridge_name = {bridge_name}
         interfaces = {interfaces}

    2. bond_create
         bond_name = {bond_name}
         mode = {bond_mode}
         members = {bond_members}

    3. for vlan in {vlans}:
         vlan_create
           parent = {bond_name}
           vlan_id = {vlan}

  requires:
    step 2 needs step 1
    step 3 needs step 2

  on failure:
    rollback all
```

#### Pathway Syntax

**Step binding**: `{reference}` substitutes pathway input parameters into step inputs.

**Pathway composition**: `->` prefix means "run this pathway" — the orchestrator recursively resolves it.

**Iteration**: `for variable in {iterable}:` repeats a step for each element.

**Conditionals**: `when step N.field:` branches based on a previous step's output.

```
  steps:
    1. check_mac_stability
         bridge_name = {bridge_name}

    2. when step 1.healthy:
         false -> mac_preserve
           device = {bridge_name}
```

**Dependencies**: `step A needs step B` declares ordering constraints.

**Failure handling**: `on failure:` declares the strategy — `rollback all`, `report partial`, etc.

### Topology Contracts

Topologies declare desired state without specifying steps.

```
topology production_server
  does:
    Standard production server: management bridge, storage bond, VLAN bridges.

  takes:
    bridge_name     string    "Management bridge name"
    bond_name       string    "Bond interface name"
    bond_mode       string    "Bond mode"
    bond_members    string[]  "Bond member interfaces"
    vlans           int[]     "VLAN IDs"

  has:
    management:
      is bridge
      uplink {uplink}
      stp enabled

    storage:
      is bond
      mode {bond_mode}
      members {bond_members}

    vm_traffic:
      is vlan_bridges
      trunk storage
      vlans {vlans}

  verify:
    check_connectivity bridge_name={bridge_name}
    check_bond_state bond_name={bond_name}
    within 60s
```

The `has` block declares resources. Each resource has a type (`is bridge`, `is bond`) and properties. The orchestrator determines which pathways to run to make the desired state true.

---

## 4. Writing Genes

### The execute() Function

Every gene must define an `execute` function:

```python
def execute(input_json: str) -> str:
    import json
    data = json.loads(input_json)

    bridge_name = data["bridge_name"]
    interfaces = data["interfaces"]

    result = gene_sdk.create_bridge(bridge_name, interfaces)

    return json.dumps({
        "success": True,
        "bridge": result,
    })
```

- Input: JSON string matching the contract's `takes` schema
- Output: JSON string matching the contract's `gives` schema
- Must always include a `success` boolean field in the output
- On failure, return `{"success": false, "error": "description"}` — don't raise exceptions

### gene_sdk: The Kernel Interface

The `gene_sdk` object is injected into the gene's namespace at load time. It provides all kernel operations:

**Bridge Operations**:
- `gene_sdk.create_bridge(name, interfaces)` → dict
- `gene_sdk.delete_bridge(name)` → None
- `gene_sdk.attach_interface(bridge, interface)` → None
- `gene_sdk.detach_interface(bridge, interface)` → None
- `gene_sdk.get_bridge(name)` → dict or None

**STP Operations**:
- `gene_sdk.set_stp(bridge_name, enabled, forward_delay)` → dict
- `gene_sdk.get_stp_state(bridge)` → dict

**MAC Operations**:
- `gene_sdk.get_device_mac(device)` → str
- `gene_sdk.set_device_mac(device, mac)` → None
- `gene_sdk.send_gratuitous_arp(interface, mac)` → None

**Bond Operations**:
- `gene_sdk.create_bond(name, mode, members)` → dict
- `gene_sdk.delete_bond(name)` → None
- `gene_sdk.get_bond(name)` → dict or None

**VLAN Operations**:
- `gene_sdk.create_vlan(parent, vlan_id)` → dict
- `gene_sdk.delete_vlan(parent, vlan_id)` → None
- `gene_sdk.get_vlan(parent, vlan_id)` → dict or None

**Diagnostic Reads** (read-only):
- `gene_sdk.read_fdb(bridge)` → list of dicts (forwarding database entries)
- `gene_sdk.get_interface_state(interface)` → dict
- `gene_sdk.get_arp_table()` → list of dicts

### Sandboxing

Genes run inside a restricted `exec()` environment:

**Blocked builtins** — these raise errors if called:
`exec`, `eval`, `compile`, `__import__`, `open`, `input`, `breakpoint`, `exit`, `quit`

**Allowed imports** — only these modules can be imported:
`json`, `math`, `re`, `hashlib`, `datetime`, `collections`, `collections.abc`, `itertools`, `functools`, `copy`, `string`, `textwrap`

**Timeout**: 30 seconds default. Gene execution is interrupted if it exceeds this.

Any attempt to import `os`, `subprocess`, `sys`, `socket`, or any other module will raise `GeneImportError`.

### Testing Genes

```bash
sg test                        # test all loci with dominant alleles
sg test bridge_create          # test specific locus
sg test bridge_create -v       # verbose: show all check details
```

Conformance tests validate:
1. Source defines an `execute()` function
2. Output is valid JSON
3. Output contains a `success` field
4. All required `gives` fields are present (when `success=True`)
5. Field types match the contract schema

---

## 5. The Evolutionary Loop

```
execute → validate → score → fallback → mutate → register → promote
```

### Step by Step

1. **Execute**: Gene runs against input inside the sandbox
2. **Validate**: Output checked against contract's `gives` schema
3. **Score**: Fitness recorded — success increments, failure increments
4. **Fallback**: On failure, try the next allele in the fallback stack
5. **Mutate**: All alleles exhausted → LLM generates a new variant from the failing gene's source, the contract, and the error context
6. **Register**: New allele registered in the content-addressed store (SHA-256 of source)
7. **Promote/Demote**: Fitness accumulation drives allele lifecycle

### Fitness Scoring

**Simple fitness**:
```
fitness = successful_invocations / max(total_invocations, 10)
```

Minimum 10 invocations before the score is meaningful.

**Temporal fitness** (when diagnostic feedback is available):
- Immediate (30%) — did the gene succeed right now?
- Convergence (50%) — does the result hold after 30 seconds?
- Resilience (20%) — does the result hold over hours?

Retroactive decay: if convergence or resilience checks fail, the original gene's fitness is penalized even though it "succeeded" at execution time.

**Distributed fitness** (with federation peers):
```
distributed = 0.7 * local_fitness + 0.3 * peer_fitness
```

Peer fitness only applied when peers have at least 10 total invocations.

### Promotion

An allele is promoted to dominant when:
- It has been invoked at least **50 times**
- Its fitness exceeds the current dominant's fitness by at least **0.1**

If no dominant exists, any allele with fitness > 0.0 can be promoted.

### Demotion

An allele is demoted after **3 consecutive failures**. It moves from dominant to recessive, and the next allele in the fallback stack takes over.

### Mutation Engines

| Engine | Flag | Requires |
|--------|------|----------|
| Mock | `--mutation-engine mock` | `fixtures/` directory |
| Claude | `--mutation-engine claude` | `ANTHROPIC_API_KEY` |
| OpenAI | `--mutation-engine openai` | `OPENAI_API_KEY` |
| DeepSeek | `--mutation-engine deepseek` | `DEEPSEEK_API_KEY` |
| Auto | `--mutation-engine auto` | Tries Claude → OpenAI → DeepSeek → Mock |

### Proactive Generation

Don't wait for failure — generate competing alleles ahead of time:

```bash
sg generate bridge_create              # generate 1 competing allele
sg generate bridge_create --count 5    # generate 5 variants
sg generate --all                      # generate for every locus
```

---

## 6. Pathways and Composition

### Running Pathways

```bash
sg run configure_bridge_with_stp \
  --input '{"bridge_name":"br0","interfaces":["eth0","eth1"],"stp_enabled":true,"forward_delay":15}'
```

The orchestrator resolves each step to the dominant allele at that locus, executes it, and passes outputs forward.

### Pathway Fusion

When a pathway succeeds **10 consecutive times** with the same allele composition (same set of alleles at every locus), the pathway is eligible for fusion.

Fusion consolidates the multi-step pathway into a single optimized gene. The fused gene is generated by the mutation engine from the constituent gene sources and the pathway contract.

If the fused gene fails, it **decomposes** back to the original multi-step pathway. This is automatic and transparent.

**Fusion lifecycle**:
1. Pathway succeeds → reinforcement count increments
2. Different allele composition → reinforcement resets
3. 10 consecutive successes → fusion triggered
4. Fused gene registered and used for subsequent executions
5. Fused gene fails → decompose back to multi-step

---

## 7. Topologies

### Deploying a Topology

```bash
sg deploy production_server \
  --input '{"bridge_name":"mgmt","bridge_ifaces":["eth0"],"bond_name":"storage","bond_mode":"802.3ad","bond_members":["eth1","eth2"],"vlans":[100,200,300],"uplink":"eth0"}'
```

The orchestrator reads the `has` block, determines which pathways to run, resolves dependencies from resource references, and executes the plan.

### Verify Blocks

Topologies can declare verification with timescales:

```
verify:
  check_connectivity bridge_name={bridge_name}
  check_bond_state bond_name={bond_name}
  within 60s
```

The `within` clause sets the convergence window — how long to wait before running diagnostic checks.

---

## 8. Safety

### Gene Sandboxing

Every gene runs inside a restricted `exec()` with:
- Blocked dangerous builtins (no `open`, `eval`, `exec`, `compile`)
- Import allowlist (only safe modules: `json`, `math`, `re`, etc.)
- 30-second execution timeout
- No filesystem access, no network access, no subprocess spawning

### Transactions and Rollback

Configuration genes are wrapped in transactions. The `SafeKernel` proxy records an undo action for every mutating operation:

- `create_bridge` → records `delete_bridge` as undo
- `set_stp` → records previous STP state as undo
- `create_bond` → records `delete_bond` as undo

If the gene fails or output validation fails, all recorded undo actions are executed in reverse order.

### Blast Radius Classification

Contracts declare risk levels that control safety behavior:

| Risk Level | Transaction Required | Shadow Execution |
|------------|---------------------|------------------|
| `none` | No | No |
| `low` | Yes | No |
| `medium` | Yes | No |
| `high` | Yes | Yes — must pass 3 shadow runs on mock kernel first |
| `critical` | Yes | Yes — must pass 3 shadow runs on mock kernel first |

### Protected Interfaces

The production kernel enforces two safety mechanisms:

1. **Safety prefix**: All resource names must start with `sg-test-` to prevent accidental modification of real infrastructure
2. **Protected interfaces**: `eth0` and `lo` are always protected. Additional interfaces via environment variable:

```bash
export SG_PROTECTED_INTERFACES="vswitch0,ens192,em1"
```

Any attempt to modify a protected interface raises `ValueError`.

### Genome Snapshots

Capture and restore complete genome state:

```bash
sg snapshot --name "before-upgrade" --description "pre-upgrade state"
sg snapshots                           # list all snapshots
sg rollback before-upgrade             # restore from snapshot
```

A snapshot captures:
- `.sg/registry/` — all allele sources and the index
- `phenotype.toml` — dominant alleles and fallback stacks
- `fusion_tracker.json` — pathway reinforcement state
- `.sg/regression.json` — fitness regression history

### Regression Detection

The system monitors fitness trends and acts on regressions:

| Drop from Peak | Classification | Action |
|----------------|---------------|--------|
| >= 0.2 | Mild regression | Proactive mutation triggered |
| >= 0.4 | Severe regression | Auto-demotion to recessive |

Requires at least 10 invocations before detection activates.

---

## 9. Dashboard

Start the web dashboard:

```bash
pip install -e ".[dashboard]"
sg dashboard                           # localhost:8420
sg dashboard --host 0.0.0.0 --port 8420  # accessible externally
```

### API Endpoints

| Route | Method | Returns |
|-------|--------|---------|
| `GET /` | GET | Interactive HTML dashboard |
| `GET /api/status` | GET | Genome counts and average fitness |
| `GET /api/loci` | GET | All loci with dominant allele and fitness |
| `GET /api/locus/{name}` | GET | Alleles, contract details, lineage |
| `GET /api/pathways` | GET | Pathway fusion state and reinforcement counts |
| `GET /api/allele/{sha}/source` | GET | Gene source code |
| `GET /api/lineage/{sha}` | GET | Mutation ancestry chain |
| `GET /api/regression` | GET | Fitness regression history |
| `GET /api/events` | GET | SSE stream for live updates |

### SSE Live Updates

The `/api/events` endpoint streams Server-Sent Events. Connect with any SSE client to get real-time notifications when genome state changes (new alleles, promotions, fitness updates).

---

## 10. Federation

Multiple organisms (running instances) can share successful alleles over HTTP. Each organism makes independent promotion decisions.

### Peer Configuration

Create `peers.json` in the project root:

```json
{
  "peers": [
    {"url": "http://server1:8420", "name": "server1", "secret": "shared-key"},
    {"url": "http://server2:8420", "name": "server2"}
  ]
}
```

The `secret` field is optional. When present, requests are signed with HMAC-SHA256 via the `X-SG-Signature` header.

### Sharing Alleles

```bash
sg share bridge_create                 # push dominant allele to all peers
sg share bridge_create --peer http://server1:8420  # push to specific peer
sg pull bridge_create                  # fetch alleles from all peers
sg pull bridge_create --peer http://server1:8420   # fetch from specific peer
```

### Integrity and Security

- **SHA-256 verification**: Every imported allele's source is hashed and compared to the declared `source_sha256`. Mismatches are rejected.
- **HMAC-SHA256 authentication**: When a peer has a `secret`, all payloads are signed. The receiving peer verifies the signature before accepting the allele.

### Distributed Fitness

When peers share alleles, they also share fitness observations. The receiving organism incorporates peer data:

```
distributed_fitness = 0.7 * local_fitness + 0.3 * peer_fitness
```

Peer fitness is only factored in when the peer has at least 10 total invocations for that allele. Each organism still makes its own promotion decisions based on its local environment.

---

## 11. CLI Reference

### Commands

| Command | Description |
|---------|-------------|
| `sg init` | Parse contracts, register seed genes, create phenotype |
| `sg run <pathway> --input '{...}'` | Execute a pathway |
| `sg deploy <topology> --input '{...}'` | Deploy a topology |
| `sg generate [locus] [--all] [--count N]` | Proactively generate competing alleles |
| `sg watch <pathway> --input '{...}' [--interval N] [--count N]` | Periodically run diagnostics |
| `sg compete <locus> --input '{...}' [--rounds N]` | Run allele competition trials |
| `sg status` | Show genome state |
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

---

## 12. Configuration

### Environment Variables

| Variable | Description |
|----------|-------------|
| `SG_PROJECT_ROOT` | Project root directory (default: `.`) |
| `ANTHROPIC_API_KEY` | Claude API key for mutations |
| `OPENAI_API_KEY` | OpenAI API key for mutations |
| `DEEPSEEK_API_KEY` | DeepSeek API key for mutations |
| `SG_PROTECTED_INTERFACES` | Comma-separated extra protected interfaces |

### Kernel Selection

- **Mock kernel** (`--kernel mock`): In-memory state, no real system calls. Use for development and testing.
- **Production kernel** (`--kernel production`): Real `ip link`, `bridge`, and sysfs operations. Requires Linux and sudo. Enforces `sg-test-*` safety prefix on all resource names.

### Mutation Engine Selection

- **Auto** (default): Tries Claude → OpenAI → DeepSeek → Mock, using whichever has an API key configured
- **Mock**: Uses fixture files at `fixtures/mutations/`. No API key needed. Good for development.
- **Claude/OpenAI/DeepSeek**: Calls the respective LLM API to generate gene mutations from the contract, failing source, error context, and execution history.

### Phenotype Map

`phenotype.toml` is the deployment-specific expression map. It's generated by `sg init` and updated by the orchestrator as alleles are promoted and demoted.

```toml
[bridge_create]
dominant = "abc123..."
fallback = ["def456...", "ghi789..."]

[bridge_stp]
dominant = "jkl012..."
fallback = []
```

---

## 13. Development

### Running Tests

```bash
pip install -e ".[dev,dashboard,federation]"
python3 -m pytest tests/ -x -q         # all tests, stop on first failure
python3 -m pytest tests/ -v            # verbose output
python3 -m pytest tests/test_sandbox.py -v  # specific test file
```

### Contract Conformance

Validate that genes conform to their contracts:

```bash
sg test                                # test all loci
sg test bridge_create -v               # test specific locus, verbose
```

### Phenotype Diffing

Compare genome states to see what changed:

```bash
sg diff                                # current vs last snapshot
sg diff --snapshot before-upgrade      # current vs named snapshot
sg diff --a snap1 --b snap2            # compare two snapshots
```

Output format:
```
+ new_locus: dominant=abc123 fitness=0.850
- removed_locus: was dominant=def456 fitness=0.700
~ changed_locus: abc123 -> ghi789 fitness 0.800 -> 0.950
+ fusion: configure_bridge_with_stp
```

### Adding a New Contract

1. Create a `.sg` file in `contracts/genes/` (or `pathways/` or `topologies/`)
2. Write the contract following the format in section 3
3. Create a seed gene in `genes/` with the same name as the locus
4. Run `sg init` to register it
5. Run `sg test <locus>` to verify conformance

### Adding a Seed Gene

1. Create `genes/<locus_name>_v1.py`
2. Implement `execute(input_json: str) -> str`
3. Use `gene_sdk` for kernel operations
4. Return JSON with all required `gives` fields
5. Handle errors gracefully — return `{"success": false, "error": "..."}` instead of raising

---

## 14. Editor Support

### Neovim

The project includes a Vim/Neovim syntax highlighting plugin for `.sg` files at `vim/sg/`.

**Install to Neovim's site directory** (works globally, no plugin manager needed):

```bash
cp vim/sg/ftdetect/sg.vim ~/.local/share/nvim/site/ftdetect/
cp vim/sg/syntax/sg.vim ~/.local/share/nvim/site/syntax/
cp vim/sg/ftplugin/sg.vim ~/.local/share/nvim/site/ftplugin/
```

**Or with a plugin manager** (Lazy.nvim):

```lua
{ dir = "/path/to/software-genomics/vim/sg", name = "sg-syntax" }
```

**Or add to runtimepath**:

```vim
set runtimepath+=/path/to/software-genomics/vim/sg
```

The plugin provides:
- Filetype detection for `.sg` files
- Syntax highlighting for keywords, types, strings, references, operators, control flow, step numbers, and duration literals
- 2-space indentation settings
