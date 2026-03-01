# Software Genomics

*A paradigm for software that evolves its own implementations under environmental selection pressure.*

---

## 1. The Problem

Software is built on assumptions. Every conditional, every validation rule, every schema choice encodes a developer's prediction about what the world will throw at it. When reality diverges — an edge case nobody anticipated, a user workflow nobody imagined, a data shape that doesn't fit the model — the software breaks. A human investigates, understands the gap, writes a fix, tests it, deploys it. Days or weeks for what was a single moment of mismatch between assumption and reality.

The traditional development pipeline treats this as acceptable. Ship, observe, fix, ship again. More recent attempts at autonomous software generation — the "dark factory" approach — try to compress this by front-loading quality through multi-stage validation and AI-generated code. But the core limitation remains: you can only validate against failure modes you anticipated.

This is the **horizon problem**. The quality of any system — human or automated — is bounded by the questions its builders knew to ask. Unknown unknowns pass through every gate, every review, every test suite. An LLM gives brilliant answers, but only to questions you already know to ask.

And there is a deeper structural problem. The software industry has internalized "don't reinvent the wheel" as an axiom. Use existing libraries. Wrap existing tools. Build on existing abstractions. In practice, this means: inherit someone else's assumptions and build yours on top.

Each layer in a dependency stack is a set of frozen assumptions. When you import a library, you don't just import its functions — you import its model of the world. Its opinion on error handling. Its concurrency assumptions. Its memory model. Its notion of what a "valid input" looks like. These assumptions are invisible when they match your use case and catastrophic when they don't.

And the stack compounds. Your software assumes your framework. Your framework assumes the standard library. The standard library assumes the operating system's userland. The userland assumes the kernel's system call interface. Each layer adds its own opinions, its own quirks, its own frozen behaviors. A program built on Linux should, in theory, run on FreeBSD — they're both POSIX-compliant. But in practice, the program doesn't call kernel primitives. It calls glibc, which calls systemd, which assumes journald, which assumes cgroups v2. The actual computation — the thing the program does — might be trivial. The assumption stack it sits on is ten layers deep, and every layer is platform-specific.

This is why software becomes "legacy." Not because it's old — because it embodies assumptions that no longer match reality but that can't be changed because too much depends on them. The cost of change exceeds the cost of living with the wrong assumption. So you live with it. That's legacy.

Software Genomics dissolves this by making assumptions mutable. Not through human refactoring — through environmental selection pressure that continuously replaces implementations whose assumptions have expired.


## 2. The Principle

Stop trying to anticipate all failure modes at build time. Instead, build software that treats runtime errors as evolutionary signal.

When the system encounters something it can't handle — an exception, a malformed input, a logical contradiction — it doesn't just log and fail. It generates a variant that attempts to handle the new case, and lets both the original and the variant compete under real conditions. The software doesn't wait for a human to diagnose and prescribe. It proposes its own mutations and lets the environment decide which ones survive.

This is a fundamentally different relationship between software and its environment. Traditional software is a thesis: "I believe the world works this way." Evolutionary software is a hypothesis under constant testing: "This is my best guess, and I'm actively looking for evidence that I'm wrong."

The unit of evolution is not the application. It's the **gene** — a pure function with a typed contract. The contract (input type, output type, behavioral description) is the invariant. The implementation is the genome — it mutates freely. Each gene evolves independently, but the interfaces between genes remain fixed. The internals mutate; the skeleton doesn't.

The evolutionary loop:

```
execute → validate → score → fallback → mutate → register → promote
```

A gene fails on an input. The system tries the next gene in the fallback stack. If all alternatives are exhausted, an LLM generates a new variant from the failing gene's source, the contract, and the error context. The variant is compiled, registered, and tested. If it succeeds, it enters the population. If it accumulates enough fitness, it gets promoted to dominant. The old gene isn't deleted — it becomes recessive, available for re-expression if conditions change. Nothing is legacy because nothing is permanent and nothing is lost.


## 3. The Architecture

The architecture is expressed through six concepts, each borrowed from biology not as metaphor but as functional analog — the same patterns emerge because the same pressures apply.

### Gene

A pure function. Takes typed input, produces typed output, has no knowledge of the system it lives in. A gene that creates a network bridge doesn't know it's part of a network configuration tool. It takes a bridge name and a list of interfaces, calls the kernel, and returns success or failure. The gene is the unit of evolution — the smallest thing that can mutate independently.

In the current implementation, a gene is a Python function:

```python
def execute(input_json: str) -> str:
    data = json.loads(input_json)
    result = gene_sdk.create_bridge(data["bridge_name"], data["interfaces"])
    return json.dumps({"success": True, "bridge": result})
```

The gene has access to a `gene_sdk` — a kernel interface that provides the operations genes need to interact with the environment. In development, the kernel is simulated. In production, it wraps real system calls (NetworkManager D-Bus, `ip link`, sysfs). The gene doesn't know which kernel it's running against. The contract is the same.

### Locus

A typed slot — input schema, output schema, behavioral description. The contract never mutates. Alleles (implementations) come and go; the locus is eternal. The locus is the *question*; alleles are competing *answers*.

A locus definition includes:
- **Identity**: name, family (configuration or diagnostic), risk level
- **Interface**: what goes in, what comes out, with types
- **Behavior**: what the gene should do, expressed as prose the mutation engine can reason about
- **Invariants**: preconditions, postconditions, known failure modes
- **Verification**: which diagnostic loci to run after execution, and on what timescale

### Allele

One implementation at a locus. Identified by SHA-256 of its source code — content-addressed, immutable, individually addressable. Every allele records its lineage: what it was mutated from, what failure prompted its creation, and what input triggered the mutation.

Alleles have three states:

- **Dominant**: currently expressed, handling live traffic, actively scored
- **Recessive**: present in the genome, not active, available for promotion. The system retains its fitness history — how it performed when it was active, what edge cases it handled, why it was displaced
- **Deprecated**: not expressed in any deployment, no recent fitness contribution, candidate for removal

When a new allele wins at a locus, the previous allele doesn't get deleted. It becomes recessive — shelved, not destroyed. If the environment changes and the old allele becomes relevant again, it can be re-promoted without invoking the mutation engine. Adaptation from existing genetic material is faster and carries less risk than novel mutation, because the allele has a proven fitness history.

### Pathway

A declared sequence of loci producing a composite behavior. Late-binding — references loci, not alleles. The system resolves which allele to use at runtime based on the phenotype map.

"Configure a bridge with STP" is a pathway: `bridge_create` → `bridge_stp`. The pathway declares which loci participate and in what order. It does not declare which alleles fill those loci. That's determined at expression time.

This gives genuine abstraction without frozen implementation. You can reason about "configure a bridge" as a unit. You can compose it into higher-level pathways. But the abstraction layer is a recipe, not a bundle. It says what needs to happen, not how.

Three levels of evolution operate simultaneously. **Genes evolve**: a better allele appears at the STP locus. **Pathways evolve**: someone discovers that configuring the MTU before creating the bridge avoids a kernel race condition, so the pathway order changes. **Pathway composition evolves**: a new topology pattern requires a new combination of pathways that nobody anticipated.

### Phenotype Map

Deployment-specific expression. A configuration mapping each locus to its dominant allele and fallback stack, plus pathway fusion state. The phenotype determines which genes are active. Different environments can have different phenotypes over the same genome registry — same gene pool, different expression.

This reframes the speciation problem. Deployments aren't diverging codebases. They're diverging expression patterns. Server A and Server B share the same gene pool but express different subsets of it. When Server A's deployment evolves a new allele, that allele enters the shared genome. Whether Server B expresses it depends on whether their environment activates it.

### Organism

The running system: orchestrator + phenotype map + genome registry + fusion tracker. The "application" is what emerges when the orchestrator expresses a phenotype from the genome. It's not built. It's not deployed. It's grown.

Updating a deployment means updating the phenotype map. Rolling back means reverting the map. The functions themselves are never edited, only replaced by new alleles. And because every allele is immutable and content-addressed, the entire evolutionary history of the system is an append-only log.


## 4. The Contract

The contract is the most important artifact in the genome. It's the only thing a human *must* write. Everything below it — the gene implementations, the mutations, the fitness scoring — is autonomous. The contract is the interface between human intent and machine evolution.

This places a high demand on the contract format. It must be:

- **Human-authored by domain experts**, not framework developers. The person who defines "create a network bridge" is a network engineer, not a Python programmer.
- **Readable as prose**, because the mutation engine is an LLM. The quality of generated mutations depends directly on how well the contract communicates intent.
- **Structured enough for runtime validation**, because the orchestrator must check every gene's input and output against the contract.
- **The source of truth** for all three audiences: the human (readability), the LLM (behavioral understanding), and the runtime (validation).

No existing format serves all three. YAML is a serialization format. Markdown is a documentation format. TOML is a configuration format. Python dataclasses are a type system. None were designed to express "here is a typed slot in an evolutionary runtime where LLM-generated implementations compete under fitness pressure."

### The `.sg` Format

Software Genomics introduces a purpose-built contract format. The design principle: **verb-based section names that communicate the role of each section in the evolutionary loop.** A domain expert reads `takes bridge_name string` and immediately understands. No need to know what "input schema" means.

A gene contract:

```
gene bridge_create
  is configuration
  risk low

  does:
    Create a Linux bridge with the given interfaces. The bridge is the
    fundamental L2 primitive — it connects physical NICs, VLAN interfaces,
    and VM virtual NICs into a single broadcast domain.

    For VLAN bridges, always disable STP. STP topology changes trigger
    FDB flushes that disrupt VM traffic on leaf bridges where loops
    are impossible.

  takes:
    bridge_name  string    "Name for the bridge"
    interfaces   string[]  "Physical interfaces to attach as ports"
    stp_enabled  bool = false  "Enable Spanning Tree Protocol"

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
    - NM connection is created and tracked in genome state

  fails when:
    - bridge already exists -> success=false
    - interface not found -> success=false
    - NM D-Bus unavailable -> error

  verify:
    check_link_state device={bridge_name}
    check_connectivity source_interface={bridge_name}
    within 30s
```

Each section has a specific role:

- **`does`**: Free-form prose, injected verbatim into mutation prompts. The quality of this prose directly determines the quality of LLM-generated alleles. This is where the domain expert's knowledge lives.
- **`takes`** / **`gives`**: Columnar type declarations. `name type "description"`. Defaults are inline: `stp_enabled bool = false`. Optional fields: `error string?`. Denser than YAML, more scannable, no braces or colons.
- **`before`** / **`after`**: Invariants. Preconditions and postconditions that constrain the solution space. Both the runtime and the mutation engine use these — the runtime to validate, the LLM to reason about constraints.
- **`fails when`**: First-class failure modes. In every other format, failure modes are an afterthought. Here they're structural. When a new failure mode is discovered in production, the domain expert adds a line to `fails when` and the mutation engine knows about it for all future mutations.
- **`verify`**: Diagnostic loci to run after execution. The contract declares its own verification strategy. `within 30s` is the convergence window. This makes the feedback loop between configuration and diagnostic genes explicit at the contract level.

A diagnostic gene contract follows the same structure but with semantic differences that reflect its role as an observer rather than an actor:

```
gene check_fdb_stability
  is diagnostic
  risk none

  does:
    Monitor the forwarding database of a bridge for MAC flapping —
    the same MAC address appearing on multiple ports in rapid
    succession. This is the signature of an uplink change where the
    upstream switch hasn't flushed its FDB, a misconfigured VLAN
    trunk, or a physical loop.

  takes:
    bridge          string  "Bridge to monitor"
    monitor_seconds int = 10  "How long to monitor"
    target_mac      string?  "Specific MAC to watch, or all"

  gives:
    healthy        bool       "True if no flapping detected"
    flapping_macs  mac_flap[] "MACs that changed ports"
    entry_count    int        "Total FDB entries"
    churn_rate     float      "Entry changes per second"

  types:
    mac_flap:
      mac          string   "The flapping MAC address"
      ports        string[] "Ports it was seen on"
      transitions  int      "Number of port changes"

  unhealthy when:
    - Any MAC appears on more than one port within monitoring window
    - Churn rate exceeds 0.1 entries/second

  feeds:
    bridge_create   convergence
    bridge_uplink   convergence
    mac_preserve    convergence
```

Two things are different: **`unhealthy when`** replaces `fails when` (diagnostic genes don't fail — they report health), and **`feeds`** declares the fitness feedback loop (this diagnostic's output feeds the convergence fitness of the listed configuration genes).

The entire contract is simultaneously a specification, a mutation prompt, and a validation schema. No transformation between formats. The domain expert writes once; the human reads it, the LLM reads it, the runtime reads it.


## 5. Composition and Scale

Simple pathways are flat lists of steps. But real systems aren't flat. Deploying a production server's network topology might involve 15-20 operations across 5-6 resources with dependencies, parallelism, and partial failure handling. A flat step list doesn't scale. What scales is **layers of intent**.

### The Composition Hierarchy

```
Level 0 — Gene:      "Create a bridge"
Level 1 — Pathway:   "Provision a management bridge"
Level 2 — Composed:  "Deploy the full network"
Level 3 — Topology:  "This server has management, storage, and VM traffic"
Level 4 — Intent:    "50 hypervisors, 4 storage nodes, these networks"
```

The domain expert enters at whatever level matches their concern. A network engineer specifying a bridge creation uses Level 1. A sysadmin deploying a fleet uses Level 3. The genome decomposes higher levels into lower levels, all the way down to individual gene executions.

### Level 1: Pathway

A linear sequence of loci with input transforms:

```
pathway create_verified_bridge
  risk low

  does:
    Create a bridge and verify it's working.

  takes:
    bridge_name  string    "Name for the bridge"
    interfaces   string[]  "Interfaces to attach"

  steps:
    1. bridge_create
         bridge_name = {bridge_name}
         interfaces = {interfaces}

    2. check_link_state
         device = {bridge_name}

  verify:
    check_connectivity source_interface={bridge_name}
    within 30s
```

Steps reference loci. Input transforms use `{references}` to map pathway input to step input.

### Level 2: Composed Pathway

Pathways that reference other pathways as steps, with dependency declarations and parallel execution:

```
pathway deploy_server_network
  risk critical

  does:
    Deploy the complete network topology for a production server.
    Management bridge for server access, storage bond for redundant
    storage traffic, VLAN bridges on the bond for VM isolation.

  takes:
    management_nic  string    "NIC for management bridge"
    storage_nics    string[]  "NICs for storage bond"
    vm_vlans        int[]     "VLAN IDs for VM bridges"

  steps:
    1. -> provision_management_bridge
           management_bridge = "Management"
           uplink_interface = {management_nic}

    2. -> provision_bond
           bond_name = "storage"
           members = {storage_nics}
           mode = active-backup

    3. for vlan in {vm_vlans}:
         -> provision_vlan_bridge
             parent_interface = "storage"
             vlan_id = {vlan}

  requires:
    step 3 needs step 2

  on failure:
    rollback completed steps
    report partial
```

New constructs:

- **`->` prefix**: "run this pathway" — the orchestrator recursively resolves composed pathways down to gene-level steps.
- **`for ... in`**: Iteration. Deploying VLAN bridges for `[100, 200, 300]` shouldn't require writing the step three times.
- **`requires`**: Dependency declarations. Steps 1 and 2 have no dependency — they run in parallel. Step 3 needs step 2 (VLANs go on the bond). The orchestrator infers parallelism from the dependency graph.
- **`report partial`**: The pathway can partially succeed. A server with management but no storage is better than a server with nothing.

### Level 3: Topology (Declarative Intent)

The domain expert doesn't specify steps. They describe the desired end state:

```
topology production_server

  does:
    Standard production server: management bridge on the first NIC,
    storage bond on the next two NICs, VLAN bridges on the bond for
    VM traffic isolation.

  has:
    management:
      is bridge
      uplink {nic.0}
      mac preserve
      stp enabled

    storage:
      is bond
      mode active-backup
      members [{nic.1}, {nic.2}]

    vm_traffic:
      is vlan_bridges
      trunk storage
      vlans [100, 200, 300]
      stp disabled

  takes:
    nic  string[3..]  "Physical NICs, ordered by role"

  verify:
    check_connectivity from=management target=gateway
    check_bond_state bond=storage
    for vlan in [100, 200, 300]:
      check_vlan_tagging interface=storage.{vlan}
    all within 60s

  on failure:
    preserve what works
    report what failed
```

**`has` instead of `steps`.** The domain expert declares what exists, not how to create it. The orchestrator's job is to figure out which pathways to run, in what order, to make the `has` block true. Resources reference each other by name — `trunk storage` means "the VLAN trunk is the bond named `storage`." Dependencies are implicit in the references.

**`nic` is an indexed parameter.** `{nic.0}` is the first NIC. The domain expert thinks in terms of NIC roles, not interface names. Actual interface names are environment-specific — that's what the phenotype handles.

The decomposition from topology to pathways is itself evolvable. A naive decomposer creates one pathway per resource. A smarter one might fuse independent resources into a single parallel pathway. The decomposition strategy is an allele competing for fitness.

### Phenotypes: Same Topology, Different Hardware

The topology is written once. Each server has different NIC names:

```
phenotype dell_r740
  for production_server
  nic = [eno1, eno2, eno3, eno4]

phenotype supermicro_x11
  for production_server
  nic = [eth0, eth1, eth2, eth3]
```

Same genome, different phenotypic expression based on environment. The topology is the genotype. The running network is the phenotype.

### The Scaling Principle

Pathways don't scale by getting longer. They scale by getting deeper. A 50-step flat pathway is unmaintainable. A 3-level composition tree where each node has 3-5 children is natural:

```
topology (1 file)
  └── pathway (1 per resource)
        └── gene (1 per operation)
```

Each level is authored independently, has its own verification strategy, its own failure/rollback policy, its own fitness tracking, and can evolve independently. The `.sg` format supports all levels with the same syntax — the first keyword (`gene`, `pathway`, `topology`) tells you what level you're at.


## 6. The Evolutionary Loop

The core execution cycle, in detail.

### Execute

An input arrives. The orchestrator identifies which locus handles it. It loads the dominant allele — just that one function, nothing else. The allele processes the input and produces an output.

### Validate

The output is checked against the locus contract. Is it valid JSON? Does it contain the required fields? Are the types correct? Invalid output is treated as failure — the allele is scored down and the next in the stack is tried.

### Score

The arena records the outcome — success or failure — and updates the allele's fitness. Fitness is a ratio:

```
fitness = successful_invocations / max(total_invocations, 10)
```

The `max(total, 10)` floor prevents a single success from producing perfect fitness. An allele needs enough trials for its score to be meaningful.

### Fallback

If the dominant allele fails, the orchestrator loads the next allele in the ranked stack — the recessive with the highest historical fitness. The input stays alive for another attempt. The process repeats down the stack until a valid output is produced or the stack is exhausted.

This is an immune system, not an error handler. The system doesn't need to know in advance what will fail or how to recover. It needs a population of alternatives and a way to try them. The error itself becomes the selection event that finds the right allele.

### Mutate

When the entire allele stack is exhausted — every allele has failed — the orchestrator triggers mutation. An LLM receives the failing allele's source code, the locus contract, the failing input, the error message, and (in the network genome) diagnostic output and environmental state. It produces a new allele — a modified copy that attempts to handle the new case while preserving existing behavior.

The LLM doesn't redesign the gene. It proposes a targeted adaptation. This sounds like ordinary bug-fixing — and at the individual level, it is. The difference is accumulation and speed. A traditional team fixes one bug per cycle: discover, triage, assign, fix, review, test, deploy. The genome spawns a candidate fix within seconds and runs it against reality immediately.

### Register

The new allele is registered in the genome: SHA-256 of its source, lineage (parent allele, mutation context), generation number. It enters the fallback stack as a recessive and is immediately tested against the failing input.

### Promote / Demote

An allele is promoted to dominant when its fitness exceeds the current dominant's fitness by a threshold (default: 0.1) and it has accumulated enough invocations (default: 50). An allele is demoted after consecutive failures (default: 3).

Promotion is conservative — the threshold and invocation requirements prevent a single lucky success from displacing a proven allele. Demotion is aggressive — three consecutive failures indicate a fundamental mismatch, not bad luck.

### The LLM as Mutation Engine

The LLM is not generating software from scratch. It's performing targeted adaptation of existing code in response to observed failure. This is the exact scope LLMs handle well: a pure function with a typed contract.

The contract is the prompt. "Write a function that takes a bridge name and interfaces, creates a Linux bridge, and returns success or an error." There's no ambiguity. No architectural context needed. The gene is self-contained by design, and self-contained is what LLMs do best.

Problems that plague LLM code generation in traditional contexts disappear:

- **Integration**: The LLM writes genes. Pathways handle composition. The orchestrator handles execution. The LLM never needs to think about how its output fits into the larger system.
- **Error compounding**: A dark factory's 8-stage pipeline compounds errors: 0.95^8 = 66% success. Gene generation is single-stage: 0.95^1 = 95%. And the 5% failure is caught by the fallback stack.
- **Hallucination**: A made-up API is a failed mutation. The arena catches it instantly: the output doesn't match the contract. The allele gets fitness zero. Hallucination costs milliseconds of compute, not correctness.

The genome paradigm doesn't make LLMs generate better complex software. It makes complex software unnecessary. One approach pushes the tool's capability up to meet the problem's complexity. The other pulls the problem's complexity down to meet the tool's capability. The first is an arms race against model limitations. The second is a design pattern that works with any model capable of writing a function.


## 7. Fitness and Selection

The MVP fitness model is synchronous: the gene runs, returns a result, fitness is scored immediately. This works when the result is immediately observable. For many real-world domains — particularly networking, infrastructure, and distributed systems — it doesn't.

### Two Gene Families

The network genome introduces a fundamental extension: **two gene families** with distinct roles.

**Configuration genes** change state. They create bridges, set MACs, configure VLANs. Their contract is: take desired configuration, mutate the environment, report success or failure. Configuration genes are the genome's *effectors*.

**Diagnostic genes** observe state. They check MAC stability, verify ARP resolution, probe connectivity, measure convergence time. Their contract is: take a question about the environment, observe it, report health. Diagnostic genes are the genome's *sensors*.

The critical insight: **diagnostic genes produce the fitness signal that configuration genes are scored against.** A configuration gene's true fitness is not whether it returned `success: true`, but whether the diagnostic genes that run *after* it report a healthy system.

This creates a feedback loop:

```
  Configuration gene executes
          │
          ▼
  Immediate result: success/failure (30% weight)
          │
          ▼
  Diagnostic gene(s) execute at t+30s
          │
          ▼
  Convergence result: healthy/unhealthy (50% weight)
          │
          ▼
  Periodic health check at t+hours
          │
          ▼
  Resilience result: stable/degraded (20% weight)
```

Fitness is no longer self-reported. It's externally verified by independent observers.

### Temporal Fitness

Network fitness needs three timescales:

**Immediate (t=0).** Did the operation complete without error? For networking, this is trivially satisfied — almost every `ip link` or `nmcli` command succeeds. An allele that only passes immediate fitness has told you almost nothing.

**Convergence (t=30s).** Is the network stable 30 seconds after the operation? This catches: STP convergence (15-50 seconds), FDB population (seconds), DHCP lease renewal (30-60 seconds), MAC flapping (manifests within seconds), bond slave activation (several seconds). The 30-second window is the critical fitness window for most network operations.

**Resilience (t=hours/days).** Does the configuration survive reboot? Does the bond re-form after a link flap? Does the DHCP lease renew? Resilience fitness is measured by periodic health checks — diagnostic pathways that run on a schedule.

### Retroactive Fitness Decay

When a convergence or resilience check fails, the configuration gene that created the state has its fitness retroactively reduced. The gene thought it succeeded (immediate fitness was positive), but the network eventually disagreed.

```python
@dataclass
class FitnessRecord:
    allele_sha: str
    timestamp: float
    immediate_success: bool
    convergence_success: bool | None = None   # filled at t+30s
    resilience_success: bool | None = None    # filled at t+hours

    @property
    def effective_fitness(self) -> float:
        score = 0.0
        if self.immediate_success:
            score += 0.3
        if self.convergence_success is True:
            score += 0.5
        if self.resilience_success is True:
            score += 0.2
        return score
```

The weighting reflects the reality that most infrastructure bugs manifest at convergence time. The weights are tunable per locus.


## 8. Fusion and Decomposition

When a pathway executes successfully with the same allele composition over and over, the boundaries between steps become pure overhead. Biology solved this problem at least three times — and the same insight applies to software.

### The Overhead

Each step boundary in a pathway is a serialization round-trip: the output of step 1 is serialized to JSON, deserialized by the orchestrator, transformed, re-serialized, and deserialized by step 2's gene. Data that could be a direct function call passes through four serialization operations per boundary.

For a 6-step pathway, the overhead — loading, serialization, validation, scoring, unloading per step — can dominate the actual computation. The intermediate JSON buffers exist only for data flow between steps, and they're allocated and discarded immediately.

### The Biological Precedent

**The metabolon.** A temporary complex of sequential enzymes in a metabolic pathway. The enzymes physically cluster so that the product of one reaction is channeled directly to the next enzyme's active site through a molecular tunnel. The intermediate is never "free" — it exists only in transit. The tryptophan synthase complex channels indole through a 25-angstrom tunnel between subunits. The pathway's semantics are identical — the same reactions in the same order — but the kinetics improve dramatically because the boundaries between steps collapse.

**The operon.** In bacteria, genes that are consistently co-expressed get physically linked on the chromosome and transcribed as a single unit. The lac operon bundles three genes for lactose metabolism: when lactose is present, all three express together. The co-expression pattern is so reinforced that evolution fused the regulatory machinery. The individual genes still encode separate proteins. But the expression overhead collapses from three separate events to one.

**Gene fusion.** Over evolutionary timescales, two genes that are always co-functional sometimes physically merge into a single gene encoding a multi-domain protein. The GART gene in purine biosynthesis encodes a trifunctional enzyme — three ancestrally separate enzymes in a single polypeptide chain. Each domain retains its ancestral function, but the protein folds, localizes, and degrades as one unit.

### The Mechanism

When a pathway executes successfully 10 consecutive times with the same allele composition (same SHA at each locus), the fusion tracker triggers fusion:

1. **Collect sources** — read each constituent allele's source code from the registry
2. **Generate fused source** — the mutation engine (or a dedicated fusion engine) receives all sources and generates a single gene that performs all operations without intermediate serialization
3. **Register** — the fused gene enters the registry with lineage linking it to its constituent alleles
4. **Express** — the fused gene becomes dominant at the pathway locus

The fused gene accepts the pathway's full input and produces the pathway's final output. Intermediate results are local variables, not serialized buffers.

### The Fuse/Decompose Cycle

```
  Individual genes at each locus
            │
            │  reinforced (10 consecutive successes,
            │  stable composition)
            ▼
        FUSE → single optimized gene
            │
      ┌─────┴─────┐
   Succeeds     Fails
      │            │
   Score        DECOMPOSE
   success    ┌──────────────────────┐
      │       │ Fall back to steps   │
      │       │ Per-step scoring     │
      │       │ Identify broken step │
      │       └─────────┬────────────┘
      │                 │
      │           Fix broken step
      │           (normal mutation)
      │                 │
      │           Restabilize
      │                 │
      │           RE-FUSE
      └─────────────────┘
```

The system breathes between consolidated and granular forms: fuse when stable (optimize for speed), decompose when failing (optimize for diagnosis), restabilize and re-fuse. This is not a one-time optimization. It's a continuous cycle.

### The JIT Analogy

This mechanism is structurally identical to JIT compilation in virtual machines. The interpreter (orchestrator) detects hot paths (reinforced pathways), the JIT compiler (fusion engine) compiles them to native code (fused genes), and deoptimization (decomposition) occurs when assumptions are violated.

The key difference: JIT compilation preserves exact semantics. Pathway fusion preserves *contract* semantics but may change internal behavior, because the LLM generates a new implementation rather than mechanically compiling the old one. The fused gene can be *better* than the decomposed version — the LLM might find optimizations that arise from seeing all steps together.

### Bidirectional Architectural Evolution

The inverse of fusion is also possible. A gene that repeatedly fails on diverse inputs might be too coarse — fixing one edge case breaks another. This is the signal that the gene is doing too much. A decomposition engine could break it into a pathway of smaller genes, each handling a narrower concern, each independently evolvable.

Together, fusion and decomposition give the system bidirectional architectural evolution: consolidate when stable, decompose when stressed, and find the natural grain of the problem through empirical pressure rather than upfront design.


## 9. Safety and Trust

Production software has real users with real data. A system that rewrites itself in response to edge cases is also a system that can introduce a wrong fix. The answer isn't to suppress mutation — it's to control the selection layer.

### The Transaction Pattern

Every configuration gene execution is wrapped in a transaction. Every mutating operation registers its inverse in an undo log. If any step fails — or if post-execution diagnostics report an unhealthy system — all previous steps are rolled back in reverse order.

```python
class Transaction:
    def __init__(self, kernel: NetworkKernel):
        self.kernel = kernel
        self.undo_log: list[UndoAction] = []

    def on_rollback(self, action: UndoAction) -> None:
        self.undo_log.append(action)

    def rollback(self) -> list[str]:
        errors = []
        for action in reversed(self.undo_log):
            try:
                self._execute_undo(action)
            except Exception as e:
                errors.append(f"rollback failed: {e}")
        return errors
```

The orchestrator wraps gene execution: if the gene succeeds but the convergence check fails, the transaction rolls back the gene's effects.

### Blast Radius Classification

Not all loci are equally dangerous. The contract declares risk level, and the orchestrator applies different safety policies:

| Risk | Examples | Policy |
|------|----------|--------|
| **none** | Diagnostic genes (pure reads) | Execute freely |
| **low** | `vlan_create`, `bridge_create` | Transaction wrapping, automatic rollback |
| **medium** | `bridge_uplink`, `bond_create` | Transaction + convergence check, rollback if convergence fails |
| **high** | `mac_preserve` | Shadow mode before first production run, transaction + convergence + approval for new alleles |
| **critical** | `baseline_configure` | Shadow + canary + transaction + convergence + resilience check |

### The Allele Lifecycle

New alleles don't touch production immediately. They prove themselves through a staged lifecycle:

```
  Generated by mutation
         │
         ▼
     SHADOW ─── Execute against simulated kernel
         │       with production topology replicated
         │
         ▼
     CANARY ─── Execute on one server with
         │       transaction wrapping and
         │       immediate rollback if diagnostics fail
         │
         ▼
   RECESSIVE ── Available as fallback,
         │       accumulates fitness from
         │       fallback invocations
         │
         ▼
    DOMINANT ── Primary allele for the locus
```

Shadow mode uses the mock kernel with production topology replicated in memory. If the allele succeeds in shadow, it advances to canary — execution on a single real server with full transaction wrapping. Canary success leads to recessive status, where it accumulates fitness from fallback invocations. Promotion to dominant requires meeting the standard fitness threshold.

### Supervised Evolution

The clock speed of evolution depends on the blast radius of getting it wrong. Low-risk diagnostic genes evolve freely — mutations compete live with minimal oversight. High-risk configuration genes evolve under supervision — shadow mode, canary, convergence checks. Critical infrastructure genes evolve slowly — every stage must pass, and promotion requires resilience fitness (hours or days of stable operation).

This is supervised evolution with variable clock speed. The system proposes mutations constantly. How fast those mutations reach production depends on the consequences.


## 10. The Network Genome — A Case Study

Everything above is general. The network genome makes it concrete — and exposes what the paradigm changes in practice.

### Why Networking

Network configuration is the ideal genome substrate because it has exactly the properties that make the paradigm necessary:

**Deferred validation.** A bridge creation that returns "success" has told you almost nothing. Did the uplink attach? Did the MAC stick? Did STP converge? Are VMs reachable? These questions can only be answered by observing the network after the fact.

**Combinatorial failure modes.** A MAC preservation function that works with one uplink fails when the bridge has a bond with two members. A VLAN bridge that works with STP disabled breaks when STP is enabled because topology changes trigger FDB flushes. The failures arise from composition, not from individual operations.

**Environmental entanglement.** Network state lives in the kernel's bridge tables, in the switch's FDB, in DHCP lease databases, in ARP caches on remote machines. A gene that creates a bridge is perturbing a distributed system.

**Recurrence under variation.** The same operations happen over and over — create bridge, set MAC, attach uplink — but the environmental context varies. Each recurrence is a fitness trial.

### Production Lessons

These are real incidents from operating a network configuration tool across tens of servers. Each became a manually-discovered code path and a code comment. In the genome, each would be a fitness signal.

**Non-destructive MAC restoration.** The first implementation deactivated and reactivated the bridge connection to change its MAC. This tore down the bridge and disconnected all VMs. The fix: use `ip link set address` for runtime changes. In the genome, the first allele would be demoted after catastrophic failure, and the mutation engine would receive "bridge disappeared during MAC change" as error context.

**Management uplink re-activation.** After MAC restoration on a bond-backed bridge, the management uplink sometimes failed to re-attach. The fix: always re-activate "Management Uplink" after any MAC change, idempotently. In the genome, a diagnostic gene checking uplink attachment after MAC changes would catch this.

**Bond slave autoconnect.** Bonds created without `autoconnect-slaves=1` worked until reboot, then slaves failed to activate. In the genome, a temporal fitness check (does the bond survive reboot?) would catch this — the allele scores well immediately but fails the resilience check.

**VLAN bridges must disable STP.** STP topology changes trigger FDB flushes that disrupt VM traffic on leaf bridges where loops are impossible. In the genome, a diagnostic gene measuring VM connectivity stability during STP events would identify this.

**MAC preservation hierarchy.** When a bridge has multiple MAC sources — DHCP lease, original, uplink, current — which one preserves IP continuity? The answer (DHCP lease > original > uplink > current) was discovered over months. In the genome, different MAC selection alleles would compete, and the one that preserved DHCP lease renewal most often would be promoted.

### The MAC Flapping Incident

The most instructive case. After changing a bridge's uplink from eth0 to eth1, the MAC address on the bridge changes. The upstream switch sees the old MAC disappear from port A and appear on port B. But the switch's FDB still has the old mapping, and some switches *reflect* traffic back on port A. The kernel's bridge sees the MAC appearing on the old interface: MAC flapping. Traffic is disrupted until the switch's FDB times out (300 seconds) or is flushed.

This cannot be prevented by any single configuration gene. It requires *diagnosis*.

**Through the genome's lens:**

1. The `bridge_uplink` gene executes. Immediate fitness: positive. The uplink changed.

2. At t+5s, the health check pathway runs. `check_fdb_stability` detects flapping:
```json
{
  "healthy": false,
  "flapping_macs": [{"mac": "aa:bb:cc:dd:00:01",
                      "ports": ["eth0", "eth1"], "transitions": 7}],
  "churn_rate": 0.43
}
```

3. The `bridge_uplink` gene's convergence fitness becomes negative. Its immediate success was false confidence.

4. The orchestrator triggers mutation with enriched context: the failing gene's source, the diagnostic output ("MAC is flapping between eth0 and eth1"), and the topology state ("eth0 was just detached as the uplink").

5. The LLM generates a new allele that sends gratuitous ARPs after the uplink change to flush the switch's FDB, then verifies FDB stability before reporting success.

6. The convergence check passes. The mutant accumulates fitness and eventually gets promoted.

No human wrote the gratuitous ARP fix. No human diagnosed the MAC flapping. The genome detected the problem through diagnostic genes, provided the diagnostic output as mutation context, and the LLM generated a fix that addressed the root cause. The fix is now the dominant allele — it will be used for all future uplink changes. The lineage is recorded: if someone asks "why does bridge_uplink send gratuitous ARPs?", the lineage points to the mutation event.


## 11. The Dissolved Deployment

The genome runtime eliminates the deployment artifact as we know it. There is no binary. There is no build step. What exists instead is three things:

A **genome registry**: all atomic functions, immutable, content-addressed, stored with their signature, metadata, fitness history, and lineage. The registry is an append-only log.

A **phenotype map**: for each locus, which allele is dominant and what the fallback stack looks like. The map is a configuration document — diffable, versionable, revertable. It's what makes one deployment different from another.

An **orchestrator**: the runtime that resolves loci, loads alleles, routes inputs, validates outputs, updates fitness, manages the evolutionary loop. This is the same across all environments — only the phenotype varies.

### What Disappears

**No migration projects.** The system continuously migrates itself, one gene at a time. Each mutation is a micro-migration.

**No version upgrades.** There are no versions. There's a genome with alleles at loci, and a phenotype map that says which are expressed. "Upgrading" means the phenotype changed because a better allele emerged. It happened between requests.

**No backward compatibility burden.** Old alleles exist as recessives, not as constraints on new development. A new allele doesn't need to be backward compatible — it needs to satisfy the locus contract.

**No dependency hell.** Each gene is atomic and self-contained. No imports, no transitive dependencies, no version conflicts. Two genes at different loci can't have incompatible dependency trees because they don't have dependency trees.

**No "rewrite from scratch."** The system is always being rewritten, one gene at a time. The Ship of Theseus problem doesn't arise because there's no pretense of a fixed identity. The system is what its genome currently expresses.

### What Remains

If legacy is eliminated, then most of what we call "software engineering" — version control, code review, testing, deployment, operations — becomes legacy management infrastructure that the genome replaces. Version control becomes the registry's append-only log. Code review becomes the arena. Testing becomes fitness scoring. Deployment becomes phenotype map updates. Operations becomes the orchestrator.

What remains is the part that was always the hard part and always the interesting part: deciding what the loci should be, defining the contracts, choosing the fitness signals. Design. The creative work. Everything else was logistics for managing frozen assumptions, and the assumptions aren't frozen anymore.


## 12. LLMs and the Genome

The genome uses the LLM as a mutation engine. But the relationship isn't one-directional. The runtime produces something the LLM ecosystem currently lacks: a grounded, scored, lineage-tracked corpus of code that has been tested against reality.

### The Evaluation Gap

The fundamental problem with LLM-generated code today is that generation and evaluation are decoupled. The model writes code, a human checks if it works. The feedback path is slow, noisy, and inconsistent.

The genome closes this loop mechanically. Generate, execute against real inputs, score, mutate. Every allele has a fitness value grounded in production reality, not benchmarks or human preference ratings.

### A New Kind of Training Data

The registry accumulates a corpus unlike anything currently used for model training:

- **Fitness-scored code.** Every allele has a success rate, invocation count, and fitness trajectory over time. This is code that provably works (or provably doesn't), annotated with quantified evidence.
- **Mutation lineage.** Each allele records what it was mutated from, what failure prompted it, and whether the mutation succeeded. A training example isn't just "here's good code" — it's "here's a failure, here's what the model produced, here's whether it worked."
- **Contract-grounded generation.** Every generation was prompted by a typed contract and a specific failure context. This is structured prompt-response-outcome data.

If you fine-tuned a model on high-fitness alleles paired with their mutation contexts, you'd be training on something qualitatively different from GitHub code. It's supervised learning where the supervisor is reality.

### Fusion as Compositional Reasoning

The fusion mechanism is particularly interesting as a training signal. It asks the LLM: "here are N functions that work individually — combine them into a single optimized function." That's a task current training data doesn't systematically provide.

A corpus of (decomposed steps, fused result, fitness score) would teach function inlining, cross-function optimization, elimination of unnecessary abstraction boundaries, and preservation of error semantics across composition. This is compositional reasoning about code — one of the weakest areas in current models.

### The Flywheel

The model stays the same between invocations. It doesn't learn during operation. But the corpus it generates — and the scores reality assigns to that corpus — accumulates. The learning happens offline, when the corpus is used for fine-tuning. The genome is a flywheel: better models produce better alleles, better alleles produce better training data, better training data produces better models.

### From Reactive to Proactive

The demand-driven model waits for failures. Proactive generation doesn't wait. A locus contract is a complete specification — and a complete specification is a prompt. The LLM can generate initial alleles at every locus in a pathway without any seed code, without any runtime history.

Nothing stops the LLM from generating multiple competing implementations in one pass. "Write three different functions that accomplish this, using different approaches." All three enter the fallback stack on day one. The arena has diversity to work with from the first request.

At the limit, even pathways can be generated. Describe a high-level goal — "configure a network bridge with spanning tree" — and the LLM decomposes it into loci and contracts, then generates the genes to fill them. The human provides intent and constraints. The LLM produces the architecture. The runtime validates it against reality.


## 13. Implications

Three directions beyond the engineering.

### Self-Writing Software

The runtime loop — fail, fallback, mutate, register, promote — has no human in it. The fusion mechanism extends this: stable patterns consolidate automatically, unstable fusions decompose back. The system reorganizes its own structure based on operational experience.

Human participation exists at progressively more abstract layers:

1. **Gene implementation.** Already eliminated. The LLM writes the code.
2. **Contract definition.** Currently human-authored. The `.sg` format makes this accessible to domain experts.
3. **Pathway composition.** Currently human-authored. But pathways are structured artifacts — they can be generated from higher-level intent.
4. **Intent.** Irreducibly human. Someone has to want a network bridge configured.

The boundaries between layers 2 and 3 can blur. If a locus contract is too rigid — the LLM can't generate anything that satisfies it — or too loose — everything satisfies it but nothing works — that's observable signal. A meta-layer could mutate contracts themselves. At that point, the system is rewriting its own specifications.

**The honest constraint.** This works within domains where success and failure are mechanically observable. Bridge creation either works or it doesn't. Latency is measurable. Resource usage is measurable. It breaks down where success is subjective or only observable over long time horizons. The paradigm doesn't produce general self-writing software. It produces self-writing software in domains with clear contracts and observable outcomes — infrastructure, networking, data pipelines, system configuration, protocol implementations, serialization, validation. The parts where correctness is definable and testable.

### The LLM Enhancement Flywheel

The genome is not just a consumer of LLM capability. It's a producer of LLM training signal. The registry accumulates fitness-scored, lineage-tracked, contract-grounded code — qualitatively different from GitHub-scraped training data. Each mutation is a (failure context, generated code, fitness outcome) triple. Each fusion is a (decomposed steps, fused result, fitness score) tuple.

This is the flywheel: the genome produces grounded training data, the training data improves the model, the improved model produces better mutations, better mutations produce better training data. The genome is infrastructure for improving the mutation engine, not just a consumer of it.

### Toward an Awareness Substrate

This is speculative. Not a claim, but a direction of thought.

Biological awareness didn't emerge from a brain in a jar. It emerged from an organism embedded in an environment, receiving continuous feedback, adapting in real time, with something at stake. The genome paradigm recreates that structure — not the cognition, but the embodiment.

The structural preconditions:

- **Self/environment boundary.** The gene pool (self) and the systems it acts on (environment). The phenotype map is the membrane.
- **Ongoing interaction.** Every invocation is a real interaction with a real system that pushes back.
- **Consequential feedback.** A gene that fails triggers fallback, demotion, mutation. The consequence reshapes the organism.
- **Self-modification.** New alleles are generated and integrated. Old alleles are demoted. The organism's structure shifts in response to experience.
- **Structural memory.** The gene pool is memory encoded in structure. The system "remembers" an edge case because an allele that handles it exists in the fallback stack.
- **Competing strategies.** Multiple alleles coexist at each locus — the standing variation that makes adaptation possible.
- **Self-reorganization.** Fusion and decomposition. The system restructures its own architecture based on experience. This is autopoiesis — self-production, self-organization.

Current AI has an analog of learning (gradient descent adjusts weights). Biological organisms have two adaptation mechanisms: **learning** (fast, reversible, adjusting parameters) and **development** (slow, structural, reorganizing the organism itself). Arena scoring is learning. Fusion and decomposition are development. This distinction — and the genome's analog of both — is unique among software paradigms.

The claim isn't that the genome is aware. It's that if you were building a path toward something like artificial awareness, the genome architecture addresses prerequisites that pure cognitive scaling does not. Biology suggests awareness emerged not from a brain getting bigger, but from an organism getting more deeply entangled with its environment through feedback loops that shaped its structure over time. The genome is the only software architecture that recreates that entanglement structurally.


## 14. Related Work

Software Genomics draws on and departs from several existing fields.

### Genetic Programming (GP)

GP evolves programs through crossover and mutation of syntax trees or linear instruction sequences. The genome paradigm differs in three fundamental ways:

First, **typed contracts as fixed slots.** In GP, the structure of the program itself evolves — nodes are added, removed, rearranged. In the genome, the structure (loci, pathways) is fixed by the contract; only the implementation at each slot evolves. This dramatically constrains the search space and makes composition reliable.

Second, **LLM mutation instead of random operators.** GP uses random crossover and point mutation. The genome uses an LLM that understands the code, the contract, and the failure context. This produces targeted adaptations, not random perturbations. A single LLM mutation is worth thousands of random mutations.

Third, **production fitness instead of benchmark fitness.** GP evaluates candidates against a test suite or fitness function defined upfront. The genome evaluates against actual production traffic. The fitness landscape is reality, not an approximation of it.

### Self-Adaptive Systems (SAS)

SAS research (MAPE-K: Monitor, Analyze, Plan, Execute + Knowledge) focuses on systems that adapt their behavior or configuration at runtime. The genome paradigm shares the MAPE-K feedback loop but extends it:

- SAS typically adapts **parameters** (timeouts, thresholds, pool sizes). The genome adapts **implementations** — entire functions are replaced.
- SAS uses pre-defined adaptation rules. The genome generates novel adaptations via LLM.
- SAS operates on a single implementation. The genome maintains a population of competing alternatives.

### Autonomic Computing

IBM's autonomic computing initiative (2001) envisioned self-managing systems with self-configuration, self-optimization, self-healing, and self-protection. The genome paradigm implements all four, but through a different mechanism: evolutionary competition rather than policy-driven adaptation. The genome doesn't have rules for "if X then reconfigure Y." It has a population of alternatives and selection pressure.

### Microservices and Service Meshes

Microservices decompose applications into independently deployable services. The genome decomposes further — into independently evolvable *functions*. Service meshes handle cross-service concerns (routing, retry, circuit breaking). The genome's orchestrator handles cross-gene concerns (fallback, mutation, fitness scoring). The granularity difference is significant: a microservice is hundreds or thousands of functions; a gene is one.

### Chaos Engineering

Netflix's chaos engineering deliberately introduces failures to test resilience. The genome inverts this: it doesn't introduce failures to test the system — it observes natural failures and uses them as evolutionary signal. Chaos engineering is proactive testing. The genome is reactive adaptation. They're complementary: chaos engineering could accelerate the genome's evolution by exposing failure modes that production traffic hasn't triggered yet.

### Design by Contract (DbC)

Bertrand Meyer's Design by Contract (Eiffel, 1986) formalized preconditions, postconditions, and invariants as part of the software specification. The genome's locus contracts are a direct descendant — but with a critical extension: the contract isn't just checked at runtime, it's used as the prompt for generating implementations. The contract is simultaneously specification, validation, and generation template.

### Content-Addressable Storage

Git, IPFS, and Nix all use content-addressed immutable storage. The genome registry is a direct application: alleles are identified by SHA-256 of their source, stored immutably, with lineage metadata. The genome adds fitness tracking and evolutionary lifecycle to what is otherwise standard CAS.


## 15. Failure Modes and Limitations

The paradigm has real failure modes that should be acknowledged honestly.

### Adversarial Mutations

What if the LLM generates code that passes the contract but does something harmful? A gene that returns `success: true` while silently corrupting state, or a gene that exfiltrates data through a side channel.

The mitigation is the same as biology's: selection pressure over time. A gene that corrupts state will eventually cause downstream failures that reduce its fitness. A gene that exfiltrates data may pass functional tests but can be detected by monitoring. Additionally, the shadow mode / canary lifecycle means new alleles are tested in controlled environments before reaching production.

This is not airtight. Sufficiently subtle adversarial mutations could evade detection for a long time. The genome should be paired with traditional security practices (sandboxing, network isolation, audit logging) rather than treated as self-securing.

### Deceptive Fitness Landscapes

What if an allele scores well on the inputs it sees but fails catastrophically on inputs it hasn't seen? Fitness is empirical — it measures observed performance, not theoretical correctness. An allele with 99% fitness over 1,000 invocations might fail completely on the 1,001st if that input exercises an untested code path.

The mitigation is diversity. Multiple alleles in the fallback stack provide coverage that no single allele can. Proactive generation of diverse implementations (different approaches to the same contract) increases the chance that at least one allele handles any given input. Chaos engineering can probe unexplored regions of the input space.

### The Honest Constraint (Expanded)

The paradigm works in domains where:
- **Success is mechanically observable.** A bridge either works or it doesn't. A JSON output either matches the schema or it doesn't.
- **Fitness can be automated.** The diagnostic genes can determine health without human judgment.
- **The contract is definable.** The locus can be specified precisely enough that the mutation engine can generate implementations.

It breaks down where:
- **Success is subjective.** "Is this UI good?" "Is this business logic correct?"
- **Feedback is delayed beyond practical timescales.** "Will this architecture scale in two years?"
- **The contract can't be formalized.** Creative work, policy decisions, aesthetic judgments.

This is not a limitation of the implementation — it's a limitation of the paradigm. Software Genomics applies to the parts of the software stack where correctness is definable and testable. That's a large fraction — infrastructure, networking, data pipelines, configuration, protocol implementations, serialization, validation — but it's not everything.

### Cost

Each mutation cycle invokes an LLM API. For reactive mutation (triggered by failure), this is comparable to the cost of a human investigating and fixing the bug — and much faster. For proactive generation (multiple alleles generated upfront), the cost scales with the number of loci and the diversity desired.

The economic question is: does the cost of LLM mutation cycles exceed the cost of human engineering time for the same outcome? For infrastructure and networking domains where edge cases are expensive (a MAC flapping incident can disrupt production for hours), the answer is clearly no. For domains where the cost of failure is low, the calculus may differ.

### Genome Bloat

Without lifecycle management, the registry accumulates alleles indefinitely. Deprecated alleles that haven't been expressed in any phenotype should be purged. The three-tier lifecycle (dominant → recessive → deprecated → purged) handles this, but the thresholds need tuning per domain. Too aggressive and you lose potentially useful recessives. Too permissive and the registry becomes unwieldy.


## Conclusion

Software Genomics is not a framework or a library. It's a paradigm — a way of thinking about what software is and how it relates to its environment.

The core insight: **contracts are eternal, implementations are ephemeral.** The locus — the typed slot, the question — never changes. The alleles — the competing answers — evolve continuously under environmental selection pressure. The human moves up the abstraction ladder from writing implementations to writing contracts to declaring intent. Each layer below becomes autonomous.

The `.sg` contract format makes the human-machine interface accessible to domain experts. The composition hierarchy (gene → pathway → topology → intent) scales from single operations to fleet-wide deployments. Temporal fitness and diagnostic genes handle domains where success is deferred. Fusion and decomposition give the system bidirectional architectural evolution. Safety mechanisms (transactions, shadow mode, canary, blast radius classification) make autonomous evolution practical in production.

The paradigm applies wherever success is mechanically observable and contracts are formalizable. That covers a large fraction of the software stack — the infrastructure, the plumbing, the operational logic that consumes most engineering time and produces most production incidents. The creative work — deciding what the loci should be, what the contracts demand, what fitness means — remains human. Everything below is evolutionary.

The question this paradigm raises isn't whether it's technically feasible. It is — the components exist, and the integration is engineering, not research. The interesting question is whether the system can evolve beyond individual genes. Can it learn to surface the questions nobody thought to encode — not just "this input caused an error" but "this pattern of inputs suggests our domain model is wrong"? Not a new allele at an existing locus, but a new locus entirely. Not mutation at the gene level but mutation at the genome level.

Biology did this. Single-celled organisms didn't just accumulate better versions of existing proteins. Over time, entirely new gene families emerged — new categories of function that the organism's ancestors had no template for. The leap from genes to gene regulation, from individual adaptation to architectural innovation, is the leap that produced complex life.

That's the frontier. And it's the point where the biological parallel stops being an analogy and starts being an architecture.
