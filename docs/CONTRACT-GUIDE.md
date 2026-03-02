# Contract Authoring Guide

Contracts define the typed slots (loci) that gene implementations compete to fill. They're written in the `.sg` format by domain experts — no Python required. Contracts are the invariants of the evolutionary system: implementations mutate freely, but contracts stay fixed.

For plugin development, see [PLUGIN-GUIDE.md](PLUGIN-GUIDE.md). For runtime operations, see [HANDBOOK.md](HANDBOOK.md).

---

## 1. The `.sg` Format

The `.sg` format uses verb-based section names that communicate each section's role in the evolutionary loop:

```
gene <name> [for <domain>]
  is <family>
  risk <level>

  does:
    <prose description>

  takes:
    <field>  <type>  "<description>"

  gives:
    <field>  <type>  "<description>"

  before:
    - <precondition>

  after:
    - <postcondition>

  fails when:
    - <failure mode> -> <consequence>

  verify:
    <diagnostic_locus> <param>={ref}
    within <duration>
```

Every section starts with a verb — `does`, `takes`, `gives`, `fails when` — because contracts describe behavior, not structure.

---

## 2. Gene Contracts

### Section Reference

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

### Field Types

| Type | Description | Example |
|------|-------------|---------|
| `string` | Text value | `name string` |
| `bool` | Boolean | `enabled bool` |
| `int` | Integer | `count int` |
| `float` | Floating point | `ratio float` |
| `string[]` | Array of strings | `items string[]` |
| `int[]` | Array of integers | `ids int[]` |
| `type?` | Optional (may be absent) | `error string?` |
| `type = val` | Default value (implies optional) | `enabled bool = false` |

### The `does:` Section

Free-form prose describing what the gene does. This text is directly injected into LLM mutation prompts, so write it as if explaining the task to a developer:

```
does:
  Fetch CSV data from a URL and load it into a database table.
  Parses the CSV response, validates column names against the table
  schema, and writes records as rows. Tracks the table as a managed
  resource for rollback.
```

Good `does:` sections:
- Explain *what* and *why*, not just *how*
- Mention domain conventions the gene should follow
- Call out edge cases the LLM should handle

### The `takes:` and `gives:` Sections

Define input and output schemas:

```
takes:
  url         string  "URL to fetch CSV data from"
  connection  string  "Database connection name"
  table       string  "Target table name"

gives:
  success       bool    "Whether the ingest completed"
  rows_written  int     "Number of rows written"
  error         string? "Error message on failure"
```

Every gene must return at least `success: bool` in its output. Use `string?` (optional) for fields only present in certain cases.

### Preconditions and Postconditions

`before:` lists what must be true before execution. `after:` lists what must be true after successful execution:

```
before:
  - Target table exists with a compatible schema
  - URL is reachable and returns CSV data

after:
  - All CSV rows are written to the target table
  - Table row count increased by the number of CSV rows
```

These are injected into mutation prompts and used by the conformance tester.

### Failure Modes

`fails when:` lists known failure scenarios and their expected behavior:

```
fails when:
  - URL unreachable -> success=false
  - table does not exist -> error
  - schema mismatch -> success=false
```

The `-> consequence` part tells the mutation engine what the gene should return when this failure occurs.

---

## 3. Gene Families

### Configuration Genes (`is configuration`)

Configuration genes **act** on the environment — they create, modify, or delete resources. They have blast radius, transactions, and rollback capability.

```
gene ingest_csv_to_table for data
  is configuration
  risk low
  ...
```

### Diagnostic Genes (`is diagnostic`)

Diagnostic genes **observe** the environment — they read state and report health. They always have `risk none`, use `unhealthy when` instead of `fails when`, and declare `feeds` to provide fitness feedback:

```
gene check_nulls for data
  is diagnostic
  risk none

  does:
    Check for null values in a specific column of a database table.

  ...

  unhealthy when:
    - null ratio is above 10%

  feeds:
    ingest_csv_to_table convergence
```

### The Diagnostic-Configuration Feedback Loop

Diagnostics produce the fitness signal for configuration genes. Without diagnostics, configuration genes are only scored on immediate success/failure. With diagnostics:

- **Immediate fitness (30%)** — did the gene succeed right now?
- **Convergence fitness (50%)** — does the result hold after the `within` window?
- **Resilience fitness (20%)** — does the result hold over hours?

A diagnostic gene that `feeds: create_widget convergence` means: "my health check result affects create_widget's convergence score."

---

## 4. Domain Annotation

The `for <domain>` clause binds a contract to a specific kernel domain:

```
gene bridge_create for network
  ...

gene ingest_csv_to_table for data
  ...
```

When loading contracts, the engine warns if a contract's domain doesn't match the active kernel's `domain_name()`. This prevents accidentally loading network contracts when running a data kernel.

Contracts without a `for` clause are domain-agnostic and work with any kernel.

---

## 5. Blast Radius Classification

The `risk` level controls safety behavior during gene execution:

| Risk Level | Transaction | Shadow Execution | When to Use |
|------------|-------------|------------------|-------------|
| `none` | No | No | Read-only operations, diagnostics |
| `low` | Yes | No | Reversible state changes |
| `medium` | Yes | No | State changes with limited scope |
| `high` | Yes | Yes (3 shadow runs) | Broad state changes |
| `critical` | Yes | Yes (3 shadow runs) | Irreversible or widely impactful changes |

**Shadow execution** means the gene is first tested against a mock kernel to verify it doesn't crash before running against the real system.

Classify risk based on your domain's impact model:
- **Data pipelines**: Writing records = `low`, dropping tables = `critical`
- **Network**: Creating a bridge = `low`, modifying routing = `high`
- **Infrastructure**: Starting a service = `medium`, modifying firewall rules = `critical`

---

## 6. The `feeds` Mechanism

The `feeds` section in diagnostic genes declares which configuration genes receive fitness feedback:

```
feeds:
  bridge_create convergence
  mac_preserve  convergence
```

Format: `<locus> <fitness_component>`

The fitness component is typically `convergence` (50% weight). When the diagnostic reports `healthy: false`, the target configuration gene's convergence fitness is penalized — even if the configuration gene "succeeded" at execution time.

This creates an evolutionary pressure toward implementations that produce *durable* results, not just immediate successes.

---

## 7. Verification Steps

The `verify` section declares diagnostic loci to run after execution:

```
verify:
  check_row_count connection={connection} table={table}
  check_nulls connection={connection} table={table} column={column}
  within 30s
```

- Each line names a diagnostic locus and binds parameters using `{reference}` substitution
- The `within` clause sets the convergence window — how long to wait before running the checks
- Verification results feed back into the gene's temporal fitness: a passing check at `t+30s` contributes convergence fitness (50% of the total score), while a failing check retroactively decays the gene's fitness even though the immediate execution succeeded

---

## 8. Pathway Contracts

Pathways compose genes into multi-step operations.

### Basic Pathway

```
pathway ingest_and_validate for data
  risk low

  does:
    Ingest CSV data then validate quality.

  takes:
    url         string  "URL to fetch CSV data from"
    connection  string  "Database connection name"
    table       string  "Target table name"
    column      string  "Column to check for nulls"

  steps:
    1. ingest_csv_to_table
         url = {url}
         connection = {connection}
         table = {table}

    2. check_row_count
         connection = {connection}
         table = {table}

    3. check_nulls
         connection = {connection}
         table = {table}
         column = {column}

  on failure:
    rollback all
```

### Step Binding

`{reference}` substitutes pathway input parameters into step inputs:

```
steps:
  1. create_widget
       name = {widget_name}
       config = {widget_config}
```

### Pathway Composition

Prefix a step name with `->` to run another pathway instead of a gene:

```
steps:
  1. -> provision_management_bridge
       bridge_name = {bridge_name}
       interfaces = {interfaces}
```

### Iteration

Repeat a step for each element:

```
steps:
  3. for vlan in {vlans}:
       vlan_create
         parent = {bond_name}
         vlan_id = {vlan}
```

### Conditionals

Branch based on a previous step's output:

```
steps:
  1. check_widget_health
       name = {name}

  2. when step 1.healthy:
       false -> repair_widget
         name = {name}
```

### Dependencies

Declare ordering constraints:

```
requires:
  step 2 needs step 1
  step 3 needs step 2
```

### Failure Handling

```
on failure:
  rollback all       # undo all completed steps in reverse
  report partial     # report what succeeded, continue
```

---

## 9. Topology Contracts

Topologies declare desired state without specifying steps. The engine figures out which pathways to run.

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

Each `has` block declares a resource with a type (`is bridge`, `is bond`) and properties. Resource types are mapped to pathways by the kernel's `resource_mappers()` method.

---

## 10. Cross-Domain Examples

### Network Domain — Configuration Gene

```
gene bridge_create for network
  is configuration
  risk low

  does:
    Create a Linux bridge with the given interfaces.

  takes:
    bridge_name  string    "Name for the bridge"
    interfaces   string[]  "Physical interfaces to attach"

  gives:
    success            bool      "Whether the bridge was created"
    resources_created  string[]  "NM connections created"
    error              string?   "Error message on failure"

  before:
    - Bridge with this name does not already exist

  after:
    - Bridge exists and is in UP state

  fails when:
    - bridge already exists -> success=false
    - interface not found -> success=false

  verify:
    check_connectivity bridge_name={bridge_name}
    within 30s
```

### Data Domain — Configuration Gene

```
gene ingest_csv_to_table for data
  is configuration
  risk low

  does:
    Fetch CSV data from a URL and load it into a database table.

  takes:
    url         string  "URL to fetch CSV data from"
    connection  string  "Database connection name"
    table       string  "Target table name"

  gives:
    success       bool    "Whether the ingest completed"
    rows_written  int     "Number of rows written"
    error         string? "Error message on failure"

  before:
    - Target table exists with a compatible schema

  after:
    - All CSV rows are written to the target table

  fails when:
    - URL unreachable -> success=false
    - table does not exist -> error

  verify:
    check_row_count connection={connection} table={table}
    within 30s
```

### Data Domain — Diagnostic Gene

```
gene check_nulls for data
  is diagnostic
  risk none

  does:
    Check for null values in a specific column.

  takes:
    connection  string  "Database connection name"
    table       string  "Table to check"
    column      string  "Column to check for nulls"

  gives:
    success     bool    "Whether the check completed"
    healthy     bool    "Whether null ratio is acceptable"
    null_count  int     "Number of null values"
    total_rows  int     "Total number of rows"
    null_ratio  float   "Ratio of nulls to total"
    error       string? "Error message on failure"

  unhealthy when:
    - null ratio is above 10%

  feeds:
    ingest_csv_to_table convergence

  fails when:
    - table does not exist -> success=false
```

The pattern is the same across domains — only the operations, field names, and domain context change.
