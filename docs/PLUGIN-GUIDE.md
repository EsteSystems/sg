# Plugin Developer Guide

How to build a domain plugin for Software Genomics.

A plugin extends the engine with a new domain — network configuration, data pipelines, monitoring, IoT, anything with state-changing operations that benefit from evolutionary adaptation. The engine provides the evolutionary loop (mutation, fitness, promotion); your plugin provides the domain operations that genes call.

For contract syntax reference, see [CONTRACT-GUIDE.md](CONTRACT-GUIDE.md). For runtime operations, see [HANDBOOK.md](HANDBOOK.md).

---

## 1. Plugin Structure

Every plugin follows this layout:

```
plugins/my_domain/
├── pyproject.toml              # Package metadata + entry-point registration
├── sg_my_domain/               # Main Python package
│   ├── __init__.py             # Exports kernel classes + path helpers
│   ├── kernel.py               # Abstract kernel interface (extends Kernel)
│   └── mock.py                 # In-memory mock kernel for dev/test
├── contracts/                  # .sg contract definitions
│   └── genes/                  # Gene contracts
├── genes/                      # Seed gene implementations
└── fixtures/                   # Test fixtures (mock mutation outputs)
```

Generate this structure automatically:

```bash
sg new-plugin my_domain
```

See the **data pipeline** plugin at `plugins/data/` for a minimal working example, or the **network** plugin at `plugins/network/` for a full production implementation.

---

## 2. Kernel Implementation

Your kernel is the bridge between genes and your domain. It extends `sg.kernel.base.Kernel`:

```python
from abc import abstractmethod
from sg.kernel.base import Kernel, mutating


class MyDomainKernel(Kernel):
    """Abstract kernel interface for my domain."""

    # Domain-specific operations that genes will call via gene_sdk:

    @mutating(undo=lambda k, snap, name: k.delete_widget(name))
    @abstractmethod
    def create_widget(self, name: str, config: dict) -> dict:
        """Create a widget. Returns its properties."""
        ...

    @abstractmethod
    def delete_widget(self, name: str) -> None:
        """Delete a widget by name."""
        ...

    @abstractmethod
    def get_widget(self, name: str) -> dict | None:
        """Get widget state. Returns None if not found."""
        ...
```

### Required Base Methods

Every kernel must implement these methods from the `Kernel` ABC:

| Method | Purpose |
|--------|---------|
| `reset()` | Clear all state. Called between test runs. |
| `track_resource(type, name)` | Track a created resource for cleanup/rollback. |
| `untrack_resource(type, name)` | Stop tracking a resource. |
| `tracked_resources()` | Return all tracked resources as `(type, name)` pairs. |

### Optional Overrides

| Method | Default | Purpose |
|--------|---------|---------|
| `delete_resource(type, name)` | Calls `untrack_resource` | Dispatch to type-specific delete operations |
| `describe_operations()` | `[]` | List operations for LLM mutation prompts |
| `mutation_prompt_context()` | `""` | Domain context for LLM prompts |
| `domain_name()` | `"generic"` | Short domain identifier |
| `resource_mappers()` | `{}` | Topology resource type mappers |
| `create_shadow()` | Raises `NotImplementedError` | Create a mock kernel for shadow execution |

### Self-Description for LLM Prompts

When the mutation engine generates or fixes a gene, it includes your kernel's self-description in the prompt. This is how the LLM knows what operations are available:

```python
def describe_operations(self) -> list[str]:
    return [
        "create_widget(name: str, config: dict) -> dict",
        "delete_widget(name: str) -> None",
        "get_widget(name: str) -> dict | None",
    ]

def mutation_prompt_context(self) -> str:
    return (
        "This gene operates on widgets. gene_sdk is a MyDomainKernel "
        "providing widget CRUD operations. Widgets have a name and a "
        "config dict. Always check if a widget exists before creating."
    )

def domain_name(self) -> str:
    return "my_domain"
```

---

## 3. The `@mutating` Decorator

Mark state-changing kernel methods with `@mutating` to enable automatic transaction recording and rollback. The engine's `SafeKernel` proxy uses this metadata to wrap calls.

### Basic Usage — Undo Only

```python
@mutating(undo=lambda k, snap, name, config: k.delete_widget(name))
@abstractmethod
def create_widget(self, name: str, config: dict) -> dict:
    ...
```

The `undo` function receives:
1. `k` — the kernel instance
2. `snap` — the snapshot value (None if no snapshot function)
3. The original method arguments (`name`, `config`)

### With Snapshot — Capture Pre-State

When the undo requires knowing the state *before* the mutation:

```python
@mutating(
    snapshot=lambda k, name: k.get_widget(name),
    undo=lambda k, snap, name: (
        k.create_widget(name, snap["config"]) if snap else None
    ),
)
@abstractmethod
def delete_widget(self, name: str) -> None:
    ...
```

The `snapshot` function is called *before* the method executes. Its return value is passed as `snap` to the undo function.

### Read-Only Methods

Methods that only read state (diagnostics, queries) do **not** need `@mutating`. They pass through the `SafeKernel` proxy unwrapped.

---

## 4. Mock Kernel

Every plugin needs a mock kernel for development and testing. It implements all abstract methods with in-memory state:

```python
from sg_my_domain.kernel import MyDomainKernel


class MockMyDomainKernel(MyDomainKernel):
    """In-memory mock for development and testing."""

    def __init__(self) -> None:
        self._widgets: dict[str, dict] = {}
        self._tracked: list[tuple[str, str]] = []

    def reset(self) -> None:
        self._widgets.clear()
        self._tracked.clear()

    def track_resource(self, resource_type: str, name: str) -> None:
        pair = (resource_type, name)
        if pair not in self._tracked:
            self._tracked.append(pair)

    def untrack_resource(self, resource_type: str, name: str) -> None:
        pair = (resource_type, name)
        if pair in self._tracked:
            self._tracked.remove(pair)

    def tracked_resources(self) -> list[tuple[str, str]]:
        return list(self._tracked)

    def create_shadow(self) -> MockMyDomainKernel:
        return MockMyDomainKernel()

    # Domain operations:

    def create_widget(self, name: str, config: dict) -> dict:
        if name in self._widgets:
            raise ValueError(f"widget '{name}' already exists")
        widget = {"name": name, "config": config, "status": "active"}
        self._widgets[name] = widget
        self.track_resource("widget", name)
        return widget

    def delete_widget(self, name: str) -> None:
        if name not in self._widgets:
            raise ValueError(f"widget '{name}' not found")
        del self._widgets[name]
        self.untrack_resource("widget", name)

    def get_widget(self, name: str) -> dict | None:
        return self._widgets.get(name)
```

### Test Setup Helpers

Add convenience methods to your mock kernel for setting up test state:

```python
def add_widget(self, name: str, config: dict) -> None:
    """Pre-populate a widget for testing."""
    self._widgets[name] = {"name": name, "config": config, "status": "active"}

def inject_failure(self, operation: str, message: str) -> None:
    """Make the next call to `operation` raise RuntimeError."""
    self._injected_failures[operation] = message
```

### Shadow Execution

The `create_shadow()` method returns a fresh mock instance used by the safety layer to test HIGH/CRITICAL risk alleles before live execution. Keep it simple — return a new empty mock:

```python
def create_shadow(self) -> MockMyDomainKernel:
    return MockMyDomainKernel()
```

---

## 5. Writing Contracts

Contracts define the typed slots (loci) that genes implement. They're written in the `.sg` format by domain experts — no Python required.

### Gene Contract

```
gene create_widget for my_domain
  is configuration
  risk low

  does:
    Create a widget with the given name and configuration.
    The widget is tracked as a managed resource.

  takes:
    name    string  "Widget name"
    config  string  "Widget configuration as JSON"

  gives:
    success  bool    "Whether the widget was created"
    widget   string? "Created widget properties as JSON"
    error    string? "Error message on failure"

  before:
    - Widget with this name does not exist

  after:
    - Widget exists and is in active state

  fails when:
    - widget already exists -> success=false

  verify:
    check_widget_health name={name}
    within 30s
```

Key points:
- **`for my_domain`** — binds this contract to your domain kernel
- **`is configuration`** — this gene acts on state (vs `is diagnostic` for read-only)
- **`risk`** — blast radius classification: `none`, `low`, `medium`, `high`, `critical`

See [CONTRACT-GUIDE.md](CONTRACT-GUIDE.md) for the complete syntax reference.

### Diagnostic Gene Contract

Diagnostics observe state and feed fitness back to configuration genes:

```
gene check_widget_health for my_domain
  is diagnostic
  risk none

  does:
    Check that a widget is in healthy state.

  takes:
    name  string  "Widget to check"

  gives:
    healthy  bool    "True if widget is healthy"
    status   string  "Current widget status"

  unhealthy when:
    - Widget status is not "active"

  feeds:
    create_widget convergence
```

The `feeds` section declares the fitness feedback loop — this diagnostic's output affects the convergence fitness (50% weight) of `create_widget`.

---

## 6. Writing Seed Genes

Every locus needs at least one seed gene — the initial implementation. Create `genes/<locus>_v1.py`:

```python
"""Seed gene: create a widget."""
import json


def execute(input_json: str) -> str:
    data = json.loads(input_json)

    name = data.get("name")
    if not name or not isinstance(name, str):
        return json.dumps({"success": False, "error": "missing or invalid name"})

    config = data.get("config", "{}")

    try:
        result = gene_sdk.create_widget(name, json.loads(config))
        return json.dumps({
            "success": True,
            "widget": json.dumps(result),
        })
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})
```

Rules:
- Define `execute(input_json: str) -> str`
- Parse input with `json.loads(input_json)`
- Use `gene_sdk` for all domain operations (it's injected into the namespace)
- Return JSON with at least a `success` boolean field
- Handle errors gracefully — return `{"success": false, "error": "..."}`, don't raise

### Allowed Imports

Genes run in a sandbox. Only these modules are importable: `json`, `math`, `re`, `hashlib`, `datetime`, `collections`, `collections.abc`, `itertools`, `functools`, `copy`, `string`, `textwrap`.

---

## 7. Packaging

### pyproject.toml

Register your kernel(s) as entry points under `sg.kernels`:

```toml
[project]
name = "sg-my-domain"
version = "0.1.0"
description = "Software Genomics — my domain plugin"
requires-python = ">=3.10"
dependencies = [
    "software-genomics>=0.2.0",
]

[project.entry-points."sg.kernels"]
my-domain-mock = "sg_my_domain.mock:MockMyDomainKernel"
# my-domain-production = "sg_my_domain.production:ProductionMyDomainKernel"

[tool.setuptools.packages.find]
include = ["sg_my_domain*"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"
```

### Path Helpers

Export path helpers in `__init__.py` so tests and the CLI can find your contracts and genes:

```python
from pathlib import Path

def contracts_path() -> Path:
    return Path(__file__).parent.parent / "contracts"

def genes_path() -> Path:
    return Path(__file__).parent.parent / "genes"

def fixtures_path() -> Path:
    return Path(__file__).parent.parent / "fixtures"
```

### Installation and Discovery

```bash
pip install -e plugins/my_domain     # install in development mode
sg kernels                           # verify kernel is discovered
```

The `sg kernels` command lists all registered kernels:

```
Available kernels:
  stub                      sg.kernel.stub:StubKernel
  my-domain-mock            sg_my_domain.mock:MockMyDomainKernel
```

### Using Your Kernel

```bash
sg init --kernel my-domain-mock
sg run my_pathway --kernel my-domain-mock --input '{...}'
```

---

## 8. Testing

### Conformance Testing

Once your contracts and seed genes are registered:

```bash
sg test                    # test all loci
sg test create_widget -v   # test specific locus, verbose
```

Conformance validates:
1. Gene defines `execute()`
2. Output is valid JSON
3. Output has a `success` field
4. All required `gives` fields are present
5. Field types match the contract schema

### End-to-End Testing

Write pytest tests that exercise the full evolutionary loop. Pattern from `tests/test_data_e2e.py`:

```python
import shutil
import pytest

import sg_my_domain
from sg_my_domain import MockMyDomainKernel
from sg.contracts import ContractStore
from sg.phenotype import PhenotypeMap
from sg.registry import Registry
from sg.orchestrator import Orchestrator
from sg.mutation import MockMutationEngine

CONTRACTS_DIR = sg_my_domain.contracts_path()
GENES_DIR = sg_my_domain.genes_path()
FIXTURES_DIR = sg_my_domain.fixtures_path()


@pytest.fixture
def project(tmp_path):
    # Copy plugin contracts to temp project
    contracts_dst = tmp_path / "contracts"
    shutil.copytree(CONTRACTS_DIR, contracts_dst)

    # Load contracts
    contract_store = ContractStore.open(contracts_dst)

    # Open registry and phenotype
    registry = Registry.open(tmp_path / ".sg" / "registry")
    phenotype = PhenotypeMap.open(tmp_path / "phenotype.toml")

    # Register seed genes
    for locus in contract_store.known_loci():
        candidates = sorted(GENES_DIR.glob(f"{locus}_*.py"))
        if candidates:
            source = candidates[0].read_text()
            sha = registry.register(source, locus)
            phenotype.promote(locus, sha)

    registry.save_index()
    phenotype.save()

    # Create orchestrator
    kernel = MockMyDomainKernel()
    mutation_engine = MockMutationEngine(FIXTURES_DIR)
    orchestrator = Orchestrator(
        registry=registry,
        phenotype=phenotype,
        contract_store=contract_store,
        mutation_engine=mutation_engine,
        kernel=kernel,
    )
    return orchestrator, kernel, registry, phenotype


def test_pathway_succeeds(project):
    orchestrator, kernel, registry, phenotype = project
    # Set up domain state in mock kernel
    kernel.add_widget("test", {"key": "value"})
    # Run pathway
    result = orchestrator.run_pathway("my_pathway", '{"name": "test"}')
    assert result["success"] is True
```

---

## 9. Quickstart

Generate a complete plugin skeleton:

```bash
sg new-plugin my_domain
```

This creates `plugins/my_domain/` with all the boilerplate. Then:

1. **Edit `sg_my_domain/kernel.py`** — add your domain's abstract operations
2. **Edit `sg_my_domain/mock.py`** — implement mock versions with in-memory state
3. **Write contracts** in `contracts/genes/` — one `.sg` file per locus
4. **Write seed genes** in `genes/` — one `<locus>_v1.py` per locus
5. **Install**: `pip install -e plugins/my_domain`
6. **Verify**: `sg kernels` should show your mock kernel
7. **Initialize**: `sg init --kernel my-domain-mock`
8. **Test**: `sg test`

### Production Kernel

For real-world use, add a production kernel that wraps actual system calls:

```python
# sg_my_domain/production.py
from sg_my_domain.kernel import MyDomainKernel


class ProductionMyDomainKernel(MyDomainKernel):
    """Production kernel with real side effects."""
    ...
```

Register it in `pyproject.toml`:

```toml
[project.entry-points."sg.kernels"]
my-domain-mock = "sg_my_domain.mock:MockMyDomainKernel"
my-domain = "sg_my_domain.production:ProductionMyDomainKernel"
```

---

## Reference

| Resource | Path |
|----------|------|
| Kernel ABC | `sg/kernel/base.py` |
| Kernel discovery | `sg/kernel/discovery.py` |
| `@mutating` decorator | `sg/kernel/base.py` |
| SafeKernel (transactions) | `sg/safety.py` |
| Contract parser | `sg/parser/` |
| Data plugin (example) | `plugins/data/` |
| Network plugin (example) | `plugins/network/` |
