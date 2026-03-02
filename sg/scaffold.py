"""Plugin scaffolding — generates skeleton plugin packages.

`sg new-plugin <name>` creates a complete plugin directory with kernel ABC,
mock kernel, example contract, example seed gene, and pyproject.toml.
"""
from __future__ import annotations

import keyword
import re
from pathlib import Path


class ScaffoldError(Exception):
    """Raised when plugin scaffolding fails."""


def validate_plugin_name(name: str) -> str:
    """Validate and normalize a plugin name. Returns the Python-safe name.

    Raises ScaffoldError for invalid names.
    """
    if not name:
        raise ScaffoldError("plugin name cannot be empty")

    # Normalize hyphens to underscores for Python package name
    py_name = name.replace("-", "_")

    if not re.match(r"^[a-z][a-z0-9_]*$", py_name):
        raise ScaffoldError(
            f"invalid plugin name '{name}': must start with a lowercase letter "
            "and contain only lowercase letters, digits, and underscores"
        )

    if keyword.iskeyword(py_name):
        raise ScaffoldError(f"invalid plugin name '{name}': is a Python keyword")

    # Reserved names (existing plugins)
    if py_name in ("network", "data"):
        raise ScaffoldError(
            f"plugin name '{name}' is reserved (an existing plugin uses this name)"
        )

    return py_name


def _title_case(name: str) -> str:
    """Convert underscore_name to TitleCase."""
    return "".join(word.capitalize() for word in name.split("_"))


# --- Templates ---

_PYPROJECT_TOML = """\
[project]
name = "sg-{name}"
version = "0.1.0"
description = "Software Genomics — {name} domain plugin"
requires-python = ">=3.10"
license = {{text = "BSD-2-Clause"}}
dependencies = [
    "software-genomics>=0.2.0",
]

[project.entry-points."sg.kernels"]
{name}-mock = "sg_{py_name}.mock:Mock{title}Kernel"

[tool.setuptools.packages.find]
include = ["sg_{py_name}*"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"
"""

_INIT_PY = """\
\"""Software Genomics — {name} domain plugin.

Provides {title}Kernel ABC, Mock{title}Kernel, and {name} contracts/genes.
\"""
from __future__ import annotations

from pathlib import Path

from sg_{py_name}.kernel import {title}Kernel
from sg_{py_name}.mock import Mock{title}Kernel

__all__ = [
    "{title}Kernel",
    "Mock{title}Kernel",
    "contracts_path",
    "genes_path",
    "fixtures_path",
]


def contracts_path() -> Path:
    \"""Return the path to the {name} contracts directory.\"""
    return Path(__file__).parent.parent / "contracts"


def genes_path() -> Path:
    \"""Return the path to the {name} seed genes directory.\"""
    return Path(__file__).parent.parent / "genes"


def fixtures_path() -> Path:
    \"""Return the path to the {name} test fixtures directory.\"""
    return Path(__file__).parent.parent / "fixtures"
"""

_KERNEL_PY = """\
\"""{title}Kernel — abstract kernel interface for {name} operations.

Extend this class with domain-specific operations that genes will call
via gene_sdk.
\"""
from __future__ import annotations

from abc import abstractmethod

from sg.kernel.base import Kernel, mutating


class {title}Kernel(Kernel):
    \"""Abstract kernel interface for the {name} domain.

    Genes use gene_sdk (a {title}Kernel instance) to perform
    domain-specific operations. Add your domain operations as
    abstract methods below.
    \"""

    # --- Example domain operation ---
    # Uncomment and modify for your domain:
    #
    # @mutating(undo=lambda k, snap, name: k.delete_thing(name))
    # @abstractmethod
    # def create_thing(self, name: str) -> dict:
    #     \\"\\"\\"Create a thing. Returns its properties.\\"\\"\\"
    #     ...
    #
    # @abstractmethod
    # def delete_thing(self, name: str) -> None:
    #     \\"\\"\\"Delete a thing by name.\\"\\"\\"
    #     ...

    # --- Self-description ---

    def describe_operations(self) -> list[str]:
        return [
            # Add your domain operations here, e.g.:
            # "create_thing(name: str) -> dict",
            # "delete_thing(name: str) -> None",
        ]

    def mutation_prompt_context(self) -> str:
        return (
            "This gene operates in the {name} domain. gene_sdk is a "
            "{title}Kernel providing domain-specific operations."
        )

    def domain_name(self) -> str:
        return "{name}"
"""

_MOCK_PY = """\
\"""Mock{title}Kernel — in-memory implementation for development and testing.\"""
from __future__ import annotations

from sg_{py_name}.kernel import {title}Kernel


class Mock{title}Kernel({title}Kernel):
    \"""In-memory mock kernel for development and testing.

    Implements all {title}Kernel operations with in-memory state.
    No real side effects.
    \"""

    def __init__(self) -> None:
        self._resources: list[tuple[str, str]] = []

    def reset(self) -> None:
        self._resources.clear()

    def track_resource(self, resource_type: str, name: str) -> None:
        pair = (resource_type, name)
        if pair not in self._resources:
            self._resources.append(pair)

    def untrack_resource(self, resource_type: str, name: str) -> None:
        pair = (resource_type, name)
        if pair in self._resources:
            self._resources.remove(pair)

    def tracked_resources(self) -> list[tuple[str, str]]:
        return list(self._resources)

    def delete_resource(self, resource_type: str, name: str) -> None:
        self.untrack_resource(resource_type, name)

    def create_shadow(self) -> Mock{title}Kernel:
        return Mock{title}Kernel()
"""

_EXAMPLE_CONTRACT = """\
gene example_action for {name}
  is configuration
  risk none

  does:
    An example action for the {name} domain.
    Replace this with a real description of what the gene does.

  takes:
    name  string  "Name of the resource to act on"

  gives:
    success  bool    "Whether the action succeeded"
    error    string? "Error message on failure"

  before:
    - Resource does not already exist

  after:
    - Resource exists and is in expected state

  fails when:
    - resource already exists -> success=false
"""

_EXAMPLE_GENE = """\
\"""Seed gene: example action for the {name} domain.\"""
import json


def execute(input_json: str) -> str:
    data = json.loads(input_json)

    name = data.get("name")
    if not name or not isinstance(name, str):
        return json.dumps({{"success": False, "error": "missing or invalid name"}})

    try:
        # Replace with actual gene_sdk calls for your domain:
        # result = gene_sdk.create_thing(name)
        gene_sdk.track_resource("example", name)
        return json.dumps({{"success": True}})
    except Exception as e:
        return json.dumps({{"success": False, "error": str(e)}})
"""


def scaffold_plugin(name: str, output_dir: Path) -> Path:
    """Generate a complete plugin skeleton.

    Args:
        name: Plugin name (e.g., "storage", "monitoring").
        output_dir: Parent directory (e.g., plugins/).

    Returns:
        Path to the created plugin directory.

    Raises:
        ScaffoldError: If the name is invalid or directory already exists.
    """
    py_name = validate_plugin_name(name)
    title = _title_case(py_name)
    plugin_dir = output_dir / name

    if plugin_dir.exists():
        raise ScaffoldError(
            f"directory already exists: {plugin_dir}\n"
            "Remove it first or choose a different name."
        )

    # Template substitution context
    ctx = {"name": name, "py_name": py_name, "title": title}

    # Create directory structure
    pkg_dir = plugin_dir / f"sg_{py_name}"
    (plugin_dir / "contracts" / "genes").mkdir(parents=True)
    (plugin_dir / "genes").mkdir(parents=True)
    (plugin_dir / "fixtures").mkdir(parents=True)
    pkg_dir.mkdir(parents=True)

    # Write files
    (plugin_dir / "pyproject.toml").write_text(_PYPROJECT_TOML.format(**ctx))
    (pkg_dir / "__init__.py").write_text(_INIT_PY.format(**ctx))
    (pkg_dir / "kernel.py").write_text(_KERNEL_PY.format(**ctx))
    (pkg_dir / "mock.py").write_text(_MOCK_PY.format(**ctx))
    (plugin_dir / "contracts" / "genes" / "example_action.sg").write_text(
        _EXAMPLE_CONTRACT.format(**ctx)
    )
    (plugin_dir / "genes" / "example_action_v1.py").write_text(
        _EXAMPLE_GENE.format(**ctx)
    )

    return plugin_dir
