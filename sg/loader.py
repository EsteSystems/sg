"""Gene loading via exec().

Genes are Python source strings with an execute(input_json: str) -> str function.
Loading is exec() into a namespace dict with the kernel injected as `gene_sdk`.
"""
from __future__ import annotations

from typing import Callable

from sg.kernel.base import NetworkKernel


def load_gene(source: str, kernel: NetworkKernel) -> Callable[[str], str]:
    """Load a gene from source code. Returns the execute function.

    The gene's namespace gets the kernel injected as `gene_sdk` so genes
    can call `gene_sdk.create_bridge(...)` etc.
    """
    namespace: dict = {"gene_sdk": kernel}
    exec(source, namespace)

    execute_fn = namespace.get("execute")
    if execute_fn is None:
        raise ValueError("gene source does not define an execute() function")
    if not callable(execute_fn):
        raise ValueError("gene source 'execute' is not callable")

    return execute_fn


def call_gene(execute_fn: Callable[[str], str], input_json: str) -> str:
    """Call a loaded gene's execute function with error wrapping."""
    try:
        result = execute_fn(input_json)
    except Exception as e:
        raise RuntimeError(f"gene execution failed: {e}") from e

    if not isinstance(result, str):
        raise RuntimeError(f"gene returned {type(result).__name__}, expected str")

    return result
