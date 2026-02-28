"""Gene sandboxing — restrict what gene code can do inside exec().

Blocks dangerous builtins (exec, eval, open, etc.), limits imports
to a safe allowlist, and provides execution timeout.
"""
from __future__ import annotations

import builtins
import signal
import sys
import threading
from typing import Callable


# Builtins that genes must not access
BLOCKED_BUILTINS = frozenset({
    "exec", "eval", "compile", "__import__", "open", "input",
    "breakpoint", "exit", "quit",
})

# Modules that genes are allowed to import
ALLOWED_MODULES = frozenset({
    "json", "math", "re", "hashlib", "datetime", "collections",
    "itertools", "functools", "copy", "string", "textwrap",
    "collections.abc",
})

DEFAULT_TIMEOUT = 30  # seconds


class GeneImportError(ImportError):
    """Raised when a gene tries to import a blocked module."""
    pass


class GeneTimeout(Exception):
    """Raised when a gene exceeds its execution timeout."""
    pass


def _restricted_import(name: str, globals=None, locals=None,
                       fromlist=(), level=0):
    """Import function that only allows modules in ALLOWED_MODULES."""
    # Allow submodule access for allowed top-level modules
    top_level = name.split(".")[0]
    if name not in ALLOWED_MODULES and top_level not in ALLOWED_MODULES:
        raise GeneImportError(
            f"gene cannot import '{name}' — not in allowed modules: "
            f"{', '.join(sorted(ALLOWED_MODULES))}"
        )
    return __builtins_original_import__(name, globals, locals, fromlist, level)


# Save the real __import__ for use by _restricted_import
__builtins_original_import__ = builtins.__import__


def make_sandbox_globals(kernel) -> dict:
    """Build a restricted globals dict for gene exec().

    Blocks dangerous builtins (exec, eval, open, etc.) and restricts
    imports to a safe allowlist. The kernel is injected as `gene_sdk`.
    """
    # Build safe builtins dict
    all_builtins = vars(builtins)
    safe_builtins = {
        k: v for k, v in all_builtins.items()
        if k not in BLOCKED_BUILTINS
    }
    # Replace __import__ with restricted version
    safe_builtins["__import__"] = _restricted_import

    return {
        "__builtins__": safe_builtins,
        "gene_sdk": kernel,
    }


def execute_with_timeout(
    fn: Callable[[str], str],
    input_json: str,
    timeout: int = DEFAULT_TIMEOUT,
) -> str:
    """Call fn(input_json) with a timeout.

    Uses signal.SIGALRM on Unix (main thread only), falls back to
    threading.Timer on other platforms or non-main threads.
    """
    if timeout <= 0:
        return fn(input_json)

    # Try signal-based timeout (Unix, main thread only)
    if hasattr(signal, "SIGALRM") and threading.current_thread() is threading.main_thread():
        return _timeout_signal(fn, input_json, timeout)

    # Fallback: threading-based timeout
    return _timeout_thread(fn, input_json, timeout)


def _timeout_signal(fn: Callable, input_json: str, timeout: int) -> str:
    """Signal-based timeout (Unix main thread)."""
    def _handler(signum, frame):
        raise GeneTimeout(f"gene execution exceeded {timeout}s timeout")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    old_alarm = signal.alarm(timeout)
    try:
        result = fn(input_json)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        if old_alarm > 0:
            signal.alarm(old_alarm)
    return result


def _timeout_thread(fn: Callable, input_json: str, timeout: int) -> str:
    """Thread-based timeout fallback."""
    result_holder: list = []
    error_holder: list = []

    def _target():
        try:
            result_holder.append(fn(input_json))
        except Exception as e:
            error_holder.append(e)

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        raise GeneTimeout(f"gene execution exceeded {timeout}s timeout")
    if error_holder:
        raise error_holder[0]
    if not result_holder:
        raise RuntimeError("gene execution produced no result")
    return result_holder[0]
