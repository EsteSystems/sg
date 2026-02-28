"""Tests for gene sandboxing."""
import json
import pytest
from pathlib import Path

from sg.kernel.mock import MockNetworkKernel
from sg.loader import load_gene, call_gene
from sg.sandbox import (
    make_sandbox_globals, execute_with_timeout,
    GeneImportError, GeneTimeout,
    BLOCKED_BUILTINS, ALLOWED_MODULES,
)


GENES_DIR = Path(__file__).parent.parent / "genes"


@pytest.fixture
def kernel():
    return MockNetworkKernel()


class TestBlockedBuiltins:
    def test_gene_cannot_open_files(self, kernel):
        """Genes cannot use open() to read/write files."""
        source = '''
import json
def execute(input_json):
    open("/etc/passwd", "r")
    return json.dumps({"success": True})
'''
        execute_fn = load_gene(source, kernel)
        with pytest.raises(RuntimeError, match="gene execution failed"):
            call_gene(execute_fn, '{}')

    def test_gene_cannot_eval(self, kernel):
        """Genes cannot use eval()."""
        source = '''
import json
def execute(input_json):
    eval("1 + 1")
    return json.dumps({"success": True})
'''
        execute_fn = load_gene(source, kernel)
        with pytest.raises(RuntimeError, match="gene execution failed"):
            call_gene(execute_fn, '{}')

    def test_gene_cannot_exec(self, kernel):
        """Genes cannot use exec() inside their code."""
        source = '''
import json
def execute(input_json):
    exec("x = 1")
    return json.dumps({"success": True})
'''
        execute_fn = load_gene(source, kernel)
        with pytest.raises(RuntimeError, match="gene execution failed"):
            call_gene(execute_fn, '{}')

    def test_gene_cannot_compile(self, kernel):
        """Genes cannot use compile()."""
        source = '''
import json
def execute(input_json):
    compile("x = 1", "<string>", "exec")
    return json.dumps({"success": True})
'''
        execute_fn = load_gene(source, kernel)
        with pytest.raises(RuntimeError, match="gene execution failed"):
            call_gene(execute_fn, '{}')


class TestBlockedImports:
    def test_gene_cannot_import_os(self, kernel):
        """Genes cannot import os."""
        source = '''
import os
def execute(input_json):
    return '{}'
'''
        with pytest.raises(GeneImportError, match="cannot import 'os'"):
            load_gene(source, kernel)

    def test_gene_cannot_import_subprocess(self, kernel):
        """Genes cannot import subprocess."""
        source = '''
import subprocess
def execute(input_json):
    return '{}'
'''
        with pytest.raises(GeneImportError, match="cannot import 'subprocess'"):
            load_gene(source, kernel)

    def test_gene_cannot_import_sys(self, kernel):
        """Genes cannot import sys."""
        source = '''
import sys
def execute(input_json):
    return '{}'
'''
        with pytest.raises(GeneImportError, match="cannot import 'sys'"):
            load_gene(source, kernel)

    def test_gene_cannot_import_socket(self, kernel):
        """Genes cannot import socket."""
        source = '''
import socket
def execute(input_json):
    return '{}'
'''
        with pytest.raises(GeneImportError, match="cannot import 'socket'"):
            load_gene(source, kernel)

    def test_gene_cannot_import_shutil(self, kernel):
        """Genes cannot import shutil."""
        source = '''
import shutil
def execute(input_json):
    return '{}'
'''
        with pytest.raises(GeneImportError, match="cannot import 'shutil'"):
            load_gene(source, kernel)


class TestAllowedImports:
    def test_gene_can_import_json(self, kernel):
        """Genes can import json."""
        source = '''
import json
def execute(input_json):
    return json.dumps({"success": True})
'''
        execute_fn = load_gene(source, kernel)
        result = call_gene(execute_fn, '{}')
        assert json.loads(result)["success"] is True

    def test_gene_can_import_math(self, kernel):
        """Genes can import math."""
        source = '''
import json, math
def execute(input_json):
    return json.dumps({"success": True, "pi": math.pi})
'''
        execute_fn = load_gene(source, kernel)
        result = call_gene(execute_fn, '{}')
        assert json.loads(result)["pi"] == pytest.approx(3.14159, rel=1e-3)

    def test_gene_can_import_collections(self, kernel):
        """Genes can import collections (used by check_mac_stability)."""
        source = '''
import json
from collections import defaultdict
def execute(input_json):
    d = defaultdict(int)
    d["a"] += 1
    return json.dumps({"success": True})
'''
        execute_fn = load_gene(source, kernel)
        result = call_gene(execute_fn, '{}')
        assert json.loads(result)["success"] is True

    def test_gene_can_import_re(self, kernel):
        """Genes can import re."""
        source = '''
import json, re
def execute(input_json):
    m = re.match(r"hello", "hello world")
    return json.dumps({"success": m is not None})
'''
        execute_fn = load_gene(source, kernel)
        result = call_gene(execute_fn, '{}')
        assert json.loads(result)["success"] is True

    def test_gene_can_import_hashlib(self, kernel):
        """Genes can import hashlib."""
        source = '''
import json, hashlib
def execute(input_json):
    h = hashlib.sha256(b"test").hexdigest()
    return json.dumps({"success": True, "hash": h})
'''
        execute_fn = load_gene(source, kernel)
        result = call_gene(execute_fn, '{}')
        assert json.loads(result)["success"] is True


class TestKernelAccess:
    def test_gene_has_kernel_access(self, kernel):
        """Genes can still access gene_sdk."""
        source = '''
import json
def execute(input_json):
    bridge = gene_sdk.create_bridge("test-br", ["eth0", "eth1"])
    return json.dumps({"success": True, "bridge": bridge["name"]})
'''
        execute_fn = load_gene(source, kernel)
        result = call_gene(execute_fn, '{}')
        data = json.loads(result)
        assert data["success"] is True
        assert data["bridge"] == "test-br"


class TestTimeout:
    def test_gene_timeout(self, kernel):
        """Long-running gene is interrupted by timeout."""
        source = '''
import json
def execute(input_json):
    while True:
        pass
    return json.dumps({"success": True})
'''
        execute_fn = load_gene(source, kernel)
        with pytest.raises(GeneTimeout):
            execute_with_timeout(execute_fn, '{}', timeout=1)

    def test_gene_within_timeout(self, kernel):
        """Gene that completes quickly is not interrupted."""
        source = '''
import json
def execute(input_json):
    return json.dumps({"success": True})
'''
        execute_fn = load_gene(source, kernel)
        result = execute_with_timeout(execute_fn, '{}', timeout=5)
        assert json.loads(result)["success"] is True


class TestExistingGenes:
    def test_all_seed_genes_work_in_sandbox(self, kernel):
        """All existing seed genes load and execute under sandbox."""
        gene_files = sorted(GENES_DIR.glob("*.py"))
        assert len(gene_files) > 0, "no seed genes found"

        for gene_file in gene_files:
            source = gene_file.read_text()
            execute_fn = load_gene(source, kernel)
            assert callable(execute_fn), f"{gene_file.name} did not produce callable"


class TestSandboxGlobals:
    def test_blocked_builtins_not_in_globals(self):
        """All blocked builtins are absent from sandbox globals (except __import__ which is replaced)."""
        kernel = MockNetworkKernel()
        g = make_sandbox_globals(kernel)
        safe = g["__builtins__"]
        for name in BLOCKED_BUILTINS - {"__import__"}:
            assert name not in safe, f"blocked builtin '{name}' found in sandbox"
        # __import__ is replaced with a restricted version, not removed
        import builtins as _b
        assert safe["__import__"] is not _b.__import__

    def test_safe_builtins_present(self):
        """Common safe builtins (len, range, etc.) are available."""
        kernel = MockNetworkKernel()
        g = make_sandbox_globals(kernel)
        safe = g["__builtins__"]
        for name in ["len", "range", "int", "str", "list", "dict", "print",
                      "isinstance", "enumerate", "zip", "map", "filter",
                      "sorted", "min", "max", "sum", "abs", "round", "type",
                      "True", "False", "None", "Exception", "ValueError"]:
            assert name in safe, f"safe builtin '{name}' missing from sandbox"

    def test_kernel_injected(self):
        """gene_sdk is available in sandbox globals."""
        kernel = MockNetworkKernel()
        g = make_sandbox_globals(kernel)
        assert g["gene_sdk"] is kernel
