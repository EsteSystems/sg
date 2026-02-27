"""Tests for gene loading via exec()."""
import json
import pytest
from sg.kernel.mock import MockNetworkKernel
from sg.loader import load_gene, call_gene


@pytest.fixture
def kernel():
    return MockNetworkKernel()


def test_load_simple_gene(kernel):
    source = '''
import json

def execute(input_json):
    data = json.loads(input_json)
    return json.dumps({"success": True, "echo": data})
'''
    fn = load_gene(source, kernel)
    result = call_gene(fn, '{"hello": "world"}')
    parsed = json.loads(result)
    assert parsed["success"] is True
    assert parsed["echo"]["hello"] == "world"


def test_load_gene_with_kernel(kernel):
    source = '''
import json

def execute(input_json):
    gene_sdk.reset()
    return json.dumps({"success": True})
'''
    fn = load_gene(source, kernel)
    result = call_gene(fn, '{}')
    assert json.loads(result)["success"] is True


def test_load_gene_missing_execute(kernel):
    with pytest.raises(ValueError, match="does not define"):
        load_gene("x = 1", kernel)


def test_load_gene_execute_not_callable(kernel):
    with pytest.raises(ValueError, match="not callable"):
        load_gene("execute = 42", kernel)


def test_call_gene_exception(kernel):
    source = '''
def execute(input_json):
    raise ValueError("intentional error")
'''
    fn = load_gene(source, kernel)
    with pytest.raises(RuntimeError, match="gene execution failed"):
        call_gene(fn, '{}')


def test_call_gene_wrong_return_type(kernel):
    source = '''
def execute(input_json):
    return 42
'''
    fn = load_gene(source, kernel)
    with pytest.raises(RuntimeError, match="expected str"):
        call_gene(fn, '{}')
