"""Tests for the MockNetworkKernel."""
import pytest
from sg.kernel.mock import MockNetworkKernel


@pytest.fixture
def kernel():
    k = MockNetworkKernel()
    yield k
    k.reset()


def test_create_bridge(kernel):
    result = kernel.create_bridge("br0", ["eth0", "eth1"])
    assert result["name"] == "br0"
    assert result["interfaces"] == ["eth0", "eth1"]
    assert result["stp_enabled"] is False
    assert result["forward_delay"] == 15


def test_create_bridge_duplicate(kernel):
    kernel.create_bridge("br0", ["eth0"])
    with pytest.raises(ValueError, match="already exists"):
        kernel.create_bridge("br0", ["eth1"])


def test_create_bridge_empty_name(kernel):
    with pytest.raises(ValueError, match="cannot be empty"):
        kernel.create_bridge("", ["eth0"])


def test_set_stp(kernel):
    kernel.create_bridge("br0", ["eth0"])
    result = kernel.set_stp("br0", True, 20)
    assert result["stp_enabled"] is True
    assert result["forward_delay"] == 20


def test_set_stp_nonexistent(kernel):
    with pytest.raises(ValueError, match="does not exist"):
        kernel.set_stp("br0", True, 15)


def test_set_stp_invalid_delay(kernel):
    kernel.create_bridge("br0", ["eth0"])
    with pytest.raises(ValueError, match="forward_delay"):
        kernel.set_stp("br0", True, 50)


def test_get_bridge(kernel):
    kernel.create_bridge("br0", ["eth0"])
    result = kernel.get_bridge("br0")
    assert result is not None
    assert result["name"] == "br0"


def test_get_bridge_nonexistent(kernel):
    assert kernel.get_bridge("br0") is None


def test_inject_failure(kernel):
    kernel.inject_failure("create_bridge", "simulated failure")
    with pytest.raises(RuntimeError, match="simulated failure"):
        kernel.create_bridge("br0", ["eth0"])
    # One-shot: second call should succeed
    kernel.create_bridge("br0", ["eth0"])


def test_reset(kernel):
    kernel.create_bridge("br0", ["eth0"])
    kernel.reset()
    assert kernel.get_bridge("br0") is None
