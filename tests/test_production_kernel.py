"""Integration tests for ProductionNetworkKernel on real Linux host.

These tests create real network interfaces using sg-test-* prefix.
Run with: pytest tests/test_production_kernel.py -v
Requires: Linux, sudo, ip/bridge commands.

Skipped automatically on non-Linux or when sudo is unavailable.
"""
import json
import platform
import subprocess
import pytest

from sg_network.production import ProductionNetworkKernel, SAFETY_PREFIX


def _has_sudo() -> bool:
    try:
        result = subprocess.run(
            ["sudo", "-n", "true"],
            capture_output=True, timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


requires_linux = pytest.mark.skipif(
    platform.system() != "Linux",
    reason="requires Linux"
)
requires_sudo = pytest.mark.skipif(
    not _has_sudo() if platform.system() == "Linux" else True,
    reason="requires passwordless sudo"
)


@requires_linux
@requires_sudo
class TestProductionBridge:
    @pytest.fixture(autouse=True)
    def cleanup(self):
        kernel = ProductionNetworkKernel()
        yield kernel
        kernel.cleanup_all_test_resources()

    def test_create_and_get_bridge(self, cleanup):
        kernel = cleanup
        result = kernel.create_bridge("sg-test-br0", [])
        assert result is not None
        assert result["name"] == "sg-test-br0"

        bridge = kernel.get_bridge("sg-test-br0")
        assert bridge is not None
        assert bridge["name"] == "sg-test-br0"

    def test_create_bridge_with_veth(self, cleanup):
        kernel = cleanup
        kernel.create_veth_pair("sg-test-v0", "sg-test-v1")
        result = kernel.create_bridge("sg-test-br0", ["sg-test-v0"])
        assert "sg-test-v0" in result["interfaces"]

    def test_delete_bridge(self, cleanup):
        kernel = cleanup
        kernel.create_bridge("sg-test-br0", [])
        kernel.delete_bridge("sg-test-br0")
        assert kernel.get_bridge("sg-test-br0") is None

    def test_attach_detach_interface(self, cleanup):
        kernel = cleanup
        kernel.create_veth_pair("sg-test-v0", "sg-test-v1")
        kernel.create_bridge("sg-test-br0", [])
        kernel.attach_interface("sg-test-br0", "sg-test-v0")

        bridge = kernel.get_bridge("sg-test-br0")
        assert "sg-test-v0" in bridge["interfaces"]

        kernel.detach_interface("sg-test-br0", "sg-test-v0")
        bridge = kernel.get_bridge("sg-test-br0")
        assert "sg-test-v0" not in bridge["interfaces"]

    def test_nonexistent_bridge_returns_none(self, cleanup):
        kernel = cleanup
        assert kernel.get_bridge("sg-test-nonexistent") is None


@requires_linux
@requires_sudo
class TestProductionSTP:
    @pytest.fixture(autouse=True)
    def cleanup(self):
        kernel = ProductionNetworkKernel()
        yield kernel
        kernel.cleanup_all_test_resources()

    def test_set_stp(self, cleanup):
        kernel = cleanup
        kernel.create_bridge("sg-test-br0", [])
        result = kernel.set_stp("sg-test-br0", True, 15)
        assert result["stp_enabled"] is True

    def test_get_stp_state(self, cleanup):
        kernel = cleanup
        kernel.create_bridge("sg-test-br0", [])
        kernel.set_stp("sg-test-br0", True, 15)
        state = kernel.get_stp_state("sg-test-br0")
        assert state["enabled"] is True
        assert state["bridge"] == "sg-test-br0"


@requires_linux
@requires_sudo
class TestProductionMAC:
    @pytest.fixture(autouse=True)
    def cleanup(self):
        kernel = ProductionNetworkKernel()
        yield kernel
        kernel.cleanup_all_test_resources()

    def test_get_device_mac(self, cleanup):
        kernel = cleanup
        kernel.create_bridge("sg-test-br0", [])
        mac = kernel.get_device_mac("sg-test-br0")
        assert ":" in mac
        assert len(mac) == 17

    def test_set_device_mac(self, cleanup):
        kernel = cleanup
        kernel.create_bridge("sg-test-br0", [])
        kernel.set_device_mac("sg-test-br0", "02:aa:bb:cc:dd:ee")
        mac = kernel.get_device_mac("sg-test-br0")
        assert mac == "02:aa:bb:cc:dd:ee"


@requires_linux
@requires_sudo
class TestProductionBond:
    @pytest.fixture(autouse=True)
    def cleanup(self):
        kernel = ProductionNetworkKernel()
        yield kernel
        kernel.cleanup_all_test_resources()

    def test_create_bond(self, cleanup):
        kernel = cleanup
        kernel.create_veth_pair("sg-test-m0", "sg-test-m0-peer")
        kernel.create_veth_pair("sg-test-m1", "sg-test-m1-peer")
        result = kernel.create_bond(
            "sg-test-bond0", "balance-rr",
            ["sg-test-m0", "sg-test-m1"],
        )
        assert result is not None
        assert result["name"] == "sg-test-bond0"

    def test_delete_bond(self, cleanup):
        kernel = cleanup
        kernel.create_bond("sg-test-bond0", "balance-rr", [])
        kernel.delete_bond("sg-test-bond0")
        assert kernel.get_bond("sg-test-bond0") is None


@requires_linux
@requires_sudo
class TestProductionVLAN:
    @pytest.fixture(autouse=True)
    def cleanup(self):
        kernel = ProductionNetworkKernel()
        yield kernel
        kernel.cleanup_all_test_resources()

    def test_create_vlan(self, cleanup):
        kernel = cleanup
        kernel.create_bridge("sg-test-br0", [])
        result = kernel.create_vlan("sg-test-br0", 100)
        assert result is not None
        assert result["vlan_id"] == 100

    def test_delete_vlan(self, cleanup):
        kernel = cleanup
        kernel.create_bridge("sg-test-br0", [])
        kernel.create_vlan("sg-test-br0", 100)
        kernel.delete_vlan("sg-test-br0", 100)
        assert kernel.get_vlan("sg-test-br0", 100) is None


@requires_linux
@requires_sudo
class TestProductionDiagnostics:
    @pytest.fixture(autouse=True)
    def cleanup(self):
        kernel = ProductionNetworkKernel()
        yield kernel
        kernel.cleanup_all_test_resources()

    def test_read_fdb(self, cleanup):
        kernel = cleanup
        kernel.create_bridge("sg-test-br0", [])
        fdb = kernel.read_fdb("sg-test-br0")
        assert isinstance(fdb, list)

    def test_get_interface_state(self, cleanup):
        kernel = cleanup
        kernel.create_bridge("sg-test-br0", [])
        state = kernel.get_interface_state("sg-test-br0")
        assert state["name"] == "sg-test-br0"
        assert "mac" in state
        assert "carrier" in state

    def test_get_arp_table(self, cleanup):
        kernel = cleanup
        arp = kernel.get_arp_table()
        assert isinstance(arp, list)


@requires_linux
@requires_sudo
class TestProductionSafety:
    @pytest.fixture(autouse=True)
    def cleanup(self):
        kernel = ProductionNetworkKernel()
        yield kernel
        kernel.cleanup_all_test_resources()

    def test_refuses_unprotected_name(self, cleanup):
        kernel = cleanup
        with pytest.raises(ValueError, match="sg-test-"):
            kernel.create_bridge("br0", [])

    def test_refuses_protected_interface(self, cleanup):
        kernel = cleanup
        kernel.create_bridge("sg-test-br0", [])
        with pytest.raises(ValueError, match="protected"):
            kernel.attach_interface("sg-test-br0", "eth0")

    def test_cleanup_removes_all(self, cleanup):
        kernel = cleanup
        kernel.create_bridge("sg-test-br0", [])
        kernel.create_bridge("sg-test-br1", [])
        kernel.cleanup_all_test_resources()
        assert kernel.get_bridge("sg-test-br0") is None
        assert kernel.get_bridge("sg-test-br1") is None

    def test_resource_tracking(self, cleanup):
        kernel = cleanup
        kernel.create_bridge("sg-test-br0", [])
        resources = kernel.tracked_resources()
        assert ("bridge", "sg-test-br0") in resources
