"""Tests for MockNetworkKernel — full network simulation."""
import pytest
from sg.kernel.mock import MockNetworkKernel


@pytest.fixture
def kernel():
    k = MockNetworkKernel()
    k.reset()
    return k


# --- Bridge operations ---

class TestBridge:
    def test_create_bridge(self, kernel):
        result = kernel.create_bridge("br0", ["eth0", "eth1"])
        assert result["name"] == "br0"
        assert result["interfaces"] == ["eth0", "eth1"]
        assert result["stp_enabled"] is False

    def test_create_bridge_duplicate(self, kernel):
        kernel.create_bridge("br0", [])
        with pytest.raises(ValueError, match="already exists"):
            kernel.create_bridge("br0", [])

    def test_create_bridge_empty_name(self, kernel):
        with pytest.raises(ValueError, match="cannot be empty"):
            kernel.create_bridge("", [])

    def test_delete_bridge(self, kernel):
        kernel.create_bridge("br0", ["eth0"])
        kernel.delete_bridge("br0")
        assert kernel.get_bridge("br0") is None

    def test_delete_bridge_nonexistent(self, kernel):
        with pytest.raises(ValueError, match="does not exist"):
            kernel.delete_bridge("br0")

    def test_delete_bridge_detaches_interfaces(self, kernel):
        kernel.create_bridge("br0", ["eth0"])
        kernel.delete_bridge("br0")
        state = kernel.get_interface_state("eth0")
        assert state["master"] == ""

    def test_attach_interface(self, kernel):
        kernel.create_bridge("br0", ["eth0"])
        kernel.attach_interface("br0", "eth1")
        bridge = kernel.get_bridge("br0")
        assert "eth1" in bridge["interfaces"]

    def test_attach_interface_duplicate(self, kernel):
        kernel.create_bridge("br0", ["eth0"])
        with pytest.raises(ValueError, match="already attached"):
            kernel.attach_interface("br0", "eth0")

    def test_detach_interface(self, kernel):
        kernel.create_bridge("br0", ["eth0", "eth1"])
        kernel.detach_interface("br0", "eth0")
        bridge = kernel.get_bridge("br0")
        assert "eth0" not in bridge["interfaces"]
        assert "eth1" in bridge["interfaces"]

    def test_detach_interface_not_attached(self, kernel):
        kernel.create_bridge("br0", ["eth0"])
        with pytest.raises(ValueError, match="not attached"):
            kernel.detach_interface("br0", "eth2")

    def test_get_bridge(self, kernel):
        kernel.create_bridge("br0", ["eth0"])
        bridge = kernel.get_bridge("br0")
        assert bridge is not None
        assert bridge["name"] == "br0"

    def test_get_bridge_nonexistent(self, kernel):
        assert kernel.get_bridge("br0") is None


# --- STP operations ---

class TestSTP:
    def test_set_stp(self, kernel):
        kernel.create_bridge("br0", [])
        result = kernel.set_stp("br0", True, 15)
        assert result["stp_enabled"] is True
        assert result["forward_delay"] == 15

    def test_set_stp_nonexistent(self, kernel):
        with pytest.raises(ValueError, match="does not exist"):
            kernel.set_stp("br0", True, 15)

    def test_set_stp_invalid_delay(self, kernel):
        kernel.create_bridge("br0", [])
        with pytest.raises(ValueError, match="forward_delay must be"):
            kernel.set_stp("br0", True, 0)

    def test_get_stp_state(self, kernel):
        kernel.create_bridge("br0", [])
        kernel.set_stp("br0", True, 20)
        state = kernel.get_stp_state("br0")
        assert state["enabled"] is True
        assert state["forward_delay"] == 20
        assert state["bridge"] == "br0"

    def test_get_stp_state_nonexistent(self, kernel):
        with pytest.raises(ValueError, match="does not exist"):
            kernel.get_stp_state("br0")


# --- MAC operations ---

class TestMAC:
    def test_get_device_mac(self, kernel):
        kernel.create_bridge("br0", ["eth0"])
        mac = kernel.get_device_mac("eth0")
        assert isinstance(mac, str)
        assert len(mac.split(":")) == 6

    def test_get_device_mac_nonexistent(self, kernel):
        with pytest.raises(ValueError, match="does not exist"):
            kernel.get_device_mac("eth99")

    def test_set_device_mac(self, kernel):
        kernel.create_bridge("br0", ["eth0"])
        kernel.set_device_mac("eth0", "aa:bb:cc:dd:ee:ff")
        assert kernel.get_device_mac("eth0") == "aa:bb:cc:dd:ee:ff"

    def test_send_gratuitous_arp(self, kernel):
        kernel.send_gratuitous_arp("eth0", "aa:bb:cc:dd:ee:ff")
        assert len(kernel._gratuitous_arps) == 1
        assert kernel._gratuitous_arps[0]["interface"] == "eth0"


# --- Bond operations ---

class TestBond:
    def test_create_bond(self, kernel):
        result = kernel.create_bond("bond0", "802.3ad", ["eth0", "eth1"])
        assert result["name"] == "bond0"
        assert result["mode"] == "802.3ad"
        assert result["members"] == ["eth0", "eth1"]
        assert result["active"] is True

    def test_create_bond_duplicate(self, kernel):
        kernel.create_bond("bond0", "active-backup", ["eth0"])
        with pytest.raises(ValueError, match="already exists"):
            kernel.create_bond("bond0", "active-backup", ["eth1"])

    def test_delete_bond(self, kernel):
        kernel.create_bond("bond0", "802.3ad", ["eth0"])
        kernel.delete_bond("bond0")
        assert kernel.get_bond("bond0") is None

    def test_delete_bond_nonexistent(self, kernel):
        with pytest.raises(ValueError, match="does not exist"):
            kernel.delete_bond("bond0")

    def test_delete_bond_frees_members(self, kernel):
        kernel.create_bond("bond0", "802.3ad", ["eth0"])
        kernel.delete_bond("bond0")
        state = kernel.get_interface_state("eth0")
        assert state["master"] == ""

    def test_get_bond(self, kernel):
        kernel.create_bond("bond0", "active-backup", ["eth0"])
        bond = kernel.get_bond("bond0")
        assert bond["name"] == "bond0"

    def test_get_bond_nonexistent(self, kernel):
        assert kernel.get_bond("bond0") is None


# --- VLAN operations ---

class TestVLAN:
    def test_create_vlan(self, kernel):
        result = kernel.create_vlan("eth0", 100)
        assert result["parent"] == "eth0"
        assert result["vlan_id"] == 100
        assert result["name"] == "eth0.100"

    def test_create_vlan_duplicate(self, kernel):
        kernel.create_vlan("eth0", 100)
        with pytest.raises(ValueError, match="already exists"):
            kernel.create_vlan("eth0", 100)

    def test_create_vlan_invalid_id(self, kernel):
        with pytest.raises(ValueError, match="VLAN ID must be"):
            kernel.create_vlan("eth0", 0)
        with pytest.raises(ValueError, match="VLAN ID must be"):
            kernel.create_vlan("eth0", 4095)

    def test_delete_vlan(self, kernel):
        kernel.create_vlan("eth0", 100)
        kernel.delete_vlan("eth0", 100)
        assert kernel.get_vlan("eth0", 100) is None

    def test_delete_vlan_nonexistent(self, kernel):
        with pytest.raises(ValueError, match="does not exist"):
            kernel.delete_vlan("eth0", 100)

    def test_get_vlan(self, kernel):
        kernel.create_vlan("eth0", 200)
        vlan = kernel.get_vlan("eth0", 200)
        assert vlan["vlan_id"] == 200

    def test_get_vlan_nonexistent(self, kernel):
        assert kernel.get_vlan("eth0", 999) is None


# --- FDB and diagnostics ---

class TestDiagnostics:
    def test_read_fdb_empty(self, kernel):
        kernel.create_bridge("br0", [])
        fdb = kernel.read_fdb("br0")
        assert fdb == []

    def test_read_fdb_nonexistent(self, kernel):
        with pytest.raises(ValueError, match="does not exist"):
            kernel.read_fdb("br0")

    def test_add_fdb_entry(self, kernel):
        kernel.create_bridge("br0", ["eth0"])
        kernel.add_fdb_entry("br0", "aa:bb:cc:dd:ee:ff", "eth0")
        fdb = kernel.read_fdb("br0")
        assert len(fdb) == 1
        assert fdb[0]["mac"] == "aa:bb:cc:dd:ee:ff"
        assert fdb[0]["port"] == "eth0"

    def test_get_interface_state(self, kernel):
        kernel.create_bridge("br0", ["eth0"])
        state = kernel.get_interface_state("eth0")
        assert state["name"] == "eth0"
        assert state["carrier"] is True
        assert state["operstate"] == "up"
        assert state["master"] == "br0"

    def test_get_interface_state_nonexistent(self, kernel):
        with pytest.raises(ValueError, match="does not exist"):
            kernel.get_interface_state("eth99")

    def test_arp_table_empty(self, kernel):
        assert kernel.get_arp_table() == []

    def test_add_arp_entry(self, kernel):
        kernel.add_arp_entry("192.168.1.1", "aa:bb:cc:dd:ee:ff", "eth0")
        table = kernel.get_arp_table()
        assert len(table) == 1
        assert table[0]["ip"] == "192.168.1.1"


# --- Anomaly injection ---

class TestAnomalies:
    def test_inject_failure(self, kernel):
        kernel.inject_failure("create_bridge", "simulated panic")
        with pytest.raises(RuntimeError, match="simulated panic"):
            kernel.create_bridge("br0", [])
        # One-shot: second call succeeds
        kernel.create_bridge("br0", [])

    def test_inject_mac_flapping(self, kernel):
        kernel.create_bridge("br0", ["eth0", "eth1"])
        kernel.inject_mac_flapping("br0", "aa:bb:cc:dd:ee:ff", ["eth0", "eth1"])

        fdb = kernel.read_fdb("br0")
        flapping_macs = [e for e in fdb if e["mac"] == "aa:bb:cc:dd:ee:ff"]
        assert len(flapping_macs) == 2
        ports = {e["port"] for e in flapping_macs}
        assert ports == {"eth0", "eth1"}

    def test_inject_mac_flapping_nonexistent_bridge(self, kernel):
        with pytest.raises(ValueError, match="does not exist"):
            kernel.inject_mac_flapping("br0", "aa:bb:cc:dd:ee:ff", ["eth0"])

    def test_inject_link_failure(self, kernel):
        kernel.create_bridge("br0", ["eth0"])
        kernel.inject_link_failure("eth0")
        state = kernel.get_interface_state("eth0")
        assert state["carrier"] is False
        assert state["operstate"] == "down"

    def test_inject_link_failure_new_interface(self, kernel):
        kernel.inject_link_failure("eth99")
        state = kernel.get_interface_state("eth99")
        assert state["carrier"] is False

    def test_set_fail_at(self, kernel):
        kernel.set_fail_at(2)
        kernel.create_bridge("br0", [])  # mutation #1 — ok
        with pytest.raises(RuntimeError, match="simulated failure"):
            kernel.create_bridge("br1", [])  # mutation #2 — fails

    def test_reset_clears_all(self, kernel):
        kernel.create_bridge("br0", ["eth0"])
        kernel.create_bond("bond0", "802.3ad", ["eth1"])
        kernel.create_vlan("eth0", 100)
        kernel.inject_failure("create_bridge", "fail")
        kernel.track_resource("bridge", "br0")
        kernel.reset()

        assert kernel.get_bridge("br0") is None
        assert kernel.get_bond("bond0") is None
        assert kernel.get_vlan("eth0", 100) is None
        assert kernel.tracked_resources() == []
        # Should not fail after reset
        kernel.create_bridge("br0", [])


# --- Resource tracking ---

class TestResourceTracking:
    def test_track_resource(self, kernel):
        kernel.track_resource("bridge", "br0")
        assert ("bridge", "br0") in kernel.tracked_resources()

    def test_track_resource_no_duplicates(self, kernel):
        kernel.track_resource("bridge", "br0")
        kernel.track_resource("bridge", "br0")
        assert len(kernel.tracked_resources()) == 1

    def test_untrack_resource(self, kernel):
        kernel.track_resource("bridge", "br0")
        kernel.untrack_resource("bridge", "br0")
        assert kernel.tracked_resources() == []

    def test_untrack_nonexistent(self, kernel):
        # Should not raise
        kernel.untrack_resource("bridge", "br0")
