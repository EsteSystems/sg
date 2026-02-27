"""ProductionNetworkKernel — real network operations.

Bridge/bond/VLAN: NetworkManager D-Bus
MAC changes: ip link set (never NM deactivate/reactivate)
FDB/ARP/STP reads: sysfs + ip neighbor
Packet capture: tcpdump subprocess
"""
