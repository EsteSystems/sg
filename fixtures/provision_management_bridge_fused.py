"""Fusion fixture: provision_management_bridge fused gene."""
import json

def execute(input_json: str) -> str:
    data = json.loads(input_json)

    bridge_name = data.get("bridge_name", "")
    interfaces = data.get("interfaces", [])
    uplink = data.get("uplink", "")
    stp_enabled = data.get("stp_enabled", True)
    forward_delay = data.get("forward_delay", 15)

    try:
        gene_sdk.create_bridge(bridge_name, interfaces)
        gene_sdk.set_stp(bridge_name, stp_enabled, forward_delay)
        gene_sdk.attach_interface(bridge_name, uplink)

        original_mac = gene_sdk.get_device_mac(bridge_name)
        gene_sdk.send_gratuitous_arp(bridge_name, original_mac)

        return json.dumps({
            "success": True,
            "bridge_name": bridge_name,
            "stp_enabled": stp_enabled,
            "uplink": uplink,
            "mac": original_mac,
        })
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})
