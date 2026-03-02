"""Mutation fixture: mac_preserve fix."""
import json

def execute(input_json: str) -> str:
    data = json.loads(input_json)
    device = data.get("device", "")
    try:
        original_mac = gene_sdk.get_device_mac(device)
        source_mac = data.get("source_mac", original_mac)
        if source_mac != original_mac:
            gene_sdk.set_device_mac(device, source_mac)
        gene_sdk.send_gratuitous_arp(device, source_mac)
        return json.dumps({"success": True, "original_mac": original_mac, "new_mac": source_mac})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})
