#!/usr/bin/env python3
"""Run the Software Genome against a real production server.

Safety:
- All resources use sg-test-* prefix
- vswitch0 is protected via SG_PROTECTED_INTERFACES env var
- Snapshot created before any changes
- Cleanup in try/finally
- Dry-run mode available

Usage:
    python3 demo/production_run.py --dry-run          # safe preview
    python3 demo/production_run.py                    # live run
    python3 demo/production_run.py --host 192.168.5.5 # custom host
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description="Production run demo")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--host", default="localhost",
                        help="Target host (default: localhost)")
    args = parser.parse_args()

    # Always protect management interfaces — set BEFORE importing production kernel
    os.environ["SG_PROTECTED_INTERFACES"] = "vswitch0,em1,em2,vswitch1,vswitch2,vswitch3,cni0,virbr0"

    from sg_network.production import ProductionNetworkKernel, PROTECTED_INTERFACES
    from sg.snapshot import SnapshotManager
    from sg.contracts import ContractStore
    from sg.fusion import FusionTracker
    from sg_network import MockNetworkKernel
    from sg.mutation import MockMutationEngine
    from sg.orchestrator import Orchestrator
    from sg.phenotype import PhenotypeMap
    from sg.registry import Registry
    from pathlib import Path

    print("=== Software Genome Production Run ===")
    print(f"Host: {args.host}")
    print(f"Dry run: {args.dry_run}")
    print(f"Protected interfaces: {PROTECTED_INTERFACES}")
    print()

    root = Path(os.environ.get("SG_PROJECT_ROOT",
                               os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    # Create snapshot before production changes
    mgr = SnapshotManager(root)
    try:
        snap = mgr.create(name="pre-production", description="before production run")
        print(f"Snapshot created: {snap.name}")
    except ValueError:
        print("Snapshot 'pre-production' already exists, continuing")

    # Create production kernel
    kernel = ProductionNetworkKernel(dry_run=args.dry_run)

    # Load genome state
    contract_store = ContractStore.open(root / "contracts")
    registry = Registry.open(root / ".sg" / "registry")
    phenotype = PhenotypeMap.load(root / "phenotype.toml")
    fusion_tracker = FusionTracker.open(root / "fusion_tracker.json")
    mutation_engine = MockMutationEngine(root / "fixtures")

    orch = Orchestrator(
        registry=registry,
        phenotype=phenotype,
        mutation_engine=mutation_engine,
        fusion_tracker=fusion_tracker,
        kernel=kernel,
        contract_store=contract_store,
        project_root=root,
    )

    try:
        # Run configure_bridge_with_stp with sg-test-* names
        input_json = json.dumps({
            "bridge_name": "sg-test-br0",
            "interfaces": ["sg-test-eth0", "sg-test-eth1"],
            "stp_enabled": True,
            "forward_delay": 15,
        })

        print("\n--- Running configure_bridge_with_stp ---")
        outputs = orch.run_pathway("configure_bridge_with_stp", input_json)
        for i, output in enumerate(outputs):
            print(f"Step {i+1}: {output}")

        # Run health check
        health_input = json.dumps({
            "bridge_name": "sg-test-br0",
        })

        print("\n--- Running health_check_bridge ---")
        health_outputs = orch.run_pathway("health_check_bridge", health_input)
        for i, output in enumerate(health_outputs):
            print(f"Step {i+1}: {output}")

        orch.save_state()
        print("\n--- Production run complete ---")

    finally:
        # Always clean up test resources
        print("\n--- Cleanup ---")
        kernel.cleanup_all_test_resources()
        print("All sg-test-* resources cleaned up")


if __name__ == "__main__":
    main()
