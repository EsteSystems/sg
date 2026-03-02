#!/usr/bin/env python3
"""Live federation demo: two organisms share alleles over HTTP.

Demonstrates multi-organism cooperation:
  1. Organism A and Organism B start with different seed genes
  2. Both start their dashboard servers (federation endpoints)
  3. Organism A pushes its dominant allele to Organism B
  4. Organism B pulls alleles from Organism A
  5. Both organisms now have each other's alleles as recessives

Usage:
  python demo/federation_demo.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sg import arena
from sg.contracts import ContractStore
from sg.federation import (
    PeerConfig, export_allele, import_allele, push_allele, pull_alleles,
)
from sg.fusion import FusionTracker
from sg_network import MockNetworkKernel
from sg.mutation import MockMutationEngine
from sg.orchestrator import Orchestrator
from sg.phenotype import PhenotypeMap
from sg.registry import Registry

import sg_network

CONTRACTS_DIR = sg_network.contracts_path()
GENES_DIR = sg_network.genes_path()
FIXTURES_DIR = sg_network.fixtures_path()


def setup_organism(name: str, base_dir: Path, loci: list[str]) -> Path:
    """Set up a project directory for one organism with specific loci seeded."""
    org_dir = base_dir / name
    org_dir.mkdir()

    shutil.copytree(CONTRACTS_DIR, org_dir / "contracts")
    shutil.copytree(FIXTURES_DIR, org_dir / "fixtures")

    contract_store = ContractStore.open(org_dir / "contracts")
    registry = Registry.open(org_dir / ".sg" / "registry")
    phenotype = PhenotypeMap()

    for locus in loci:
        candidates = sorted(GENES_DIR.glob(f"{locus}_*.py"))
        if candidates:
            source = candidates[0].read_text()
            sha = registry.register(source, locus)
            phenotype.promote(locus, sha)
            allele = registry.get(sha)
            allele.state = "dominant"

    registry.save_index()
    phenotype.save(org_dir / "phenotype.toml")
    FusionTracker.open(org_dir / "fusion_tracker.json").save(
        org_dir / "fusion_tracker.json"
    )
    return org_dir


def start_dashboard(org_dir: Path, port: int) -> threading.Thread:
    """Start a dashboard server in a background thread."""
    import sg.dashboard as dash
    import uvicorn

    dash._project_root = org_dir

    # Create a fresh app instance for each organism
    from fastapi import FastAPI, Body
    from fastapi.responses import HTMLResponse, JSONResponse

    app = FastAPI(title=f"Organism @ {port}")

    # Re-register routes pointing at this organism's state
    @app.get("/api/federation/alleles/{locus}")
    def federation_alleles(locus: str):
        cs = ContractStore.open(org_dir / "contracts")
        reg = Registry.open(org_dir / ".sg" / "registry")
        alleles = reg.alleles_for_locus(locus)
        result = []
        for a in alleles[:5]:
            data = export_allele(reg, a.sha256)
            if data:
                result.append(data)
        return {"alleles": result}

    @app.post("/api/federation/receive")
    def federation_receive(data: dict = Body(...)):
        reg = Registry.open(org_dir / ".sg" / "registry")
        pheno = PhenotypeMap.load(org_dir / "phenotype.toml")
        sha = import_allele(reg, data)
        locus = data.get("locus", "")
        pheno.add_to_fallback(locus, sha)
        allele = reg.get(sha)
        if allele:
            allele.state = "recessive"
        reg.save_index()
        pheno.save(org_dir / "phenotype.toml")
        return {"status": "ok", "sha": sha[:12]}

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait for server to be ready
    import httpx
    for _ in range(20):
        time.sleep(0.25)
        try:
            httpx.get(f"http://127.0.0.1:{port}/api/federation/alleles/test",
                      timeout=1.0)
            break
        except Exception:
            pass
    return thread


def show_organism_state(name: str, org_dir: Path):
    """Print the allele state of an organism."""
    registry = Registry.open(org_dir / ".sg" / "registry")
    phenotype = PhenotypeMap.load(org_dir / "phenotype.toml")

    print(f"  [{name}] alleles:")
    for sha, allele in registry.alleles.items():
        dom = phenotype.get_dominant(allele.locus)
        marker = " (dominant)" if sha == dom else " (recessive)"
        fitness = arena.compute_fitness(allele)
        print(f"    {allele.locus}: {sha[:12]} gen={allele.generation} "
              f"state={allele.state} fitness={fitness:.3f}{marker}")


def run_demo():
    print("=" * 60)
    print("  Software Genome — Federation Demo")
    print("  Two organisms sharing alleles over HTTP")
    print("=" * 60)
    print()

    with tempfile.TemporaryDirectory(prefix="sg-federation-") as tmp:
        base_dir = Path(tmp)

        # --- Phase 1: Setup ---
        print("--- Phase 1: Setting up two organisms ---")
        print()

        # Organism A: has bridge_create and bridge_stp
        org_a_dir = setup_organism(
            "organism-alpha", base_dir, ["bridge_create", "bridge_stp"]
        )
        print("  Organism Alpha: bridge_create, bridge_stp")

        # Organism B: has bond_create and bridge_create
        org_b_dir = setup_organism(
            "organism-beta", base_dir, ["bond_create", "bridge_create"]
        )
        print("  Organism Beta:  bond_create, bridge_create")
        print()

        show_organism_state("Alpha", org_a_dir)
        print()
        show_organism_state("Beta", org_b_dir)
        print()

        # --- Phase 2: Start dashboards ---
        print("--- Phase 2: Starting federation servers ---")
        port_a, port_b = 8421, 8422
        start_dashboard(org_a_dir, port_a)
        print(f"  Alpha listening on :{port_a}")
        start_dashboard(org_b_dir, port_b)
        print(f"  Beta  listening on :{port_b}")
        print()

        peer_a = PeerConfig(url=f"http://127.0.0.1:{port_a}", name="alpha")
        peer_b = PeerConfig(url=f"http://127.0.0.1:{port_b}", name="beta")

        # --- Phase 3: Push alleles ---
        print("--- Phase 3: Alpha pushes bridge_stp to Beta ---")
        reg_a = Registry.open(org_a_dir / ".sg" / "registry")
        pheno_a = PhenotypeMap.load(org_a_dir / "phenotype.toml")

        stp_sha = pheno_a.get_dominant("bridge_stp")
        if stp_sha:
            allele_data = export_allele(reg_a, stp_sha)
            if allele_data:
                import httpx
                try:
                    resp = httpx.post(
                        f"{peer_b.url}/api/federation/receive",
                        json=allele_data, timeout=10.0,
                    )
                    ok = resp.status_code == 200
                    print(f"  Push bridge_stp ({stp_sha[:12]}) to Beta: "
                          f"{'success' if ok else f'FAILED ({resp.status_code}: {resp.text})'}")
                except Exception as e:
                    print(f"  Push bridge_stp ({stp_sha[:12]}) to Beta: "
                          f"FAILED ({type(e).__name__}: {e})")
        print()

        # --- Phase 4: Pull alleles ---
        print("--- Phase 4: Alpha pulls bond_create from Beta ---")
        alleles = pull_alleles(peer_b, "bond_create")
        print(f"  Received {len(alleles)} allele(s) from Beta")

        reg_a_fresh = Registry.open(org_a_dir / ".sg" / "registry")
        pheno_a_fresh = PhenotypeMap.load(org_a_dir / "phenotype.toml")
        for data in alleles:
            sha = import_allele(reg_a_fresh, data)
            pheno_a_fresh.add_to_fallback("bond_create", sha)
            allele = reg_a_fresh.get(sha)
            if allele:
                allele.state = "recessive"
            print(f"  Imported {sha[:12]} (bond_create) into Alpha as recessive")
        if alleles:
            reg_a_fresh.save_index()
            pheno_a_fresh.save(org_a_dir / "phenotype.toml")
        print()

        # --- Phase 5: Final state ---
        print("--- Phase 5: Final state after federation ---")
        print()
        show_organism_state("Alpha", org_a_dir)
        print()
        show_organism_state("Beta", org_b_dir)
        print()

        # --- Phase 6: Execute shared alleles ---
        print("--- Phase 6: Execute shared alleles ---")
        print()

        # Beta now has bridge_stp — execute it
        print("  Beta executes bridge_stp (received from Alpha):")
        cs_b = ContractStore.open(org_b_dir / "contracts")
        reg_b = Registry.open(org_b_dir / ".sg" / "registry")
        pheno_b = PhenotypeMap.load(org_b_dir / "phenotype.toml")
        ft_b = FusionTracker.open(org_b_dir / "fusion_tracker.json")
        kernel_b = MockNetworkKernel()
        me_b = MockMutationEngine(org_b_dir / "fixtures")

        orch_b = Orchestrator(
            registry=reg_b, phenotype=pheno_b,
            mutation_engine=me_b, fusion_tracker=ft_b,
            kernel=kernel_b, contract_store=cs_b, project_root=org_b_dir,
        )

        # Create bridge first (Beta has bridge_create)
        orch_b.execute_locus("bridge_create", json.dumps({
            "bridge_name": "br0", "interfaces": ["eth0"],
        }))

        # Now try bridge_stp from Alpha
        stp_dom = pheno_b.get_dominant("bridge_stp")
        if stp_dom is None:
            # Promote the received allele
            stp_alleles = reg_b.alleles_for_locus("bridge_stp")
            if stp_alleles:
                pheno_b.promote("bridge_stp", stp_alleles[0].sha256)
                stp_alleles[0].state = "dominant"

        result = orch_b.execute_locus("bridge_stp", json.dumps({
            "bridge_name": "br0", "stp_enabled": True, "forward_delay": 15,
        }))

        if result:
            output, used_sha = result
            print(f"    Success! Used allele {used_sha[:12]}")
            print(f"    Output: {output}")
        else:
            print("    Failed (allele may not be compatible with Beta's kernel state)")
        print()

    print("=" * 60)
    print("  Federation demo complete.")
    print("  Two organisms successfully shared alleles over HTTP.")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
