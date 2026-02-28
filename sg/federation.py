"""Multi-organism federation — allele sharing over HTTP.

Organisms share successful alleles with peers. Each organism makes
independent promotion decisions. No consensus required.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from sg import arena
from sg.registry import Registry, AlleleMetadata

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]


@dataclass
class PeerConfig:
    """A federation peer."""
    url: str
    name: str = ""


def load_peers(config_path: Path) -> list[PeerConfig]:
    """Load peer list from peers.json."""
    if not config_path.exists():
        return []
    data = json.loads(config_path.read_text())
    return [PeerConfig(**p) for p in data.get("peers", [])]


def export_allele(registry: Registry, sha: str) -> dict | None:
    """Package an allele for sharing: metadata + source."""
    allele = registry.get(sha)
    if allele is None:
        return None
    source = registry.load_source(sha)
    if source is None:
        return None
    return {
        "sha256": allele.sha256,
        "locus": allele.locus,
        "generation": allele.generation,
        "source": source,
        "fitness": arena.compute_fitness(allele),
        "successful_invocations": allele.successful_invocations,
        "total_invocations": allele.total_invocations,
    }


def import_allele(registry: Registry, data: dict) -> str:
    """Import a shared allele. Returns SHA."""
    sha = registry.register(
        data["source"],
        data["locus"],
        generation=data.get("generation", 0),
    )
    return sha


def push_allele(peer: PeerConfig, allele_data: dict) -> bool:
    """Push an allele to a peer. Returns success."""
    try:
        resp = httpx.post(
            f"{peer.url}/api/federation/receive",
            json=allele_data,
            timeout=30.0,
        )
        return resp.status_code == 200
    except Exception:
        return False


def pull_alleles(peer: PeerConfig, locus: str) -> list[dict]:
    """Pull alleles for a locus from a peer."""
    try:
        resp = httpx.get(
            f"{peer.url}/api/federation/alleles/{locus}",
            timeout=30.0,
        )
        if resp.status_code == 200:
            return resp.json().get("alleles", [])
    except Exception:
        pass
    return []
