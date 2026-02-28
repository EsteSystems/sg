"""Multi-organism federation — allele sharing over HTTP.

Organisms share successful alleles with peers. Each organism makes
independent promotion decisions. No consensus required.

Security: SHA-256 integrity verification on import, optional shared
secret for peer authentication.
"""
from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass, field
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
    secret: str = ""


def load_peers(config_path: Path) -> list[PeerConfig]:
    """Load peer list from peers.json."""
    if not config_path.exists():
        return []
    data = json.loads(config_path.read_text())
    return [PeerConfig(**p) for p in data.get("peers", [])]


def compute_source_sha(source: str) -> str:
    """SHA-256 of source code, used for integrity verification."""
    return hashlib.sha256(source.encode()).hexdigest()


def sign_payload(payload: dict, secret: str) -> str:
    """HMAC-SHA256 signature over the JSON payload."""
    body = json.dumps(payload, sort_keys=True).encode()
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def verify_signature(payload: dict, signature: str, secret: str) -> bool:
    """Verify HMAC-SHA256 signature."""
    expected = sign_payload(payload, secret)
    return hmac.compare_digest(expected, signature)


def export_allele(registry: Registry, sha: str) -> dict | None:
    """Package an allele for sharing: metadata + source + integrity hash."""
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
        "source_sha256": compute_source_sha(source),
        "fitness": arena.compute_fitness(allele),
        "successful_invocations": allele.successful_invocations,
        "total_invocations": allele.total_invocations,
    }


def verify_allele_integrity(data: dict) -> bool:
    """Verify that source_sha256 matches the actual source content."""
    source = data.get("source", "")
    expected = data.get("source_sha256", "")
    if not expected:
        return True  # no hash provided — accept (backwards compat)
    return compute_source_sha(source) == expected


def import_allele(registry: Registry, data: dict) -> str:
    """Import a shared allele after integrity check. Returns SHA."""
    if not verify_allele_integrity(data):
        raise ValueError("allele integrity check failed: source_sha256 mismatch")
    sha = registry.register(
        data["source"],
        data["locus"],
        generation=data.get("generation", 0),
    )
    return sha


def push_allele(peer: PeerConfig, allele_data: dict) -> bool:
    """Push an allele to a peer. Returns success."""
    try:
        headers = {}
        if peer.secret:
            headers["X-SG-Signature"] = sign_payload(allele_data, peer.secret)
        resp = httpx.post(
            f"{peer.url}/api/federation/receive",
            json=allele_data,
            headers=headers,
            timeout=30.0,
        )
        return resp.status_code == 200
    except Exception:
        return False


def pull_alleles(peer: PeerConfig, locus: str) -> list[dict]:
    """Pull alleles for a locus from a peer."""
    try:
        headers = {}
        if peer.secret:
            headers["X-SG-Signature"] = sign_payload({"locus": locus}, peer.secret)
        resp = httpx.get(
            f"{peer.url}/api/federation/alleles/{locus}",
            headers=headers,
            timeout=30.0,
        )
        if resp.status_code == 200:
            return resp.json().get("alleles", [])
    except Exception:
        pass
    return []
