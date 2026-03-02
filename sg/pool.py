"""Gene pool — share high-fitness alleles across organisms.

Pools are HTTP services where organisms push and pull alleles.
Push eligibility: high fitness, sufficient invocations, dominant or formerly dominant.
Pull integration: pulled alleles enter as recessive and compete on local fitness.

Contribution tracking: organisms must push to pull (reciprocity enforcement).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from sg import arena
from sg.registry import Registry, AlleleMetadata
from sg.phenotype import PhenotypeMap

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]

try:
    import tomli
except ImportError:
    try:
        import tomllib as tomli  # type: ignore[no-redef]
    except ImportError:
        tomli = None  # type: ignore[assignment]


# --- Configuration ---


@dataclass
class PoolConfig:
    """Configuration for a gene pool."""
    name: str
    url: str
    token_env: str = ""

    @property
    def token(self) -> str | None:
        if self.token_env:
            return os.environ.get(self.token_env)
        return None


def load_pool_configs(config_path: Path) -> list[PoolConfig]:
    """Load pool configurations from pools.toml."""
    if not config_path.exists():
        return []
    text = config_path.read_text()
    if tomli is not None:
        data = tomli.loads(text)
    else:
        data = {}
    pools = data.get("pool", [])
    if isinstance(pools, dict):
        pools = [pools]
    return [PoolConfig(**p) for p in pools]


# --- Membership tracking ---


@dataclass
class PoolMembership:
    """Tracks an organism's relationship with a pool."""
    pool_name: str
    pool_url: str
    organism_id: str = ""
    last_push: float | None = None
    last_pull: float | None = None
    total_pushed: int = 0
    total_pulled: int = 0
    access_granted: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> PoolMembership:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class MembershipStore:
    """Persists pool membership state."""

    def __init__(self, path: Path):
        self.path = path
        self._memberships: dict[str, PoolMembership] = {}
        if path.exists():
            data = json.loads(path.read_text())
            for name, m in data.items():
                self._memberships[name] = PoolMembership.from_dict(m)

    def get(self, pool_name: str) -> PoolMembership | None:
        return self._memberships.get(pool_name)

    def get_or_create(self, pool_name: str, pool_url: str) -> PoolMembership:
        if pool_name not in self._memberships:
            self._memberships[pool_name] = PoolMembership(
                pool_name=pool_name, pool_url=pool_url,
            )
        return self._memberships[pool_name]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {name: m.to_dict() for name, m in self._memberships.items()}
        self.path.write_text(json.dumps(data, indent=2))


# --- Push eligibility ---

DEFAULT_MIN_FITNESS = 0.85
DEFAULT_MIN_INVOCATIONS = 50


def is_push_eligible(
    allele: AlleleMetadata,
    min_fitness: float = DEFAULT_MIN_FITNESS,
    min_invocations: int = DEFAULT_MIN_INVOCATIONS,
) -> bool:
    """Check if an allele is eligible for pool push.

    Criteria:
    - Fitness above threshold
    - Minimum invocation count
    - Currently dominant or formerly dominant (not deprecated)
    """
    if allele.total_invocations < min_invocations:
        return False
    fitness = arena.compute_fitness(allele)
    if fitness < min_fitness:
        return False
    if allele.state in ("deprecated",):
        return False
    return True


# --- Pool client ---


class PoolClient:
    """Client for interacting with gene pools."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.configs = load_pool_configs(project_root / "pools.toml")
        self.memberships = MembershipStore(
            project_root / ".sg" / "pool_memberships.json"
        )

    def _get_config(self, pool_name: str) -> PoolConfig:
        for cfg in self.configs:
            if cfg.name == pool_name:
                return cfg
        raise ValueError(f"no pool configured with name: {pool_name}")

    def _headers(self, config: PoolConfig) -> dict[str, str]:
        headers: dict[str, str] = {}
        token = config.token
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def push(
        self,
        locus: str,
        registry: Registry,
        phenotype: PhenotypeMap,
        pool_name: str,
    ) -> bool:
        """Push the dominant allele for a locus to a pool.

        Returns True if push succeeded, False otherwise.
        """
        config = self._get_config(pool_name)
        dominant_sha = phenotype.get_dominant(locus)
        if dominant_sha is None:
            return False

        allele = registry.get(dominant_sha)
        if allele is None:
            return False

        if not is_push_eligible(allele):
            return False

        source = registry.load_source(dominant_sha)
        if source is None:
            return False

        payload = {
            "locus": locus,
            "sha256": allele.sha256,
            "source": source,
            "generation": allele.generation,
            "fitness": arena.compute_fitness(allele),
            "successful_invocations": allele.successful_invocations,
            "total_invocations": allele.total_invocations,
        }

        if httpx is None:
            raise RuntimeError("httpx required for pool operations: pip install httpx")

        try:
            resp = httpx.post(
                f"{config.url}/pool/push",
                json=payload,
                headers=self._headers(config),
                timeout=30.0,
            )
            if resp.status_code == 200:
                membership = self.memberships.get_or_create(pool_name, config.url)
                membership.last_push = time.time()
                membership.total_pushed += 1
                self.memberships.save()
                return True
        except Exception:
            pass
        return False

    def pull(
        self,
        locus: str,
        registry: Registry,
        phenotype: PhenotypeMap,
        pool_name: str,
    ) -> list[str]:
        """Pull alleles for a locus from a pool.

        Returns list of imported allele SHAs.
        Pulled alleles enter as recessive.
        """
        config = self._get_config(pool_name)

        if httpx is None:
            raise RuntimeError("httpx required for pool operations: pip install httpx")

        try:
            resp = httpx.get(
                f"{config.url}/pool/pull/{locus}",
                headers=self._headers(config),
                timeout=30.0,
            )
            if resp.status_code != 200:
                return []
            alleles_data = resp.json().get("alleles", [])
        except Exception:
            return []

        imported: list[str] = []
        for data in alleles_data:
            sha = registry.register(
                data["source"],
                data["locus"],
                generation=data.get("generation", 0),
            )
            allele = registry.get(sha)
            if allele:
                allele.state = "recessive"
            phenotype.add_to_fallback(locus, sha)
            imported.append(sha)

        if imported:
            membership = self.memberships.get_or_create(pool_name, config.url)
            membership.last_pull = time.time()
            membership.total_pulled += len(imported)
            self.memberships.save()

        return imported

    def status(self, pool_name: str) -> PoolMembership | None:
        """Get membership status for a pool."""
        return self.memberships.get(pool_name)

    def list_pools(self) -> list[PoolConfig]:
        """Return all configured pools."""
        return list(self.configs)

    def auto(
        self,
        registry: Registry,
        phenotype: PhenotypeMap,
        contract_store: Any,
        pool_name: str,
    ) -> dict[str, Any]:
        """Automatic push/pull cycle.

        Pushes all eligible dominant alleles, then pulls new alleles
        for all known loci.

        Returns summary dict with push/pull counts.
        """
        config = self._get_config(pool_name)
        pushed = 0
        pulled = 0
        push_errors: list[str] = []
        pull_errors: list[str] = []

        # Push all eligible
        for locus in contract_store.known_loci():
            dominant_sha = phenotype.get_dominant(locus)
            if dominant_sha is None:
                continue
            allele = registry.get(dominant_sha)
            if allele is None:
                continue
            if not is_push_eligible(allele):
                continue
            try:
                if self.push(locus, registry, phenotype, pool_name):
                    pushed += 1
            except Exception as e:
                push_errors.append(f"{locus}: {e}")

        # Pull for all known loci
        for locus in contract_store.known_loci():
            try:
                shas = self.pull(locus, registry, phenotype, pool_name)
                pulled += len(shas)
            except Exception as e:
                pull_errors.append(f"{locus}: {e}")

        return {
            "pushed": pushed,
            "pulled": pulled,
            "push_errors": push_errors,
            "pull_errors": pull_errors,
        }
