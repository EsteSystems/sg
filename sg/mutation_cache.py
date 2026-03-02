"""LLM mutation call cache — disk-backed with TTL expiry.

Caches the source code returned by LLM mutation calls, keyed by a hash
of (contract_text, error_message, parent_sha). Avoids redundant LLM
calls when the same failure/locus combination is retried.

Usage::

    from sg.mutation_cache import MutationCache

    cache = MutationCache(Path(".sg/mutation_cache"))
    key = MutationCache.cache_key(contract, error, parent_sha)
    hit = cache.get(key)
    if hit is None:
        source = llm_call(...)
        cache.put(key, source)
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path


class MutationCache:
    """Disk-backed cache for LLM mutation results.

    Each entry is a JSON file named by key, containing the generated
    source and an expiry timestamp. Expired entries are ignored on read
    and can be cleaned up with :meth:`clear`.
    """

    def __init__(self, cache_path: Path, ttl: float = 3600.0) -> None:
        self.cache_path = cache_path
        self.ttl = ttl

    @staticmethod
    def cache_key(contract_text: str, error_message: str,
                  parent_sha: str) -> str:
        """Compute a deterministic cache key from mutation inputs."""
        combined = f"{contract_text}|{error_message}|{parent_sha}"
        return hashlib.sha256(combined.encode()).hexdigest()[:32]

    def get(self, key: str) -> str | None:
        """Return the cached source code, or None on miss/expiry."""
        path = self.cache_path / f"{key}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return None
        if time.time() > data.get("expires", 0):
            return None
        return data.get("source")

    def put(self, key: str, source: str) -> None:
        """Store a mutation result in the cache."""
        self.cache_path.mkdir(parents=True, exist_ok=True)
        path = self.cache_path / f"{key}.json"
        data = {
            "source": source,
            "expires": time.time() + self.ttl,
            "created": time.time(),
        }
        path.write_text(json.dumps(data))

    def clear(self) -> int:
        """Remove all cache entries. Returns count of files removed."""
        if not self.cache_path.exists():
            return 0
        count = 0
        for path in self.cache_path.glob("*.json"):
            path.unlink()
            count += 1
        return count
