"""Tests for sg.mutation_cache — LLM call caching."""
from __future__ import annotations

import json
import time

import pytest

from sg.mutation_cache import MutationCache


class TestCacheKey:
    def test_deterministic(self):
        k1 = MutationCache.cache_key("contract", "error", "sha")
        k2 = MutationCache.cache_key("contract", "error", "sha")
        assert k1 == k2

    def test_different_inputs_different_keys(self):
        k1 = MutationCache.cache_key("a", "b", "c")
        k2 = MutationCache.cache_key("a", "b", "d")
        assert k1 != k2

    def test_key_length(self):
        k = MutationCache.cache_key("x", "y", "z")
        assert len(k) == 32


class TestMutationCache:
    def test_miss_returns_none(self, tmp_path):
        cache = MutationCache(tmp_path / "cache")
        assert cache.get("nonexistent") is None

    def test_put_and_get(self, tmp_path):
        cache = MutationCache(tmp_path / "cache")
        cache.put("key1", "def execute(x): return x")
        assert cache.get("key1") == "def execute(x): return x"

    def test_expiry(self, tmp_path):
        cache = MutationCache(tmp_path / "cache", ttl=0.1)
        cache.put("key1", "source")
        assert cache.get("key1") == "source"
        time.sleep(0.15)
        assert cache.get("key1") is None

    def test_clear(self, tmp_path):
        cache = MutationCache(tmp_path / "cache")
        cache.put("a", "source_a")
        cache.put("b", "source_b")
        removed = cache.clear()
        assert removed == 2
        assert cache.get("a") is None
        assert cache.get("b") is None

    def test_clear_empty(self, tmp_path):
        cache = MutationCache(tmp_path / "cache")
        assert cache.clear() == 0

    def test_corrupted_file_returns_none(self, tmp_path):
        cache = MutationCache(tmp_path / "cache")
        (tmp_path / "cache").mkdir(parents=True)
        (tmp_path / "cache" / "badkey.json").write_text("not json")
        assert cache.get("badkey") is None

    def test_creates_directory(self, tmp_path):
        cache = MutationCache(tmp_path / "deep" / "cache")
        cache.put("k", "v")
        assert (tmp_path / "deep" / "cache").is_dir()
