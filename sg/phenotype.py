"""Phenotype map — deployment-specific expression.

TOML file mapping loci to dominant alleles + fallback stacks,
plus pathway fusion state.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[no-redef]

import tomli_w


@dataclass
class LocusConfig:
    dominant: str | None = None
    fallback: list[str] = field(default_factory=list)


@dataclass
class PathwayFusionConfig:
    fused_sha: str | None = None
    fused_fallback: str | None = None
    composition_fingerprint: str | None = None


class PhenotypeMap:
    """In-memory representation of the phenotype TOML."""

    def __init__(self) -> None:
        self.loci: dict[str, LocusConfig] = {}
        self.pathway_fusions: dict[str, PathwayFusionConfig] = {}

    def ensure_locus(self, locus: str) -> LocusConfig:
        if locus not in self.loci:
            self.loci[locus] = LocusConfig()
        return self.loci[locus]

    def promote(self, locus: str, sha: str) -> None:
        """Set sha as dominant for locus. Old dominant goes to fallback head."""
        config = self.ensure_locus(locus)
        if config.dominant is not None and config.dominant != sha:
            if config.dominant not in config.fallback:
                config.fallback.insert(0, config.dominant)
        config.dominant = sha
        if sha in config.fallback:
            config.fallback.remove(sha)

    def add_to_fallback(self, locus: str, sha: str) -> None:
        config = self.ensure_locus(locus)
        if sha != config.dominant and sha not in config.fallback:
            config.fallback.append(sha)

    def get_stack(self, locus: str) -> list[str]:
        """Return [dominant, ...fallback] for a locus."""
        config = self.loci.get(locus)
        if config is None:
            return []
        stack = []
        if config.dominant:
            stack.append(config.dominant)
        stack.extend(config.fallback)
        return stack

    def get_dominant(self, locus: str) -> str | None:
        config = self.loci.get(locus)
        return config.dominant if config else None

    # --- Fusion state ---

    def get_fused(self, pathway: str) -> PathwayFusionConfig | None:
        return self.pathway_fusions.get(pathway)

    def set_fused(self, pathway: str, sha: str, fingerprint: str) -> None:
        self.pathway_fusions[pathway] = PathwayFusionConfig(
            fused_sha=sha,
            composition_fingerprint=fingerprint,
        )

    def set_fused_fallback(self, pathway: str, sha: str) -> None:
        config = self.pathway_fusions.get(pathway)
        if config:
            config.fused_fallback = sha

    def clear_fused(self, pathway: str) -> None:
        self.pathway_fusions.pop(pathway, None)

    # --- Persistence ---

    def save(self, path: Path) -> None:
        data: dict = {"locus": {}, "pathway_fusion": {}}
        for key, config in self.loci.items():
            entry: dict = {}
            if config.dominant:
                entry["dominant"] = config.dominant
            if config.fallback:
                entry["fallback"] = config.fallback
            data["locus"][key] = entry
        for key, fusion in self.pathway_fusions.items():
            entry = {}
            if fusion.fused_sha:
                entry["fused_sha"] = fusion.fused_sha
            if fusion.fused_fallback:
                entry["fused_fallback"] = fusion.fused_fallback
            if fusion.composition_fingerprint:
                entry["composition_fingerprint"] = fusion.composition_fingerprint
            data["pathway_fusion"][key] = entry
        path.write_bytes(tomli_w.dumps(data).encode())

    @classmethod
    def load(cls, path: Path) -> PhenotypeMap:
        pm = cls()
        if not path.exists():
            return pm
        with open(path, "rb") as f:
            data = tomllib.load(f)
        for key, entry in data.get("locus", {}).items():
            pm.loci[key] = LocusConfig(
                dominant=entry.get("dominant"),
                fallback=entry.get("fallback", []),
            )
        for key, entry in data.get("pathway_fusion", {}).items():
            pm.pathway_fusions[key] = PathwayFusionConfig(
                fused_sha=entry.get("fused_sha"),
                fused_fallback=entry.get("fused_fallback"),
                composition_fingerprint=entry.get("composition_fingerprint"),
            )
        return pm
