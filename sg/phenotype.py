"""Phenotype map — deployment-specific expression.

TOML file mapping loci to dominant alleles + fallback stacks,
plus pathway fusion state.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

from sg.filelock import atomic_write_bytes, file_lock
from sg.log import get_logger

logger = get_logger("phenotype")

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


@dataclass
class PathwayAlleleConfig:
    dominant: str | None = None
    fallback: list[str] = field(default_factory=list)


@dataclass
class TopologyAlleleConfig:
    dominant: str | None = None
    fallback: list[str] = field(default_factory=list)


class PhenotypeMap:
    """In-memory representation of the phenotype TOML."""

    def __init__(self) -> None:
        self.loci: dict[str, LocusConfig] = {}
        self.pathway_fusions: dict[str, PathwayFusionConfig] = {}
        self.pathway_alleles: dict[str, PathwayAlleleConfig] = {}
        self.topology_alleles: dict[str, TopologyAlleleConfig] = {}

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

    # --- Pathway alleles ---

    def ensure_pathway_allele(self, pathway_name: str) -> PathwayAlleleConfig:
        if pathway_name not in self.pathway_alleles:
            self.pathway_alleles[pathway_name] = PathwayAlleleConfig()
        return self.pathway_alleles[pathway_name]

    def promote_pathway(self, pathway_name: str, sha: str) -> None:
        """Set sha as dominant pathway allele. Old dominant goes to fallback head."""
        config = self.ensure_pathway_allele(pathway_name)
        if config.dominant is not None and config.dominant != sha:
            if config.dominant not in config.fallback:
                config.fallback.insert(0, config.dominant)
        config.dominant = sha
        if sha in config.fallback:
            config.fallback.remove(sha)

    def add_pathway_fallback(self, pathway_name: str, sha: str) -> None:
        config = self.ensure_pathway_allele(pathway_name)
        if sha != config.dominant and sha not in config.fallback:
            config.fallback.append(sha)

    def get_pathway_stack(self, pathway_name: str) -> list[str]:
        """Return [dominant, ...fallback] pathway allele SHAs."""
        config = self.pathway_alleles.get(pathway_name)
        if config is None:
            return []
        stack = []
        if config.dominant:
            stack.append(config.dominant)
        stack.extend(config.fallback)
        return stack

    def get_pathway_dominant(self, pathway_name: str) -> str | None:
        config = self.pathway_alleles.get(pathway_name)
        return config.dominant if config else None

    # --- Topology alleles ---

    def ensure_topology_allele(self, topology_name: str) -> TopologyAlleleConfig:
        if topology_name not in self.topology_alleles:
            self.topology_alleles[topology_name] = TopologyAlleleConfig()
        return self.topology_alleles[topology_name]

    def promote_topology(self, topology_name: str, sha: str) -> None:
        """Set sha as dominant topology allele. Old dominant goes to fallback head."""
        config = self.ensure_topology_allele(topology_name)
        if config.dominant is not None and config.dominant != sha:
            if config.dominant not in config.fallback:
                config.fallback.insert(0, config.dominant)
        config.dominant = sha
        if sha in config.fallback:
            config.fallback.remove(sha)

    def add_topology_fallback(self, topology_name: str, sha: str) -> None:
        config = self.ensure_topology_allele(topology_name)
        if sha != config.dominant and sha not in config.fallback:
            config.fallback.append(sha)

    def get_topology_stack(self, topology_name: str) -> list[str]:
        """Return [dominant, ...fallback] topology allele SHAs."""
        config = self.topology_alleles.get(topology_name)
        if config is None:
            return []
        stack = []
        if config.dominant:
            stack.append(config.dominant)
        stack.extend(config.fallback)
        return stack

    def get_topology_dominant(self, topology_name: str) -> str | None:
        config = self.topology_alleles.get(topology_name)
        return config.dominant if config else None

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
        data["pathway_allele"] = {}
        for key, config in self.pathway_alleles.items():
            entry = {}
            if config.dominant:
                entry["dominant"] = config.dominant
            if config.fallback:
                entry["fallback"] = config.fallback
            data["pathway_allele"][key] = entry
        data["topology_allele"] = {}
        for key, config in self.topology_alleles.items():
            entry = {}
            if config.dominant:
                entry["dominant"] = config.dominant
            if config.fallback:
                entry["fallback"] = config.fallback
            data["topology_allele"][key] = entry
        with file_lock(path):
            atomic_write_bytes(path, tomli_w.dumps(data).encode())

    @classmethod
    def load(cls, path: Path) -> PhenotypeMap:
        pm = cls()
        if not path.exists():
            return pm
        try:
            with open(path, "rb") as f:
                data = tomllib.load(f)
        except Exception:
            logger.warning("phenotype map corrupted at %s, starting fresh", path)
            return pm
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
        for key, entry in data.get("pathway_allele", {}).items():
            pm.pathway_alleles[key] = PathwayAlleleConfig(
                dominant=entry.get("dominant"),
                fallback=entry.get("fallback", []),
            )
        for key, entry in data.get("topology_allele", {}).items():
            pm.topology_alleles[key] = TopologyAlleleConfig(
                dominant=entry.get("dominant"),
                fallback=entry.get("fallback", []),
            )
        return pm
