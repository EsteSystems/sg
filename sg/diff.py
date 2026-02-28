"""Phenotype diffing — compare two genome states.

Compares phenotype maps (current vs snapshot, or two snapshots) and
reports which loci changed dominants, which were added/removed, and
which pathway fusions changed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from sg.phenotype import PhenotypeMap
from sg.registry import Registry
from sg import arena


@dataclass
class LocusDiff:
    locus: str
    change: str  # "added", "removed", "dominant_changed", "fallback_changed"
    old_dominant: str | None = None
    new_dominant: str | None = None
    old_fitness: float = 0.0
    new_fitness: float = 0.0


@dataclass
class PhenotypeDiff:
    loci_changes: list[LocusDiff] = field(default_factory=list)
    fusions_added: list[str] = field(default_factory=list)
    fusions_removed: list[str] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        return bool(self.loci_changes or self.fusions_added or self.fusions_removed)


def diff_phenotypes(
    old: PhenotypeMap,
    new: PhenotypeMap,
    old_reg: Registry | None = None,
    new_reg: Registry | None = None,
) -> PhenotypeDiff:
    """Compare two phenotype maps. Optionally include fitness from registries."""
    diff = PhenotypeDiff()

    old_loci = set(old.loci.keys())
    new_loci = set(new.loci.keys())

    # Added loci
    for locus in sorted(new_loci - old_loci):
        new_dom = new.get_dominant(locus)
        fitness = _get_fitness(new_reg, new_dom) if new_dom else 0.0
        diff.loci_changes.append(LocusDiff(
            locus=locus, change="added",
            new_dominant=new_dom[:12] if new_dom else None,
            new_fitness=fitness,
        ))

    # Removed loci
    for locus in sorted(old_loci - new_loci):
        old_dom = old.get_dominant(locus)
        fitness = _get_fitness(old_reg, old_dom) if old_dom else 0.0
        diff.loci_changes.append(LocusDiff(
            locus=locus, change="removed",
            old_dominant=old_dom[:12] if old_dom else None,
            old_fitness=fitness,
        ))

    # Changed loci
    for locus in sorted(old_loci & new_loci):
        old_dom = old.get_dominant(locus)
        new_dom = new.get_dominant(locus)
        old_stack = old.get_stack(locus)
        new_stack = new.get_stack(locus)

        if old_dom != new_dom:
            diff.loci_changes.append(LocusDiff(
                locus=locus, change="dominant_changed",
                old_dominant=old_dom[:12] if old_dom else None,
                new_dominant=new_dom[:12] if new_dom else None,
                old_fitness=_get_fitness(old_reg, old_dom) if old_dom else 0.0,
                new_fitness=_get_fitness(new_reg, new_dom) if new_dom else 0.0,
            ))
        elif old_stack != new_stack:
            diff.loci_changes.append(LocusDiff(
                locus=locus, change="fallback_changed",
                old_dominant=old_dom[:12] if old_dom else None,
                new_dominant=new_dom[:12] if new_dom else None,
            ))

    # Fusion changes
    old_fused = {
        name for name, cfg in old.pathway_fusions.items()
        if cfg.fused_sha
    }
    new_fused = {
        name for name, cfg in new.pathway_fusions.items()
        if cfg.fused_sha
    }
    diff.fusions_added = sorted(new_fused - old_fused)
    diff.fusions_removed = sorted(old_fused - new_fused)

    return diff


def _get_fitness(reg: Registry | None, sha: str | None) -> float:
    """Get fitness for an allele SHA, or 0.0 if unavailable."""
    if reg is None or sha is None:
        return 0.0
    allele = reg.get(sha)
    if allele is None:
        return 0.0
    return round(arena.compute_fitness(allele), 3)


def format_diff(diff: PhenotypeDiff) -> str:
    """Format a PhenotypeDiff as human-readable text."""
    if not diff.has_changes:
        return "No changes."

    lines = []

    for lc in diff.loci_changes:
        if lc.change == "added":
            lines.append(f"+ {lc.locus}: dominant={lc.new_dominant or 'none'}"
                        f"  fitness={lc.new_fitness:.3f}")
        elif lc.change == "removed":
            lines.append(f"- {lc.locus}: was dominant={lc.old_dominant or 'none'}"
                        f"  fitness={lc.old_fitness:.3f}")
        elif lc.change == "dominant_changed":
            lines.append(f"~ {lc.locus}: {lc.old_dominant or 'none'} -> {lc.new_dominant or 'none'}"
                        f"  fitness {lc.old_fitness:.3f} -> {lc.new_fitness:.3f}")
        elif lc.change == "fallback_changed":
            lines.append(f"~ {lc.locus}: fallback stack changed"
                        f"  (dominant={lc.old_dominant or 'none'})")

    for name in diff.fusions_added:
        lines.append(f"+ fusion: {name}")
    for name in diff.fusions_removed:
        lines.append(f"- fusion: {name}")

    return "\n".join(lines)
