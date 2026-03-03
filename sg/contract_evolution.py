"""Contract evolution — tightening, relaxation, and feeds discovery.

Analyzes runtime observations of dominant alleles against their contracts
to propose contract refinements.  Follows the FailureDiscovery proposal
pattern: threshold-based, pending/accepted/rejected lifecycle, JSON
persistence with bounded growth.

Three analysis modes:
1. **Tightening** — detect extra fields, always-present optionals, narrow ranges
2. **Relaxation** — detect high mutation failure rates and violated constraints
3. **Feeds discovery** — Pearson correlation between diagnostic outputs and
   configuration gene fitness
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sg.filelock import atomic_write_text, file_lock, file_lock_shared
from sg.log import get_logger

logger = get_logger("contract_evolution")

TIGHTENING_THRESHOLD = 50
RELAXATION_THRESHOLD = 0.3
MIN_CORRELATION_SAMPLES = 30
CORRELATION_THRESHOLD = 0.6
MAX_OBSERVATIONS = 200
MAX_PROPOSALS_PER_LOCUS = 20
MAX_RANGE_SAMPLES = 50
MAX_CORRELATIONS = 200


# --- Data structures ---


@dataclass
class ContractProposal:
    """A proposed change to a gene contract."""
    locus: str
    proposal_type: str  # tighten_gives, tighten_before, relax_constraint, add_feeds
    description: str
    current_value: str = ""
    proposed_value: str = ""
    evidence_count: int = 0
    created_at: float = field(default_factory=time.time)
    status: str = "pending"

    def to_dict(self) -> dict:
        return {
            "locus": self.locus,
            "proposal_type": self.proposal_type,
            "description": self.description,
            "current_value": self.current_value,
            "proposed_value": self.proposed_value,
            "evidence_count": self.evidence_count,
            "created_at": self.created_at,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ContractProposal:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class OutputObservation:
    """Tracks fields actually produced by a dominant allele's outputs."""
    locus: str = ""
    field_counts: dict[str, int] = field(default_factory=dict)
    total_observations: int = 0
    extra_fields: dict[str, int] = field(default_factory=dict)
    field_value_ranges: dict[str, list[float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "locus": self.locus,
            "field_counts": self.field_counts,
            "total_observations": self.total_observations,
            "extra_fields": self.extra_fields,
            "field_value_ranges": {
                k: v[:MAX_RANGE_SAMPLES]
                for k, v in self.field_value_ranges.items()
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> OutputObservation:
        return cls(
            locus=d.get("locus", ""),
            field_counts=d.get("field_counts", {}),
            total_observations=d.get("total_observations", 0),
            extra_fields=d.get("extra_fields", {}),
            field_value_ranges=d.get("field_value_ranges", {}),
        )


@dataclass
class MutationFailureTracker:
    """Tracks mutation failure rates and which constraints are violated."""
    locus: str = ""
    total_mutations: int = 0
    failed_mutations: int = 0
    constraint_violations: dict[str, int] = field(default_factory=dict)

    @property
    def failure_rate(self) -> float:
        if self.total_mutations == 0:
            return 0.0
        return self.failed_mutations / self.total_mutations

    def to_dict(self) -> dict:
        return {
            "locus": self.locus,
            "total_mutations": self.total_mutations,
            "failed_mutations": self.failed_mutations,
            "constraint_violations": self.constraint_violations,
        }

    @classmethod
    def from_dict(cls, d: dict) -> MutationFailureTracker:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class FitnessCorrelation:
    """Tracks correlation between diagnostic outputs and config gene fitness."""
    diagnostic_locus: str = ""
    config_locus: str = ""
    diagnostic_values: list[float] = field(default_factory=list)
    config_fitness_values: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "diagnostic_locus": self.diagnostic_locus,
            "config_locus": self.config_locus,
            "diagnostic_values": self.diagnostic_values[-MAX_RANGE_SAMPLES:],
            "config_fitness_values": self.config_fitness_values[-MAX_RANGE_SAMPLES:],
        }

    @classmethod
    def from_dict(cls, d: dict) -> FitnessCorrelation:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# --- Main class ---


class ContractEvolution:
    """Tracks output observations and proposes contract changes."""

    def __init__(self) -> None:
        self.observations: dict[str, OutputObservation] = {}
        self.proposals: dict[str, list[ContractProposal]] = {}
        self.mutation_trackers: dict[str, MutationFailureTracker] = {}
        self.correlations: dict[str, FitnessCorrelation] = {}
        self._path: Path | None = None

    # --- Tightening (2A) ---

    def record_output(
        self, locus: str, output_json: str, contract,
    ) -> ContractProposal | None:
        """Record an output observation.  Returns proposal if threshold met."""
        try:
            data = json.loads(output_json)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(data, dict):
            return None

        obs = self.observations.get(locus)
        if obs is None:
            obs = OutputObservation(locus=locus)
            self.observations[locus] = obs
        obs.total_observations += 1

        gives_names = {f.name for f in (contract.gives if contract else [])}
        for key, value in data.items():
            obs.field_counts[key] = obs.field_counts.get(key, 0) + 1
            if key not in gives_names and key != "success":
                obs.extra_fields[key] = obs.extra_fields.get(key, 0) + 1
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                ranges = obs.field_value_ranges.setdefault(key, [])
                ranges.append(float(value))
                if len(ranges) > MAX_RANGE_SAMPLES:
                    obs.field_value_ranges[key] = ranges[-MAX_RANGE_SAMPLES:]

        if obs.total_observations >= TIGHTENING_THRESHOLD:
            proposals = self.analyze_tightening(locus, contract)
            if proposals:
                return proposals[0]
        return None

    def analyze_tightening(self, locus: str, contract) -> list[ContractProposal]:
        """Analyze whether the gives spec can be tightened."""
        obs = self.observations.get(locus)
        if obs is None or obs.total_observations < TIGHTENING_THRESHOLD:
            return []

        results: list[ContractProposal] = []
        existing = {p.description for p in self.proposals.get(locus, [])}

        # 1. Extra fields consistently produced beyond gives
        for field_name, count in obs.extra_fields.items():
            ratio = count / obs.total_observations
            if ratio >= 0.9:
                desc = f"Gene consistently produces extra field '{field_name}'"
                if desc not in existing:
                    results.append(ContractProposal(
                        locus=locus,
                        proposal_type="tighten_gives",
                        description=desc,
                        current_value="not in gives",
                        proposed_value=f"add '{field_name}' to gives",
                        evidence_count=count,
                    ))

        # 2. Optional fields always present -> propose required
        gives_fields = {f.name: f for f in (contract.gives if contract else [])}
        for field_name, fdef in gives_fields.items():
            if getattr(fdef, 'optional', False):
                count = obs.field_counts.get(field_name, 0)
                ratio = count / obs.total_observations if obs.total_observations > 0 else 0
                if ratio >= 0.95:
                    desc = f"Optional field '{field_name}' is always present"
                    if desc not in existing:
                        results.append(ContractProposal(
                            locus=locus,
                            proposal_type="tighten_gives",
                            description=desc,
                            current_value=f"{field_name} (optional)",
                            proposed_value=f"{field_name} (required)",
                            evidence_count=count,
                        ))

        self._store_proposals(locus, results)
        return results

    # --- Relaxation (2B) ---

    def record_mutation_failure(
        self, locus: str, error: str, contract=None,
    ) -> ContractProposal | None:
        """Track mutation failure.  Returns proposal if failure rate too high."""
        tracker = self.mutation_trackers.get(locus)
        if tracker is None:
            tracker = MutationFailureTracker(locus=locus)
            self.mutation_trackers[locus] = tracker
        tracker.total_mutations += 1
        tracker.failed_mutations += 1

        # Classify the constraint violation
        violation_key = self._classify_violation(error)
        if violation_key:
            tracker.constraint_violations[violation_key] = (
                tracker.constraint_violations.get(violation_key, 0) + 1
            )

        if (tracker.total_mutations >= 10
                and tracker.failure_rate >= RELAXATION_THRESHOLD):
            proposals = self.analyze_relaxation(locus, contract)
            if proposals:
                return proposals[0]
        return None

    def record_mutation_success(self, locus: str) -> None:
        """Record a successful mutation for rate tracking."""
        tracker = self.mutation_trackers.get(locus)
        if tracker is None:
            tracker = MutationFailureTracker(locus=locus)
            self.mutation_trackers[locus] = tracker
        tracker.total_mutations += 1

    def analyze_relaxation(
        self, locus: str, contract=None,
    ) -> list[ContractProposal]:
        """Propose relaxation for constraints mutations consistently violate."""
        tracker = self.mutation_trackers.get(locus)
        if tracker is None or tracker.total_mutations < 10:
            return []
        if tracker.failure_rate < RELAXATION_THRESHOLD:
            return []

        results: list[ContractProposal] = []
        existing = {p.description for p in self.proposals.get(locus, [])}

        for violation, count in sorted(
            tracker.constraint_violations.items(),
            key=lambda x: -x[1],
        ):
            ratio = count / tracker.failed_mutations if tracker.failed_mutations else 0
            if ratio >= 0.3:
                desc = f"Mutations consistently fail: {violation}"
                if desc not in existing:
                    results.append(ContractProposal(
                        locus=locus,
                        proposal_type="relax_constraint",
                        description=desc,
                        current_value=violation,
                        proposed_value="consider relaxing this constraint",
                        evidence_count=count,
                    ))

        self._store_proposals(locus, results)
        return results

    @staticmethod
    def _classify_violation(error: str) -> str:
        """Classify a mutation error into a constraint violation category."""
        error_lower = error.lower()
        if "missing" in error_lower and "field" in error_lower:
            return "gives_missing_field"
        if "type" in error_lower and ("mismatch" in error_lower or "wrong" in error_lower):
            return "gives_wrong_type"
        if "execute" not in error_lower and "function" in error_lower:
            return "missing_execute"
        if "json" in error_lower:
            return "invalid_json"
        if "success" in error_lower:
            return "missing_success_field"
        return ""

    # --- Feeds Discovery (2C) ---

    def record_diagnostic_output(
        self, diagnostic_locus: str, output_data: dict,
    ) -> None:
        """Record a diagnostic output value for correlation analysis."""
        healthy = output_data.get("healthy")
        if healthy is None:
            return
        value = 1.0 if healthy else 0.0
        for key in list(self.correlations):
            corr = self.correlations[key]
            if corr.diagnostic_locus == diagnostic_locus:
                corr.diagnostic_values.append(value)
                if len(corr.diagnostic_values) > MAX_RANGE_SAMPLES:
                    corr.diagnostic_values = corr.diagnostic_values[-MAX_RANGE_SAMPLES:]

    def record_config_fitness(
        self, config_locus: str, fitness: float,
    ) -> None:
        """Record config gene fitness for correlation analysis."""
        for key in list(self.correlations):
            corr = self.correlations[key]
            if corr.config_locus == config_locus:
                corr.config_fitness_values.append(fitness)
                if len(corr.config_fitness_values) > MAX_RANGE_SAMPLES:
                    corr.config_fitness_values = corr.config_fitness_values[-MAX_RANGE_SAMPLES:]

    def ensure_correlation_pair(
        self, diagnostic_locus: str, config_locus: str,
    ) -> None:
        """Ensure a correlation tracker exists for a (diagnostic, config) pair."""
        key = f"{diagnostic_locus}:{config_locus}"
        if key not in self.correlations:
            if len(self.correlations) >= MAX_CORRELATIONS:
                return
            self.correlations[key] = FitnessCorrelation(
                diagnostic_locus=diagnostic_locus,
                config_locus=config_locus,
            )

    def analyze_feeds(self, contract_store) -> list[ContractProposal]:
        """Compute Pearson correlation and propose feeds declarations."""
        results: list[ContractProposal] = []

        # Build set of existing feeds declarations
        existing_feeds: set[tuple[str, str]] = set()
        for locus_name in contract_store.known_loci():
            gc = contract_store.get_gene(locus_name)
            if gc and gc.feeds:
                for fd in gc.feeds:
                    existing_feeds.add((locus_name, fd.target))

        existing_descs = set()
        for proposals in self.proposals.values():
            for p in proposals:
                existing_descs.add(p.description)

        for key, corr in self.correlations.items():
            n = min(len(corr.diagnostic_values), len(corr.config_fitness_values))
            if n < MIN_CORRELATION_SAMPLES:
                continue

            x = corr.diagnostic_values[-n:]
            y = corr.config_fitness_values[-n:]
            r = _pearson(x, y)

            if abs(r) >= CORRELATION_THRESHOLD:
                pair = (corr.diagnostic_locus, corr.config_locus)
                if pair in existing_feeds:
                    continue
                desc = (f"Diagnostic '{corr.diagnostic_locus}' correlates "
                        f"(r={r:.2f}) with '{corr.config_locus}' fitness")
                if desc in existing_descs:
                    continue
                results.append(ContractProposal(
                    locus=corr.diagnostic_locus,
                    proposal_type="add_feeds",
                    description=desc,
                    proposed_value=f"feeds {corr.config_locus} convergence",
                    evidence_count=n,
                ))

        for p in results:
            self._store_proposals(p.locus, [p])
        return results

    # --- Proposal management ---

    def get_proposals(
        self, locus: str | None = None, status: str = "pending",
    ) -> list[ContractProposal]:
        """Get proposals, optionally filtered by locus and status."""
        if locus is not None:
            return [
                p for p in self.proposals.get(locus, []) if p.status == status
            ]
        result = []
        for proposals in self.proposals.values():
            result.extend(p for p in proposals if p.status == status)
        return result

    def accept_proposal(self, locus: str, index: int) -> bool:
        """Accept a pending proposal by index."""
        proposals = [
            p for p in self.proposals.get(locus, []) if p.status == "pending"
        ]
        if 0 <= index < len(proposals):
            proposals[index].status = "accepted"
            return True
        return False

    def reject_proposal(self, locus: str, index: int) -> bool:
        """Reject a pending proposal by index."""
        proposals = [
            p for p in self.proposals.get(locus, []) if p.status == "pending"
        ]
        if 0 <= index < len(proposals):
            proposals[index].status = "rejected"
            return True
        return False

    def _store_proposals(
        self, locus: str, new_proposals: list[ContractProposal],
    ) -> None:
        """Add proposals, enforcing per-locus cap."""
        existing = self.proposals.setdefault(locus, [])
        for p in new_proposals:
            existing.append(p)
        if len(existing) > MAX_PROPOSALS_PER_LOCUS:
            # Keep most recent
            self.proposals[locus] = existing[-MAX_PROPOSALS_PER_LOCUS:]

    # --- Persistence ---

    def save(self, path: Path | None = None) -> None:
        p = path or self._path
        if p is None:
            return
        data = {
            "observations": {
                k: v.to_dict() for k, v in self.observations.items()
            },
            "proposals": {
                k: [p.to_dict() for p in v]
                for k, v in self.proposals.items()
            },
            "mutation_trackers": {
                k: v.to_dict() for k, v in self.mutation_trackers.items()
            },
            "correlations": {
                k: v.to_dict() for k, v in self.correlations.items()
            },
        }
        p.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(p, json.dumps(data, indent=2))

    @classmethod
    def open(cls, path: Path) -> ContractEvolution:
        ce = cls()
        ce._path = path
        if not path.exists():
            return ce
        try:
            with file_lock_shared(path):
                data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            logger.warning("corrupted contract evolution state, starting fresh")
            return ce
        for k, v in data.get("observations", {}).items():
            ce.observations[k] = OutputObservation.from_dict(v)
        for k, v in data.get("proposals", {}).items():
            ce.proposals[k] = [ContractProposal.from_dict(p) for p in v]
        for k, v in data.get("mutation_trackers", {}).items():
            ce.mutation_trackers[k] = MutationFailureTracker.from_dict(v)
        for k, v in data.get("correlations", {}).items():
            ce.correlations[k] = FitnessCorrelation.from_dict(v)
        return ce


# --- Utility ---


def _pearson(x: list[float], y: list[float]) -> float:
    """Pearson correlation coefficient.  Returns 0.0 if computation fails."""
    n = len(x)
    if n != len(y) or n < 2:
        return 0.0
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    var_x = sum((xi - mean_x) ** 2 for xi in x)
    var_y = sum((yi - mean_y) ** 2 for yi in y)
    denom = math.sqrt(var_x * var_y)
    if denom == 0:
        return 0.0
    return cov / denom
