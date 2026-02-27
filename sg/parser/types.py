"""AST node types for the .sg contract format."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class GeneFamily(str, Enum):
    CONFIGURATION = "configuration"
    DIAGNOSTIC = "diagnostic"


class BlastRadius(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class FieldDef:
    """A field in a takes/gives block."""
    name: str
    type: str
    required: bool = True
    default: str | None = None
    optional: bool = False
    description: str = ""


@dataclass
class TypeDef:
    """An inline type definition in a types block."""
    name: str
    fields: list[FieldDef] = field(default_factory=list)


@dataclass
class VerifyStep:
    """A verification step: locus + parameter bindings."""
    locus: str
    params: dict[str, str] = field(default_factory=dict)


@dataclass
class FeedsDef:
    """A feeds declaration: target locus + fitness timescale."""
    target_locus: str
    timescale: str  # "convergence", "resilience", "immediate"


@dataclass
class GeneContract:
    """Parsed gene contract from a .sg file."""
    name: str
    family: GeneFamily
    risk: BlastRadius
    does: str
    takes: list[FieldDef] = field(default_factory=list)
    gives: list[FieldDef] = field(default_factory=list)
    types: list[TypeDef] = field(default_factory=list)
    before: list[str] = field(default_factory=list)
    after: list[str] = field(default_factory=list)
    fails_when: list[str] = field(default_factory=list)
    unhealthy_when: list[str] = field(default_factory=list)
    verify: list[VerifyStep] = field(default_factory=list)
    verify_within: str | None = None
    feeds: list[FeedsDef] = field(default_factory=list)


@dataclass
class PathwayStep:
    """A step in a pathway definition."""
    index: int
    locus: str
    is_pathway_ref: bool = False  # True if -> prefix (composed pathway)
    params: dict[str, str] = field(default_factory=dict)


@dataclass
class ForStep:
    """A for-loop step in a pathway."""
    index: int
    variable: str
    iterable: str
    body: PathwayStep | None = None


@dataclass
class ConditionalStep:
    """A when-step: conditional locus binding."""
    index: int
    condition_step: int
    condition_field: str
    branches: dict[str, PathwayStep] = field(default_factory=dict)


@dataclass
class Dependency:
    """A step dependency declaration."""
    step: int
    needs: int


@dataclass
class PathwayContract:
    """Parsed pathway contract from a .sg file."""
    name: str
    risk: BlastRadius
    does: str
    takes: list[FieldDef] = field(default_factory=list)
    steps: list[PathwayStep | ForStep | ConditionalStep] = field(default_factory=list)
    requires: list[Dependency] = field(default_factory=list)
    verify: list[VerifyStep] = field(default_factory=list)
    verify_within: str | None = None
    on_failure: str = "rollback all"


@dataclass
class TopologyResource:
    """A resource in a topology's has block."""
    name: str
    resource_type: str  # "bridge", "bond", "vlan_bridges"
    properties: dict[str, str] = field(default_factory=dict)


@dataclass
class TopologyContract:
    """Parsed topology contract from a .sg file."""
    name: str
    does: str
    takes: list[FieldDef] = field(default_factory=list)
    has: list[TopologyResource] = field(default_factory=list)
    verify: list[VerifyStep] = field(default_factory=list)
    verify_within: str | None = None
    on_failure: str = "preserve what works"
