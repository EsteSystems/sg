"""Central genome pool server — FastAPI app for cross-organism allele sharing.

Organisms push high-fitness alleles and pull alleles for their loci.
Supports cross-domain matching (e.g., network alleles usable in data pipelines)
via structural contract compatibility checking.

Storage is filesystem-backed: JSON indexes + content-addressed source files,
same pattern as Registry.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, Query, Request
from fastapi.responses import JSONResponse


# --- Data models ---


@dataclass
class PoolAllele:
    sha256: str
    locus: str
    generation: int
    fitness: float
    normalized_fitness: float
    domain: str
    organism_id: str
    pushed_at: float
    source_path: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> PoolAllele:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ContractMetadata:
    locus: str
    domain: str
    family: str
    takes: list[dict] = field(default_factory=list)
    gives: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ContractMetadata:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class OrganismRecord:
    organism_id: str
    total_pushed: int = 0
    total_pulled: int = 0
    last_push: float | None = None
    last_pull: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> OrganismRecord:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class DomainStats:
    domain: str
    allele_count: int = 0
    fitness_sum: float = 0.0
    fitness_sum_sq: float = 0.0

    @property
    def avg(self) -> float:
        if self.allele_count == 0:
            return 0.0
        return self.fitness_sum / self.allele_count

    @property
    def stddev(self) -> float:
        if self.allele_count == 0:
            return 0.0
        variance = (self.fitness_sum_sq / self.allele_count) - (self.avg ** 2)
        return math.sqrt(max(0.0, variance))

    def to_dict(self) -> dict:
        return {
            "domain": self.domain,
            "allele_count": self.allele_count,
            "fitness_sum": self.fitness_sum,
            "fitness_sum_sq": self.fitness_sum_sq,
        }

    @classmethod
    def from_dict(cls, d: dict) -> DomainStats:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# --- Cross-domain compatibility (dict-based, no parser import) ---


def _contracts_compatible_dicts(a: dict, b: dict) -> bool:
    """Check structural compatibility between two contract metadata dicts.

    Same semantics as sg.contracts.contracts_compatible() but operates on
    plain dicts so the server doesn't import the parser.

    Compatible means: every required field in a's takes/gives has a matching
    field in b with the same name and type.
    """
    def _fields_compat(a_fields: list[dict], b_fields: list[dict]) -> bool:
        b_map = {f["name"]: f for f in b_fields}
        for f in a_fields:
            if not f.get("required", True) or f.get("optional", False):
                continue
            if f["name"] not in b_map:
                return False
            if f.get("type") != b_map[f["name"]].get("type"):
                return False
        return True

    return (
        _fields_compat(a.get("takes", []), b.get("takes", []))
        and _fields_compat(a.get("gives", []), b.get("gives", []))
    )


# --- Pool store ---


class PoolStore:
    """Filesystem-backed storage for the central gene pool."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.alleles_dir = data_dir / "alleles"
        self.index_dir = data_dir / "index"
        self.contracts_dir = data_dir / "contracts"

        self._alleles: dict[str, PoolAllele] = {}
        self._loci: dict[str, list[str]] = {}  # locus -> [sha, ...]
        self._organisms: dict[str, OrganismRecord] = {}
        self._domains: dict[str, DomainStats] = {}
        self._contracts: dict[str, ContractMetadata] = {}  # locus -> metadata

    def ensure_dirs(self) -> None:
        self.alleles_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.contracts_dir.mkdir(parents=True, exist_ok=True)

    def store_allele(self, data: dict, organism_id: str) -> str:
        """Store an allele from a push request. Returns the sha256."""
        sha = data["sha256"]
        locus = data["locus"]
        source = data["source"]
        fitness = data.get("fitness", 0.0)
        generation = data.get("generation", 0)
        domain = data.get("domain", "unknown")

        # Write source file
        source_path = self.alleles_dir / f"{sha}.py"
        source_path.write_text(source)

        # Normalize fitness
        normalized = self.normalize_fitness(fitness, domain)

        # Update domain stats
        stats = self._domains.setdefault(domain, DomainStats(domain=domain))
        if sha not in self._alleles:
            stats.allele_count += 1
            stats.fitness_sum += fitness
            stats.fitness_sum_sq += fitness * fitness

        # Create allele record
        allele = PoolAllele(
            sha256=sha,
            locus=locus,
            generation=generation,
            fitness=fitness,
            normalized_fitness=normalized,
            domain=domain,
            organism_id=organism_id,
            pushed_at=time.time(),
            source_path=str(source_path),
        )
        self._alleles[sha] = allele

        # Update locus index
        if locus not in self._loci:
            self._loci[locus] = []
        if sha not in self._loci[locus]:
            self._loci[locus].append(sha)

        return sha

    def get_alleles_for_locus(self, locus: str, limit: int = 10) -> list[PoolAllele]:
        """Get alleles for a locus, sorted by fitness descending."""
        shas = self._loci.get(locus, [])
        alleles = [self._alleles[s] for s in shas if s in self._alleles]
        alleles.sort(key=lambda a: a.fitness, reverse=True)
        return alleles[:limit]

    def get_compatible_alleles(
        self, contract: ContractMetadata, limit: int = 10,
    ) -> list[PoolAllele]:
        """Get alleles from other domains that are structurally compatible."""
        target_dict = contract.to_dict()
        compatible: list[PoolAllele] = []

        for locus, meta in self._contracts.items():
            if meta.domain == contract.domain:
                continue  # skip same domain — not cross-domain
            source_dict = meta.to_dict()
            if _contracts_compatible_dicts(target_dict, source_dict):
                compatible.extend(self.get_alleles_for_locus(locus, limit))

        compatible.sort(key=lambda a: a.normalized_fitness, reverse=True)
        return compatible[:limit]

    def store_contract(self, locus: str, meta: ContractMetadata) -> None:
        """Store contract metadata for a locus."""
        self._contracts[locus] = meta
        path = self.contracts_dir / f"{locus}.json"
        path.write_text(json.dumps(meta.to_dict(), indent=2))

    def record_push(self, organism_id: str) -> None:
        rec = self._organisms.setdefault(
            organism_id, OrganismRecord(organism_id=organism_id)
        )
        rec.total_pushed += 1
        rec.last_push = time.time()

    def record_pull(self, organism_id: str, count: int) -> None:
        rec = self._organisms.setdefault(
            organism_id, OrganismRecord(organism_id=organism_id)
        )
        rec.total_pulled += count
        rec.last_pull = time.time()

    def check_reciprocity(self, organism_id: str, min_pushes: int = 1) -> bool:
        """Check if an organism has pushed enough to be allowed to pull."""
        if min_pushes <= 0:
            return True
        rec = self._organisms.get(organism_id)
        if rec is None:
            return False
        return rec.total_pushed >= min_pushes

    def normalize_fitness(self, fitness: float, domain: str) -> float:
        """Normalize fitness using z-score within domain. Stddev floored at 0.001."""
        stats = self._domains.get(domain)
        if stats is None or stats.allele_count == 0:
            return fitness
        stddev = max(stats.stddev, 0.001)
        return (fitness - stats.avg) / stddev

    def load(self) -> None:
        """Load all indexes from disk."""
        alleles_path = self.index_dir / "alleles.json"
        if alleles_path.exists():
            data = json.loads(alleles_path.read_text())
            self._alleles = {k: PoolAllele.from_dict(v) for k, v in data.items()}

        loci_path = self.index_dir / "loci.json"
        if loci_path.exists():
            self._loci = json.loads(loci_path.read_text())

        organisms_path = self.index_dir / "organisms.json"
        if organisms_path.exists():
            data = json.loads(organisms_path.read_text())
            self._organisms = {
                k: OrganismRecord.from_dict(v) for k, v in data.items()
            }

        domains_path = self.index_dir / "domains.json"
        if domains_path.exists():
            data = json.loads(domains_path.read_text())
            self._domains = {k: DomainStats.from_dict(v) for k, v in data.items()}

        # Load contracts
        if self.contracts_dir.exists():
            for path in self.contracts_dir.glob("*.json"):
                data = json.loads(path.read_text())
                locus = path.stem
                self._contracts[locus] = ContractMetadata.from_dict(data)

    def save(self) -> None:
        """Persist all indexes to disk."""
        self.ensure_dirs()

        alleles_path = self.index_dir / "alleles.json"
        alleles_path.write_text(json.dumps(
            {k: v.to_dict() for k, v in self._alleles.items()}, indent=2
        ))

        loci_path = self.index_dir / "loci.json"
        loci_path.write_text(json.dumps(self._loci, indent=2))

        organisms_path = self.index_dir / "organisms.json"
        organisms_path.write_text(json.dumps(
            {k: v.to_dict() for k, v in self._organisms.items()}, indent=2
        ))

        domains_path = self.index_dir / "domains.json"
        domains_path.write_text(json.dumps(
            {k: v.to_dict() for k, v in self._domains.items()}, indent=2
        ))


# --- FastAPI app ---


def create_pool_app(
    data_dir: Path,
    token: str | None = None,
    reciprocity: int = 1,
) -> FastAPI:
    """Create the pool server FastAPI app.

    Args:
        data_dir: Directory for pool storage.
        token: Optional bearer token for authentication.
        reciprocity: Minimum pushes required before pulls allowed (0 disables).
    """
    app = FastAPI(title="Software Genome Pool Server")
    store = PoolStore(data_dir)
    store.ensure_dirs()
    store.load()

    def _check_auth(authorization: str | None) -> JSONResponse | None:
        if token is None:
            return None
        if not authorization or not authorization.startswith("Bearer "):
            return JSONResponse(
                {"error": "authentication required"}, status_code=401,
            )
        if authorization[7:] != token:
            return JSONResponse(
                {"error": "invalid token"}, status_code=401,
            )
        return None

    def _get_organism(x_sg_organism: str | None) -> str:
        return x_sg_organism or "anonymous"

    @app.post("/pool/push")
    async def pool_push(
        request: Request,
        authorization: str | None = Header(None),
        x_sg_organism: str | None = Header(None),
    ):
        auth_err = _check_auth(authorization)
        if auth_err:
            return auth_err

        data = await request.json()

        # Validate required fields
        for field_name in ("locus", "sha256", "source"):
            if field_name not in data:
                return JSONResponse(
                    {"error": f"missing required field: {field_name}"},
                    status_code=400,
                )

        organism_id = _get_organism(x_sg_organism)
        sha = store.store_allele(data, organism_id)

        # Store contract metadata if provided
        if "contract" in data and data["contract"]:
            contract_data = data["contract"]
            meta = ContractMetadata(
                locus=data["locus"],
                domain=data.get("domain", "unknown"),
                family=contract_data.get("family", ""),
                takes=contract_data.get("takes", []),
                gives=contract_data.get("gives", []),
            )
            store.store_contract(data["locus"], meta)

        store.record_push(organism_id)
        store.save()

        return {"status": "ok", "sha256": sha}

    @app.get("/pool/pull/{locus}")
    async def pool_pull(
        locus: str,
        cross_domain: bool = Query(False),
        contract: str | None = Query(None),
        limit: int = Query(10),
        authorization: str | None = Header(None),
        x_sg_organism: str | None = Header(None),
    ):
        auth_err = _check_auth(authorization)
        if auth_err:
            return auth_err

        organism_id = _get_organism(x_sg_organism)

        # Check reciprocity
        if not store.check_reciprocity(organism_id, reciprocity):
            return JSONResponse(
                {"error": "reciprocity required: push alleles before pulling"},
                status_code=403,
            )

        alleles: list[PoolAllele]
        if cross_domain:
            if not contract:
                return JSONResponse(
                    {"error": "contract metadata required for cross-domain pull"},
                    status_code=400,
                )
            try:
                contract_dict = json.loads(contract)
            except json.JSONDecodeError:
                return JSONResponse(
                    {"error": "invalid contract JSON"}, status_code=400,
                )
            meta = ContractMetadata(
                locus=locus,
                domain=contract_dict.get("domain", "unknown"),
                family=contract_dict.get("family", ""),
                takes=contract_dict.get("takes", []),
                gives=contract_dict.get("gives", []),
            )
            alleles = store.get_compatible_alleles(meta, limit)
        else:
            alleles = store.get_alleles_for_locus(locus, limit)

        # Build response with source code
        result = []
        for a in alleles:
            source_path = store.alleles_dir / f"{a.sha256}.py"
            source = ""
            if source_path.exists():
                source = source_path.read_text()
            result.append({
                "sha256": a.sha256,
                "locus": a.locus,
                "source": source,
                "generation": a.generation,
                "fitness": a.fitness,
                "normalized_fitness": a.normalized_fitness,
                "domain": a.domain,
            })

        if result:
            store.record_pull(organism_id, len(result))
            store.save()

        return {"alleles": result}

    @app.get("/pool/status")
    async def pool_status(
        authorization: str | None = Header(None),
    ):
        auth_err = _check_auth(authorization)
        if auth_err:
            return auth_err

        return {
            "allele_count": len(store._alleles),
            "locus_count": len(store._loci),
            "organism_count": len(store._organisms),
            "domain_count": len(store._domains),
            "domains": {
                d: {"allele_count": s.allele_count, "avg_fitness": round(s.avg, 3)}
                for d, s in store._domains.items()
            },
        }

    @app.get("/pool/organisms")
    async def pool_organisms(
        authorization: str | None = Header(None),
    ):
        auth_err = _check_auth(authorization)
        if auth_err:
            return auth_err

        return {
            "organisms": [
                rec.to_dict() for rec in store._organisms.values()
            ]
        }

    return app


# --- Entry point ---


def run_pool_server(
    data_dir: Path,
    host: str = "127.0.0.1",
    port: int = 9420,
    token: str | None = None,
    reciprocity: int = 1,
) -> None:
    """Start the pool server."""
    import uvicorn
    app = create_pool_app(data_dir, token=token, reciprocity=reciprocity)
    print(f"Starting pool server at http://{host}:{port}")
    print(f"  Data directory: {data_dir}")
    print(f"  Authentication: {'enabled' if token else 'disabled'}")
    print(f"  Reciprocity threshold: {reciprocity}")
    uvicorn.run(app, host=host, port=port, log_level="warning")
