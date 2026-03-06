"""Web dashboard — REST API + single-page HTML frontend.

Serves genome state via JSON endpoints and a lightweight HTML dashboard.
Requires: pip install 'sg[dashboard]' (fastapi + uvicorn).
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import time
import uuid as _uuid
from collections import OrderedDict
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.responses import StreamingResponse

from sg import arena
from sg.contracts import ContractStore
from sg.fusion import FusionTracker
from sg.log import get_logger
from sg.pathway_fitness import PathwayFitnessTracker
from sg.pathway_registry import PathwayRegistry
from sg.phenotype import PhenotypeMap
from sg.registry import Registry

logger = get_logger("dashboard")


app = FastAPI(title="Software Genome Dashboard")

# Project root — set at startup
_project_root: Path = Path(".")
_metrics_collector = None  # type: ignore


_contract_store_cache = None
_contract_store_mtime = 0.0

def _load_contracts():
    """Load contracts, with mtime-based caching to auto-reload on file changes."""
    global _contract_store_cache, _contract_store_mtime
    root = _project_root
    contracts_dir = root / "contracts"
    try:
        current_mtime = max(
            (f.stat().st_mtime for f in contracts_dir.rglob("*.sg")),
            default=0.0,
        )
    except Exception:
        current_mtime = 0.0
    if _contract_store_cache is None or current_mtime > _contract_store_mtime:
        _contract_store_cache = ContractStore.open(contracts_dir)
        _contract_store_mtime = current_mtime
    return _contract_store_cache


def _load_state():
    """Load all state from disk (fresh on each request)."""
    root = _project_root
    contract_store = _load_contracts()
    registry = Registry.open(root / ".sg" / "registry")
    phenotype = PhenotypeMap.load(root / "phenotype.toml")
    fusion_tracker = FusionTracker.open(root / "fusion_tracker.json")
    pathway_fitness = PathwayFitnessTracker.open(root / "pathway_fitness.json")
    pathway_registry = PathwayRegistry.open(root / ".sg" / "pathway_registry")
    from sg.meta_params import MetaParamTracker
    meta_tracker = MetaParamTracker.open(root / ".sg" / "meta_params.json")
    return contract_store, registry, phenotype, fusion_tracker, pathway_fitness, pathway_registry, meta_tracker


@app.get("/api/status")
def api_status():
    cs, reg, pheno, ft, _pft, _pr, mpt = _load_state()
    allele_count = len(reg.alleles)
    loci_count = len(cs.known_loci())
    pathway_count = len(cs.known_pathways())
    topology_count = len(cs.known_topologies())
    fused_count = sum(
        1 for name in cs.known_pathways()
        if (f := pheno.get_fused(name)) and f.fused_sha
    )
    fitnesses = [arena.compute_fitness(a, params=mpt.get_params(a.locus)) for a in reg.alleles.values()]
    avg_fitness = sum(fitnesses) / len(fitnesses) if fitnesses else 0.0
    return {
        "loci_count": loci_count,
        "allele_count": allele_count,
        "pathway_count": pathway_count,
        "topology_count": topology_count,
        "fused_count": fused_count,
        "avg_fitness": round(avg_fitness, 3),
    }


@app.get("/api/loci")
def api_loci():
    cs, reg, pheno, _, _, _, mpt = _load_state()
    result = []
    for locus in cs.known_loci():
        alleles = reg.alleles_for_locus(locus)
        dominant_sha = pheno.get_dominant(locus)
        dominant_fitness = 0.0
        if dominant_sha:
            dom = reg.get(dominant_sha)
            if dom:
                dominant_fitness = arena.compute_fitness(dom, params=mpt.get_params(locus))
        result.append({
            "name": locus,
            "dominant_sha": dominant_sha[:12] if dominant_sha else None,
            "allele_count": len(alleles),
            "dominant_fitness": round(dominant_fitness, 3),
        })
    return result


@app.get("/api/locus/{name}")
def api_locus(name: str):
    cs, reg, pheno, _, _, _, mpt = _load_state()
    alleles = reg.alleles_for_locus(name)
    dominant_sha = pheno.get_dominant(name)
    params = mpt.get_params(name)

    allele_list = []
    for a in alleles:
        allele_list.append({
            "sha": a.sha256[:12],
            "sha_full": a.sha256,
            "generation": a.generation,
            "fitness": round(arena.compute_fitness(a, params=params), 3),
            "state": a.state,
            "successful_invocations": a.successful_invocations,
            "failed_invocations": a.failed_invocations,
            "parent_sha": a.parent_sha[:12] if a.parent_sha else None,
            "is_dominant": a.sha256 == dominant_sha,
        })

    contract = cs.get_gene(name)
    contract_info = None
    if contract:
        contract_info = {
            "does": contract.does,
            "family": contract.family.value,
            "risk": contract.risk.value,
            "takes": [{"name": f.name, "type": f.type} for f in contract.takes],
            "gives": [{"name": f.name, "type": f.type} for f in contract.gives],
        }

    return {"name": name, "alleles": allele_list, "contract": contract_info}


@app.get("/api/pathways")
def api_pathways():
    cs, _, pheno, ft, pft, pr, _ = _load_state()
    result = []
    for name in cs.known_pathways():
        fusion = pheno.get_fused(name)
        track = ft.get_track(name)
        fitness_rec = pft.get_record(name)
        pw_alleles = pr.get_for_pathway(name)
        pw_dominant = pheno.get_pathway_dominant(name)
        pw_contract = cs.get_pathway(name)
        defaults = {}
        if pw_contract:
            for f in pw_contract.takes:
                if f.default is not None:
                    defaults[f.name] = f.default
        result.append({
            "name": name,
            "fused": bool(fusion and fusion.fused_sha),
            "fused_sha": fusion.fused_sha[:12] if fusion and fusion.fused_sha else None,
            "reinforcement_count": track.reinforcement_count if track else 0,
            "total_successes": track.total_successes if track else 0,
            "total_failures": track.total_failures if track else 0,
            "fitness": round(pft.compute_fitness(name), 3),
            "total_executions": fitness_rec.total_executions if fitness_rec else 0,
            "avg_time_ms": round(fitness_rec.avg_execution_time_ms, 1) if fitness_rec else 0,
            "consecutive_failures": fitness_rec.consecutive_failures if fitness_rec else 0,
            "pathway_allele_count": len(pw_alleles),
            "dominant_pathway_allele": pw_dominant[:12] if pw_dominant else None,
            "defaults": defaults,
        })
    return result


@app.get("/api/allele/{sha}/source")
def api_allele_source(sha: str):
    _, reg, _, _, _, _, _ = _load_state()
    # Try exact match first, then prefix
    source = reg.load_source(sha)
    if source is None:
        for full_sha in reg.alleles:
            if full_sha.startswith(sha):
                source = reg.load_source(full_sha)
                break
    if source is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    return {"source": source}


@app.get("/api/lineage/{sha}")
def api_lineage(sha: str):
    """Return the lineage chain for an allele (child → parent → ...)."""
    _, reg, _, _, _, _, mpt = _load_state()
    # Resolve prefix
    full_sha = sha
    if sha not in reg.alleles:
        for s in reg.alleles:
            if s.startswith(sha):
                full_sha = s
                break

    chain = []
    current = full_sha
    seen = set()
    while current and current not in seen:
        seen.add(current)
        allele = reg.get(current)
        if allele is None:
            break
        chain.append({
            "sha": allele.sha256[:12],
            "sha_full": allele.sha256,
            "locus": allele.locus,
            "generation": allele.generation,
            "fitness": round(arena.compute_fitness(allele, params=mpt.get_params(allele.locus)), 3),
            "state": allele.state,
        })
        current = allele.parent_sha
    return {"lineage": chain}


@app.get("/api/regression")
def api_regression():
    """Return regression history for all tracked alleles."""
    root = _project_root
    regression_path = root / ".sg" / "regression.json"
    from sg.regression import RegressionDetector
    det = RegressionDetector.open(regression_path)
    result = []
    for sha, h in det.history.items():
        result.append({
            "sha": sha[:12],
            "sha_full": sha,
            "peak_fitness": round(h.peak_fitness, 3),
            "last_fitness": round(h.last_fitness, 3),
            "samples": h.samples,
            "drop": round(h.peak_fitness - h.last_fitness, 3),
        })
    return {"history": result}


@app.get("/api/pathway/{name}/fitness")
def api_pathway_fitness(name: str):
    """Return detailed pathway fitness data."""
    _, _, _, _, pft, _, _ = _load_state()
    rec = pft.get_record(name)
    if rec is None:
        return {"pathway": name, "fitness": 0.0, "executions": 0}
    return {
        "pathway": name,
        "fitness": round(pft.compute_fitness(name), 3),
        "total_executions": rec.total_executions,
        "successful_executions": rec.successful_executions,
        "failed_executions": rec.failed_executions,
        "avg_time_ms": round(rec.avg_execution_time_ms, 1),
        "consecutive_failures": rec.consecutive_failures,
        "failure_distribution": pft.get_failure_distribution(name),
        "timing_anomalies": [a.to_dict() for a in pft.get_timing_anomalies(name)],
        "step_timings": {
            step: {"avg": round(sum(t) / len(t), 1), "count": len(t)}
            for step, t in rec.step_timings.items()
        },
    }


@app.get("/api/pathway/{name}/lineage")
def api_pathway_lineage(name: str):
    """Return pathway allele lineage."""
    _, _, pheno, _, _, pr, _ = _load_state()
    from sg.pathway_arena import compute_pathway_fitness
    alleles = pr.get_for_pathway(name)
    dominant_sha = pheno.get_pathway_dominant(name)
    result = []
    for a in alleles:
        result.append({
            "sha": a.structure_sha[:12],
            "sha_full": a.structure_sha,
            "pathway_name": a.pathway_name,
            "fitness": round(compute_pathway_fitness(a), 3),
            "state": a.state,
            "total_executions": a.total_executions,
            "successful_executions": a.successful_executions,
            "failed_executions": a.failed_executions,
            "parent_sha": a.parent_sha[:12] if a.parent_sha else None,
            "mutation_operator": a.mutation_operator,
            "is_dominant": a.structure_sha == dominant_sha,
            "steps": [s.to_dict() for s in a.steps],
        })
    return {"pathway": name, "alleles": result}


@app.get("/api/events")
async def api_events():
    """SSE stream — yields update events when files change or daemon ticks."""
    root = _project_root

    async def event_stream():
        last_mtime = 0.0
        last_tick = 0
        while True:
            current = 0.0
            for f in [root / "phenotype.toml",
                      root / ".sg" / "registry" / "registry.json",
                      root / "fusion_tracker.json",
                      root / "pathway_fitness.json",
                      root / ".sg" / "pathway_registry" / "pathway_registry.json"]:
                if f.exists():
                    current = max(current, f.stat().st_mtime)
            if current > last_mtime and last_mtime > 0:
                yield f"data: {{\"type\": \"update\", \"time\": {current}}}\n\n"
            last_mtime = current

            if _daemon.tick_count != last_tick:
                last_tick = _daemon.tick_count
                d = json.dumps({"type": "daemon_tick", **_daemon.to_dict()})
                yield f"data: {d}\n\n"

            await asyncio.sleep(2)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# Dashboard analysis endpoints

@app.get("/api/feeds")
def api_feeds():
    """Return the feeds and verify relationship graph across all loci."""
    cs, _, _, _, _, _, _ = _load_state()
    edges = []
    verify_links = []
    for locus_name in cs.known_loci():
        gene = cs.get_gene(locus_name)
        if gene is None:
            continue
        if gene.feeds:
            for fd in gene.feeds:
                edges.append({
                    "source": locus_name,
                    "target": fd.target_locus,
                    "timescale": fd.timescale,
                    "source_family": gene.family.value,
                })
        if gene.verify:
            for vs in gene.verify:
                verify_links.append({
                    "source": locus_name,
                    "target": vs.locus,
                    "delay": gene.verify_within or "",
                })
    return {"feeds": edges, "verify_links": verify_links}


@app.get("/api/audit")
def api_audit(count: int = 100):
    """Return recent audit log entries."""
    from sg.audit import AuditLog
    audit = AuditLog(_project_root / ".sg" / "audit.jsonl")
    entries = audit.read_recent(count)
    return {"entries": [e.to_dict() for e in entries]}


@app.get("/api/contract_evolution")
def api_contract_evolution(locus: str | None = None):
    """Return contract evolution proposals."""
    from sg.contract_evolution import ContractEvolution
    ce = ContractEvolution.open(_project_root / ".sg" / "contract_evolution.json")
    proposals = ce.get_proposals(locus=locus)
    return {"proposals": [p.to_dict() for p in proposals]}


# ── Contract editor endpoints ─────────────────────────────────────────

@app.get("/api/contract/{name}/raw")
def api_contract_raw(name: str):
    """Return raw .sg source and parsed structure for the contract editor."""
    cs, _, _, _, _, _, _ = _load_state()
    path = cs.file_path(name)
    if path is None or not path.exists():
        return JSONResponse({"error": f"contract '{name}' not found"}, status_code=404)
    source = path.read_text()

    gene = cs.get_gene(name)
    pathway = cs.get_pathway(name)
    topology = cs.get_topology(name)
    contract = gene or pathway or topology

    ctype = "gene" if gene else ("pathway" if pathway else "topology")
    parsed = {"type": ctype, "name": name}

    if gene:
        parsed["family"] = gene.family.value
        parsed["risk"] = gene.risk.value
        parsed["domain"] = gene.domain
        parsed["does"] = gene.does
        parsed["takes"] = [{"name": f.name, "type": f.type, "description": f.description,
                            "default": f.default, "optional": f.optional} for f in gene.takes]
        parsed["gives"] = [{"name": f.name, "type": f.type, "description": f.description,
                            "optional": f.optional} for f in gene.gives]
        parsed["connects"] = [{"param": c.param, "interface": c.interface,
                               "description": c.description} for c in gene.connects]
        parsed["before"] = gene.before
        parsed["after"] = gene.after
        parsed["fails_when"] = gene.fails_when
        parsed["unhealthy_when"] = gene.unhealthy_when
        parsed["verify"] = [{"locus": v.locus, "params": v.params} for v in gene.verify]
        parsed["verify_within"] = gene.verify_within
        parsed["feeds"] = [{"target_locus": f.target_locus, "timescale": f.timescale}
                           for f in gene.feeds]
    elif pathway:
        parsed["risk"] = pathway.risk.value
        parsed["domain"] = pathway.domain
        parsed["does"] = pathway.does
        parsed["takes"] = [{"name": f.name, "type": f.type, "description": f.description,
                            "default": f.default, "optional": f.optional} for f in pathway.takes]
        parsed["steps"] = []
        for s in pathway.steps:
            from sg.parser.types import PathwayStep as ASTPathwayStep
            if isinstance(s, ASTPathwayStep):
                parsed["steps"].append({"index": s.index, "locus": s.locus, "params": s.params})
        parsed["on_failure"] = pathway.on_failure
        parsed["verify"] = [{"locus": v.locus, "params": v.params} for v in pathway.verify]
        parsed["verify_within"] = pathway.verify_within

    return {"source": source, "parsed": parsed}


@app.put("/api/contract/{name}")
async def api_contract_save(name: str, request: Request):
    """Save a modified .sg contract source back to disk and reload."""
    data = await request.json()
    source = data.get("source", "")
    if not source.strip():
        return JSONResponse({"error": "empty contract source"}, status_code=400)

    cs, _, _, _, _, _, _ = _load_state()
    path = cs.file_path(name)

    # Validate by parsing before saving
    try:
        from sg.parser.parser import parse_sg
        contract = parse_sg(source)
    except Exception as e:
        return JSONResponse({"error": f"parse error: {e}"}, status_code=400)

    new_name = contract.name
    if path is None:
        # New contract — determine directory from type
        from sg.parser.types import GeneContract as GC, PathwayContract as PC
        if isinstance(contract, GC):
            subdir = "genes"
        elif isinstance(contract, PC):
            subdir = "pathways"
        else:
            subdir = "topologies"
        path = _project_root / "contracts" / subdir / f"{new_name}.sg"
        path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(source)
    global _contract_store_cache, _contract_store_mtime
    _contract_store_cache = None
    _contract_store_mtime = 0.0
    return {"ok": True, "name": new_name, "path": str(path)}


@app.post("/api/pathway/draft")
async def api_pathway_draft(request: Request):
    """Generate a pathway .sg contract from a natural language intent."""
    data = await request.json()
    intent = data.get("intent", "").strip()
    engine_name = data.get("mutation_engine", "deepseek")
    kernel_name = data.get("kernel", "data-mock")

    if not intent:
        return JSONResponse({"error": "intent is required"}, status_code=400)

    job_id = _uuid.uuid4().hex[:12]
    _jobs[job_id] = {"status": "running", "type": "draft_pathway"}
    _prune_jobs()

    def _do_draft():
        root = _project_root
        try:
            from sg.cli import load_contract_store, make_mutation_engine
            from sg.kernel.discovery import load_kernel
            import argparse

            contract_store = load_contract_store(root)
            kernel = load_kernel(kernel_name)
            args_ns = argparse.Namespace(mutation_engine=engine_name, model=None, kernel=kernel_name)
            mutation_engine = make_mutation_engine(args_ns, root, contract_store, kernel=kernel)

            gene_summaries = []
            for locus in contract_store.known_loci():
                gene = contract_store.get_gene(locus)
                if gene is None:
                    continue
                takes = ", ".join(f"{f.name}: {f.type}" for f in gene.takes)
                gives = ", ".join(f"{f.name}: {f.type}" for f in gene.gives)
                defaults = {f.name: f.default for f in gene.takes if f.default is not None}
                gene_summaries.append(
                    f"  gene {locus}: {gene.does.strip()}\n"
                    f"    takes({takes})\n"
                    f"    gives({gives})"
                    + (f"\n    defaults: {defaults}" if defaults else "")
                )

            genes_text = "\n".join(gene_summaries)

            example = (
                'pathway example_name for data\n'
                '  risk low\n\n'
                '  does:\n'
                '    Description of what this pathway does.\n\n'
                '  takes:\n'
                '    url         string  "URL to fetch data"  default="https://example.com/data.csv"\n'
                '    connection  string  "Database connection" default="warehouse"\n\n'
                '  steps:\n'
                '    1. first_gene\n'
                '         param1 = {url}\n'
                '         param2 = {connection}\n\n'
                '    2. second_gene\n'
                '         connection = {connection}\n\n'
                '  on failure:\n'
                '    rollback all\n'
            )

            prompt = (
                f"Generate a Software Genomics pathway contract in .sg format.\n\n"
                f"User intent: {intent}\n\n"
                f"Available genes:\n{genes_text}\n\n"
                f"Rules:\n"
                f"- Use ONLY genes from the list above\n"
                f"- Wire step parameters using {{param_name}} bindings from the takes section\n"
                f"- Include default= values for all takes fields\n"
                f"- Use 'on failure: rollback all' unless the intent suggests otherwise\n"
                f"- Pick a descriptive snake_case name for the pathway\n"
                f"- Set the domain to match the genes' domain\n\n"
                f"Example format:\n```\n{example}```\n\n"
                f"Return ONLY the .sg contract text, no explanations."
            )

            response = mutation_engine._call_api(prompt)

            # Extract .sg content from response
            sg_source = response.strip()
            if "```" in sg_source:
                blocks = sg_source.split("```")
                for block in blocks[1:]:
                    cleaned = block.strip()
                    if cleaned.startswith("sg\n") or cleaned.startswith("sg\r"):
                        cleaned = cleaned[2:].strip()
                    elif cleaned.startswith("\n") or cleaned.startswith("pathway"):
                        pass
                    if "pathway " in cleaned:
                        sg_source = cleaned.split("```")[0].strip()
                        break

            # Validate by parsing
            from sg.parser.parser import parse_sg
            from sg.parser.types import PathwayContract as PC
            contract = parse_sg(sg_source)
            if not isinstance(contract, PC):
                _jobs[job_id] = {"status": "done", "type": "draft_pathway", "success": False,
                                 "error": "LLM generated a non-pathway contract"}
                return

            parsed = {
                "type": "pathway",
                "name": contract.name,
                "domain": contract.domain or "",
                "risk": contract.risk or "low",
                "does": contract.does or "",
                "takes": [{"name": f.name, "type": f.type_name, "description": f.description or "",
                           "default": f.default, "optional": f.optional}
                          for f in contract.takes],
                "steps": [{"index": s.index, "locus": s.locus, "params": s.params or {}}
                          for s in contract.steps],
                "on_failure": contract.on_failure or "",
            }

            _jobs[job_id] = {"status": "done", "type": "draft_pathway", "success": True,
                             "source": sg_source, "parsed": parsed}
        except Exception as e:
            _jobs[job_id] = {"status": "done", "type": "draft_pathway", "success": False,
                             "error": str(e)}

    asyncio.get_event_loop().run_in_executor(None, _do_draft)
    return {"job_id": job_id, "status": "running"}


# ── Action endpoints (control plane) ──────────────────────────────────

_jobs: OrderedDict[str, dict] = OrderedDict()
_MAX_JOBS = 50


def _prune_jobs() -> None:
    while len(_jobs) > _MAX_JOBS:
        _jobs.popitem(last=False)


@app.get("/api/job/{job_id}")
def api_job(job_id: str):
    """Poll for job result."""
    job = _jobs.get(job_id)
    if job is None:
        return JSONResponse({"error": "job not found"}, status_code=404)
    return job


@app.get("/api/kernels")
def api_kernels():
    """List available kernel names."""
    from sg.kernel.discovery import discover_kernels
    result = []
    for name, ep in sorted(discover_kernels().items()):
        result.append({"name": name, "entry_point": ep.value})
    return {"kernels": result}


@app.post("/api/init")
async def api_init(request: Request):
    """Seed all loci from contracts — hand-written files first, then LLM for the rest."""
    data = await request.json()
    kernel_name = data.get("kernel", "data-mock")
    engine_name = data.get("mutation_engine", "deepseek")

    job_id = _uuid.uuid4().hex[:12]
    _jobs[job_id] = {"status": "running", "type": "init"}
    _prune_jobs()

    def _do_init():
        root = _project_root
        try:
            from sg.kernel.discovery import load_kernel
            from sg.cli import load_contract_store, _discover_seed_genes, make_mutation_engine
            from sg.pathway import pathway_from_contract
            from sg.pathway_registry import steps_from_pathway
            import argparse

            contract_store = load_contract_store(root)
            registry = Registry.open(root / ".sg" / "registry")
            phenotype = PhenotypeMap()

            genes_dir = root / "genes"
            seeds = {}
            if genes_dir.exists():
                seeds = _discover_seed_genes(genes_dir, contract_store.known_loci())

            seeded = []
            for locus, gene_path in seeds.items():
                source = gene_path.read_text()
                sha = registry.register(source, locus)
                phenotype.promote(locus, sha)
                allele = registry.get(sha)
                if allele:
                    allele.state = "dominant"
                seeded.append({"locus": locus, "sha": sha[:12], "source": "file"})

            unseeded = [l for l in contract_store.known_loci() if l not in seeds]
            if unseeded and engine_name != "mock":
                kernel = load_kernel(kernel_name)
                args_ns = argparse.Namespace(mutation_engine=engine_name, model=data.get("model"), kernel=kernel_name)
                mutation_engine = make_mutation_engine(args_ns, root, contract_store, kernel=kernel)

                for locus in unseeded:
                    gene_contract = contract_store.get_gene(locus)
                    if gene_contract is None:
                        continue
                    if hasattr(mutation_engine, '_contract_prompt'):
                        contract_prompt = mutation_engine._contract_prompt(locus)
                    else:
                        info = contract_store.contract_info(locus)
                        contract_prompt = f"Locus: {locus}\nDescription: {info.description}"
                    try:
                        sources = mutation_engine.generate(locus, contract_prompt, 1)
                        if sources:
                            sha = registry.register(sources[0], locus, generation=0)
                            phenotype.promote(locus, sha)
                            allele = registry.get(sha)
                            if allele:
                                allele.state = "dominant"
                            seeded.append({"locus": locus, "sha": sha[:12], "source": "llm"})
                    except Exception as e:
                        seeded.append({"locus": locus, "sha": None, "source": "error", "error": str(e)})

            # Register pathway alleles
            pw_count = 0
            for pw_name in contract_store.known_pathways():
                pw_contract = contract_store.get_pathway(pw_name)
                if pw_contract is None:
                    continue
                pathway = pathway_from_contract(pw_contract)
                pr = PathwayRegistry.open(root / ".sg" / "pathway_registry")
                steps = steps_from_pathway(pathway)
                sha = pr.register(pw_name, steps)
                pw_allele = pr.get(sha)
                if pw_allele and pw_allele.state != "dominant":
                    pw_allele.state = "dominant"
                phenotype.promote_pathway(pw_name, sha)
                pr.save_index()
                pw_count += 1

            registry.save_index()
            phenotype.save(root / "phenotype.toml")
            _jobs[job_id] = {"status": "done", "type": "init", "success": True,
                             "seeded": seeded, "pathways": pw_count}
        except Exception as e:
            _jobs[job_id] = {"status": "done", "type": "init", "success": False,
                             "error": str(e)}

    asyncio.get_event_loop().run_in_executor(None, _do_init)
    return {"job_id": job_id, "status": "running"}


@app.post("/api/run")
async def api_run(request: Request):
    """Run a pathway in background with optional iterations. Uses contract defaults."""
    data = await request.json()
    pathway_name = data.get("pathway")
    kernel_name = data.get("kernel", "data-mock")
    iterations = data.get("iterations", 1)
    input_override = data.get("input")

    if not pathway_name:
        return JSONResponse({"error": "pathway is required"}, status_code=400)

    job_id = _uuid.uuid4().hex[:12]
    _jobs[job_id] = {"status": "running", "type": "run", "pathway": pathway_name,
                     "progress": 0, "total": iterations}
    _prune_jobs()

    def _do_run():
        root = _project_root
        try:
            from sg.kernel.discovery import load_kernel
            from sg.cli import load_contract_store
            from sg.orchestrator import Orchestrator
            from sg.mutation import MockMutationEngine
            from sg.meta_params import MetaParamTracker

            contract_store = load_contract_store(root)

            defaults = {}
            pw_contract = contract_store.get_pathway(pathway_name)
            if pw_contract:
                for f in pw_contract.takes:
                    if f.default is not None:
                        defaults[f.name] = f.default
            if input_override and isinstance(input_override, dict):
                defaults.update(input_override)
            final_input = json.dumps(defaults)

            registry = Registry.open(root / ".sg" / "registry")
            phenotype = PhenotypeMap.load(root / "phenotype.toml")
            fusion_tracker = FusionTracker.open(root / "fusion_tracker.json")
            kernel = load_kernel(kernel_name)
            pft = PathwayFitnessTracker.open(root / "pathway_fitness.json")
            pr = PathwayRegistry.open(root / ".sg" / "pathway_registry")
            meta_tracker = MetaParamTracker.open(root / ".sg" / "meta_params.json")
            mutation_engine = MockMutationEngine(root / "fixtures")

            orch = Orchestrator(
                registry=registry, phenotype=phenotype,
                mutation_engine=mutation_engine, fusion_tracker=fusion_tracker,
                kernel=kernel, contract_store=contract_store,
                project_root=root,
                pathway_fitness_tracker=pft, pathway_registry=pr,
                meta_param_tracker=meta_tracker,
            )

            all_runs = []
            successes = 0
            for iteration in range(iterations):
                _jobs[job_id]["progress"] = iteration + 1
                try:
                    outputs = orch.run_pathway(pathway_name, final_input)
                    steps = []
                    run_ok = True
                    for i, (out, sha) in enumerate(outputs):
                        try:
                            parsed = json.loads(out)
                            if isinstance(parsed, dict) and not parsed.get("success", True):
                                run_ok = False
                        except Exception:
                            parsed = out
                        steps.append({"step": i + 1, "sha": sha[:12], "output": parsed})
                    if run_ok:
                        successes += 1
                    all_runs.append({"iteration": iteration + 1, "success": run_ok, "steps": steps})
                except Exception as e:
                    all_runs.append({"iteration": iteration + 1, "success": False, "error": str(e)})

            orch.verify_scheduler.wait()
            orch.save_state()

            _jobs[job_id] = {"status": "done", "type": "run", "success": successes > 0,
                             "pathway": pathway_name, "iterations": iterations,
                             "successes": successes, "failures": iterations - successes,
                             "runs": all_runs[-1:] if iterations > 1 else all_runs}
        except Exception as e:
            _jobs[job_id] = {"status": "done", "type": "run", "success": False,
                             "error": str(e)}

    asyncio.get_event_loop().run_in_executor(None, _do_run)
    return {"job_id": job_id, "status": "running"}


@app.post("/api/generate")
async def api_generate(request: Request):
    """Generate competing alleles via LLM in background."""
    data = await request.json()
    locus = data.get("locus")
    count = data.get("count", 1)
    kernel_name = data.get("kernel", "data-mock")
    engine_name = data.get("mutation_engine", "mock")

    if not locus:
        return JSONResponse({"error": "locus is required"}, status_code=400)

    job_id = _uuid.uuid4().hex[:12]
    _jobs[job_id] = {"status": "running", "type": "generate", "locus": locus}
    _prune_jobs()

    def _do_generate():
        root = _project_root
        try:
            from sg.kernel.discovery import load_kernel
            from sg.cli import load_contract_store, make_mutation_engine
            import argparse

            contract_store = load_contract_store(root)
            registry = Registry.open(root / ".sg" / "registry")
            phenotype = PhenotypeMap.load(root / "phenotype.toml")
            kernel = load_kernel(kernel_name)

            args = argparse.Namespace(
                mutation_engine=engine_name,
                model=data.get("model"),
                kernel=kernel_name,
            )
            mutation_engine = make_mutation_engine(args, root, contract_store, kernel=kernel)

            gene = contract_store.get_gene(locus)
            if gene is None:
                _jobs[job_id] = {"status": "done", "type": "generate", "success": False,
                                 "error": f"unknown locus: {locus}"}
                return

            if hasattr(mutation_engine, '_contract_prompt'):
                contract_prompt = mutation_engine._contract_prompt(locus)
            else:
                contract_prompt = f"Locus: {locus}\nDescription: {gene.does}"

            dominant_sha = phenotype.get_dominant(locus)
            parent_gen = 0
            if dominant_sha:
                parent = registry.get(dominant_sha)
                if parent:
                    parent_gen = parent.generation

            sources = mutation_engine.generate(locus, contract_prompt, count=count)

            registered = []
            for src in sources:
                sha = registry.register(src, locus, generation=parent_gen + 1,
                                        parent_sha=dominant_sha)
                allele = registry.get(sha)
                if allele:
                    allele.state = "recessive"
                phenotype.add_to_fallback(locus, sha)
                registered.append(sha[:12])

            registry.save_index()
            phenotype.save(root / "phenotype.toml")
            _jobs[job_id] = {"status": "done", "type": "generate", "success": True,
                             "locus": locus, "registered": registered}
        except Exception as e:
            _jobs[job_id] = {"status": "done", "type": "generate", "success": False,
                             "error": str(e)}

    asyncio.get_event_loop().run_in_executor(None, _do_generate)
    return {"job_id": job_id, "status": "running"}


@app.post("/api/compete")
async def api_compete(request: Request):
    """Run allele competition trials in background. Builds input from contract defaults."""
    data = await request.json()
    locus = data.get("locus")
    rounds = data.get("rounds", 10)
    kernel_name = data.get("kernel", "data-mock")

    if not locus:
        return JSONResponse({"error": "locus is required"}, status_code=400)

    job_id = _uuid.uuid4().hex[:12]
    _jobs[job_id] = {"status": "running", "type": "compete", "locus": locus}
    _prune_jobs()

    def _do_compete():
        root = _project_root
        try:
            from sg.kernel.discovery import load_kernel
            from sg.cli import load_contract_store
            from sg.loader import load_gene, call_gene
            from sg.contracts import validate_output
            from sg.meta_params import MetaParamTracker

            contract_store = load_contract_store(root)
            registry = Registry.open(root / ".sg" / "registry")
            phenotype = PhenotypeMap.load(root / "phenotype.toml")
            kernel = load_kernel(kernel_name)
            mpt = MetaParamTracker.open(root / ".sg" / "meta_params.json")

            # Build input from pathway contract defaults that use this locus
            defaults = {}
            for pw_name in contract_store.known_pathways():
                pw = contract_store.get_pathway(pw_name)
                if pw:
                    for f in pw.takes:
                        if f.default is not None and f.name not in defaults:
                            defaults[f.name] = f.default
            gene = contract_store.get_gene(locus)
            if gene:
                for f in gene.takes:
                    if f.default is not None and f.name not in defaults:
                        defaults[f.name] = f.default
            input_json = json.dumps(defaults)

            alleles = registry.alleles_for_locus(locus)
            if not alleles:
                _jobs[job_id] = {"status": "done", "type": "compete", "success": False,
                                 "error": f"no alleles for locus: {locus}"}
                return

            results = []
            for allele in alleles:
                source = registry.load_source(allele.sha256)
                if source is None:
                    continue
                passed = 0
                last_error = None
                last_output = None
                for _ in range(rounds):
                    try:
                        fn = load_gene(source, kernel)
                        result = call_gene(fn, input_json)
                        last_output = result[:200] if result else None
                        if validate_output(locus, result, contract_store):
                            passed += 1
                            arena.record_success(allele)
                        else:
                            arena.record_failure(allele)
                            last_error = f"validation failed: {result[:120]}" if result else "empty output"
                    except Exception as exc:
                        arena.record_failure(allele)
                        last_error = str(exc)[:150]

                results.append({
                    "sha": allele.sha256[:12],
                    "state": allele.state,
                    "passed": passed,
                    "total": rounds,
                    "fitness": round(arena.compute_fitness(allele, params=mpt.get_params(locus)), 3),
                    "last_error": last_error,
                    "last_output": last_output,
                })

            # Promote the best performer if it beats the current dominant
            results.sort(key=lambda r: r["fitness"], reverse=True)
            best = results[0] if results else None
            dominant_sha = phenotype.get_dominant(locus)
            promoted = None
            if best and dominant_sha:
                dom_result = next((r for r in results if r["sha"] == dominant_sha[:12]), None)
                if dom_result and best["sha"] != dom_result["sha"] and best["fitness"] > dom_result["fitness"] + 0.05:
                    best_full = next((a.sha256 for a in alleles if a.sha256.startswith(best["sha"])), None)
                    if best_full:
                        phenotype.promote(locus, best_full)
                        winner = registry.get(best_full)
                        if winner:
                            winner.state = "dominant"
                        loser = registry.get(dominant_sha)
                        if loser:
                            loser.state = "recessive"
                        promoted = best["sha"]
                        phenotype.save(root / "phenotype.toml")

            registry.save_index()
            _jobs[job_id] = {
                "status": "done", "type": "compete", "success": True,
                "locus": locus, "rounds": rounds, "results": results,
                "dominant": dominant_sha[:12] if dominant_sha else None,
                "promoted": promoted,
            }
        except Exception as e:
            _jobs[job_id] = {"status": "done", "type": "compete", "success": False,
                             "error": str(e)}

    asyncio.get_event_loop().run_in_executor(None, _do_compete)
    return {"job_id": job_id, "status": "running"}


# ── Allele deletion endpoints ────────────────────────────────────────

@app.delete("/api/allele/{sha}")
def api_delete_allele(sha: str):
    """Delete a single allele by SHA (prefix match supported)."""
    root = _project_root
    registry = Registry.open(root / ".sg" / "registry")
    phenotype = PhenotypeMap.load(root / "phenotype.toml")

    full_sha = None
    for s in registry.alleles:
        if s == sha or s.startswith(sha):
            full_sha = s
            break
    if full_sha is None:
        return JSONResponse({"error": f"allele not found: {sha}"}, status_code=404)

    meta = registry.get(full_sha)
    locus = meta.locus if meta else "unknown"
    registry.delete_allele(full_sha)
    phenotype.remove_allele(locus, full_sha)
    registry.save_index()
    phenotype.save(root / "phenotype.toml")
    return {"ok": True, "deleted": full_sha[:12], "locus": locus}


@app.delete("/api/locus/{name}/alleles")
def api_delete_locus_alleles(name: str):
    """Delete all alleles for a specific locus."""
    root = _project_root
    registry = Registry.open(root / ".sg" / "registry")
    phenotype = PhenotypeMap.load(root / "phenotype.toml")

    count = registry.delete_locus(name)
    if count == 0:
        return JSONResponse({"error": f"no alleles for locus: {name}"}, status_code=404)

    phenotype.clear_locus(name)
    registry.save_index()
    phenotype.save(root / "phenotype.toml")
    return {"ok": True, "locus": name, "deleted": count}


@app.delete("/api/alleles")
def api_delete_all_alleles():
    """Delete all alleles from all loci — full genome reset."""
    root = _project_root
    registry = Registry.open(root / ".sg" / "registry")
    phenotype = PhenotypeMap.load(root / "phenotype.toml")

    count = registry.delete_all()
    phenotype.clear_all_loci()
    registry.save_index()
    phenotype.save(root / "phenotype.toml")
    return {"ok": True, "deleted": count}


@dataclasses.dataclass
class DaemonState:
    running: bool = False
    paused: bool = False
    tick_count: int = 0
    tick_interval: float = 30.0
    kernel: str = "data-mock"
    mutation_engine: str = "deepseek"
    auto_mutate: bool = True
    auto_compete: bool = True
    compete_every: int = 5
    compete_rounds: int = 10
    last_pathway: str = ""
    last_tick_error: str = ""
    _task: asyncio.Task | None = dataclasses.field(default=None, repr=False)

    def to_dict(self):
        return {
            "running": self.running,
            "paused": self.paused,
            "tick_count": self.tick_count,
            "tick_interval": self.tick_interval,
            "kernel": self.kernel,
            "mutation_engine": self.mutation_engine,
            "auto_mutate": self.auto_mutate,
            "auto_compete": self.auto_compete,
            "compete_every": self.compete_every,
            "compete_rounds": self.compete_rounds,
            "last_pathway": self.last_pathway,
            "last_tick_error": self.last_tick_error,
        }

_daemon = DaemonState()


async def _daemon_loop():
    """Autonomous evolutionary loop — runs as an asyncio task."""
    root = _project_root
    pathway_index = 0

    while _daemon.running:
        if _daemon.paused:
            await asyncio.sleep(1)
            continue

        await asyncio.sleep(_daemon.tick_interval)
        if not _daemon.running:
            break

        _daemon.tick_count += 1
        tick = _daemon.tick_count
        logger.info("daemon tick %d", tick)

        try:
            result = await asyncio.to_thread(_daemon_tick, root, pathway_index)
            pathway_index = result.get("next_index", pathway_index + 1)
            _daemon.last_pathway = result.get("pathway", "")
            _daemon.last_tick_error = ""
        except (SystemExit, BaseException) as e:
            _daemon.last_tick_error = str(e)
            logger.error("daemon tick %d failed: %s", tick, e, exc_info=True)

    logger.info("daemon loop exited (tick_count=%d)", _daemon.tick_count)


def _daemon_tick(root: Path, pathway_index: int) -> dict:
    """Single daemon tick — runs in a thread. Returns state for next iteration."""
    from sg.kernel.discovery import load_kernel
    from sg.cli import load_contract_store, make_mutation_engine
    from sg.orchestrator import Orchestrator
    from sg.meta_params import MetaParamTracker
    from sg.loader import load_gene, call_gene
    from sg.contracts import validate_output
    import argparse

    contract_store = load_contract_store(root)
    registry = Registry.open(root / ".sg" / "registry")
    phenotype = PhenotypeMap.load(root / "phenotype.toml")
    fusion_tracker = FusionTracker.open(root / "fusion_tracker.json")
    kernel = load_kernel(_daemon.kernel)
    pft = PathwayFitnessTracker.open(root / "pathway_fitness.json")
    pr = PathwayRegistry.open(root / ".sg" / "pathway_registry")
    meta_tracker = MetaParamTracker.open(root / ".sg" / "meta_params.json")

    args_ns = argparse.Namespace(
        mutation_engine=_daemon.mutation_engine,
        model=None, kernel=_daemon.kernel,
    )
    mutation_engine = make_mutation_engine(args_ns, root, contract_store, kernel=kernel)

    orch = Orchestrator(
        registry=registry, phenotype=phenotype,
        mutation_engine=mutation_engine, fusion_tracker=fusion_tracker,
        kernel=kernel, contract_store=contract_store,
        project_root=root,
        pathway_fitness_tracker=pft, pathway_registry=pr,
        meta_param_tracker=meta_tracker,
    )

    pathways = list(contract_store.known_pathways())
    if not pathways:
        orch.save_state()
        return {"pathway": "", "next_index": 0}

    idx = pathway_index % len(pathways)
    pw_name = pathways[idx]

    # Build input from contract defaults
    defaults = {}
    pw_contract = contract_store.get_pathway(pw_name)
    if pw_contract:
        for f in pw_contract.takes:
            if f.default is not None:
                defaults[f.name] = f.default
    input_json = json.dumps(defaults)

    # Phase 1: run the pathway
    try:
        outputs = orch.run_pathway(pw_name, input_json)
        logger.info("daemon tick pathway '%s': %d steps ok", pw_name, len(outputs))
    except Exception as e:
        logger.warning("daemon tick pathway '%s' failed: %s", pw_name, e)

    # Phase 2: auto-mutate — for any locus with 3+ consecutive failures, generate a variant
    if _daemon.auto_mutate:
        for locus in contract_store.known_loci():
            dom_sha = phenotype.get_dominant(locus)
            if not dom_sha:
                continue
            allele = registry.get(dom_sha)
            if allele and allele.consecutive_failures >= 3:
                gene = contract_store.get_gene(locus)
                if gene is None:
                    continue
                try:
                    if hasattr(mutation_engine, '_contract_prompt'):
                        prompt = mutation_engine._contract_prompt(locus)
                    else:
                        prompt = f"Locus: {locus}\nDescription: {gene.does}"
                    sources = mutation_engine.generate(locus, prompt, count=1)
                    for src in sources:
                        sha = registry.register(src, locus, generation=allele.generation + 1,
                                                parent_sha=dom_sha)
                        new_allele = registry.get(sha)
                        if new_allele:
                            new_allele.state = "recessive"
                        phenotype.add_to_fallback(locus, sha)
                    logger.info("daemon auto-mutated %s (failures=%d)", locus, allele.consecutive_failures)
                except Exception as e:
                    logger.warning("daemon auto-mutate %s failed: %s", locus, e)

    # Phase 3: auto-compete — every N ticks, run competition for all loci
    if _daemon.auto_compete and _daemon.tick_count % _daemon.compete_every == 0:
        for locus in contract_store.known_loci():
            alleles = registry.alleles_for_locus(locus)
            if len(alleles) < 2:
                continue

            locus_defaults = {}
            gene = contract_store.get_gene(locus)
            if gene:
                for f in gene.takes:
                    if f.default is not None:
                        locus_defaults[f.name] = f.default
            locus_defaults.update(defaults)
            comp_input = json.dumps(locus_defaults)

            best_sha = None
            best_fitness = -1.0
            for a in alleles:
                source = registry.load_source(a.sha256)
                if source is None:
                    continue
                for _ in range(_daemon.compete_rounds):
                    try:
                        fn = load_gene(source, kernel)
                        result = call_gene(fn, comp_input)
                        if validate_output(locus, result, contract_store):
                            arena.record_success(a)
                        else:
                            arena.record_failure(a)
                    except Exception:
                        arena.record_failure(a)
                fitness = arena.compute_fitness(a, params=meta_tracker.get_params(locus))
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_sha = a.sha256

            dom_sha = phenotype.get_dominant(locus)
            if best_sha and dom_sha and best_sha != dom_sha:
                dom_allele = registry.get(dom_sha)
                dom_fitness = arena.compute_fitness(dom_allele, params=meta_tracker.get_params(locus)) if dom_allele else 0.0
                if best_fitness > dom_fitness + 0.05:
                    phenotype.promote(locus, best_sha)
                    winner = registry.get(best_sha)
                    if winner:
                        winner.state = "dominant"
                    if dom_allele:
                        dom_allele.state = "recessive"
                    logger.info("daemon promoted %s for %s (fitness %.3f > %.3f)",
                                best_sha[:12], locus, best_fitness, dom_fitness)

    orch.verify_scheduler.wait()
    orch.save_state()
    return {"pathway": pw_name, "next_index": idx + 1}


@app.get("/api/daemon/status")
def api_daemon_status():
    return _daemon.to_dict()


@app.post("/api/daemon/start")
async def api_daemon_start(request: Request):
    data = await request.json()
    if _daemon.running:
        return {"ok": False, "error": "daemon already running"}

    _daemon.kernel = data.get("kernel", _daemon.kernel)
    _daemon.mutation_engine = data.get("mutation_engine", _daemon.mutation_engine)
    _daemon.tick_interval = data.get("tick_interval", _daemon.tick_interval)
    _daemon.auto_mutate = data.get("auto_mutate", _daemon.auto_mutate)
    _daemon.auto_compete = data.get("auto_compete", _daemon.auto_compete)
    _daemon.compete_every = data.get("compete_every", _daemon.compete_every)
    _daemon.compete_rounds = data.get("compete_rounds", _daemon.compete_rounds)
    _daemon.running = True
    _daemon.paused = False
    _daemon.tick_count = 0
    _daemon.last_tick_error = ""
    _daemon._task = asyncio.create_task(_daemon_loop())
    logger.info("daemon started (interval=%.1fs, kernel=%s, engine=%s)",
                _daemon.tick_interval, _daemon.kernel, _daemon.mutation_engine)
    return {"ok": True}


@app.post("/api/daemon/stop")
async def api_daemon_stop():
    if not _daemon.running:
        return {"ok": False, "error": "daemon not running"}
    _daemon.running = False
    if _daemon._task:
        _daemon._task.cancel()
        _daemon._task = None
    logger.info("daemon stopped at tick %d", _daemon.tick_count)
    return {"ok": True, "tick_count": _daemon.tick_count}


@app.post("/api/daemon/pause")
async def api_daemon_pause():
    if not _daemon.running:
        return {"ok": False, "error": "daemon not running"}
    _daemon.paused = not _daemon.paused
    logger.info("daemon %s at tick %d", "paused" if _daemon.paused else "resumed", _daemon.tick_count)
    return {"ok": True, "paused": _daemon.paused}


@app.post("/api/daemon/configure")
async def api_daemon_configure(request: Request):
    data = await request.json()
    for key in ("tick_interval", "auto_mutate", "auto_compete", "compete_every",
                "compete_rounds", "kernel", "mutation_engine"):
        if key in data:
            setattr(_daemon, key, data[key])
    return {"ok": True, **_daemon.to_dict()}


# Federation endpoints (used by sg share/pull)

@app.post("/api/federation/receive")
async def federation_receive(request: Request):
    """Accept an allele from a peer with integrity verification."""
    data = await request.json()
    _, reg, pheno, _, _, _, _ = _load_state()
    from sg.federation import import_allele, verify_allele_integrity
    if not verify_allele_integrity(data):
        return JSONResponse({"error": "integrity check failed"}, status_code=400)
    try:
        sha = import_allele(reg, data)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    locus = data.get("locus", "")
    pheno.add_to_fallback(locus, sha)
    allele = reg.get(sha)
    if allele:
        allele.state = "recessive"
    reg.save_index()
    pheno.save(_project_root / "phenotype.toml")
    return {"status": "ok", "sha": sha[:12]}


@app.post("/api/federation/fitness")
async def federation_fitness(request: Request):
    """Accept fitness observations from a peer for an allele."""
    data = await request.json()
    _, reg, _, _, _, _, _ = _load_state()
    sha = data.get("sha256", "")
    peer_name = data.get("peer", "unknown")

    # Resolve prefix
    allele = reg.get(sha)
    if allele is None:
        for full_sha in reg.alleles:
            if full_sha.startswith(sha):
                allele = reg.get(full_sha)
                break

    if allele is None:
        return JSONResponse({"error": "allele not found"}, status_code=404)

    from sg.federation import merge_peer_observation
    merge_peer_observation(allele, peer_name, data)
    reg.save_index()
    return {"status": "ok", "peer_observations": len(allele.peer_observations)}


@app.get("/api/federation/alleles/{locus}")
def federation_alleles(locus: str):
    """Serve alleles for a locus to a peer."""
    _, reg, _, _, _, _, mpt = _load_state()
    from sg.federation import export_allele
    alleles = reg.alleles_for_locus(locus)
    result = []
    for a in alleles[:5]:  # limit to top 5 by fitness
        data = export_allele(reg, a.sha256, meta_param_tracker=mpt)
        if data:
            result.append(data)
    return {"alleles": result}


_DASHBOARD_HTML = r"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Software Genome Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<style>
:root {
  --bg-deep:#0d0d1a; --bg-panel:#1a1a2e; --bg-card:#222244; --border:#333366;
  --text:#e0e0e0; --text-dim:#888; --accent:#0ff; --accent2:#0af;
  --success:#0f0; --warning:#f80; --danger:#f44; --recessive:#ff0;
  --convergence:#0ff; --resilience:#0f0; --immediate:#f80;
}
* { box-sizing:border-box; margin:0; padding:0; }
body { font-family:'JetBrains Mono','Fira Code',monospace; background:var(--bg-deep);
  color:var(--text); display:grid; grid-template-columns:220px 1fr 0px;
  grid-template-rows:auto 1fr; height:100vh; overflow:hidden; font-size:12px; }
body.detail-open { grid-template-columns:220px 1fr 300px; }

/* Command Bar */
#command-bar { grid-column:1/-1; background:var(--bg-panel); border-bottom:1px solid var(--border);
  padding:8px 16px; display:flex; align-items:center; gap:12px; flex-wrap:wrap; z-index:10; }
#command-bar select, #command-bar input, #command-bar button {
  font-family:inherit; font-size:11px; border-radius:3px; }
#command-bar select, #command-bar input {
  background:var(--bg-card); border:1px solid var(--border); color:var(--text);
  padding:5px 8px; }
#command-bar select { min-width:140px; }
#command-bar input.cmd-count { width:50px; }
#command-bar .sep { width:1px; height:24px; background:var(--border); }
#command-bar button {
  padding:5px 14px; cursor:pointer; border:1px solid var(--border);
  font-weight:bold; transition:all 0.15s; }
#command-bar .btn-init { background:#a0f1; color:#c0a0ff; border-color:#c0a0ff; }
#command-bar .btn-init:hover { background:#a0f3; }
#command-bar .btn-run { background:#0f03; color:var(--success); border-color:var(--success); }
#command-bar .btn-run:hover { background:#0f06; }
#command-bar .btn-gen { background:#0ff1; color:var(--accent); border-color:var(--accent); }
#command-bar .btn-gen:hover { background:#0ff3; }
#command-bar .btn-compete { background:#f801; color:var(--warning); border-color:var(--warning); }
#command-bar .btn-compete:hover { background:#f803; }
#command-bar label { color:var(--text-dim); font-size:10px; text-transform:uppercase; }
#command-bar .daemon-controls { display:flex; align-items:center; gap:8px; margin-left:auto; }
#command-bar .daemon-indicator { display:flex; align-items:center; gap:5px; font-size:10px; color:var(--text-dim); }
#command-bar .daemon-dot { width:8px; height:8px; border-radius:50%; background:var(--danger); flex-shrink:0; }
#command-bar .daemon-dot.on { background:var(--success); box-shadow:0 0 6px var(--success); }
#command-bar .daemon-dot.paused { background:var(--warning); box-shadow:0 0 4px var(--warning); }
#command-bar .daemon-tick { color:var(--accent); font-variant-numeric:tabular-nums; }
#command-bar .daemon-btns { display:flex; gap:4px; }
#command-bar .btn-daemon-start { background:#0f03; color:var(--success); border-color:var(--success); }
#command-bar .btn-daemon-start:hover { background:#0f06; }
#command-bar .btn-daemon-start.running { background:#f003; color:var(--danger); border-color:var(--danger); }
#command-bar .btn-daemon-start.running:hover { background:#f006; }
#command-bar .btn-daemon-pause { background:#ff01; color:var(--warning); border-color:var(--warning); }
#command-bar .btn-daemon-pause:hover { background:#ff03; }
.cmd-group { display:flex; align-items:center; gap:6px; }
.cmd-count { width:40px; text-align:center; }

/* Output Toast */
#output-toast { position:fixed; bottom:16px; right:16px; max-width:520px; max-height:60vh;
  background:var(--bg-panel); border:1px solid var(--accent); border-radius:6px;
  padding:12px; z-index:200; display:none; overflow:auto; font-size:11px;
  box-shadow:0 4px 20px rgba(0,0,0,0.5); }
#output-toast.show { display:block; }
#output-toast .toast-header { display:flex; justify-content:space-between; align-items:center;
  margin-bottom:8px; }
#output-toast .toast-header h4 { color:var(--accent); font-size:12px; }
#output-toast .toast-close { background:none; border:none; color:var(--danger); cursor:pointer;
  font-size:16px; font-family:inherit; }
#output-toast pre { white-space:pre-wrap; color:var(--text); line-height:1.5; }
#output-toast .toast-success { color:var(--success); }
#output-toast .toast-error { color:var(--danger); }

/* Sidebar */
#sidebar { background:var(--bg-panel); border-right:1px solid var(--border);
  display:flex; flex-direction:column; overflow:hidden; }
#sidebar .logo { padding:12px 14px 8px; color:var(--accent); font-size:14px; font-weight:bold;
  border-bottom:1px solid var(--border); }
#sidebar .search { padding:8px 10px; }
#sidebar .search input { width:100%; background:var(--bg-card); border:1px solid var(--border);
  color:var(--text); padding:5px 8px; font-family:inherit; font-size:11px; border-radius:3px; }
#sidebar .perspectives { padding:8px 10px; border-bottom:1px solid var(--border); }
#sidebar .perspectives .label { color:var(--text-dim); font-size:10px; text-transform:uppercase;
  letter-spacing:1px; margin-bottom:6px; }
.persp-btn { display:block; width:100%; text-align:left; background:none; border:none;
  color:var(--text-dim); padding:5px 8px; cursor:pointer; font-family:inherit; font-size:12px;
  border-radius:3px; margin-bottom:2px; }
.persp-btn:hover { background:var(--bg-card); }
.persp-btn.active { background:var(--bg-card); color:var(--accent); border-left:2px solid var(--accent); }
#entity-list { flex:1; overflow-y:auto; padding:6px 0; }
.entity-section .section-header { padding:4px 14px; color:var(--text-dim); font-size:10px;
  text-transform:uppercase; cursor:pointer; display:flex; justify-content:space-between; }
.entity-item { padding:4px 14px; cursor:pointer; display:flex; align-items:center; gap:6px;
  border-left:2px solid transparent; }
.entity-item:hover { background:var(--bg-card); }
.entity-item.selected { background:var(--bg-card); border-left-color:var(--accent); }
.entity-item .name { flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.entity-item .fitness-bar { width:40px; height:6px; background:var(--border); border-radius:3px;
  overflow:hidden; }
.entity-item .fitness-bar .fill { height:100%; border-radius:3px; }
.entity-item .fitness-val { color:var(--text-dim); font-size:10px; width:28px; text-align:right; }
#status-bar { padding:8px 14px; border-top:1px solid var(--border); font-size:10px;
  color:var(--text-dim); line-height:1.6; }
#status-bar .val { color:var(--accent); }

/* Main canvas */
#canvas { overflow:auto; padding:16px; }
.view { display:none; height:100%; }
.view.active { display:flex; flex-direction:column; }
.tab-bar { display:flex; gap:2px; margin-bottom:12px; flex-shrink:0; }
.tab-btn { background:var(--bg-panel); border:1px solid var(--border); color:var(--text-dim);
  padding:6px 14px; cursor:pointer; font-family:inherit; font-size:11px; border-radius:3px 3px 0 0; }
.tab-btn:hover { color:var(--text); }
.tab-btn.active { background:var(--bg-card); color:var(--accent); border-bottom-color:var(--bg-card); }
.tab-content { display:none; flex:1; overflow:auto; }
.tab-content.active { display:block; }
.view-title { color:var(--accent); font-size:13px; margin-bottom:10px; }
.empty-state { color:var(--text-dim); text-align:center; padding:40px; font-size:13px; }

/* Cards and tables */
.card { background:var(--bg-card); border:1px solid var(--border); border-radius:4px;
  padding:12px; margin-bottom:8px; }
.card h4 { color:var(--accent); margin-bottom:6px; font-size:12px; }
table { border-collapse:collapse; width:100%; }
th,td { padding:5px 10px; text-align:left; border-bottom:1px solid var(--border); font-size:11px; }
th { color:var(--accent); font-size:10px; text-transform:uppercase; }
a { color:var(--accent2); cursor:pointer; text-decoration:none; }
a:hover { text-decoration:underline; }

/* D3 graph styling */
.node rect,.node circle,.node polygon { stroke:var(--border); stroke-width:1; }
.node text { fill:var(--text); font-size:10px; font-family:inherit; }
.edge line,.edge path { stroke:var(--text-dim); stroke-width:1.5; }
.edge text { fill:var(--text-dim); font-size:9px; }
.edge.feeds { stroke:var(--convergence); }
.edge.verify { stroke:var(--text-dim); stroke-dasharray:4,3; }
marker { fill:var(--text-dim); }

/* Pathway mini cards */
.pw-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(280px,1fr)); gap:10px; }
.pw-card { background:var(--bg-card); border:1px solid var(--border); border-radius:4px;
  padding:12px; cursor:pointer; transition:border-color 0.2s; }
.pw-card:hover { border-color:var(--accent); }
.pw-card .pw-name { color:var(--accent); font-size:12px; margin-bottom:6px; }
.pw-card .pw-steps { color:var(--text-dim); font-size:11px; }
.pw-card .pw-stats { display:flex; gap:12px; margin-top:6px; font-size:10px; color:var(--text-dim); }
.pw-card .pw-stats .val { color:var(--text); }

/* Allele stack */
.allele-stack { display:flex; flex-direction:column; gap:4px; max-width:400px; }
.allele-card { background:var(--bg-card); border:1px solid var(--border); border-radius:4px;
  padding:10px; position:relative; }
.allele-card.dominant { border-left:3px solid var(--success); }
.allele-card.recessive { border-left:3px solid var(--recessive); }
.allele-card.deprecated { border-left:3px solid var(--danger); }
.allele-card .sha { font-size:11px; color:var(--accent2); cursor:pointer; }
.allele-card .meta { font-size:10px; color:var(--text-dim); margin-top:4px; }
.btn-del { background:none; color:var(--danger); border:1px solid var(--danger); border-radius:3px;
  cursor:pointer; font-family:inherit; font-size:11px; padding:4px 12px; opacity:0.7; }
.btn-del:hover { opacity:1; background:rgba(255,60,60,0.15); }
.btn-del-sm { background:none; border:none; color:var(--danger); cursor:pointer; font-size:14px;
  line-height:1; padding:0 2px; opacity:0.4; font-family:inherit; }
.btn-del-sm:hover { opacity:1; }
.allele-connector { width:2px; height:12px; background:var(--border); margin-left:20px; }
.btn-compete { background:var(--recessive); color:#fff; border:none; border-radius:3px; cursor:pointer;
  font-size:11px; padding:4px 12px; }
.btn-compete:hover { background:#c59000; }
.btn-gen { background:var(--accent2); color:#fff; border:none; border-radius:3px; cursor:pointer;
  font-size:11px; padding:4px 12px; }
.btn-gen:hover { filter:brightness(1.2); }

/* Kanban */
.kanban { display:grid; grid-template-columns:repeat(3,1fr); gap:10px; }
.kanban-col { background:var(--bg-panel); border:1px solid var(--border); border-radius:4px;
  padding:8px; min-height:200px; }
.kanban-col h4 { color:var(--text-dim); font-size:10px; text-transform:uppercase;
  text-align:center; margin-bottom:8px; padding-bottom:6px; border-bottom:1px solid var(--border); }
.kanban-card { background:var(--bg-card); border:1px solid var(--border); border-radius:3px;
  padding:8px; margin-bottom:6px; font-size:11px; }
.kanban-card .type { font-size:9px; text-transform:uppercase; padding:1px 4px; border-radius:2px;
  display:inline-block; margin-bottom:4px; }
.kanban-card .type.tighten { background:#0aa3; color:var(--convergence); }
.kanban-card .type.relax { background:#f803; color:var(--warning); }
.kanban-card .type.feeds { background:#0f03; color:var(--success); }

/* Metrics grid */
.metrics-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(160px,1fr)); gap:10px; }
.metric-card { background:var(--bg-card); border:1px solid var(--border); border-radius:4px;
  padding:12px; text-align:center; }
.metric-card .val { font-size:20px; color:var(--accent); margin-bottom:2px; }
.metric-card .label { font-size:10px; color:var(--text-dim); }

/* Audit stream */
.audit-list { max-height:500px; overflow-y:auto; }
.audit-entry { padding:6px 10px; border-bottom:1px solid var(--border); font-size:11px;
  display:flex; gap:8px; align-items:center; }
.audit-entry .badge { font-size:9px; padding:1px 5px; border-radius:2px; text-transform:uppercase;
  flex-shrink:0; }
.badge.promotion { background:#0f03; color:var(--success); }
.badge.demotion { background:#f443; color:var(--danger); }
.badge.mutation { background:#0ff3; color:var(--accent); }
.badge.regression { background:#f803; color:var(--warning); }
.badge.default { background:#8883; color:var(--text-dim); }
.audit-entry .time { color:var(--text-dim); font-size:10px; flex-shrink:0; }
.audit-entry .detail { flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }

/* Progress bar */
.progress { width:100%; height:6px; background:var(--border); border-radius:3px; overflow:hidden; }
.progress .fill { height:100%; border-radius:3px; transition:width 0.3s; }

/* Detail panel */
#detail { background:var(--bg-panel); border-left:1px solid var(--border); overflow-y:auto;
  padding:0; display:none; }
body.detail-open #detail { display:block; }
#detail .header { padding:12px; border-bottom:1px solid var(--border); }
#detail .header .entity-name { color:var(--accent); font-size:13px; }
#detail .header .entity-type { font-size:10px; color:var(--text-dim); text-transform:uppercase; }
#detail .section { padding:10px 12px; border-bottom:1px solid var(--border); }
#detail .section h5 { color:var(--text-dim); font-size:10px; text-transform:uppercase;
  margin-bottom:6px; }
#detail .kv { display:flex; justify-content:space-between; font-size:11px; margin-bottom:3px; }
#detail .kv .k { color:var(--text-dim); }
#detail .close-detail { position:absolute; top:8px; right:8px; color:var(--text-dim);
  cursor:pointer; font-size:16px; background:none; border:none; font-family:inherit; }

/* Source modal */
.modal { display:none; position:fixed; top:3%; left:8%; width:84%; height:90%;
  background:var(--bg-panel); border:1px solid var(--accent); border-radius:6px;
  padding:16px; overflow:auto; z-index:100; }
.modal.open { display:block; }
.modal pre { white-space:pre-wrap; font-size:11px; line-height:1.5; color:var(--text); }
.modal .close-btn { float:right; cursor:pointer; color:var(--danger); font-size:18px;
  background:none; border:none; font-family:inherit; }
.modal h3 { color:var(--accent); font-size:13px; margin-bottom:10px; }
.kw { color:#c792ea; } .str { color:#c3e88d; } .num { color:#f78c6c; }
.cm { color:#546e7a; } .fn { color:#82aaff; }

/* Heatmap */
.heatmap { overflow-x:auto; }
.heatmap svg text { fill:var(--text); font-size:10px; font-family:inherit; }

/* ── Contract Editor ── */
#command-bar .btn-new-pw { background:#0af1; color:#40c0ff; border-color:#40c0ff; }
#command-bar .btn-new-pw:hover { background:#0af3; }
.intent-overlay { display:none; position:fixed; inset:0; background:rgba(0,0,0,0.75); z-index:299; }
.intent-overlay.open { display:block; }
.intent-dialog { display:none; position:fixed; top:50%; left:50%; transform:translate(-50%,-50%);
  background:var(--bg-deep); border:1px solid var(--accent); border-radius:8px;
  width:500px; z-index:300; box-shadow:0 8px 30px rgba(0,0,0,0.6); }
.intent-dialog.open { display:block; }
.intent-header { padding:12px 16px; border-bottom:1px solid var(--border); color:var(--accent);
  font-weight:bold; font-size:13px; }
.intent-body { padding:16px; }
.intent-label { display:block; color:var(--text-dim); font-size:11px; margin-bottom:8px; }
.intent-input { width:100%; background:var(--bg-card); border:1px solid var(--border); color:var(--text);
  padding:10px; font-family:inherit; font-size:12px; border-radius:4px; resize:vertical; }
.intent-input:focus { border-color:var(--accent); outline:none; }
.intent-status { font-size:11px; margin-top:8px; min-height:16px; }
.intent-footer { padding:12px 16px; border-top:1px solid var(--border); display:flex; gap:8px;
  justify-content:flex-end; }
.intent-footer button { padding:6px 16px; font-family:inherit; font-size:11px; border-radius:4px;
  cursor:pointer; }
.ce-overlay { display:none; position:fixed; inset:0; background:rgba(0,0,0,0.75);
  z-index:109; }
.ce-overlay.open { display:block; }
.ce-modal { display:none; position:fixed; top:2%; left:5%; width:90%; height:94%;
  background:var(--bg-deep); border:1px solid var(--accent); border-radius:6px;
  z-index:110; flex-direction:column; box-shadow:0 8px 40px rgba(0,0,0,0.6); }
.ce-modal.open { display:flex; }
.ce-bar { display:flex; align-items:center; gap:12px; padding:8px 16px;
  border-bottom:1px solid var(--border); background:var(--bg-panel); border-radius:6px 6px 0 0; }
.ce-bar .ce-title { flex:1; font-size:13px; color:var(--accent); font-weight:600; }
.ce-bar button { background:none; border:1px solid var(--border); color:var(--text);
  padding:4px 14px; border-radius:3px; cursor:pointer; font-size:11px; font-family:inherit; }
.ce-bar .btn-save { border-color:var(--success); color:var(--success); }
.ce-bar .btn-save:hover { background:rgba(0,255,100,0.08); }
.ce-bar .btn-close { border-color:var(--danger); color:var(--danger); }
.ce-bar .btn-close:hover { background:rgba(255,80,80,0.08); }
.ce-bar .ce-status { font-size:10px; color:var(--text-dim); }
.ce-body { flex:1; overflow:auto; padding:20px 32px; font-family:'SF Mono',Menlo,Consolas,monospace;
  font-size:12px; line-height:1.8; color:var(--text); }

/* Structural keywords — static, dim */
.ce-kw { color:#c792ea; user-select:none; }
.ce-verb { color:#546e7a; user-select:none; }
.ce-punct { color:#546e7a; user-select:none; }
.ce-idx { color:#f78c6c; user-select:none; }

/* Inline inputs — bottom-border only */
.ce-input { background:none; border:none; border-bottom:1px solid #ffffff18;
  color:var(--text); font-family:inherit; font-size:inherit; line-height:inherit;
  padding:0 2px; outline:none; transition:border-color 0.15s; }
.ce-input:hover { border-bottom-color:#ffffff30; background:rgba(255,255,255,0.02); }
.ce-input:focus { border-bottom-color:var(--accent); background:rgba(100,180,255,0.03); }
.ce-input.ce-name { color:#ffcb6b; font-weight:600; }
.ce-input.ce-type { color:#82aaff; width:60px; }
.ce-input.ce-desc { color:#c3e88d; }
.ce-input.ce-default { color:#f78c6c; }
.ce-input.ce-interface { color:#89ddff; width:80px; }
.ce-input.ce-value { color:var(--text); }
.ce-input.ce-domain { color:#c3e88d; }

/* Select styled as input */
.ce-select { background:none; border:none; border-bottom:1px solid #ffffff18;
  color:var(--text); font-family:inherit; font-size:inherit; line-height:inherit;
  padding:0 2px; outline:none; cursor:pointer; -webkit-appearance:none; appearance:none;
  transition:border-color 0.15s; }
.ce-select:hover { border-bottom-color:#ffffff30; }
.ce-select:focus { border-bottom-color:var(--accent); }
.ce-select.ce-family { color:#c792ea; }
.ce-select.ce-risk { color:#f78c6c; }
.ce-select option { background:var(--bg-panel); color:var(--text); }

/* Prose textarea */
.ce-prose { background:none; border:none; border-bottom:1px solid #ffffff18;
  color:var(--text); font-family:inherit; font-size:inherit; line-height:inherit;
  padding:0 2px; outline:none; width:100%; resize:none; overflow-y:hidden;
  min-height:1.8em; transition:border-color 0.15s; white-space:pre-wrap; }
.ce-prose:hover { border-bottom-color:#ffffff30; background:rgba(255,255,255,0.02); }
.ce-prose:focus { border-bottom-color:var(--accent); background:rgba(100,180,255,0.03); }

/* Section structure */
.ce-section { margin-bottom:4px; }
.ce-line { white-space:pre; min-height:1.8em; }
.ce-indent { display:inline; }
.ce-add-btn { display:inline-block; color:var(--text-dim); cursor:pointer; font-size:10px;
  border:1px dashed #ffffff15; padding:1px 8px; border-radius:2px; margin-left:4px;
  user-select:none; transition:all 0.15s; }
.ce-add-btn:hover { color:var(--accent); border-color:var(--accent); background:rgba(100,180,255,0.04); }
.ce-remove-btn { display:inline-block; color:var(--danger); opacity:0; cursor:pointer;
  font-size:9px; margin-left:4px; user-select:none; transition:opacity 0.15s; }
.ce-line:hover .ce-remove-btn { opacity:0.5; }
.ce-remove-btn:hover { opacity:1 !important; }

/* Section add buttons at bottom */
.ce-add-section { margin-top:8px; }
.ce-add-section .ce-add-btn { font-size:11px; padding:2px 12px; }
</style>
</head><body>

<!-- Command Bar -->
<div id="command-bar">
  <div class="cmd-group">
    <label>Kernel</label>
    <select id="cmd-kernel"></select>
  </div>
  <div class="cmd-group">
    <label>Engine</label>
    <select id="cmd-engine">
      <option value="deepseek">deepseek</option>
      <option value="claude">claude</option>
      <option value="openai">openai</option>
    </select>
  </div>
  <div class="sep"></div>
  <button class="btn-init" onclick="cmdInit()" title="Seed all loci from contracts via LLM">Init Genome</button>
  <button class="btn-del" onclick="resetGenome()" title="Delete all alleles from all loci" style="font-size:11px;padding:4px 10px">Reset</button>
  <button class="btn-new-pw" onclick="openNewPathwayDialog()" title="Create a pathway from natural language intent">+ Pathway</button>
  <div class="sep"></div>
  <div class="cmd-group">
    <label>Pathway</label>
    <select id="cmd-pathway"></select>
  </div>
  <div class="cmd-group">
    <label>Iterations</label>
    <input id="cmd-iterations" class="cmd-count" type="number" value="1" min="1" max="100">
  </div>
  <button class="btn-run" onclick="cmdRun()" title="Run pathway using contract defaults">Run</button>
  <div class="sep"></div>
  <div class="cmd-group">
    <label>Locus</label>
    <select id="cmd-locus"></select>
  </div>
  <div class="cmd-group">
    <label>Variants</label>
    <input id="cmd-count" class="cmd-count" type="number" value="1" min="1" max="10">
  </div>
  <button class="btn-gen" onclick="cmdGenerate()" title="Generate competing alleles from contract">Generate</button>
  <div class="cmd-group">
    <label>Rounds</label>
    <input id="cmd-rounds" class="cmd-count" type="number" value="10" min="1" max="100">
  </div>
  <button class="btn-compete" onclick="cmdCompete()" title="Run competition trials">Compete</button>
  <div class="sep"></div>
  <div class="daemon-controls">
    <div class="daemon-indicator">
      <span class="daemon-dot" id="daemon-dot"></span>
      <span id="daemon-label">stopped</span>
      <span id="daemon-tick" class="daemon-tick"></span>
    </div>
    <div class="daemon-btns">
      <button class="btn-daemon-start" id="btn-daemon-toggle" onclick="daemonToggle()" title="Start/stop the evolutionary daemon">Start</button>
      <button class="btn-daemon-pause" id="btn-daemon-pause" onclick="daemonPause()" title="Pause/resume" disabled>Pause</button>
    </div>
    <div class="cmd-group">
      <label>Tick (s)</label>
      <input id="daemon-interval" class="cmd-count" type="number" value="30" min="5" max="600" step="5">
    </div>
  </div>
</div>


<!-- Output Toast -->
<div id="output-toast">
  <div class="toast-header">
    <h4 id="toast-title">Output</h4>
    <button class="toast-close" onclick="closeToast()">&times;</button>
  </div>
  <pre id="toast-body"></pre>
</div>

<!-- Sidebar -->
<div id="sidebar">
  <div class="logo">Software Genome</div>
  <div class="search"><input id="search-input" type="text" placeholder="Search entities..."></div>
  <div class="perspectives">
    <div class="label">Perspectives</div>
    <button class="persp-btn active" data-view="logical">Logical</button>
    <button class="persp-btn" data-view="structural">Structural</button>
    <button class="persp-btn" data-view="evolutionary">Evolutionary</button>
    <button class="persp-btn" data-view="operational">Operational</button>
  </div>
  <div id="entity-list"></div>
  <div id="status-bar"></div>
</div>

<!-- Main Canvas -->
<div id="canvas">
  <!-- Logical View -->
  <div class="view" id="view-logical">
    <div class="tab-bar">
      <button class="tab-btn active" data-tab="logical-flow">Pathway Flow</button>
      <button class="tab-btn" data-tab="logical-feeds">Feeds Network</button>
      <button class="tab-btn" data-tab="logical-contract">Contract</button>
    </div>
    <div class="tab-content active" id="logical-flow"></div>
    <div class="tab-content" id="logical-feeds"></div>
    <div class="tab-content" id="logical-contract"></div>
  </div>
  <!-- Structural View -->
  <div class="view" id="view-structural">
    <div class="tab-bar">
      <button class="tab-btn active" data-tab="struct-hierarchy">Hierarchy</button>
      <button class="tab-btn" data-tab="struct-alleles">Allele Stack</button>
      <button class="tab-btn" data-tab="struct-fusion">Fusion</button>
    </div>
    <div class="tab-content active" id="struct-hierarchy"></div>
    <div class="tab-content" id="struct-alleles"></div>
    <div class="tab-content" id="struct-fusion"></div>
  </div>
  <!-- Evolutionary View -->
  <div class="view" id="view-evolutionary">
    <div class="tab-bar">
      <button class="tab-btn active" data-tab="evo-fitness">Fitness</button>
      <button class="tab-btn" data-tab="evo-lineage">Lineage</button>
      <button class="tab-btn" data-tab="evo-contracts">Contracts</button>
    </div>
    <div class="tab-content active" id="evo-fitness"></div>
    <div class="tab-content" id="evo-lineage"></div>
    <div class="tab-content" id="evo-contracts"></div>
  </div>
  <!-- Operational View -->
  <div class="view" id="view-operational">
    <div class="tab-bar">
      <button class="tab-btn active" data-tab="ops-metrics">Metrics</button>
      <button class="tab-btn" data-tab="ops-timing">Timing</button>
      <button class="tab-btn" data-tab="ops-audit">Audit Log</button>
    </div>
    <div class="tab-content active" id="ops-metrics"></div>
    <div class="tab-content" id="ops-timing"></div>
    <div class="tab-content" id="ops-audit"></div>
  </div>
</div>

<!-- Detail Panel -->
<div id="detail"></div>

<!-- Source Modal -->
<div class="modal" id="source-modal">
  <button class="close-btn" onclick="closeModal()">&times;</button>
  <h3 id="source-title"></h3>
  <pre id="source-code"></pre>
</div>

<!-- Contract Editor Modal -->
<div class="intent-overlay" id="intent-overlay" onclick="closeNewPathwayDialog()"></div>
<div class="intent-dialog" id="intent-dialog">
  <div class="intent-header">New Pathway</div>
  <div class="intent-body">
    <label class="intent-label">Describe what the pathway should do:</label>
    <textarea id="intent-text" class="intent-input" rows="3" placeholder="e.g. Ingest CSV data, validate the schema, clean null values, then verify data quality"></textarea>
    <div class="intent-status" id="intent-status"></div>
  </div>
  <div class="intent-footer">
    <button class="btn-gen" onclick="submitNewPathway()">Generate</button>
    <button class="btn-close" onclick="closeNewPathwayDialog()">Cancel</button>
  </div>
</div>

<div class="ce-overlay" id="ce-overlay" onclick="ceEditorClose()"></div>
<div class="ce-modal" id="contract-editor">
  <div class="ce-bar">
    <span class="ce-title" id="ce-title">Contract Editor</span>
    <span class="ce-status" id="ce-status"></span>
    <button class="btn-save" onclick="ceEditorSave()">Save</button>
    <button class="btn-close" onclick="ceEditorClose()">Close</button>
  </div>
  <div class="ce-body" id="ce-body"></div>
</div>

<script>
// ── Global State ──
const state = {
  perspective: 'logical',
  selectedEntity: null,  // {type:'locus'|'pathway', name:'...'}
  selectedAllele: null,
  cache: {},
  loci: [], pathways: [], status: {}, feeds: {feeds:[],verify_links:[]},
  metrics: '', regression: {history:[]}, fitnessChart: null,
};

// ── Data fetching ──
async function fetchJSON(url) {
  try { return await (await fetch(url)).json(); } catch(e) { return null; }
}
async function fetchText(url) {
  try { return await (await fetch(url)).text(); } catch(e) { return ''; }
}

async function refreshData() {
  const [s, l, p, f, r] = await Promise.all([
    fetchJSON('/api/status'), fetchJSON('/api/loci'), fetchJSON('/api/pathways'),
    fetchJSON('/api/feeds'), fetchJSON('/api/regression'),
  ]);
  if(s) state.status = s;
  if(l) state.loci = l;
  if(p) state.pathways = p;
  if(f) state.feeds = f;
  if(r) state.regression = r;
  state.metrics = await fetchText('/metrics');
  renderSidebar();
  renderCurrentView();
}

// ── SSE with polling fallback ──
function connectSSE() {
  try {
    const es = new EventSource('/api/events');
    es.onmessage = (ev) => {
      try {
        const d = JSON.parse(ev.data);
        if(d.type === 'daemon_tick') { updateDaemonUI(); }
      } catch(e) {}
      refreshData();
    };
    es.onerror = () => { es.close(); setTimeout(connectSSE, 10000); };
  } catch(e) { setInterval(refreshData, 5000); }
}

// ── Fitness color ──
function fitnessColor(f) {
  if(f >= 0.8) return 'var(--success)';
  if(f >= 0.5) return 'var(--warning)';
  if(f > 0) return 'var(--danger)';
  return 'var(--text-dim)';
}

// ── Sidebar ──
function renderSidebar() {
  const search = (document.getElementById('search-input').value||'').toLowerCase();
  const loci = state.loci.filter(l => !search || l.name.includes(search));
  const pws = state.pathways.filter(p => !search || p.name.includes(search));

  document.getElementById('entity-list').innerHTML =
    `<div class="entity-section">
      <div class="section-header">Loci (${loci.length})</div>
      ${loci.map(l => {
        const sel = state.selectedEntity?.type==='locus' && state.selectedEntity.name===l.name;
        return `<div class="entity-item${sel?' selected':''}" onclick="selectEntity('locus','${l.name}')">
          <span class="name">${l.name}</span>
          <span class="fitness-bar"><span class="fill" style="width:${Math.round(l.dominant_fitness*100)}%;background:${fitnessColor(l.dominant_fitness)}"></span></span>
          <span class="fitness-val">${l.dominant_fitness.toFixed(2)}</span>
        </div>`;
      }).join('')}
    </div>
    <div class="entity-section">
      <div class="section-header">Pathways (${pws.length})</div>
      ${pws.map(p => {
        const sel = state.selectedEntity?.type==='pathway' && state.selectedEntity.name===p.name;
        return `<div class="entity-item${sel?' selected':''}" onclick="selectEntity('pathway','${p.name}')">
          <span class="name">${p.name}</span>
          <span class="fitness-bar"><span class="fill" style="width:${Math.round(p.fitness*100)}%;background:${fitnessColor(p.fitness)}"></span></span>
          <span class="fitness-val">${p.fitness.toFixed(2)}</span>
        </div>`;
      }).join('')}
    </div>`;

  const s = state.status;
  const m = parseMetrics(state.metrics);
  document.getElementById('status-bar').innerHTML =
    `Fitness: <span class="val">${(s.avg_fitness||0).toFixed(3)}</span><br>
     Ticks: <span class="val">${m.sg_daemon_ticks_total||0}</span>
     Mutations: <span class="val">${m.sg_mutations_total||0}</span><br>
     Alleles: <span class="val">${s.allele_count||0}</span>
     Loci: <span class="val">${s.loci_count||0}</span>`;
}

function parseMetrics(text) {
  const m = {};
  (text||'').split('\n').forEach(line => {
    if(line.startsWith('#') || !line.trim()) return;
    const [k,v] = line.split(/\s+/);
    if(k && v) m[k] = parseFloat(v);
  });
  return m;
}

// ── Entity Selection ──
function selectEntity(type, name) {
  state.selectedEntity = {type, name};
  state.selectedAllele = null;
  renderSidebar();
  renderCurrentView();
  renderDetail();
  updateHash();
}

// ── Perspective Switching ──
document.querySelectorAll('.persp-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.persp-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    state.perspective = btn.dataset.view;
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    document.getElementById('view-'+state.perspective).classList.add('active');
    renderCurrentView();
    updateHash();
  });
});

// ── Tab Switching ──
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const parent = btn.closest('.view');
    parent.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    parent.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(btn.dataset.tab).classList.add('active');
    renderCurrentView();
  });
});

// ── Search ──
document.getElementById('search-input').addEventListener('input', renderSidebar);

// ── Render Dispatcher ──
function renderCurrentView() {
  const v = state.perspective;
  if(v==='logical') renderLogical();
  else if(v==='structural') renderStructural();
  else if(v==='evolutionary') renderEvolutionary();
  else if(v==='operational') renderOperational();
}

// ══════════════════════════════════════════
// LOGICAL VIEW
// ══════════════════════════════════════════
async function renderLogical() {
  const activeTab = document.querySelector('#view-logical .tab-content.active')?.id;
  if(activeTab==='logical-flow') await renderPathwayFlow();
  else if(activeTab==='logical-feeds') renderFeedsNetwork();
  else if(activeTab==='logical-contract') await renderContractInspector();
}

async function renderPathwayFlow() {
  const el = document.getElementById('logical-flow');
  const sel = state.selectedEntity;

  if(!sel || sel.type!=='pathway') {
    // Show pathway grid
    el.innerHTML = `<div class="view-title">Pathways Overview</div><div class="pw-grid">${
      state.pathways.map(p =>
        `<div class="pw-card" onclick="selectEntity('pathway','${p.name}')">
          <div class="pw-name">${p.name}</div>
          <div class="pw-stats">
            <span>Fitness: <span class="val">${p.fitness.toFixed(3)}</span></span>
            <span>${p.total_successes}/${p.total_successes+p.total_failures} ok</span>
            <span>${p.fused?'Fused':'Steps'}</span>
          </div>
          <div class="progress" style="margin-top:8px">
            <span class="fill" style="width:${Math.round(p.fitness*100)}%;background:${fitnessColor(p.fitness)}"></span>
          </div>
        </div>`
      ).join('')
    }</div>`;
    return;
  }

  // Fetch pathway lineage for steps
  const data = await fetchJSON(`/api/pathway/${sel.name}/lineage`);
  const fitness = await fetchJSON(`/api/pathway/${sel.name}/fitness`);
  if(!data || !data.alleles || data.alleles.length===0) {
    el.innerHTML = '<div class="empty-state">No pathway allele data</div>';
    return;
  }
  const dom = data.alleles.find(a => a.is_dominant) || data.alleles[0];
  const steps = dom.steps || [];
  const timings = fitness?.step_timings || {};

  // Build D3 directed graph
  el.innerHTML = `<div class="view-title">Pathway: ${sel.name}</div><svg id="flow-svg" width="100%" height="400"></svg>`;
  const svg = d3.select('#flow-svg');
  const width = el.clientWidth - 32;
  const nodeW = 140, nodeH = 36, gap = 60;
  const startX = 40, startY = 40;

  const nodes = steps.map((s,i) => ({
    id: i, label: s.target, type: s.step_type,
    x: startX + (i % 4) * (nodeW + gap),
    y: startY + Math.floor(i / 4) * (nodeH + gap + 20),
  }));

  // Find fitness for each locus
  const lociMap = {};
  state.loci.forEach(l => lociMap[l.name] = l);

  // Draw edges
  for(let i=0; i<nodes.length-1; i++) {
    const a=nodes[i], b=nodes[i+1];
    svg.append('line').attr('x1',a.x+nodeW).attr('y1',a.y+nodeH/2)
      .attr('x2',b.x).attr('y2',b.y+nodeH/2)
      .attr('stroke','var(--text-dim)').attr('stroke-width',1.5)
      .attr('marker-end','url(#arrow)');
    const t = timings[b.label];
    if(t) {
      svg.append('text').attr('x',(a.x+nodeW+b.x)/2).attr('y',a.y+nodeH/2-6)
        .attr('text-anchor','middle').attr('fill','var(--text-dim)').attr('font-size',9)
        .text(`${t.avg.toFixed(1)}ms`);
    }
  }

  // Arrow marker
  svg.append('defs').append('marker').attr('id','arrow').attr('viewBox','0 0 10 10')
    .attr('refX',10).attr('refY',5).attr('markerWidth',6).attr('markerHeight',6)
    .attr('orient','auto').append('path').attr('d','M0,0L10,5L0,10Z').attr('fill','var(--text-dim)');

  // Draw nodes
  nodes.forEach(n => {
    const g = svg.append('g').attr('transform',`translate(${n.x},${n.y})`)
      .style('cursor','pointer').on('click', () => selectEntity('locus', n.label));
    const locus = lociMap[n.label];
    const fill = locus ? fitnessColor(locus.dominant_fitness) + '33' : 'var(--bg-card)';
    const stroke = locus ? fitnessColor(locus.dominant_fitness) : 'var(--border)';

    if(n.type==='loop') {
      g.append('polygon').attr('points',`${nodeW/2},0 ${nodeW},${nodeH/2} ${nodeW/2},${nodeH} 0,${nodeH/2}`)
        .attr('fill',fill).attr('stroke',stroke);
    } else if(n.type==='conditional') {
      g.append('polygon').attr('points',`${nodeW/2},0 ${nodeW},${nodeH/2} ${nodeW/2},${nodeH} 0,${nodeH/2}`)
        .attr('fill',fill).attr('stroke',stroke);
    } else if(n.type==='composed') {
      g.append('rect').attr('width',nodeW).attr('height',nodeH).attr('rx',4)
        .attr('fill',fill).attr('stroke',stroke).attr('stroke-width',2);
    } else {
      g.append('rect').attr('width',nodeW).attr('height',nodeH).attr('rx',4)
        .attr('fill',fill).attr('stroke',stroke);
    }
    g.append('text').attr('x',nodeW/2).attr('y',nodeH/2+4)
      .attr('text-anchor','middle').attr('fill','var(--text)').attr('font-size',10)
      .text(n.label.length>18 ? n.label.slice(0,16)+'..' : n.label);
  });

  // Adjust SVG height
  const maxY = Math.max(...nodes.map(n => n.y)) + nodeH + 40;
  svg.attr('height', maxY);
}

// Cached feeds graph state — survives re-renders
let _feedsSim = null;
let _feedsKey = '';

function renderFeedsNetwork() {
  const el = document.getElementById('logical-feeds');
  const {feeds, verify_links} = state.feeds;
  if(!feeds.length && !verify_links.length) {
    el.innerHTML = '<div class="empty-state">No feeds or verify relationships defined</div>';
    _feedsSim = null; _feedsKey = '';
    return;
  }

  // Only rebuild if the graph structure changed
  const key = JSON.stringify([feeds.map(f=>f.source+'>'+f.target).sort(),
    verify_links.map(v=>v.source+'>'+v.target).sort()]);
  if(key === _feedsKey && _feedsSim && document.getElementById('feeds-svg')) {
    // Just update fitness colors in place
    const lociMap = {}; state.loci.forEach(l => lociMap[l.name] = l);
    d3.select('#feeds-svg').selectAll('.node').each(function(d) {
      const l = lociMap[d.id];
      if(l) d.fitness = l.dominant_fitness;
      const g = d3.select(this);
      const fill = fitnessColor(d.fitness)+'33';
      const stroke = fitnessColor(d.fitness);
      g.select('circle').attr('fill',fill).attr('stroke',stroke);
      g.select('rect').attr('fill',fill).attr('stroke',stroke);
    });
    return;
  }
  _feedsKey = key;

  const nodeSet = new Set();
  feeds.forEach(e => { nodeSet.add(e.source); nodeSet.add(e.target); });
  verify_links.forEach(e => { nodeSet.add(e.source); nodeSet.add(e.target); });
  const lociMap = {};
  state.loci.forEach(l => lociMap[l.name] = l);

  const nodeArr = [...nodeSet];
  const width = el.clientWidth - 32 || 600, height = 450;
  const cx = width/2, cy = height/2, radius = Math.min(width, height) * 0.35;

  // Circular initial layout to avoid overlap
  const nodes = nodeArr.map((name, i) => {
    const angle = (2 * Math.PI * i) / nodeArr.length - Math.PI/2;
    const l = lociMap[name];
    const isDiag = feeds.some(f => f.source===name && f.source_family==='diagnostic');
    return {id:name, fitness: l?.dominant_fitness||0, isDiag,
            x: cx + radius * Math.cos(angle), y: cy + radius * Math.sin(angle)};
  });
  const links = [
    ...feeds.map(f => ({source:f.source, target:f.target, type:'feeds', timescale:f.timescale})),
    ...verify_links.map(v => ({source:v.source, target:v.target, type:'verify', delay:v.delay})),
  ];

  el.innerHTML = `<div class="view-title">Feeds & Verify Network</div><svg id="feeds-svg" width="100%" height="450"></svg>`;
  const svg = d3.select('#feeds-svg');

  const defs = svg.append('defs');
  ['feeds','verify'].forEach(t => {
    defs.append('marker').attr('id',`arr-${t}`).attr('viewBox','0 0 10 10')
      .attr('refX',20).attr('refY',5).attr('markerWidth',5).attr('markerHeight',5)
      .attr('orient','auto').append('path').attr('d','M0,0L10,5L0,10Z')
      .attr('fill', t==='feeds'?'var(--convergence)':'var(--text-dim)');
  });

  if(_feedsSim) _feedsSim.stop();
  _feedsSim = d3.forceSimulation(nodes)
    .force('link', d3.forceLink(links).id(d=>d.id).distance(160).strength(0.6))
    .force('charge', d3.forceManyBody().strength(-500))
    .force('center', d3.forceCenter(cx, cy))
    .force('collide', d3.forceCollide(55))
    .force('x', d3.forceX(cx).strength(0.05))
    .force('y', d3.forceY(cy).strength(0.05))
    .alphaDecay(0.03);

  const link = svg.selectAll('.edge').data(links).join('line')
    .attr('class', d => `edge ${d.type}`)
    .attr('stroke', d => {
      if(d.type==='verify') return 'var(--text-dim)';
      if(d.timescale==='convergence') return 'var(--convergence)';
      if(d.timescale==='resilience') return 'var(--resilience)';
      return 'var(--immediate)';
    })
    .attr('stroke-dasharray', d => d.type==='verify'?'5,3':null)
    .attr('marker-end', d => `url(#arr-${d.type})`);

  const node = svg.selectAll('.node').data(nodes).join('g')
    .attr('class','node').style('cursor','pointer')
    .on('click', (e,d) => selectEntity('locus', d.id))
    .call(d3.drag()
      .on('start',(e,d)=>{if(!e.active)_feedsSim.alphaTarget(0.3).restart();d.fx=d.x;d.fy=d.y;})
      .on('drag',(e,d)=>{d.fx=e.x;d.fy=e.y;})
      .on('end',(e,d)=>{if(!e.active)_feedsSim.alphaTarget(0);})
    );

  node.each(function(d) {
    const g = d3.select(this);
    const fill = fitnessColor(d.fitness)+'33';
    const stroke = fitnessColor(d.fitness);
    if(d.isDiag) {
      g.append('circle').attr('r',16).attr('fill',fill).attr('stroke',stroke);
    } else {
      g.append('rect').attr('x',-16).attr('y',-16).attr('width',32).attr('height',32)
        .attr('rx',3).attr('fill',fill).attr('stroke',stroke);
    }
    g.append('text').attr('dy',28).attr('text-anchor','middle').attr('font-size',9)
      .attr('fill','var(--text)').text(d.id.length>20?d.id.slice(0,18)+'..':d.id);
  });

  _feedsSim.on('tick', () => {
    link.attr('x1',d=>d.source.x).attr('y1',d=>d.source.y)
      .attr('x2',d=>d.target.x).attr('y2',d=>d.target.y);
    node.attr('transform',d=>`translate(${d.x},${d.y})`);
  });
}

async function renderContractInspector() {
  const el = document.getElementById('logical-contract');
  const sel = state.selectedEntity;
  if(!sel || sel.type!=='locus') {
    el.innerHTML = '<div class="empty-state">Select a locus to view its contract</div>';
    return;
  }
  const data = await fetchJSON(`/api/locus/${sel.name}`);
  if(!data || !data.contract) {
    el.innerHTML = '<div class="empty-state">No contract data available</div>';
    return;
  }
  const c = data.contract;
  el.innerHTML = `
    <div class="view-title">Contract: ${sel.name}</div>
    <div class="card">
      <h4>gene ${sel.name}</h4>
      <div style="margin-bottom:8px;color:var(--text-dim)">
        <span style="color:${c.family==='configuration'?'var(--accent)':'var(--success)'}">${c.family}</span>
        &middot; risk: <span style="color:${c.risk==='critical'?'var(--danger)':c.risk==='high'?'var(--warning)':'var(--text)'}">${c.risk}</span>
      </div>
      <div style="margin-bottom:10px;font-style:italic;color:var(--text-dim)">${c.does||''}</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
        <div>
          <div style="color:var(--accent);font-size:10px;text-transform:uppercase;margin-bottom:4px">Takes</div>
          ${(c.takes||[]).map(f=>`<div style="font-size:11px">${f.name} <span style="color:var(--text-dim)">${f.type}</span></div>`).join('')}
        </div>
        <div>
          <div style="color:var(--accent);font-size:10px;text-transform:uppercase;margin-bottom:4px">Gives</div>
          ${(c.gives||[]).map(f=>`<div style="font-size:11px">${f.name} <span style="color:var(--text-dim)">${f.type}</span></div>`).join('')}
        </div>
      </div>
    </div>`;
}

// ══════════════════════════════════════════
// STRUCTURAL VIEW
// ══════════════════════════════════════════
async function renderStructural() {
  const activeTab = document.querySelector('#view-structural .tab-content.active')?.id;
  if(activeTab==='struct-hierarchy') renderHierarchy();
  else if(activeTab==='struct-alleles') await renderAlleleStack();
  else if(activeTab==='struct-fusion') renderFusionState();
}

function renderHierarchy() {
  const el = document.getElementById('struct-hierarchy');
  const pws = state.pathways;
  const loci = state.loci;
  const lociInPathways = new Set();

  // Build tree data
  const children = pws.map(pw => {
    const pwLoci = [];
    // We'd need steps data per pathway — approximate from loci for now
    return {name: pw.name, type:'pathway', fitness: pw.fitness,
      children: [], fused: pw.fused, reinforcement: pw.reinforcement_count};
  });

  // Loci not in pathways
  const orphanLoci = loci.map(l => ({name:l.name, type:'locus', fitness:l.dominant_fitness,
    allele_count:l.allele_count, dominant_sha:l.dominant_sha}));

  el.innerHTML = `<div class="view-title">Composition Hierarchy</div>
    <div class="card"><h4>Pathways</h4>
    ${pws.map(pw => `
      <div style="margin-bottom:8px;cursor:pointer" onclick="selectEntity('pathway','${pw.name}')">
        <div style="color:var(--accent2)">${pw.name}</div>
        <div style="font-size:10px;color:var(--text-dim)">
          Fitness: ${pw.fitness.toFixed(3)} | ${pw.fused?'Fused':'Decomposed'} |
          ${pw.total_successes+pw.total_failures} executions
        </div>
        <div class="progress" style="width:200px;margin-top:3px">
          <span class="fill" style="width:${Math.round(pw.fitness*100)}%;background:${fitnessColor(pw.fitness)}"></span>
        </div>
      </div>
    `).join('')}
    </div>
    <div class="card"><h4>Loci</h4>
    ${loci.map(l => `
      <div style="margin-bottom:6px;display:flex;align-items:center;gap:8px;cursor:pointer"
        onclick="selectEntity('locus','${l.name}')">
        <span style="color:${fitnessColor(l.dominant_fitness)};font-size:10px">●</span>
        <span>${l.name}</span>
        <span style="color:var(--text-dim);font-size:10px">${l.allele_count} allele${l.allele_count!==1?'s':''}</span>
        <span style="color:var(--text-dim);font-size:10px">${l.dominant_fitness.toFixed(3)}</span>
      </div>
    `).join('')}
    </div>`;
}

async function renderAlleleStack() {
  const el = document.getElementById('struct-alleles');
  const sel = state.selectedEntity;
  if(!sel || sel.type!=='locus') {
    el.innerHTML = '<div class="empty-state">Select a locus to view its allele stack</div>';
    return;
  }
  const data = await fetchJSON(`/api/locus/${sel.name}`);
  if(!data) { el.innerHTML = '<div class="empty-state">Loading...</div>'; return; }

  const alleles = data.alleles || [];
  const hasCompetitors = alleles.length > 1;
  el.innerHTML = `<div style="display:flex;align-items:center;gap:12px;margin-bottom:10px">
      <div class="view-title" style="margin:0">Allele Stack: ${sel.name}</div>
      ${hasCompetitors ? `<button class="btn-compete" style="font-size:11px;padding:4px 12px"
        onclick="competeLocus('${sel.name}')">Compete (${alleles.length} alleles)</button>` : ''}
      <button class="btn-gen" style="font-size:11px;padding:4px 12px"
        onclick="generateForLocus('${sel.name}')">+ Generate</button>
      <button class="btn-del" style="font-size:11px;padding:4px 12px"
        onclick="clearLocusAlleles('${sel.name}')">Clear All</button>
    </div>
    <div class="allele-stack">
    ${alleles.map((a,i) => `
      ${i>0?'<div class="allele-connector"></div>':''}
      <div class="allele-card ${a.state}">
        <div style="display:flex;justify-content:space-between;align-items:center">
          <span class="sha" onclick="showSource('${a.sha}')">${a.sha}</span>
          <div style="display:flex;align-items:center;gap:8px">
            <span style="font-size:10px;color:${a.is_dominant?'var(--success)':a.state==='recessive'?'var(--recessive)':'var(--danger)'}">${a.state}</span>
            <button class="btn-del-sm" onclick="deleteAllele('${a.sha}','${sel.name}')" title="Delete this allele">&times;</button>
          </div>
        </div>
        <div class="meta">
          gen ${a.generation} | fitness ${a.fitness.toFixed(3)} |
          ${a.successful_invocations}/${a.successful_invocations+a.failed_invocations} ok
          ${a.parent_sha?` | parent: <a onclick="showSource('${a.parent_sha}')">${a.parent_sha}</a>`:''}
        </div>
        <div class="progress" style="margin-top:4px">
          <span class="fill" style="width:${Math.round(a.fitness*100)}%;background:${fitnessColor(a.fitness)}"></span>
        </div>
      </div>
    `).join('')}
    </div>`;
}

function renderFusionState() {
  const el = document.getElementById('struct-fusion');
  const sel = state.selectedEntity;
  if(!sel || sel.type!=='pathway') {
    el.innerHTML = '<div class="empty-state">Select a pathway to view fusion state</div>';
    return;
  }
  const pw = state.pathways.find(p => p.name===sel.name);
  if(!pw) { el.innerHTML = '<div class="empty-state">Pathway not found</div>'; return; }

  const pct = Math.min(pw.reinforcement_count / 10 * 100, 100);
  el.innerHTML = `<div class="view-title">Fusion: ${sel.name}</div>
    <div class="card">
      <h4>${pw.fused?'Fused':'Decomposed (individual steps)'}</h4>
      ${pw.fused?`<div style="margin:8px 0">Fused SHA: <a onclick="showSource('${pw.fused_sha}')">${pw.fused_sha}</a></div>`:''}
      <div style="margin:8px 0">
        <div style="font-size:10px;color:var(--text-dim);margin-bottom:3px">
          Reinforcement: ${pw.reinforcement_count}/10
        </div>
        <div class="progress" style="width:200px">
          <span class="fill" style="width:${pct}%;background:${pct>=100?'var(--success)':'var(--accent)'}"></span>
        </div>
      </div>
      <div style="font-size:11px;color:var(--text-dim);margin-top:8px">
        Successes: ${pw.total_successes} | Failures: ${pw.total_failures} |
        Avg: ${pw.avg_time_ms}ms
      </div>
    </div>`;
}

// ══════════════════════════════════════════
// EVOLUTIONARY VIEW
// ══════════════════════════════════════════
async function renderEvolutionary() {
  const activeTab = document.querySelector('#view-evolutionary .tab-content.active')?.id;
  if(activeTab==='evo-fitness') await renderFitnessTimeline();
  else if(activeTab==='evo-lineage') await renderLineageTree();
  else if(activeTab==='evo-contracts') await renderContractEvolution();
}

async function renderFitnessTimeline() {
  const el = document.getElementById('evo-fitness');
  const sel = state.selectedEntity;

  if(!sel) {
    // Show regression overview
    const reg = state.regression;
    el.innerHTML = `<div class="view-title">Regression Monitor</div>
      ${reg.history.length===0?'<div class="empty-state">No regression data yet</div>':
      `<table><thead><tr><th>Allele</th><th>Peak</th><th>Current</th><th>Drop</th><th>Status</th></tr></thead>
      <tbody>${reg.history.map(h => {
        let st = '<span style="color:var(--success)">stable</span>';
        if(h.drop>=0.4) st='<span style="color:var(--danger)">SEVERE</span>';
        else if(h.drop>=0.2) st='<span style="color:var(--warning)">mild</span>';
        return `<tr><td><a onclick="showSource('${h.sha}')">${h.sha}</a></td>
          <td>${h.peak_fitness.toFixed(3)}</td><td>${h.last_fitness.toFixed(3)}</td>
          <td>${h.drop.toFixed(3)}</td><td>${st}</td></tr>`;
      }).join('')}</tbody></table>`}`;
    return;
  }

  if(sel.type==='locus') {
    const data = await fetchJSON(`/api/locus/${sel.name}`);
    if(!data || !data.alleles.length) {
      el.innerHTML = '<div class="empty-state">No allele data</div>'; return;
    }
    const dom = data.alleles.find(a=>a.is_dominant) || data.alleles[0];
    el.innerHTML = `<div class="view-title">Fitness: ${sel.name}</div>
      <div class="card">
        <div style="display:flex;gap:20px;margin-bottom:10px">
          <div><span style="color:var(--text-dim)">Dominant:</span> <a onclick="showSource('${dom.sha}')">${dom.sha}</a></div>
          <div><span style="color:var(--text-dim)">Gen:</span> ${dom.generation}</div>
          <div><span style="color:var(--text-dim)">Fitness:</span>
            <span style="color:${fitnessColor(dom.fitness)}">${dom.fitness.toFixed(3)}</span></div>
        </div>
        <div style="display:flex;gap:20px;font-size:11px">
          <div>Success: <span style="color:var(--success)">${dom.successful_invocations}</span></div>
          <div>Failed: <span style="color:var(--danger)">${dom.failed_invocations}</span></div>
          <div>State: <span style="color:${dom.state==='dominant'?'var(--success)':'var(--recessive)'}">${dom.state}</span></div>
        </div>
        <div class="progress" style="margin-top:10px;height:8px">
          <span class="fill" style="width:${Math.round(dom.fitness*100)}%;background:${fitnessColor(dom.fitness)}"></span>
        </div>
      </div>
      <div class="card"><h4>All Alleles</h4>
      <table><thead><tr><th>SHA</th><th>Gen</th><th>Fitness</th><th>State</th><th>Invocations</th></tr></thead>
      <tbody>${data.alleles.map(a=>`<tr>
        <td><a onclick="showSource('${a.sha}')">${a.sha}</a></td>
        <td>${a.generation}</td>
        <td style="color:${fitnessColor(a.fitness)}">${a.fitness.toFixed(3)}</td>
        <td style="color:${a.state==='dominant'?'var(--success)':a.state==='recessive'?'var(--recessive)':'var(--danger)'}">${a.state}</td>
        <td>${a.successful_invocations}/${a.successful_invocations+a.failed_invocations}</td>
      </tr>`).join('')}</tbody></table></div>`;
  } else {
    const data = await fetchJSON(`/api/pathway/${sel.name}/fitness`);
    if(!data) { el.innerHTML='<div class="empty-state">No data</div>'; return; }
    el.innerHTML = `<div class="view-title">Pathway Fitness: ${sel.name}</div>
      <div class="card">
        <div style="display:flex;gap:20px;font-size:11px">
          <div>Fitness: <span style="color:${fitnessColor(data.fitness)}">${data.fitness.toFixed(3)}</span></div>
          <div>Executions: ${data.total_executions}</div>
          <div>Success: <span style="color:var(--success)">${data.successful_executions}</span></div>
          <div>Failed: <span style="color:var(--danger)">${data.failed_executions}</span></div>
          <div>Avg: ${data.avg_time_ms}ms</div>
        </div>
      </div>`;
  }
}

async function renderLineageTree() {
  const el = document.getElementById('evo-lineage');
  const sel = state.selectedEntity;
  if(!sel) { el.innerHTML='<div class="empty-state">Select an entity to view lineage</div>'; return; }

  if(sel.type==='locus') {
    const dom = state.loci.find(l=>l.name===sel.name);
    if(!dom?.dominant_sha) { el.innerHTML='<div class="empty-state">No dominant allele</div>'; return; }
    const data = await fetchJSON(`/api/lineage/${dom.dominant_sha}`);
    if(!data) { el.innerHTML='<div class="empty-state">No lineage data</div>'; return; }

    el.innerHTML = `<div class="view-title">Lineage: ${sel.name}</div>
      <div style="display:flex;align-items:center;gap:4px;flex-wrap:wrap;padding:8px">
      ${data.lineage.map((n,i) =>
        `${i>0?'<span style="color:var(--text-dim);font-size:16px">&larr;</span>':''}
        <div class="card" style="display:inline-block;margin:0;padding:8px;min-width:120px;cursor:pointer"
          onclick="showSource('${n.sha}')">
          <div style="color:var(--accent2);font-size:11px">${n.sha}</div>
          <div style="font-size:10px;color:var(--text-dim)">
            gen ${n.generation} | ${n.fitness.toFixed(3)} |
            <span style="color:${n.state==='dominant'?'var(--success)':n.state==='recessive'?'var(--recessive)':'var(--danger)'}">${n.state}</span>
          </div>
        </div>`
      ).join('')}
      </div>`;
  } else {
    const data = await fetchJSON(`/api/pathway/${sel.name}/lineage`);
    if(!data || !data.alleles.length) { el.innerHTML='<div class="empty-state">No pathway alleles</div>'; return; }
    el.innerHTML = `<div class="view-title">Pathway Lineage: ${sel.name}</div>
      <table><thead><tr><th>SHA</th><th>State</th><th>Fitness</th><th>Exec</th><th>Mutation</th><th>Steps</th></tr></thead>
      <tbody>${data.alleles.map(a=>`<tr>
        <td>${a.sha}${a.is_dominant?' *':''}</td>
        <td style="color:${a.state==='dominant'?'var(--success)':'var(--recessive)'}">${a.state}</td>
        <td>${a.fitness.toFixed(3)}</td>
        <td>${a.total_executions}</td>
        <td>${a.mutation_operator||'seed'}</td>
        <td>${(a.steps||[]).map(s=>s.target).join(' -> ')}</td>
      </tr>`).join('')}</tbody></table>`;
  }
}

async function renderContractEvolution() {
  const el = document.getElementById('evo-contracts');
  const data = await fetchJSON('/api/contract_evolution');
  if(!data) { el.innerHTML='<div class="empty-state">No contract evolution data</div>'; return; }

  const pending = data.proposals.filter(p=>p.status==='pending');
  const accepted = data.proposals.filter(p=>p.status==='accepted');
  const rejected = data.proposals.filter(p=>p.status==='rejected');

  function renderCard(p) {
    const typeClass = p.proposal_type.includes('tighten')?'tighten':
      p.proposal_type.includes('relax')?'relax':'feeds';
    return `<div class="kanban-card">
      <span class="type ${typeClass}">${p.proposal_type}</span>
      <div style="margin-top:4px"><strong>${p.locus}</strong></div>
      <div style="color:var(--text-dim);margin-top:2px">${p.description}</div>
      <div style="color:var(--text-dim);font-size:10px;margin-top:4px">${p.evidence_count} observations</div>
    </div>`;
  }

  el.innerHTML = `<div class="view-title">Contract Evolution</div>
    <div class="kanban">
      <div class="kanban-col"><h4>Pending (${pending.length})</h4>${pending.map(renderCard).join('')}</div>
      <div class="kanban-col"><h4>Accepted (${accepted.length})</h4>${accepted.map(renderCard).join('')}</div>
      <div class="kanban-col"><h4>Rejected (${rejected.length})</h4>${rejected.map(renderCard).join('')}</div>
    </div>`;
}

// ══════════════════════════════════════════
// OPERATIONAL VIEW
// ══════════════════════════════════════════
async function renderOperational() {
  const activeTab = document.querySelector('#view-operational .tab-content.active')?.id;
  if(activeTab==='ops-metrics') renderMetrics();
  else if(activeTab==='ops-timing') await renderTiming();
  else if(activeTab==='ops-audit') await renderAuditLog();
}

function renderMetrics() {
  const el = document.getElementById('ops-metrics');
  const m = parseMetrics(state.metrics);

  const items = [
    ['Daemon Ticks', m.sg_daemon_ticks_total||0, ''],
    ['Promotions', m.sg_promotions_total||0, ''],
    ['Demotions', m.sg_demotions_total||0, ''],
    ['Mutations', m.sg_mutations_total||0, ''],
    ['Pathway Execs', m.sg_pathway_executions_total||0, ''],
    ['Pathway Fails', m.sg_pathway_failures_total||0, ''],
    ['Avg Fitness', (m.sg_avg_fitness||0).toFixed(3), ''],
    ['Active Loci', m.sg_active_loci||0, ''],
    ['Tick Duration', (m.sg_last_tick_duration_ms||0).toFixed(1)+'ms', ''],
  ];

  el.innerHTML = `<div class="view-title">System Metrics</div>
    <div class="metrics-grid">
    ${items.map(([label,val])=>
      `<div class="metric-card"><div class="val">${val}</div><div class="label">${label}</div></div>`
    ).join('')}
    </div>`;
}

async function renderTiming() {
  const el = document.getElementById('ops-timing');
  const rows = [];
  for(const pw of state.pathways) {
    const data = await fetchJSON(`/api/pathway/${pw.name}/fitness`);
    if(data?.step_timings) {
      for(const [step, info] of Object.entries(data.step_timings)) {
        rows.push({pathway:pw.name, step, avg:info.avg, count:info.count});
      }
    }
  }
  if(!rows.length) {
    el.innerHTML = '<div class="empty-state">No timing data available</div>';
    return;
  }

  // Build heatmap
  const steps = [...new Set(rows.map(r=>r.step))];
  const pathways = [...new Set(rows.map(r=>r.pathway))];
  const maxTime = Math.max(...rows.map(r=>r.avg));

  el.innerHTML = `<div class="view-title">Step Timing Heatmap</div>
    <table><thead><tr><th>Pathway</th>${steps.map(s=>`<th>${s}</th>`).join('')}</tr></thead>
    <tbody>${pathways.map(pw => {
      return `<tr><td>${pw}</td>${steps.map(s => {
        const r = rows.find(x=>x.pathway===pw && x.step===s);
        if(!r) return '<td>-</td>';
        const ratio = r.avg / maxTime;
        const bg = ratio > 0.7 ? 'var(--danger)' : ratio > 0.4 ? 'var(--warning)' : 'var(--success)';
        return `<td style="background:${bg}33;color:${bg}">${r.avg.toFixed(1)}ms</td>`;
      }).join('')}</tr>`;
    }).join('')}</tbody></table>`;
}

async function renderAuditLog() {
  const el = document.getElementById('ops-audit');
  const data = await fetchJSON('/api/audit?count=100');
  if(!data || !data.entries.length) {
    el.innerHTML = '<div class="empty-state">No audit entries</div>';
    return;
  }

  el.innerHTML = `<div class="view-title">Audit Log</div>
    <div class="audit-list">
    ${data.entries.map(e => {
      const t = new Date(e.timestamp*1000);
      const time = t.toLocaleTimeString();
      let badgeClass = 'default';
      if(e.event.includes('promotion')) badgeClass='promotion';
      else if(e.event.includes('demotion')) badgeClass='demotion';
      else if(e.event.includes('mutation')) badgeClass='mutation';
      else if(e.event.includes('regression')) badgeClass='regression';
      return `<div class="audit-entry">
        <span class="time">${time}</span>
        <span class="badge ${badgeClass}">${e.event}</span>
        <span class="detail">${e.locus}${e.sha?' '+e.sha.slice(0,12):''}</span>
      </div>`;
    }).join('')}
    </div>`;
}

// ══════════════════════════════════════════
// DETAIL PANEL
// ══════════════════════════════════════════
async function renderDetail() {
  const sel = state.selectedEntity;
  if(!sel) { document.body.classList.remove('detail-open'); return; }
  document.body.classList.add('detail-open');

  const dp = document.getElementById('detail');
  if(sel.type==='locus') {
    const data = await fetchJSON(`/api/locus/${sel.name}`);
    const l = state.loci.find(x=>x.name===sel.name);
    const dom = data?.alleles?.find(a=>a.is_dominant);
    const c = data?.contract;
    dp.innerHTML = `
      <div class="header" style="position:relative">
        <button class="close-detail" onclick="closeDetail()">&times;</button>
        <div class="entity-type">Locus</div>
        <div class="entity-name">${sel.name}</div>
        <div class="progress" style="margin-top:6px;height:4px">
          <span class="fill" style="width:${Math.round((l?.dominant_fitness||0)*100)}%;background:${fitnessColor(l?.dominant_fitness||0)}"></span>
        </div>
      </div>
      ${c?`<div class="section"><h5>Contract</h5>
        <div style="color:var(--text-dim);font-size:11px;margin-bottom:4px">${c.does||''}</div>
        <div class="kv"><span class="k">Family</span><span>${c.family}</span></div>
        <div class="kv"><span class="k">Risk</span><span>${c.risk}</span></div>
        <button style="margin-top:6px;background:none;border:1px solid var(--accent);color:var(--accent);padding:3px 12px;border-radius:3px;cursor:pointer;font-size:10px;font-family:inherit" onclick="ceEditorOpen('${sel.name}')">Edit Contract</button>
      </div>`:''}
      ${dom?`<div class="section"><h5>Dominant Allele</h5>
        <div class="kv"><span class="k">SHA</span><a onclick="showSource('${dom.sha}')">${dom.sha}</a></div>
        <div class="kv"><span class="k">Generation</span><span>${dom.generation}</span></div>
        <div class="kv"><span class="k">Fitness</span><span style="color:${fitnessColor(dom.fitness)}">${dom.fitness.toFixed(3)}</span></div>
        <div class="kv"><span class="k">Invocations</span><span>${dom.successful_invocations+dom.failed_invocations}</span></div>
        <div class="kv"><span class="k">Success Rate</span><span>${dom.successful_invocations}/${dom.successful_invocations+dom.failed_invocations}</span></div>
      </div>`:''}
      <div class="section"><h5>Quick Stats</h5>
        <div class="kv"><span class="k">Alleles</span><span>${data?.alleles?.length||0}</span></div>
        <div class="kv"><span class="k">Dominant Fitness</span><span>${(l?.dominant_fitness||0).toFixed(3)}</span></div>
      </div>`;
  } else {
    const pw = state.pathways.find(p=>p.name===sel.name);
    dp.innerHTML = `
      <div class="header" style="position:relative">
        <button class="close-detail" onclick="closeDetail()">&times;</button>
        <div class="entity-type">Pathway</div>
        <div class="entity-name">${sel.name}</div>
        <div class="progress" style="margin-top:6px;height:4px">
          <span class="fill" style="width:${Math.round((pw?.fitness||0)*100)}%;background:${fitnessColor(pw?.fitness||0)}"></span>
        </div>
      </div>
      <div class="section"><h5>Stats</h5>
        <div class="kv"><span class="k">Fitness</span><span style="color:${fitnessColor(pw?.fitness||0)}">${(pw?.fitness||0).toFixed(3)}</span></div>
        <div class="kv"><span class="k">Executions</span><span>${pw?.total_successes+pw?.total_failures||0}</span></div>
        <div class="kv"><span class="k">Successes</span><span style="color:var(--success)">${pw?.total_successes||0}</span></div>
        <div class="kv"><span class="k">Failures</span><span style="color:var(--danger)">${pw?.total_failures||0}</span></div>
        <div class="kv"><span class="k">Avg Time</span><span>${pw?.avg_time_ms||0}ms</span></div>
        <div class="kv"><span class="k">Fused</span><span>${pw?.fused?'Yes':'No'}</span></div>
        <div class="kv"><span class="k">Reinforcement</span><span>${pw?.reinforcement_count||0}/10</span></div>
        <div class="kv"><span class="k">Alleles</span><span>${pw?.pathway_allele_count||0}</span></div>
        <button style="margin-top:6px;background:none;border:1px solid var(--accent);color:var(--accent);padding:3px 12px;border-radius:3px;cursor:pointer;font-size:10px;font-family:inherit" onclick="ceEditorOpen('${sel.name}')">Edit Contract</button>
      </div>`;
  }
}

function closeDetail() {
  document.body.classList.remove('detail-open');
}

// ══════════════════════════════════════════
// SOURCE MODAL
// ══════════════════════════════════════════
async function showSource(sha) {
  if(!sha || sha==='none') return;
  const d = await fetchJSON('/api/allele/'+sha+'/source');
  document.getElementById('source-title').textContent = 'Source: ' + sha;
  const src = d?.source || d?.error || 'not found';
  // Python highlighting — tokenize to avoid corrupting HTML tags
  const escaped = src.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  const tokens = [];
  const kws = 'def|class|return|import|from|if|else|elif|try|except|for|in|while|with|as|raise|not|and|or|is|None|True|False|break|continue|pass|lambda|yield|async|await';
  const strPat = '"(?:[^"\\\\]|\\\\.)*"|' + "'(?:[^'\\\\]|\\\\.)*'";
  const re = new RegExp('(' + strPat + ')|(#.*$)|(\\b(?:' + kws + ')\\b)|(\\b\\d+\\.?\\d*\\b)', 'gm');
  let last = 0, m;
  while((m = re.exec(escaped)) !== null) {
    if(m.index > last) tokens.push(escaped.slice(last, m.index));
    if(m[1]) tokens.push('<span class="str">'+m[1]+'</span>');
    else if(m[2]) tokens.push('<span class="cm">'+m[2]+'</span>');
    else if(m[3]) tokens.push('<span class="kw">'+m[3]+'</span>');
    else if(m[4]) tokens.push('<span class="num">'+m[4]+'</span>');
    last = re.lastIndex;
  }
  if(last < escaped.length) tokens.push(escaped.slice(last));
  document.getElementById('source-code').innerHTML = tokens.join('');
  document.getElementById('source-modal').classList.add('open');
}

function closeModal() {
  document.getElementById('source-modal').classList.remove('open');
}

// ══════════════════════════════════════════
// CONTRACT EDITOR
// ══════════════════════════════════════════
let _ceData = null;

async function ceEditorOpen(name) {
  const resp = await fetchJSON('/api/contract/' + name + '/raw');
  if(!resp || resp.error) { showToast('Error', resp?.error || 'Failed to load', true); return; }
  _ceData = resp;
  document.getElementById('ce-title').textContent = resp.parsed.type + ' : ' + name;
  document.getElementById('ce-status').textContent = '';
  ceRender(resp.parsed);
  document.getElementById('ce-overlay').classList.add('open');
  document.getElementById('contract-editor').classList.add('open');
}

function ceEditorClose() {
  document.getElementById('contract-editor').classList.remove('open');
  document.getElementById('ce-overlay').classList.remove('open');
  _ceData = null;
}

function ceRender(p) {
  const body = document.getElementById('ce-body');
  let html = '';
  if(p.type === 'gene') {
    html = ceRenderGene(p);
  } else if(p.type === 'pathway') {
    html = ceRenderPathway(p);
  }
  body.innerHTML = html;
  requestAnimationFrame(ceAutosize);
}

function _esc(s) { return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }
function _inp(cls, name, val, extra) { extra=extra||''; return `<input class="ce-input ${cls}" data-field="${name}" value="${_esc(val)}" ${extra}>`; }
function _sel(cls, name, val, opts) {
  let h = `<select class="ce-select ${cls}" data-field="${name}">`;
  opts.forEach(o => { h += `<option value="${o}"${o===val?' selected':''}>${o}</option>`; });
  return h + '</select>';
}

function ceRenderGene(p) {
  let h = '';
  // Header: gene name for domain
  h += `<div class="ce-line"><span class="ce-kw">gene</span> ${_inp('ce-name','name',p.name)} <span class="ce-kw">for</span> ${_inp('ce-domain','domain',p.domain||'')}</div>`;
  // is
  h += `<div class="ce-line">  ${_sel('ce-family','family',p.family,['configuration','diagnostic'])}</div>`;
  // risk
  h += `<div class="ce-line">  <span class="ce-kw">risk</span> ${_sel('ce-risk','risk',p.risk,['none','low','medium','high','critical'])}</div>`;
  h += '<div class="ce-line"></div>';
  // does
  h += `<div class="ce-line">  <span class="ce-verb">does:</span></div>`;
  h += `<div class="ce-line">    <textarea class="ce-prose" data-field="does" oninput="ceAutosize()">${_esc(p.does||'')}</textarea></div>`;
  h += '<div class="ce-line"></div>';
  // takes
  if(p.takes && p.takes.length > 0) {
    h += `<div class="ce-line">  <span class="ce-verb">takes:</span></div>`;
    p.takes.forEach((f,i) => {
      h += '<div class="ce-line">    ';
      h += _inp('ce-name','takes.'+i+'.name',f.name,'style="width:'+Math.max(80,f.name.length*8)+'px"');
      h += '  ' + _inp('ce-type','takes.'+i+'.type',f.type);
      h += '  <span class="ce-punct">&quot;</span>' + _inp('ce-desc','takes.'+i+'.description',f.description,'style="width:'+Math.max(120,(f.description||'').length*7)+'px"') + '<span class="ce-punct">&quot;</span>';
      if(f.default !== null && f.default !== undefined) {
        h += '  <span class="ce-punct">default=</span><span class="ce-punct">&quot;</span>' + _inp('ce-default','takes.'+i+'.default',f.default,'style="width:'+Math.max(100,(f.default||'').length*7)+'px"') + '<span class="ce-punct">&quot;</span>';
      }
      h += `<span class="ce-remove-btn" onclick="ceRemoveField('takes',${i})">&#x2715;</span>`;
      h += '</div>';
    });
    h += `<div class="ce-line">    <span class="ce-add-btn" onclick="ceAddField('takes')">+ field</span></div>`;
    h += '<div class="ce-line"></div>';
  }
  // gives
  if(p.gives && p.gives.length > 0) {
    h += `<div class="ce-line">  <span class="ce-verb">gives:</span></div>`;
    p.gives.forEach((f,i) => {
      h += '<div class="ce-line">    ';
      h += _inp('ce-name','gives.'+i+'.name',f.name,'style="width:'+Math.max(80,f.name.length*8)+'px"');
      h += '  ' + _inp('ce-type','gives.'+i+'.type',f.type);
      if(f.optional) h += '<span class="ce-punct">?</span>';
      h += '  <span class="ce-punct">&quot;</span>' + _inp('ce-desc','gives.'+i+'.description',f.description,'style="width:'+Math.max(120,(f.description||'').length*7)+'px"') + '<span class="ce-punct">&quot;</span>';
      h += `<span class="ce-remove-btn" onclick="ceRemoveField('gives',${i})">&#x2715;</span>`;
      h += '</div>';
    });
    h += `<div class="ce-line">    <span class="ce-add-btn" onclick="ceAddField('gives')">+ field</span></div>`;
    h += '<div class="ce-line"></div>';
  }
  // connects
  if(p.connects && p.connects.length > 0) {
    h += `<div class="ce-line">  <span class="ce-verb">connects:</span></div>`;
    p.connects.forEach((c,i) => {
      h += '<div class="ce-line">    ';
      h += _inp('ce-name','connects.'+i+'.param',c.param,'style="width:'+Math.max(80,c.param.length*8)+'px"');
      h += '  ' + _inp('ce-interface','connects.'+i+'.interface',c.interface);
      h += '  <span class="ce-punct">&quot;</span>' + _inp('ce-desc','connects.'+i+'.description',c.description,'style="width:'+Math.max(120,(c.description||'').length*7)+'px"') + '<span class="ce-punct">&quot;</span>';
      h += `<span class="ce-remove-btn" onclick="ceRemoveField('connects',${i})">&#x2715;</span>`;
      h += '</div>';
    });
    h += `<div class="ce-line">    <span class="ce-add-btn" onclick="ceAddField('connects')">+ interface</span></div>`;
    h += '<div class="ce-line"></div>';
  }
  // before
  h += ceRenderBulletSection(p, 'before', 'before');
  // after
  h += ceRenderBulletSection(p, 'after', 'after');
  // fails when
  h += ceRenderBulletSection(p, 'fails_when', 'fails when');
  // unhealthy when
  h += ceRenderBulletSection(p, 'unhealthy_when', 'unhealthy when');
  // verify
  if(p.verify && p.verify.length > 0) {
    h += `<div class="ce-line">  <span class="ce-verb">verify:</span></div>`;
    p.verify.forEach((v,i) => {
      let params = Object.entries(v.params||{}).map(([k,val]) => k+'='+val).join(' ');
      h += '<div class="ce-line">    ';
      h += _inp('ce-value','verify.'+i+'.locus',v.locus,'style="width:'+Math.max(100,v.locus.length*8)+'px"');
      h += ' ' + _inp('ce-value','verify.'+i+'.params',params,'style="width:'+Math.max(150,params.length*7)+'px"');
      h += '</div>';
    });
    if(p.verify_within) {
      h += `<div class="ce-line">    <span class="ce-kw">within</span> ${_inp('ce-value','verify_within',p.verify_within,'style="width:40px"')}</div>`;
    }
    h += '<div class="ce-line"></div>';
  }
  // feeds
  if(p.feeds && p.feeds.length > 0) {
    h += `<div class="ce-line">  <span class="ce-verb">feeds:</span></div>`;
    p.feeds.forEach((f,i) => {
      h += '<div class="ce-line">    ';
      h += _inp('ce-name','feeds.'+i+'.target_locus',f.target_locus,'style="width:'+Math.max(100,f.target_locus.length*8)+'px"');
      h += ' ' + _sel('ce-risk','feeds.'+i+'.timescale',f.timescale,['immediate','convergence','resilience']);
      h += `<span class="ce-remove-btn" onclick="ceRemoveField('feeds',${i})">&#x2715;</span>`;
      h += '</div>';
    });
    h += `<div class="ce-line">    <span class="ce-add-btn" onclick="ceAddField('feeds')">+ feed</span></div>`;
    h += '<div class="ce-line"></div>';
  }
  // Add missing sections
  h += '<div class="ce-add-section">';
  if(!p.connects || p.connects.length === 0) h += '<span class="ce-add-btn" onclick="ceAddSection(\'connects\')">+ connects</span> ';
  if(!p.before || p.before.length === 0) h += '<span class="ce-add-btn" onclick="ceAddSection(\'before\')">+ before</span> ';
  if(!p.after || p.after.length === 0) h += '<span class="ce-add-btn" onclick="ceAddSection(\'after\')">+ after</span> ';
  if(!p.fails_when || p.fails_when.length === 0) h += '<span class="ce-add-btn" onclick="ceAddSection(\'fails_when\')">+ fails when</span> ';
  if(!p.feeds || p.feeds.length === 0) h += '<span class="ce-add-btn" onclick="ceAddSection(\'feeds\')">+ feeds</span> ';
  if(!p.verify || p.verify.length === 0) h += '<span class="ce-add-btn" onclick="ceAddSection(\'verify\')">+ verify</span> ';
  h += '</div>';
  return h;
}

function ceRenderPathway(p) {
  let h = '';
  h += `<div class="ce-line"><span class="ce-kw">pathway</span> ${_inp('ce-name','name',p.name)} <span class="ce-kw">for</span> ${_inp('ce-domain','domain',p.domain||'')}</div>`;
  h += `<div class="ce-line">  <span class="ce-kw">risk</span> ${_sel('ce-risk','risk',p.risk,['none','low','medium','high','critical'])}</div>`;
  h += '<div class="ce-line"></div>';
  h += `<div class="ce-line">  <span class="ce-verb">does:</span></div>`;
  h += `<div class="ce-line">    <textarea class="ce-prose" data-field="does" oninput="ceAutosize()">${_esc(p.does||'')}</textarea></div>`;
  h += '<div class="ce-line"></div>';
  // takes with defaults
  if(p.takes && p.takes.length > 0) {
    h += `<div class="ce-line">  <span class="ce-verb">takes:</span></div>`;
    p.takes.forEach((f,i) => {
      h += '<div class="ce-line">    ';
      h += _inp('ce-name','takes.'+i+'.name',f.name,'style="width:'+Math.max(80,f.name.length*8)+'px"');
      h += '  ' + _inp('ce-type','takes.'+i+'.type',f.type);
      h += '  <span class="ce-punct">&quot;</span>' + _inp('ce-desc','takes.'+i+'.description',f.description,'style="width:'+Math.max(120,(f.description||'').length*7)+'px"') + '<span class="ce-punct">&quot;</span>';
      if(f.default !== null && f.default !== undefined) {
        h += '  <span class="ce-punct">default=</span><span class="ce-punct">&quot;</span>' + _inp('ce-default','takes.'+i+'.default',f.default,'style="width:'+Math.max(100,(f.default||'').length*7)+'px"') + '<span class="ce-punct">&quot;</span>';
      } else {
        h += `  <span class="ce-add-btn" onclick="ceAddDefault(${i})" style="font-size:9px">+default</span>`;
      }
      h += `<span class="ce-remove-btn" onclick="ceRemoveField('takes',${i})">&#x2715;</span>`;
      h += '</div>';
    });
    h += `<div class="ce-line">    <span class="ce-add-btn" onclick="ceAddField('takes')">+ field</span></div>`;
    h += '<div class="ce-line"></div>';
  }
  if(p.steps && p.steps.length > 0) {
    h += `<div class="ce-line">  <span class="ce-verb">steps:</span></div>`;
    p.steps.forEach((s,i) => {
      h += `<div class="ce-line">    <span class="ce-idx">${s.index}.</span> ${_inp('ce-name','steps.'+i+'.locus',s.locus,'style="width:'+Math.max(100,s.locus.length*8)+'px"')}`;
      h += `<span class="ce-remove-btn" onclick="ceRemoveStep(${i})">&#x2715;</span></div>`;
      if(s.params) {
        Object.entries(s.params).forEach(([k,v]) => {
          h += `<div class="ce-line">         ${_inp('ce-value','steps.'+i+'.params.'+k+'._key',k,'style="width:'+Math.max(60,k.length*8)+'px"')} <span class="ce-punct">=</span> ${_inp('ce-value','steps.'+i+'.params.'+k+'._val',v,'style="width:'+Math.max(60,v.length*8)+'px"')}</div>`;
        });
      }
      h += `<div class="ce-line">         <span class="ce-add-btn" onclick="ceAddStepParam(${i})">+ param</span></div>`;
    });
    h += `<div class="ce-line">    <span class="ce-add-btn" onclick="ceAddStep()">+ step</span></div>`;
    h += '<div class="ce-line"></div>';
  } else {
    h += `<div class="ce-line">  <span class="ce-verb">steps:</span></div>`;
    h += `<div class="ce-line">    <span class="ce-add-btn" onclick="ceAddStep()">+ step</span></div>`;
    h += '<div class="ce-line"></div>';
  }
  // on failure
  if(p.on_failure) {
    h += `<div class="ce-line">  <span class="ce-verb">on failure:</span></div>`;
    h += `<div class="ce-line">    ${_inp('ce-value','on_failure',p.on_failure,'style="width:200px"')}</div>`;
  }
  return h;
}

function ceRenderBulletSection(p, field, label) {
  const items = p[field];
  if(!items || items.length === 0) return '';
  let h = `<div class="ce-line">  <span class="ce-verb">${_esc(label)}:</span></div>`;
  items.forEach((item,i) => {
    h += '<div class="ce-line">    <span class="ce-punct">-</span> ';
    h += _inp('ce-value',field+'.'+i,item,'style="width:'+Math.max(200,item.length*7)+'px"');
    h += `<span class="ce-remove-btn" onclick="ceRemoveBullet('${field}',${i})">&#x2715;</span>`;
    h += '</div>';
  });
  h += `<div class="ce-line">    <span class="ce-add-btn" onclick="ceAddBullet('${field}')">+ item</span></div>`;
  h += '<div class="ce-line"></div>';
  return h;
}

function ceAutosize() {
  document.querySelectorAll('.ce-prose').forEach(ta => {
    ta.style.height = 'auto';
    ta.style.height = Math.max(ta.scrollHeight, 20) + 'px';
  });
}

// ── Mutations ──
function ceCollectParsed() {
  const p = JSON.parse(JSON.stringify(_ceData.parsed));
  document.querySelectorAll('.ce-input, .ce-select, .ce-prose').forEach(el => {
    const field = el.dataset.field;
    if(!field) return;
    const val = el.value;
    const parts = field.split('.');
    if(parts.length === 1) {
      p[parts[0]] = val;
    } else if(parts.length === 3) {
      const [sec, idx, key] = parts;
      if(!p[sec]) return;
      const i = parseInt(idx);
      if(p[sec][i] !== undefined) p[sec][i][key] = val;
    } else if(parts.length === 5 && parts[3] === '_key') {
      // step param key rename: steps.0.params.old_key._key = new_key
      const [sec, idx, , oldKey] = [parts[0], parseInt(parts[1]), parts[2], parts[4]];
      // handled together with _val
    } else if(parts.length === 5 && parts[3] === '_val') {
      // step param value: steps.0.params.param_name._val = value
      const sec = parts[0], idx = parseInt(parts[1]), paramKey = parts[2];
      if(p[sec] && p[sec][idx] && p[sec][idx].params) {
        const keyEl = document.querySelector(`[data-field="${sec}.${idx}.params.${paramKey}._key"]`);
        const newKey = keyEl ? keyEl.value : paramKey;
        if(newKey !== paramKey) {
          delete p[sec][idx].params[paramKey];
        }
        p[sec][idx].params[newKey] = val;
      }
    }
  });
  return p;
}

function ceReconstructSource(p) {
  let lines = [];
  if(p.type === 'gene') {
    lines.push('gene ' + p.name + (p.domain ? ' for ' + p.domain : ''));
    lines.push('  is ' + p.family);
    lines.push('  risk ' + p.risk);
    lines.push('');
    if(p.does) {
      lines.push('  does:');
      p.does.split('\\n').forEach(l => lines.push('    ' + l));
      lines.push('');
    }
    if(p.takes && p.takes.length) {
      lines.push('  takes:');
      p.takes.forEach(f => {
        let line = '    ' + f.name + '  ' + f.type;
        if(f.optional && f.type && !f.type.endsWith('?')) line = '    ' + f.name + '  ' + f.type + '?';
        if(f.description) line += '  "' + f.description + '"';
        if(f.default !== null && f.default !== undefined && f.default !== '') line += '  default="' + f.default + '"';
        lines.push(line);
      });
      lines.push('');
    }
    if(p.gives && p.gives.length) {
      lines.push('  gives:');
      p.gives.forEach(f => {
        let t = f.type + (f.optional ? '?' : '');
        let line = '    ' + f.name + '  ' + t;
        if(f.description) line += '  "' + f.description + '"';
        lines.push(line);
      });
      lines.push('');
    }
    if(p.connects && p.connects.length) {
      lines.push('  connects:');
      p.connects.forEach(c => {
        let line = '    ' + c.param + '  ' + c.interface;
        if(c.description) line += '  "' + c.description + '"';
        lines.push(line);
      });
      lines.push('');
    }
    ['before','after','fails_when','unhealthy_when'].forEach(sec => {
      if(p[sec] && p[sec].length) {
        const label = sec.replace('_',' ');
        lines.push('  ' + label + ':');
        p[sec].forEach(item => lines.push('    - ' + item));
        lines.push('');
      }
    });
    if(p.verify && p.verify.length) {
      lines.push('  verify:');
      p.verify.forEach(v => {
        let params = Object.entries(v.params||{}).map(([k,val]) => k + '=' + val).join(' ');
        lines.push('    ' + v.locus + (params ? ' ' + params : ''));
      });
      if(p.verify_within) lines.push('    within ' + p.verify_within);
      lines.push('');
    }
    if(p.feeds && p.feeds.length) {
      lines.push('  feeds:');
      p.feeds.forEach(f => lines.push('    ' + f.target_locus + ' ' + f.timescale));
      lines.push('');
    }
  } else if(p.type === 'pathway') {
    lines.push('pathway ' + p.name + (p.domain ? ' for ' + p.domain : ''));
    lines.push('  risk ' + p.risk);
    lines.push('');
    if(p.does) {
      lines.push('  does:');
      p.does.split('\\n').forEach(l => lines.push('    ' + l));
      lines.push('');
    }
    if(p.takes && p.takes.length) {
      lines.push('  takes:');
      p.takes.forEach(f => {
        let line = '    ' + f.name + '  ' + f.type;
        if(f.description) line += '  "' + f.description + '"';
        if(f.default !== null && f.default !== undefined && f.default !== '') line += '  default="' + f.default + '"';
        lines.push(line);
      });
      lines.push('');
    }
    if(p.steps && p.steps.length) {
      lines.push('  steps:');
      p.steps.forEach(s => {
        lines.push('    ' + s.index + '. ' + s.locus);
        if(s.params) {
          Object.entries(s.params).forEach(([k,v]) => {
            lines.push('         ' + k + ' = ' + v);
          });
        }
      });
      lines.push('');
    }
    if(p.on_failure) {
      lines.push('  on failure:');
      lines.push('    ' + p.on_failure);
    }
  }
  return lines.join('\\n') + '\\n';
}

function ceAddField(section) {
  const p = ceCollectParsed();
  if(!p[section]) p[section] = [];
  if(section === 'takes' || section === 'gives') {
    p[section].push({name:'new_field', type:'string', description:'', default:null, optional:false});
  } else if(section === 'connects') {
    p[section].push({param:'param', interface:'https', description:''});
  } else if(section === 'feeds') {
    p[section].push({target_locus:'locus_name', timescale:'convergence'});
  }
  _ceData.parsed = p;
  ceRender(p);
}

function ceRemoveField(section, idx) {
  const p = ceCollectParsed();
  if(p[section]) { p[section].splice(idx, 1); }
  _ceData.parsed = p;
  ceRender(p);
}

function ceAddBullet(field) {
  const p = ceCollectParsed();
  if(!p[field]) p[field] = [];
  p[field].push('new condition');
  _ceData.parsed = p;
  ceRender(p);
}

function ceRemoveBullet(field, idx) {
  const p = ceCollectParsed();
  if(p[field]) { p[field].splice(idx, 1); }
  _ceData.parsed = p;
  ceRender(p);
}

function ceAddSection(section) {
  const p = ceCollectParsed();
  if(section === 'connects') p.connects = [{param:'param', interface:'https', description:''}];
  else if(section === 'feeds') p.feeds = [{target_locus:'locus_name', timescale:'convergence'}];
  else if(section === 'verify') p.verify = [{locus:'check_name', params:{}}];
  else p[section] = ['condition'];
  _ceData.parsed = p;
  ceRender(p);
}

function ceAddStep() {
  const p = ceCollectParsed();
  if(!p.steps) p.steps = [];
  const idx = p.steps.length + 1;
  p.steps.push({index: idx, locus: 'gene_name', params: {}});
  _ceData.parsed = p;
  ceRender(p);
}

function ceRemoveStep(idx) {
  const p = ceCollectParsed();
  if(p.steps) {
    p.steps.splice(idx, 1);
    p.steps.forEach((s, i) => { s.index = i + 1; });
  }
  _ceData.parsed = p;
  ceRender(p);
}

function ceAddStepParam(stepIdx) {
  const p = ceCollectParsed();
  if(p.steps && p.steps[stepIdx]) {
    if(!p.steps[stepIdx].params) p.steps[stepIdx].params = {};
    p.steps[stepIdx].params['param'] = '{value}';
  }
  _ceData.parsed = p;
  ceRender(p);
}

function ceAddDefault(idx) {
  const p = ceCollectParsed();
  if(p.takes && p.takes[idx]) { p.takes[idx].default = ''; }
  _ceData.parsed = p;
  ceRender(p);
}

async function ceEditorSave() {
  const p = ceCollectParsed();
  const source = ceReconstructSource(p);
  const statusEl = document.getElementById('ce-status');
  statusEl.textContent = 'Saving...';

  try {
    const resp = await fetch('/api/contract/' + _ceData.parsed.name, {
      method: 'PUT',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({source}),
    });
    const data = await resp.json();
    if(data.error) {
      statusEl.textContent = 'Error: ' + data.error;
      statusEl.style.color = 'var(--danger)';
    } else {
      statusEl.textContent = 'Saved';
      statusEl.style.color = 'var(--success)';
      _ceData.parsed.name = data.name;
      setTimeout(() => { statusEl.textContent = ''; statusEl.style.color = ''; }, 2000);
      refreshData();
    }
  } catch(e) {
    statusEl.textContent = 'Error: ' + e.message;
    statusEl.style.color = 'var(--danger)';
  }
}

// ── URL Hash State ──
function updateHash() {
  const parts = [`view=${state.perspective}`];
  if(state.selectedEntity) parts.push(`entity=${state.selectedEntity.type}:${state.selectedEntity.name}`);
  location.hash = parts.join('&');
}

function loadHash() {
  const h = location.hash.slice(1);
  if(!h) return;
  const params = {};
  h.split('&').forEach(p => { const [k,v]=p.split('='); params[k]=v; });
  if(params.view) {
    state.perspective = params.view;
    document.querySelectorAll('.persp-btn').forEach(b => {
      b.classList.toggle('active', b.dataset.view===state.perspective);
    });
    document.querySelectorAll('.view').forEach(v => {
      v.classList.toggle('active', v.id==='view-'+state.perspective);
    });
  }
  if(params.entity) {
    const [type,name] = params.entity.split(':');
    state.selectedEntity = {type, name};
  }
}

// ══════════════════════════════════════════
// COMMAND BAR
// ══════════════════════════════════════════

async function initCommandBar() {
  const kernelData = await fetchJSON('/api/kernels');
  const kernelSel = document.getElementById('cmd-kernel');
  kernelSel.innerHTML = '';
  if(kernelData?.kernels) {
    kernelData.kernels.forEach(k => {
      const opt = document.createElement('option');
      opt.value = k.name; opt.textContent = k.name;
      if(k.name === 'data-production') opt.selected = true;
      kernelSel.appendChild(opt);
    });
  }
}

function updateCommandBarSelectors() {
  const pwSel = document.getElementById('cmd-pathway');
  const curPw = pwSel.value;
  pwSel.innerHTML = '';
  state.pathways.forEach(p => {
    const opt = document.createElement('option');
    opt.value = p.name; opt.textContent = p.name;
    if(p.name === curPw) opt.selected = true;
    pwSel.appendChild(opt);
  });

  const locusSel = document.getElementById('cmd-locus');
  const curLocus = locusSel.value;
  locusSel.innerHTML = '';
  state.loci.forEach(l => {
    const opt = document.createElement('option');
    opt.value = l.name; opt.textContent = l.name;
    if(l.name === curLocus) opt.selected = true;
    locusSel.appendChild(opt);
  });
}

function showToast(title, body, isError) {
  document.getElementById('toast-title').textContent = title;
  const el = document.getElementById('toast-body');
  el.className = isError ? 'toast-error' : 'toast-success';
  el.style.whiteSpace = 'pre-wrap';
  el.textContent = typeof body === 'string' ? body : JSON.stringify(body, null, 2);
  document.getElementById('output-toast').classList.add('show');
}
function closeToast() { document.getElementById('output-toast').classList.remove('show'); }

async function pollJob(jobId, title) {
  const poll = async () => {
    const data = await fetchJSON('/api/job/' + jobId);
    if(!data) { showToast(title + ' Error', 'Lost connection', true); return; }
    if(data.status === 'running') {
      const prog = data.progress ? ' (' + data.progress + '/' + data.total + ')' : '';
      document.getElementById('toast-title').textContent = title + prog;
      setTimeout(poll, 500);
      return;
    }
    if(data.success) {
      let msg = JSON.stringify(data);
      if(data.type === 'init' && data.seeded) {
        const lines = data.seeded.map(s =>
          s.locus + ' -> ' + (s.sha || 'FAILED') + ' (' + s.source + ')' + (s.error ? ' ' + s.error : '')
        );
        lines.push('Pathways: ' + data.pathways);
        msg = lines.join('\\n');
      } else if(data.type === 'run') {
        if(data.iterations > 1) {
          msg = data.pathway + ': ' + data.successes + '/' + data.iterations + ' succeeded';
        } else if(data.runs && data.runs[0]) {
          const steps = data.runs[0].steps || [];
          msg = steps.map(s => 'Step ' + s.step + ' [' + s.sha + ']: ' + JSON.stringify(s.output).substring(0,100)).join('\\n');
        }
      } else if(data.type === 'compete' && data.results) {
        const lines = data.results.map(r => {
          let line = r.sha + ' [' + r.state + '] ' + r.passed + '/' + r.total + ' fitness=' + r.fitness;
          if(r.passed === 0 && r.last_error) line += '\\n  error: ' + r.last_error;
          return line;
        });
        if(data.promoted) lines.push('PROMOTED: ' + data.promoted + ' is the new dominant');
        msg = lines.join('\\n');
      } else if(data.type === 'generate' && data.registered) {
        msg = 'Registered: ' + data.registered.join(', ');
      }
      showToast(title, msg, false);
    }
    else { showToast(title + ' Failed', data.error || JSON.stringify(data), true); }
    refreshData();
  };
  poll();
}

async function cmdInit() {
  const kernel = document.getElementById('cmd-kernel').value;
  const engine = document.getElementById('cmd-engine').value;
  showToast('Initializing genome...', 'Seeding all loci from contracts via ' + engine, false);
  try {
    const resp = await fetch('/api/init', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({kernel, mutation_engine: engine}),
    });
    const data = await resp.json();
    if(data.error) { showToast('Init Failed', data.error, true); }
    else { pollJob(data.job_id, 'Init Genome'); }
  } catch(e) { showToast('Init Error', e.message, true); }
}

async function cmdRun() {
  const pathway = document.getElementById('cmd-pathway').value;
  const kernel = document.getElementById('cmd-kernel').value;
  const iterations = parseInt(document.getElementById('cmd-iterations').value) || 1;
  if(!pathway) { showToast('Error', 'Select a pathway', true); return; }

  showToast('Running...', pathway + ' x' + iterations + ' with ' + kernel, false);
  try {
    const resp = await fetch('/api/run', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({pathway, kernel, iterations}),
    });
    const data = await resp.json();
    if(data.error) { showToast('Run Failed', data.error, true); }
    else { pollJob(data.job_id, 'Run: ' + pathway + ' x' + iterations); }
  } catch(e) { showToast('Run Error', e.message, true); }
}

async function cmdGenerate() {
  const locus = document.getElementById('cmd-locus').value;
  const kernel = document.getElementById('cmd-kernel').value;
  const engine = document.getElementById('cmd-engine').value;
  const count = parseInt(document.getElementById('cmd-count').value) || 1;
  if(!locus) { showToast('Error', 'Select a locus', true); return; }

  showToast('Generating...', count + ' variant(s) for ' + locus + ' via ' + engine, false);
  try {
    const resp = await fetch('/api/generate', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({locus, kernel, mutation_engine: engine, count}),
    });
    const data = await resp.json();
    if(data.error) { showToast('Generate Failed', data.error, true); }
    else { pollJob(data.job_id, 'Generate: ' + locus); }
  } catch(e) { showToast('Generate Error', e.message, true); }
}

async function cmdCompete() {
  const locus = document.getElementById('cmd-locus').value;
  const kernel = document.getElementById('cmd-kernel').value;
  const rounds = parseInt(document.getElementById('cmd-rounds').value) || 10;
  if(!locus) { showToast('Error', 'Select a locus', true); return; }

  showToast('Competing...', locus + ' x' + rounds + ' rounds', false);
  try {
    const resp = await fetch('/api/compete', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({locus, kernel, rounds}),
    });
    const data = await resp.json();
    if(data.error) { showToast('Compete Failed', data.error, true); }
    else { pollJob(data.job_id, 'Competition: ' + locus); }
  } catch(e) { showToast('Compete Error', e.message, true); }
}

async function competeLocus(locus) {
  const kernel = document.getElementById('cmd-kernel').value || 'data-mock';
  const rounds = parseInt(document.getElementById('cmd-rounds')?.value) || 10;
  showToast('Competing...', locus + ' (' + rounds + ' rounds, all alleles)', false);
  try {
    const resp = await fetch('/api/compete', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({locus, kernel, rounds}),
    });
    const data = await resp.json();
    if(data.error) { showToast('Compete Failed', data.error, true); }
    else { pollJob(data.job_id, 'Competition: ' + locus); }
  } catch(e) { showToast('Compete Error', e.message, true); }
}

async function generateForLocus(locus) {
  const kernel = document.getElementById('cmd-kernel').value || 'data-mock';
  const engine = document.getElementById('cmd-engine').value || 'deepseek';
  showToast('Generating...', '1 variant for ' + locus + ' via ' + engine, false);
  try {
    const resp = await fetch('/api/generate', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({locus, kernel, mutation_engine: engine, count: 1}),
    });
    const data = await resp.json();
    if(data.error) { showToast('Generate Failed', data.error, true); }
    else { pollJob(data.job_id, 'Generate: ' + locus); }
  } catch(e) { showToast('Generate Error', e.message, true); }
}

function openNewPathwayDialog() {
  document.getElementById('intent-text').value = '';
  document.getElementById('intent-status').textContent = '';
  document.getElementById('intent-overlay').classList.add('open');
  document.getElementById('intent-dialog').classList.add('open');
  document.getElementById('intent-text').focus();
}

function closeNewPathwayDialog() {
  document.getElementById('intent-overlay').classList.remove('open');
  document.getElementById('intent-dialog').classList.remove('open');
}

async function submitNewPathway() {
  const intent = document.getElementById('intent-text').value.trim();
  if(!intent) return;
  const engine = document.getElementById('cmd-engine').value;
  const kernel = document.getElementById('cmd-kernel').value;
  const statusEl = document.getElementById('intent-status');
  statusEl.textContent = 'Generating pathway via ' + engine + '...';
  statusEl.style.color = 'var(--accent)';

  try {
    const resp = await fetch('/api/pathway/draft', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({intent, mutation_engine: engine, kernel}),
    });
    const data = await resp.json();
    if(data.error) {
      statusEl.textContent = data.error;
      statusEl.style.color = 'var(--danger)';
      return;
    }
    // Poll the job
    const pollDraft = async () => {
      const job = await fetchJSON('/api/job/' + data.job_id);
      if(!job) { statusEl.textContent = 'Lost connection'; statusEl.style.color='var(--danger)'; return; }
      if(job.status === 'running') { setTimeout(pollDraft, 500); return; }
      if(job.success) {
        closeNewPathwayDialog();
        _ceData = {parsed: job.parsed, source: job.source};
        document.getElementById('ce-title').textContent = 'New Pathway: ' + job.parsed.name;
        document.getElementById('ce-status').textContent = '';
        ceRender(job.parsed);
        document.getElementById('ce-overlay').classList.add('open');
        document.getElementById('contract-editor').classList.add('open');
        showToast('Pathway Draft', 'Review and save the generated pathway', false);
      } else {
        statusEl.textContent = job.error || 'Generation failed';
        statusEl.style.color = 'var(--danger)';
      }
    };
    pollDraft();
  } catch(e) {
    statusEl.textContent = e.message;
    statusEl.style.color = 'var(--danger)';
  }
}

async function deleteAllele(sha, locus) {
  if(!confirm('Delete allele ' + sha + ' from ' + locus + '?')) return;
  try {
    const resp = await fetch('/api/allele/' + sha, {method:'DELETE'});
    const data = await resp.json();
    if(data.ok) { showToast('Deleted', sha + ' removed from ' + locus, false); refreshData(); }
    else { showToast('Delete Failed', data.error, true); }
  } catch(e) { showToast('Delete Error', e.message, true); }
}

async function clearLocusAlleles(locus) {
  if(!confirm('Delete ALL alleles from ' + locus + '? This cannot be undone.')) return;
  try {
    const resp = await fetch('/api/locus/' + locus + '/alleles', {method:'DELETE'});
    const data = await resp.json();
    if(data.ok) { showToast('Cleared', data.deleted + ' allele(s) removed from ' + locus, false); refreshData(); }
    else { showToast('Clear Failed', data.error, true); }
  } catch(e) { showToast('Clear Error', e.message, true); }
}

async function resetGenome() {
  if(!confirm('DELETE ALL ALLELES from ALL loci? This is a full genome reset and cannot be undone.')) return;
  try {
    const resp = await fetch('/api/alleles', {method:'DELETE'});
    const data = await resp.json();
    if(data.ok) { showToast('Genome Reset', data.deleted + ' allele(s) deleted', false); refreshData(); }
    else { showToast('Reset Failed', data.error, true); }
  } catch(e) { showToast('Reset Error', e.message, true); }
}

async function daemonToggle() {
  const data = await fetchJSON('/api/daemon/status');
  if(data && data.running) {
    await fetch('/api/daemon/stop', {method:'POST', headers:{'Content-Type':'application/json'}, body:'{}'});
    showToast('Daemon', 'Stopped after ' + data.tick_count + ' ticks', false);
  } else {
    const kernel = document.getElementById('cmd-kernel').value;
    const engine = document.getElementById('cmd-engine').value;
    const tick_interval = parseFloat(document.getElementById('daemon-interval').value) || 30;
    await fetch('/api/daemon/start', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({kernel, mutation_engine: engine, tick_interval}),
    });
    showToast('Daemon', 'Started (tick every ' + tick_interval + 's, engine: ' + engine + ')', false);
  }
  updateDaemonUI();
}

async function daemonPause() {
  const resp = await fetch('/api/daemon/pause', {method:'POST', headers:{'Content-Type':'application/json'}, body:'{}'});
  const data = await resp.json();
  if(data.ok) showToast('Daemon', data.paused ? 'Paused' : 'Resumed', false);
  updateDaemonUI();
}

async function updateDaemonUI() {
  const data = await fetchJSON('/api/daemon/status');
  if(!data) return;
  const dot = document.getElementById('daemon-dot');
  const label = document.getElementById('daemon-label');
  const tick = document.getElementById('daemon-tick');
  const toggle = document.getElementById('btn-daemon-toggle');
  const pause = document.getElementById('btn-daemon-pause');

  if(data.running) {
    dot.className = data.paused ? 'daemon-dot paused' : 'daemon-dot on';
    label.textContent = data.paused ? 'paused' : 'evolving';
    tick.textContent = '#' + data.tick_count;
    toggle.textContent = 'Stop';
    toggle.className = 'btn-daemon-start running';
    pause.disabled = false;
    pause.textContent = data.paused ? 'Resume' : 'Pause';
  } else {
    dot.className = 'daemon-dot';
    label.textContent = 'stopped';
    tick.textContent = data.tick_count > 0 ? '(ran ' + data.tick_count + ')' : '';
    toggle.textContent = 'Start';
    toggle.className = 'btn-daemon-start';
    pause.disabled = true;
    pause.textContent = 'Pause';
  }
  if(data.last_tick_error) {
    label.textContent += ' \u26a0';
    label.title = data.last_tick_error;
  } else {
    label.title = '';
  }
}

const _origRefreshData = refreshData;
refreshData = async function() {
  await _origRefreshData();
  updateCommandBarSelectors();
  updateDaemonUI();
};

// ── Init ──
loadHash();
initCommandBar();
refreshData();
connectSSE();
setInterval(refreshData, 5000);
</script>
</body></html>"""


@app.get("/metrics")
def prometheus_metrics():
    """Export Prometheus metrics (in-process collector or disk snapshot)."""
    from fastapi.responses import PlainTextResponse
    collector = _metrics_collector
    if collector is None:
        # Try loading the snapshot written by the daemon process
        from sg.metrics import MetricsCollector
        snapshot_path = _project_root / ".sg" / "metrics.json"
        collector = MetricsCollector.load(snapshot_path)
    if collector is None:
        return PlainTextResponse(
            "# No metrics available (daemon not running?)\n",
            media_type="text/plain",
        )
    return PlainTextResponse(
        collector.export(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@app.get("/", response_class=HTMLResponse)
def dashboard_html():
    return _DASHBOARD_HTML


def run_dashboard(
    root: Path,
    host: str = "127.0.0.1",
    port: int = 8420,
    metrics_collector=None,
) -> None:
    """Start the dashboard server."""
    global _project_root, _metrics_collector
    _project_root = root
    _metrics_collector = metrics_collector
    import signal
    import uvicorn

    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning"))

    def _handle_exit(sig, frame):
        print("\nShutting down dashboard...")
        server.should_exit = True

    signal.signal(signal.SIGINT, _handle_exit)
    signal.signal(signal.SIGTERM, _handle_exit)

    logger.info("Starting dashboard at http://%s:%d (Ctrl+C to stop)", host, port)
    print(f"Dashboard running at http://{host}:{port}  (Ctrl+C to stop)")
    server.run()
