"""Web dashboard — REST API + single-page HTML frontend.

Serves genome state via JSON endpoints and a lightweight HTML dashboard.
Requires: pip install 'sg[dashboard]' (fastapi + uvicorn).
"""
from __future__ import annotations

import json
import os
import time
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


def _load_state():
    """Load all state from disk (fresh on each request)."""
    root = _project_root
    contract_store = ContractStore.open(root / "contracts")
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
    """SSE stream — yields update events when files change."""
    root = _project_root

    async def event_stream():
        last_mtime = 0.0
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
            import asyncio
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


# ── Action endpoints (control plane) ──────────────────────────────────

import asyncio
import uuid as _uuid
from collections import OrderedDict

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


@app.post("/api/run")
async def api_run(request: Request):
    """Run a pathway in background, return job ID for polling."""
    data = await request.json()
    pathway_name = data.get("pathway")
    input_json = json.dumps(data.get("input", {}))
    kernel_name = data.get("kernel", "data-mock")

    if not pathway_name:
        return JSONResponse({"error": "pathway is required"}, status_code=400)

    job_id = _uuid.uuid4().hex[:12]
    _jobs[job_id] = {"status": "running", "type": "run", "pathway": pathway_name}
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

            outputs = orch.run_pathway(pathway_name, input_json)
            orch.verify_scheduler.wait()
            orch.save_state()

            steps = []
            for i, (out, sha) in enumerate(outputs):
                try:
                    parsed = json.loads(out)
                except Exception:
                    parsed = out
                steps.append({"step": i + 1, "sha": sha[:12], "output": parsed})

            _jobs[job_id] = {"status": "done", "type": "run", "success": True,
                             "pathway": pathway_name, "steps": steps}
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
    """Run allele competition trials in background."""
    data = await request.json()
    locus = data.get("locus")
    input_json = json.dumps(data.get("input", {}))
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


@app.get("/api/daemon/status")
def api_daemon_status():
    """Return daemon state."""
    return {
        "running": _daemon_ref is not None and _daemon_ref.get("running", False),
        "tick_count": _daemon_ref.get("tick_count", 0) if _daemon_ref else 0,
        "kernel": _daemon_ref.get("kernel", "") if _daemon_ref else "",
        "mutation_engine": _daemon_ref.get("mutation_engine", "") if _daemon_ref else "",
    }

_daemon_ref: dict | None = None


# Pipeline endpoints

@app.get("/api/pipelines")
def api_pipelines():
    """List all defined pipelines with source/sink/through info."""
    cs, _, _, _, _, _, _ = _load_state()
    from sg.pipeline import build_pipeline_input
    result = []
    for name in cs.known_pipelines():
        p = cs.get_pipeline(name)
        if p is None:
            continue
        constructed_input = build_pipeline_input(p)
        result.append({
            "name": p.name,
            "domain": p.domain,
            "does": p.does,
            "source_type": p.source.source_type.value if p.source else None,
            "source_props": p.source.properties if p.source else {},
            "sink_type": p.sink.sink_type.value if p.sink else None,
            "sink_props": p.sink.properties if p.sink else {},
            "through": p.through,
            "bind": p.bind,
            "schedule": p.schedule,
            "constructed_input": constructed_input,
        })
    return {"pipelines": result}


@app.get("/api/pipeline/{name}")
def api_pipeline_detail(name: str):
    """Get full details for a single pipeline."""
    cs, _, _, _, _, _, _ = _load_state()
    from sg.pipeline import build_pipeline_input
    p = cs.get_pipeline(name)
    if p is None:
        return JSONResponse({"error": f"pipeline '{name}' not found"}, status_code=404)
    return {
        "name": p.name,
        "domain": p.domain,
        "does": p.does,
        "source_type": p.source.source_type.value if p.source else None,
        "source_props": p.source.properties if p.source else {},
        "sink_type": p.sink.sink_type.value if p.sink else None,
        "sink_props": p.sink.properties if p.sink else {},
        "through": p.through,
        "bind": p.bind,
        "schedule": p.schedule,
        "constructed_input": build_pipeline_input(p),
    }


@app.post("/api/pipeline/run")
async def api_pipeline_run(request: Request):
    """Run a pipeline in background — auto-constructs input from source/sink/bind."""
    data = await request.json()
    pipeline_name = data.get("pipeline")
    kernel_name = data.get("kernel", "data-mock")
    input_override = data.get("override")

    if not pipeline_name:
        return JSONResponse({"error": "pipeline is required"}, status_code=400)

    job_id = _uuid.uuid4().hex[:12]
    _jobs[job_id] = {"status": "running", "type": "pipeline", "pipeline": pipeline_name}
    _prune_jobs()

    def _do_pipeline_run():
        root = _project_root
        try:
            from sg.kernel.discovery import load_kernel
            from sg.cli import load_contract_store
            from sg.orchestrator import Orchestrator
            from sg.mutation import MockMutationEngine
            from sg.meta_params import MetaParamTracker
            from sg.pipeline import build_pipeline_input, run_pipeline

            contract_store = load_contract_store(root)
            p = contract_store.get_pipeline(pipeline_name)
            if p is None:
                _jobs[job_id] = {"status": "done", "type": "pipeline", "success": False,
                                 "error": f"pipeline '{pipeline_name}' not found"}
                return

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

            result = run_pipeline(p, orch, input_override)
            try:
                orch.verify_scheduler.wait()
                orch.save_state()
            except Exception:
                pass

            steps = []
            for i, out in enumerate(result.outputs):
                try:
                    parsed = json.loads(out)
                except Exception:
                    parsed = out
                steps.append({"step": i + 1, "output": parsed})

            _jobs[job_id] = {
                "status": "done", "type": "pipeline", "success": result.success,
                "pipeline": pipeline_name, "pathway": result.pathway_name,
                "input": json.loads(result.input_json) if result.input_json else {},
                "steps": steps,
                "error": result.error,
            }
        except Exception as e:
            _jobs[job_id] = {"status": "done", "type": "pipeline", "success": False,
                             "error": str(e)}

    asyncio.get_event_loop().run_in_executor(None, _do_pipeline_run)
    return {"job_id": job_id, "status": "running"}


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
#command-bar input.json-input { flex:1; min-width:200px; max-width:400px; }
#command-bar .sep { width:1px; height:24px; background:var(--border); }
#command-bar button {
  padding:5px 14px; cursor:pointer; border:1px solid var(--border);
  font-weight:bold; transition:all 0.15s; }
#command-bar .btn-run { background:#0f03; color:var(--success); border-color:var(--success); }
#command-bar .btn-run:hover { background:#0f06; }
#command-bar .btn-gen { background:#0ff1; color:var(--accent); border-color:var(--accent); }
#command-bar .btn-gen:hover { background:#0ff3; }
#command-bar .btn-compete { background:#f801; color:var(--warning); border-color:var(--warning); }
#command-bar .btn-compete:hover { background:#f803; }
#command-bar .daemon-status { margin-left:auto; font-size:10px; color:var(--text-dim);
  display:flex; align-items:center; gap:6px; }
#command-bar .daemon-dot { width:8px; height:8px; border-radius:50%; }
#command-bar .daemon-dot.on { background:var(--success); box-shadow:0 0 6px var(--success); }
#command-bar .daemon-dot.off { background:var(--danger); }
#command-bar label { color:var(--text-dim); font-size:10px; text-transform:uppercase; }
.cmd-group { display:flex; align-items:center; gap:6px; }
.cmd-count { width:40px; text-align:center; }
#command-bar .btn-pipeline { background:#06f1; color:#4af; border-color:#4af; }
#command-bar .btn-pipeline:hover { background:#06f3; }

/* Pipeline Info Banner */
#pipeline-info { grid-column:1/-1; background:var(--bg-panel); border-bottom:1px solid var(--border);
  padding:6px 16px; display:flex; align-items:center; gap:16px; font-size:11px; }
.pipeline-flow { display:flex; align-items:center; gap:8px; }
.pipe-node { padding:4px 10px; border-radius:3px; font-weight:bold; }
.pipe-node.source { background:#06f2; color:#4af; border:1px solid #4af5; }
.pipe-node.pathway { background:#0f02; color:var(--success); border:1px solid var(--success); }
.pipe-node.sink { background:#f802; color:var(--warning); border:1px solid var(--warning); }
.pipe-arrow { color:var(--text-dim); font-size:14px; }
.pipe-details { color:var(--text-dim); font-size:10px; flex:1; }

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
</style>
</head><body>

<!-- Command Bar -->
<div id="command-bar">
  <div class="cmd-group">
    <label>Pipeline</label>
    <select id="cmd-pipeline" onchange="onPipelineSelect()">
      <option value="">(manual)</option>
    </select>
  </div>
  <div class="cmd-group">
    <label>Kernel</label>
    <select id="cmd-kernel"></select>
  </div>
  <button class="btn-pipeline" onclick="cmdPipelineRun()" title="Run selected pipeline">Run Pipeline</button>
  <div class="sep"></div>
  <div class="cmd-group">
    <label>Pathway</label>
    <select id="cmd-pathway"></select>
  </div>
  <div class="cmd-group">
    <label>Input</label>
    <input id="cmd-input" class="json-input" type="text" placeholder='{"key": "value"}' value="{}">
  </div>
  <button class="btn-run" onclick="cmdRun()" title="Run pathway (manual input)">Run</button>
  <div class="sep"></div>
  <div class="cmd-group">
    <label>Locus</label>
    <select id="cmd-locus"></select>
  </div>
  <div class="cmd-group">
    <label>Engine</label>
    <select id="cmd-engine">
      <option value="mock">mock</option>
      <option value="deepseek">deepseek</option>
      <option value="claude">claude</option>
      <option value="openai">openai</option>
    </select>
  </div>
  <div class="cmd-group">
    <input id="cmd-count" class="cmd-count" type="number" value="1" min="1" max="10">
  </div>
  <button class="btn-gen" onclick="cmdGenerate()" title="Generate competing alleles">Generate</button>
  <button class="btn-compete" onclick="cmdCompete()" title="Run competition trials">Compete</button>
  <div class="daemon-status">
    <span class="daemon-dot off" id="daemon-dot"></span>
    <span id="daemon-label">daemon stopped</span>
  </div>
</div>

<!-- Pipeline Info Banner -->
<div id="pipeline-info" style="display:none">
  <div class="pipeline-flow">
    <span class="pipe-node source" id="pipe-src">source</span>
    <span class="pipe-arrow">&rarr;</span>
    <span class="pipe-node pathway" id="pipe-through">pathway</span>
    <span class="pipe-arrow">&rarr;</span>
    <span class="pipe-node sink" id="pipe-sink">sink</span>
  </div>
  <div class="pipe-details" id="pipe-details"></div>
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
    es.onmessage = () => refreshData();
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
    </div>
    <div class="allele-stack">
    ${alleles.map((a,i) => `
      ${i>0?'<div class="allele-connector"></div>':''}
      <div class="allele-card ${a.state}">
        <div style="display:flex;justify-content:space-between;align-items:center">
          <span class="sha" onclick="showSource('${a.sha}')">${a.sha}</span>
          <span style="font-size:10px;color:${a.is_dominant?'var(--success)':a.state==='recessive'?'var(--recessive)':'var(--danger)'}">${a.state}</span>
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

let _pipelines = [];

async function initCommandBar() {
  // Populate kernels
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

  // Populate pipelines
  const pipeData = await fetchJSON('/api/pipelines');
  const pipeSel = document.getElementById('cmd-pipeline');
  pipeSel.innerHTML = '<option value="">(manual)</option>';
  if(pipeData?.pipelines) {
    _pipelines = pipeData.pipelines;
    pipeData.pipelines.forEach(p => {
      const opt = document.createElement('option');
      opt.value = p.name;
      opt.textContent = p.name;
      pipeSel.appendChild(opt);
    });
  }
}

function updateCommandBarSelectors() {
  // Populate pathways
  const pwSel = document.getElementById('cmd-pathway');
  const curPw = pwSel.value;
  pwSel.innerHTML = '';
  state.pathways.forEach(p => {
    const opt = document.createElement('option');
    opt.value = p.name; opt.textContent = p.name;
    if(p.name === curPw) opt.selected = true;
    pwSel.appendChild(opt);
  });

  // Populate loci
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
      setTimeout(poll, 500);
      return;
    }
    if(data.success) {
      let msg = data;
      if(data.type === 'compete' && data.results) {
        const lines = data.results.map(r => {
          let line = r.sha + ' [' + r.state + '] ' + r.passed + '/' + r.total + ' fitness=' + r.fitness;
          if(r.passed === 0 && r.last_error) line += '\\n  error: ' + r.last_error;
          return line;
        });
        if(data.promoted) lines.push('PROMOTED: ' + data.promoted + ' is the new dominant');
        msg = lines.join('\\n');
      } else if(data.type === 'pipeline' && data.steps) {
        const lines = ['Pipeline: ' + (data.pipeline || '') + ' via ' + (data.pathway || '')];
        data.steps.forEach(s => {
          const out = typeof s.output === 'object' ? JSON.stringify(s.output) : s.output;
          lines.push('Step ' + s.step + ': ' + (out || '').substring(0, 120));
        });
        msg = lines.join('\\n');
      }
      showToast(title, msg, false);
    }
    else { showToast(title + ' Failed', data.error || data, true); }
    refreshData();
  };
  poll();
}

function onPipelineSelect() {
  const sel = document.getElementById('cmd-pipeline');
  const name = sel.value;
  const infoEl = document.getElementById('pipeline-info');
  const inputEl = document.getElementById('cmd-input');

  if(!name) {
    infoEl.style.display = 'none';
    return;
  }

  const p = _pipelines.find(x => x.name === name);
  if(!p) { infoEl.style.display = 'none'; return; }

  document.getElementById('pipe-src').textContent = (p.source_type || 'none') +
    (p.source_props?.url ? ': ' + p.source_props.url.split('/').pop() : '');
  document.getElementById('pipe-through').textContent = p.through || 'none';
  document.getElementById('pipe-sink').textContent = (p.sink_type || 'none') +
    (p.sink_props?.table ? '.' + p.sink_props.table : '');
  document.getElementById('pipe-details').textContent = p.does ? p.does.substring(0, 120) : '';

  inputEl.value = JSON.stringify(p.constructed_input || {});
  infoEl.style.display = 'flex';

  // Also set the pathway selector to match
  const pwSel = document.getElementById('cmd-pathway');
  if(p.through) {
    for(let i = 0; i < pwSel.options.length; i++) {
      if(pwSel.options[i].value === p.through) { pwSel.selectedIndex = i; break; }
    }
  }
}

async function cmdPipelineRun() {
  const pipeline = document.getElementById('cmd-pipeline').value;
  const kernel = document.getElementById('cmd-kernel').value;
  if(!pipeline) { showToast('Pipeline Error', 'Select a pipeline first', true); return; }

  showToast('Running pipeline...', pipeline + ' with ' + kernel, false);
  try {
    const resp = await fetch('/api/pipeline/run', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({pipeline, kernel}),
    });
    const data = await resp.json();
    if(data.error) { showToast('Pipeline Failed', data.error, true); }
    else { pollJob(data.job_id, 'Pipeline: ' + pipeline); }
  } catch(e) { showToast('Pipeline Error', e.message, true); }
}

async function cmdRun() {
  const pathway = document.getElementById('cmd-pathway').value;
  const kernel = document.getElementById('cmd-kernel').value;
  let input;
  try { input = JSON.parse(document.getElementById('cmd-input').value || '{}'); }
  catch(e) { showToast('Input Error', 'Invalid JSON: ' + e.message, true); return; }

  showToast('Running...', pathway + ' with ' + kernel, false);
  try {
    const resp = await fetch('/api/run', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({pathway, kernel, input}),
    });
    const data = await resp.json();
    if(data.error) { showToast('Run Failed', data.error, true); }
    else { pollJob(data.job_id, 'Run: ' + pathway); }
  } catch(e) { showToast('Run Error', e.message, true); }
}

async function cmdGenerate() {
  const locus = document.getElementById('cmd-locus').value;
  const kernel = document.getElementById('cmd-kernel').value;
  const engine = document.getElementById('cmd-engine').value;
  const count = parseInt(document.getElementById('cmd-count').value) || 1;

  showToast('Generating...', count + ' variant(s) for ' + locus + ' via ' + engine, false);
  try {
    const resp = await fetch('/api/generate', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({locus, kernel, mutation_engine: engine, count}),
    });
    const data = await resp.json();
    if(data.error) { showToast('Generate Failed', data.error, true); }
    else { pollJob(data.job_id, 'Generated: ' + locus); }
  } catch(e) { showToast('Generate Error', e.message, true); }
}

async function cmdCompete() {
  const locus = document.getElementById('cmd-locus').value;
  const kernel = document.getElementById('cmd-kernel').value;
  let input;
  try { input = JSON.parse(document.getElementById('cmd-input').value || '{}'); }
  catch(e) { showToast('Input Error', 'Invalid JSON: ' + e.message, true); return; }

  const rounds = parseInt(document.getElementById('cmd-count').value) || 10;
  showToast('Competing...', locus + ' (' + rounds + ' rounds)', false);
  try {
    const resp = await fetch('/api/compete', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({locus, kernel, input, rounds}),
    });
    const data = await resp.json();
    if(data.error) { showToast('Compete Failed', data.error, true); }
    else { pollJob(data.job_id, 'Competition: ' + locus); }
  } catch(e) { showToast('Compete Error', e.message, true); }
}

async function competeLocus(locus) {
  const kernel = document.getElementById('cmd-kernel').value || 'data-mock';
  let input;
  try { input = JSON.parse(document.getElementById('cmd-input').value || '{}'); }
  catch(e) { input = {}; }

  const rounds = 10;
  showToast('Competing...', locus + ' (' + rounds + ' rounds, all alleles)', false);
  try {
    const resp = await fetch('/api/compete', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({locus, kernel, input, rounds}),
    });
    const data = await resp.json();
    if(data.error) { showToast('Compete Failed', data.error, true); }
    else { pollJob(data.job_id, 'Competition: ' + locus); }
  } catch(e) { showToast('Compete Error', e.message, true); }
}

async function generateForLocus(locus) {
  const kernel = document.getElementById('cmd-kernel').value || 'data-mock';
  const engine = document.getElementById('cmd-engine').value || 'mock';
  showToast('Generating...', '1 variant for ' + locus + ' via ' + engine, false);
  try {
    const resp = await fetch('/api/generate', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({locus, kernel, mutation_engine: engine, count: 1}),
    });
    const data = await resp.json();
    if(data.error) { showToast('Generate Failed', data.error, true); }
    else { pollJob(data.job_id, 'Generated: ' + locus); }
  } catch(e) { showToast('Generate Error', e.message, true); }
}

async function updateDaemonStatus() {
  const data = await fetchJSON('/api/daemon/status');
  if(!data) return;
  const dot = document.getElementById('daemon-dot');
  const label = document.getElementById('daemon-label');
  if(data.running) {
    dot.className = 'daemon-dot on';
    label.textContent = 'daemon tick #' + data.tick_count;
  } else {
    dot.className = 'daemon-dot off';
    label.textContent = 'daemon stopped';
  }
}

// Extend refreshData to also update command bar selectors and daemon status
const _origRefreshData = refreshData;
refreshData = async function() {
  await _origRefreshData();
  updateCommandBarSelectors();
  updateDaemonStatus();
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
    import uvicorn
    logger.info("Starting dashboard at http://%s:%d", host, port)
    uvicorn.run(app, host=host, port=port, log_level="warning")
