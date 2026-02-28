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
from sg.phenotype import PhenotypeMap
from sg.registry import Registry


app = FastAPI(title="Software Genome Dashboard")

# Project root — set at startup
_project_root: Path = Path(".")


def _load_state():
    """Load all state from disk (fresh on each request)."""
    root = _project_root
    contract_store = ContractStore.open(root / "contracts")
    registry = Registry.open(root / ".sg" / "registry")
    phenotype = PhenotypeMap.load(root / "phenotype.toml")
    fusion_tracker = FusionTracker.open(root / "fusion_tracker.json")
    return contract_store, registry, phenotype, fusion_tracker


@app.get("/api/status")
def api_status():
    cs, reg, pheno, ft = _load_state()
    allele_count = len(reg.alleles)
    loci_count = len(cs.known_loci())
    pathway_count = len(cs.known_pathways())
    topology_count = len(cs.known_topologies())
    fused_count = sum(
        1 for name in cs.known_pathways()
        if (f := pheno.get_fused(name)) and f.fused_sha
    )
    fitnesses = [arena.compute_fitness(a) for a in reg.alleles.values()]
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
    cs, reg, pheno, _ = _load_state()
    result = []
    for locus in cs.known_loci():
        alleles = reg.alleles_for_locus(locus)
        dominant_sha = pheno.get_dominant(locus)
        dominant_fitness = 0.0
        if dominant_sha:
            dom = reg.get(dominant_sha)
            if dom:
                dominant_fitness = arena.compute_fitness(dom)
        result.append({
            "name": locus,
            "dominant_sha": dominant_sha[:12] if dominant_sha else None,
            "allele_count": len(alleles),
            "dominant_fitness": round(dominant_fitness, 3),
        })
    return result


@app.get("/api/locus/{name}")
def api_locus(name: str):
    cs, reg, pheno, _ = _load_state()
    alleles = reg.alleles_for_locus(name)
    dominant_sha = pheno.get_dominant(name)

    allele_list = []
    for a in alleles:
        allele_list.append({
            "sha": a.sha256[:12],
            "sha_full": a.sha256,
            "generation": a.generation,
            "fitness": round(arena.compute_fitness(a), 3),
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
    cs, _, pheno, ft = _load_state()
    result = []
    for name in cs.known_pathways():
        fusion = pheno.get_fused(name)
        track = ft.get_track(name)
        result.append({
            "name": name,
            "fused": bool(fusion and fusion.fused_sha),
            "fused_sha": fusion.fused_sha[:12] if fusion and fusion.fused_sha else None,
            "reinforcement_count": track.reinforcement_count if track else 0,
            "total_successes": track.total_successes if track else 0,
            "total_failures": track.total_failures if track else 0,
        })
    return result


@app.get("/api/allele/{sha}/source")
def api_allele_source(sha: str):
    _, reg, _, _ = _load_state()
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
    _, reg, _, _ = _load_state()
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
            "fitness": round(arena.compute_fitness(allele), 3),
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
                      root / "fusion_tracker.json"]:
                if f.exists():
                    current = max(current, f.stat().st_mtime)
            if current > last_mtime and last_mtime > 0:
                yield f"data: {{\"type\": \"update\", \"time\": {current}}}\n\n"
            last_mtime = current
            import asyncio
            await asyncio.sleep(2)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# Federation endpoints (used by sg share/pull)

@app.post("/api/federation/receive")
async def federation_receive(request: Request):
    """Accept an allele from a peer with integrity verification."""
    data = await request.json()
    _, reg, pheno, _ = _load_state()
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


@app.get("/api/federation/alleles/{locus}")
def federation_alleles(locus: str):
    """Serve alleles for a locus to a peer."""
    _, reg, _, _ = _load_state()
    from sg.federation import export_allele
    alleles = reg.alleles_for_locus(locus)
    result = []
    for a in alleles[:5]:  # limit to top 5 by fitness
        data = export_allele(reg, a.sha256)
        if data:
            result.append(data)
    return {"alleles": result}


_DASHBOARD_HTML = """<!DOCTYPE html>
<html><head>
<title>Software Genome Dashboard</title>
<style>
  body { font-family: monospace; background: #1a1a2e; color: #e0e0e0; margin: 2em; }
  h1 { color: #0ff; }
  h2 { color: #0aa; margin-top: 1.5em; }
  table { border-collapse: collapse; width: 100%; margin: 0.5em 0; }
  th, td { padding: 6px 12px; text-align: left; border-bottom: 1px solid #333; }
  th { color: #0ff; }
  .dominant { color: #0f0; }
  .recessive { color: #ff0; }
  .deprecated { color: #f44; }
  .stat { display: inline-block; margin: 0 2em 0.5em 0; }
  .stat-val { font-size: 2em; color: #0ff; }
  .stat-label { color: #888; }
  .modal { display:none; position:fixed; top:5%; left:10%; width:80%; height:85%;
    background:#111; border:1px solid #0ff; padding:1em; overflow:auto; z-index:10; }
  .modal pre { white-space: pre-wrap; }
  .close-btn { float:right; cursor:pointer; color:#f44; font-size:1.5em; }
  a { color: #0af; cursor: pointer; }
  .lineage-chain { padding: 0.5em; }
  .lineage-node { display: inline-block; padding: 4px 8px; margin: 2px;
    border: 1px solid #0aa; border-radius: 3px; }
  .lineage-arrow { color: #666; margin: 0 4px; }
  .bar { display: inline-block; height: 12px; background: #0aa; }
  .bar-bg { display: inline-block; height: 12px; background: #333; width: 80px; }
  .warn { color: #f80; }
</style>
</head><body>
<h1>Software Genome Dashboard</h1>
<div id="stats"></div>
<h2>Loci</h2>
<table id="loci-table"><thead><tr>
  <th>Locus</th><th>Dominant</th><th>Alleles</th><th>Fitness</th><th>Lineage</th>
</tr></thead><tbody></tbody></table>
<h2>Pathways</h2>
<table id="pathways-table"><thead><tr>
  <th>Pathway</th><th>Fused</th><th>Reinforcement</th><th>Successes</th><th>Failures</th>
</tr></thead><tbody></tbody></table>
<h2>Regression Monitor</h2>
<table id="regression-table"><thead><tr>
  <th>Allele</th><th>Peak</th><th>Current</th><th>Drop</th><th>Samples</th><th>Status</th>
</tr></thead><tbody></tbody></table>
<div id="source-modal" class="modal"><span class="close-btn" onclick="closeModal('source-modal')">&times;</span>
<h3 id="source-title"></h3><pre id="source-code"></pre></div>
<div id="lineage-modal" class="modal"><span class="close-btn" onclick="closeModal('lineage-modal')">&times;</span>
<h3 id="lineage-title"></h3><div id="lineage-chain"></div></div>
<script>
async function load() {
  const s = await (await fetch('/api/status')).json();
  document.getElementById('stats').innerHTML = [
    ['Loci', s.loci_count], ['Alleles', s.allele_count],
    ['Pathways', s.pathway_count], ['Fused', s.fused_count],
    ['Avg Fitness', s.avg_fitness.toFixed(3)]
  ].map(([l,v])=>`<div class="stat"><div class="stat-val">${v}</div><div class="stat-label">${l}</div></div>`).join('');

  const loci = await (await fetch('/api/loci')).json();
  document.querySelector('#loci-table tbody').innerHTML = loci.map(l =>
    `<tr><td>${l.name}</td><td><a onclick="showSource('${l.dominant_sha}')">${l.dominant_sha||'none'}</a></td>` +
    `<td>${l.allele_count}</td><td>${l.dominant_fitness.toFixed(3)}</td>` +
    `<td><a onclick="showLineage('${l.dominant_sha}')">view</a></td></tr>`
  ).join('');

  const pw = await (await fetch('/api/pathways')).json();
  document.querySelector('#pathways-table tbody').innerHTML = pw.map(p =>
    `<tr><td>${p.name}</td><td>${p.fused?p.fused_sha:'no'}</td>` +
    `<td>${p.reinforcement_count}/10</td><td>${p.total_successes}</td><td>${p.total_failures}</td></tr>`
  ).join('');

  const reg = await (await fetch('/api/regression')).json();
  const tbody = document.querySelector('#regression-table tbody');
  if (reg.history.length === 0) {
    tbody.innerHTML = '<tr><td colspan="6" style="color:#666">No regression data yet</td></tr>';
  } else {
    tbody.innerHTML = reg.history.map(h => {
      let status = '<span style="color:#0f0">stable</span>';
      if (h.drop >= 0.4) status = '<span style="color:#f44">SEVERE</span>';
      else if (h.drop >= 0.2) status = '<span class="warn">mild</span>';
      const barW = Math.round(h.last_fitness * 80);
      return `<tr><td><a onclick="showSource('${h.sha}')">${h.sha}</a></td>` +
        `<td>${h.peak_fitness.toFixed(3)}</td><td>${h.last_fitness.toFixed(3)}</td>` +
        `<td>${h.drop.toFixed(3)}</td><td>${h.samples}</td><td>${status}</td></tr>`;
    }).join('');
  }
}
async function showSource(sha) {
  if (!sha || sha==='none') return;
  const d = await (await fetch('/api/allele/'+sha+'/source')).json();
  document.getElementById('source-title').textContent = sha;
  document.getElementById('source-code').textContent = d.source||d.error||'not found';
  document.getElementById('source-modal').style.display = 'block';
}
async function showLineage(sha) {
  if (!sha || sha==='none') return;
  const d = await (await fetch('/api/lineage/'+sha)).json();
  document.getElementById('lineage-title').textContent = 'Lineage: ' + sha;
  document.getElementById('lineage-chain').innerHTML = d.lineage.map((n,i) =>
    (i>0?'<span class="lineage-arrow">&larr;</span>':'') +
    `<span class="lineage-node" title="gen ${n.generation}, fitness ${n.fitness}">`+
    `<a onclick="showSource('${n.sha}')">${n.sha}</a> g${n.generation}</span>`
  ).join('');
  document.getElementById('lineage-modal').style.display = 'block';
}
function closeModal(id) { document.getElementById(id).style.display='none'; }
load(); setInterval(load, 5000);
</script>
</body></html>"""


@app.get("/", response_class=HTMLResponse)
def dashboard_html():
    return _DASHBOARD_HTML


def run_dashboard(root: Path, host: str = "127.0.0.1", port: int = 8420) -> None:
    """Start the dashboard server."""
    global _project_root
    _project_root = root
    import uvicorn
    print(f"Starting dashboard at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")
