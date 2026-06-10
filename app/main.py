"""Zydus Drug Intelligence Platform — FastAPI main application."""
from __future__ import annotations

import asyncio
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config import (
    CACHE_DIR,
    CORS_ALLOWED_ORIGINS,
    FABRIC_CONTAINER,
    FABRIC_FOLDER,
    FABRIC_ONELAKE_URL,
    FABRIC_STORAGE_KEY,
    SOURCE_TIMEOUT,
)
from app.consistency import check_cross_source_consistency
from app.enrichment.data_protection import fetch_data_protection_table
from app.enrichment.labeling import enrich_labeling_batch
from app.enrichment.patents import enrich_patents
from app.enrichment.store import get_labeling_for_din, reset_labeling_table, reset_patents_table
from app.enrichment.workbook import (
    _is_excluded_din,
    build_exclusion_list,
    build_sheet1,
    build_sheet2,
    build_workbook,
)
from app.export_job import run_export_job
from app.jobs import create_job, get_job
from app.match import generate_summary
from app.models import SearchMetadata, SearchResponse, SourceResult
from app.normalize import normalize_query
from app.sources.dpd import search_dpd
from app.sources.generic_submissions import search_generic_submissions
from app.sources.noc import search_noc
from app.cache import cache_clear_all
from app.sources.patent_register import search_patent_register

app = FastAPI(title="Zydus Drug Intelligence Platform", version="1.0.0")

# CORS — lets Power BI Service, Fabric notebooks, and other browser clients call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

import pathlib as _pathlib
_static_dir = _pathlib.Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


async def _timed_source(coro, source_name: str) -> SourceResult:
    try:
        result = await asyncio.wait_for(coro, timeout=SOURCE_TIMEOUT)
        return result
    except asyncio.TimeoutError:
        return SourceResult(
            source=source_name,
            status="timeout",
            error_message=f"Source timed out after {SOURCE_TIMEOUT}s.",
        )
    except Exception as e:
        return SourceResult(source=source_name, status="error", error_message=str(e))


@app.get("/api/search", response_model=SearchResponse)
async def search(
    q: str = Query(..., description="Search term"),
    field: str = Query("ingredient", description="ingredient | brand | company | din"),
    summary: bool = Query(False, description="Include AI-generated plain-language summary"),
) -> SearchResponse:
    q = q.strip()
    if not q:
        return SearchResponse(
            metadata=SearchMetadata(
                query=q,
                field=field,
                timestamp=datetime.now(timezone.utc).isoformat(),
            ),
            sources=[],
        )

    canonical, extra_terms = await normalize_query(q, field)

    dpd_task = _timed_source(search_dpd(canonical, field, extra_terms), "DPD")
    gen_task = _timed_source(search_generic_submissions(canonical, field, extra_terms), "GenericSubmissions")
    noc_task = _timed_source(search_noc(canonical, field, extra_terms), "NOC")
    pr_task = _timed_source(search_patent_register(canonical, field, extra_terms), "PatentRegister")

    dpd_result, gen_result, noc_result, pr_result = await asyncio.gather(
        dpd_task, gen_task, noc_task, pr_task
    )

    sources = [dpd_result, gen_result, noc_result, pr_result]
    all_records = [rec for s in sources for rec in s.records]
    check_cross_source_consistency(all_records)

    ai_summary: Optional[str] = None
    if summary:
        ai_summary = await generate_summary(q, sources)

    metadata = SearchMetadata(
        query=q,
        field=field,
        timestamp=datetime.now(timezone.utc).isoformat(),
        normalized_terms=[canonical] + extra_terms,
        per_source_status={s.source: s.status for s in sources},
    )
    return SearchResponse(metadata=metadata, sources=sources, ai_summary=ai_summary)


@app.get("/api/export")
async def export(
    q: str = Query(...),
    field: str = Query("ingredient"),
    allow_partial: bool = Query(False),
) -> Response:
    """Synchronous two-sheet enriched workbook download (blocks until complete)."""
    result = await search(q=q, field=field, summary=False)

    error_sources: dict[str, Optional[str]] = {
        s.source: s.error_message for s in result.sources if s.status == "error"
    }
    if error_sources and not allow_partial:
        names = ", ".join(error_sources.keys())
        details = "; ".join(f"{k}: {v or 'unknown error'}" for k, v in error_sources.items())
        raise HTTPException(
            status_code=409,
            detail=(
                f"Source(s) failed: {names} — refusing to build a partial workbook. "
                f"Pass allow_partial=true to override. Details: {details}"
            ),
        )

    all_valid_dins = [
        r.din for s in result.sources for r in s.records if not _is_excluded_din(r.din)
    ]
    if all_valid_dins:
        await enrich_patents(all_valid_dins)

    din_map: dict[str, tuple[int, Optional[str]]] = {}
    for s in result.sources:
        if s.source != "DPD":
            continue
        for r in s.records:
            if _is_excluded_din(r.din):
                continue
            drug_code_raw = r.source_specific.get("drug_code")
            if drug_code_raw is None:
                continue
            try:
                din_key = r.din.strip()  # type: ignore[union-attr]
                if get_labeling_for_din(din_key) is None:
                    din_map[din_key] = (int(drug_code_raw), r.strength)
            except (ValueError, TypeError):
                pass
    if din_map:
        await enrich_labeling_batch(din_map)

    dp_table = await fetch_data_protection_table()
    xlsx_bytes = build_workbook(
        result,
        source_errors=error_sources if allow_partial and error_sources else None,
        dp_table=dp_table,
    )

    # Write sidecar exclusion list alongside the workbook.
    exclusion_df = build_exclusion_list(result, ingredient_name=q)
    excl_path = os.path.join(CACHE_DIR, f"{q.replace(' ', '_')}_{field}_excluded.csv")
    os.makedirs(CACHE_DIR, exist_ok=True)
    exclusion_df.to_csv(excl_path, index=False)
    if not exclusion_df.empty:
        import logging as _logging
        _logging.getLogger(__name__).info(
            "Exclusion list (%d DIN(s)) saved to: %s", len(exclusion_df), excl_path
        )

    filename = f"canadian_drugs_{q.replace(' ', '_')}_{field}.xlsx"
    return Response(
        content=xlsx_bytes,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-Exclusion-List": excl_path,
        },
    )


@app.get("/api/exclusions")
async def exclusions(
    q: str = Query(..., description="Ingredient name (same as used for /api/export)"),
    field: str = Query("ingredient"),
) -> Response:
    """Download the exclusion list (CSV) for the last export of this ingredient.

    Returns the DPD DINs that were dropped because they are not present in NOC
    for the queried ingredient.  The file is written by /api/export; call that
    first if no file exists yet.
    """
    excl_path = os.path.join(CACHE_DIR, f"{q.replace(' ', '_')}_{field}_excluded.csv")
    if not os.path.exists(excl_path):
        raise HTTPException(
            status_code=404,
            detail=(
                f"No exclusion list found for {q!r}. "
                "Call /api/export first to generate it."
            ),
        )
    with open(excl_path, "rb") as fh:
        csv_bytes = fh.read()
    filename = f"{q.replace(' ', '_')}_{field}_excluded.csv"
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── Power BI / Microsoft Fabric integration ───────────────────────────────────

def _df_to_table(df: "pd.DataFrame") -> dict:
    """Convert a DataFrame to {columns, records} dict suitable for JSON consumers."""
    if df is None or df.empty:
        return {"columns": [], "records": []}
    clean = df.where(pd.notna(df), None)
    return {"columns": list(clean.columns), "records": clean.to_dict("records")}


async def _enrich_for_export(
    result: SearchResponse,
    allow_partial: bool,
) -> tuple[dict, "pd.DataFrame", "pd.DataFrame"]:
    """Run patents + labeling enrichment and return (error_sources, sheet1_df, sheet2_df)."""
    error_sources: dict[str, Optional[str]] = {
        s.source: s.error_message for s in result.sources if s.status == "error"
    }
    if error_sources and not allow_partial:
        names = ", ".join(error_sources.keys())
        raise HTTPException(
            status_code=409,
            detail=f"Source(s) failed: {names}. Pass allow_partial=true to override.",
        )

    all_valid_dins = [
        r.din for s in result.sources for r in s.records if not _is_excluded_din(r.din)
    ]
    if all_valid_dins:
        await enrich_patents(all_valid_dins)

    din_map: dict[str, tuple[int, Optional[str]]] = {}
    for s in result.sources:
        if s.source != "DPD":
            continue
        for r in s.records:
            if _is_excluded_din(r.din):
                continue
            drug_code_raw = r.source_specific.get("drug_code")
            if drug_code_raw is None:
                continue
            try:
                din_key = r.din.strip()  # type: ignore[union-attr]
                if get_labeling_for_din(din_key) is None:
                    din_map[din_key] = (int(drug_code_raw), r.strength)
            except (ValueError, TypeError):
                pass
    if din_map:
        await enrich_labeling_batch(din_map)

    dp_table = await fetch_data_protection_table()
    s1_df = build_sheet1(result, dp_table=dp_table)
    s2_df = build_sheet2(result)
    return error_sources, s1_df, s2_df


@app.get("/api/powerbi")
async def powerbi_data(
    q: str = Query(..., description="Search term (ingredient, brand, company, or DIN)"),
    field: str = Query("ingredient", description="ingredient | brand | company | din"),
    allow_partial: bool = Query(True, description="Return data even if a source errors"),
) -> dict:
    """Power BI & Microsoft Fabric JSON data endpoint.

    Returns Sheet 1 (DPD + NOC + Patents + Labeling) and Sheet 2 (Generic Submissions)
    as flat JSON arrays.  Consumed directly by the Power BI Web connector or a Fabric
    notebook — no job polling required.

    First call performs live enrichment (30–120 s); subsequent calls return cached data
    in under 2 s.

    Power BI Web connector URL pattern:
        GET http://<host>:8000/api/powerbi?q=alpelisib&field=ingredient
    """
    result = await search(q=q, field=field, summary=False)
    _, s1_df, s2_df = await _enrich_for_export(result, allow_partial)
    return {
        "query": q,
        "field": field,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sheet1": _df_to_table(s1_df),
        "sheet2": _df_to_table(s2_df),
    }


@app.post("/api/fabric/push")
async def fabric_push(
    q: str = Query(..., description="Search term"),
    field: str = Query("ingredient"),
    allow_partial: bool = Query(True),
) -> dict:
    """Push enriched XLSX directly to Azure Data Lake Storage Gen2 / OneLake.

    Requires FABRIC_ONELAKE_URL and FABRIC_CONTAINER env vars to be set.
    Uses azure-storage-blob + azure-identity (install separately).
    Falls back to a 501 if those packages are not installed or vars are missing.

    Returns {"status": "ok", "path": "<datalake path>"} on success.
    """
    if not FABRIC_ONELAKE_URL or not FABRIC_CONTAINER:
        raise HTTPException(
            status_code=501,
            detail=(
                "Fabric push not configured. "
                "Set FABRIC_ONELAKE_URL and FABRIC_CONTAINER environment variables."
            ),
        )

    try:
        from azure.identity import DefaultAzureCredential  # type: ignore[import]
        from azure.storage.blob import BlobServiceClient  # type: ignore[import]
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail=(
                "azure-storage-blob and azure-identity are required for Fabric push. "
                "Run: pip install azure-storage-blob azure-identity"
            ),
        )

    result = await search(q=q, field=field, summary=False)
    error_sources, s1_df, s2_df = await _enrich_for_export(result, allow_partial)

    dp_table = await fetch_data_protection_table()
    xlsx_bytes = build_workbook(
        result,
        source_errors=error_sources if allow_partial and error_sources else None,
        dp_table=dp_table,
    )

    # Authenticate — Managed Identity on Fabric, fallback to storage key if provided
    if FABRIC_STORAGE_KEY:
        client = BlobServiceClient(
            account_url=FABRIC_ONELAKE_URL,
            credential=FABRIC_STORAGE_KEY,
        )
    else:
        client = BlobServiceClient(
            account_url=FABRIC_ONELAKE_URL,
            credential=DefaultAzureCredential(),
        )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    blob_name = f"{FABRIC_FOLDER}/{q.replace(' ', '_')}_{field}_{ts}.xlsx"
    blob_client = client.get_blob_client(container=FABRIC_CONTAINER, blob=blob_name)
    blob_client.upload_blob(
        xlsx_bytes,
        overwrite=True,
        content_settings={"content_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"},
    )

    full_path = f"{FABRIC_ONELAKE_URL}/{FABRIC_CONTAINER}/{blob_name}"
    return {"status": "ok", "path": full_path, "rows_sheet1": len(s1_df), "rows_sheet2": len(s2_df)}


# ── Cache management ──────────────────────────────────────────────────────────

@app.post("/api/reset-all-caches")
async def reset_all_caches() -> dict:
    """Clear all cached data: HTTP cache, patents, and labeling."""
    http_rows = cache_clear_all()
    patent_rows = reset_patents_table()
    labeling_rows = reset_labeling_table()
    return {"status": "ok", "http_rows_cleared": http_rows, "patent_rows_cleared": patent_rows, "labeling_rows_cleared": labeling_rows}


# ── Async export: start / stream / result ─────────────────────────────────────

class ExportStartRequest(BaseModel):
    q: str = ""                  # single-query backward compat
    queries: list[str] = []      # multi-product list (preferred)
    field: str = "ingredient"
    allow_partial: bool = False
    enable_ocr: bool = True


def _resolve_queries(req: ExportStartRequest) -> list[str]:
    """Return deduplicated, non-empty list of queries (case-insensitive dedup, order preserved)."""
    raw = req.queries if req.queries else ([req.q.strip()] if req.q.strip() else [])
    seen: set[str] = set()
    out: list[str] = []
    for q in raw:
        q = q.strip()
        if q and q.lower() not in seen:
            seen.add(q.lower())
            out.append(q)
    return out


@app.post("/export/start")
async def export_start(req: ExportStartRequest) -> dict:
    """Create a background export job and return its job_id immediately.

    Accepts either ``q`` (single ingredient, backward compat) or ``queries``
    (list of ingredients for a multi-product side-by-side workbook).
    Exact duplicates (case-insensitive) are silently deduplicated; the
    deduplicated list is returned so the caller knows what was accepted.
    Input order is preserved — block order in the workbook matches entry order.
    """
    qs = _resolve_queries(req)
    if not qs:
        raise HTTPException(400, "No query provided — set q or queries")
    job_id = uuid.uuid4().hex
    job = create_job(job_id, qs[0], req.field, queries=qs)
    asyncio.create_task(
        run_export_job(job, req.allow_partial, req.enable_ocr)
    )
    return {"job_id": job_id, "queries": qs}


@app.get("/export/stream/{job_id}")
async def export_stream(job_id: str) -> StreamingResponse:
    """Server-Sent Events stream for export job progress.

    Each event is a JSON object:
      progress: {stage, done, total, pct, elapsed_s, eta_s, log}
      complete: {status:"complete", download_url, elapsed_s, log}
      error:    {status:"error", message, elapsed_s}

    A `: keepalive` comment is sent every 15 s to prevent proxy timeouts.
    Reconnecting clients receive all buffered events from the start.
    """
    job = get_job(job_id)

    async def _gen():
        if job is None:
            yield f"data: {json.dumps({'status': 'error', 'message': 'Job not found'})}\n\n"
            return

        idx = 0
        while True:
            # Drain all buffered events
            while idx < len(job.events):
                evt = job.events[idx]
                idx += 1
                yield f"data: {json.dumps(evt)}\n\n"
                if evt.get("status") in ("complete", "error"):
                    return

            if job.status in ("complete", "error"):
                return

            # Race fix: clear notify flag, then re-check list length before waiting
            job._notify.clear()
            if idx < len(job.events):
                continue

            try:
                await asyncio.wait_for(job._notify.wait(), timeout=15.0)
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/export/result/{job_id}")
async def export_result(job_id: str) -> FileResponse:
    """Download the finished XLSX. Returns 409 if the job is still running."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job.status != "complete" or not job.result_path:
        raise HTTPException(409, f"Job not complete (status={job.status})")
    qs = job.queries or [job.query]
    if len(qs) == 1:
        filename = f"canadian_drugs_{qs[0].replace(' ', '_')}_{job.field}.xlsx"
    else:
        filename = f"canadian_drugs_{len(qs)}_products_{job.field}.xlsx"
    return FileResponse(
        path=job.result_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=filename,
    )


@app.get("/api/export-data/{job_id}")
async def export_data_json(job_id: str) -> dict:
    """Return Sheet 1 and Sheet 2 as JSON — the exact dataset written to the XLSX.

    This is the dashboard data endpoint.  The dashboard must call this instead of
    re-running any search or enrichment; it consumes the finished job snapshot.
    """
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job.status == "running":
        raise HTTPException(409, "Job still running — wait for SSE complete event")
    if job.status == "error":
        raise HTTPException(422, f"Job failed: {job.error}")
    return {
        "query": job.query,
        "queries": job.queries or [job.query],
        "field": job.field,
        "sheet1": {"columns": job.sheet1_columns, "records": job.sheet1_records},
        "sheet2": {"columns": job.sheet2_columns, "records": job.sheet2_records},
    }


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse(content=_HTML_UI)


# ── Embedded single-page UI ────────────────────────────────────────────────────
_HTML_UI = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Zydus Drug Intelligence Platform</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
<link href="https://fonts.googleapis.com/css2?family=Exo:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet"/>
<style>
  /* ── Design tokens ───────────────────────────────────────────────────── */
  :root {
    --primary:      #AA55A0;
    --primary-dark: #3D226E;
    --teal:         #00A5A5;
    --teal-dark:    #008BAD;
    --nav-bg:       #3D226E;
    --bg:           #FAFAFA;
    --card:         #FFFFFF;
    --border:       #D1D1D1;
    --text:         #333333;
    --muted:        #58595B;
    --ok:           #00A5A5;
    --warn:         #B45309;
    --err:          #C0392B;
    --badge-dpd:    #3D226E;
    --badge-gen:    #AA55A0;
    --badge-noc:    #00A5A5;
    --badge-pr:     #008BAD;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: "Inter", sans-serif; background: var(--bg); color: var(--text); font-size: 15px; line-height: 1.5; }

  /* ── Top nav bar ─────────────────────────────────────────────────────── */
  .site-nav {
    background: var(--nav-bg);
    padding: 0 24px;
    display: flex;
    align-items: center;
    height: 52px;
    gap: 16px;
  }
  .site-nav-brand-wrap { display: flex; flex-direction: column; gap: 0; }
  .site-nav-brand {
    font-family: "Exo", sans-serif;
    font-weight: 700;
    font-size: 0.84rem;
    color: rgba(255,255,255,0.92);
    text-transform: uppercase;
    letter-spacing: .1em;
    line-height: 1.15;
    white-space: nowrap;
  }
  .site-nav-sub {
    font-size: 0.62rem;
    font-weight: 400;
    color: rgba(255,255,255,0.42);
    text-transform: uppercase;
    letter-spacing: .09em;
  }
  .site-nav-divider { flex: 1; }
  .site-nav-link {
    font-size: 0.78rem;
    color: rgba(255,255,255,0.65);
    text-decoration: none;
    letter-spacing: .04em;
  }
  .site-nav-link:hover { color: #fff; }

  /* ── Header ──────────────────────────────────────────────────────────── */
  header {
    background: var(--primary);
    color: white;
    padding: 24px 24px 22px;
    border-bottom: 3px solid var(--teal);
  }
  .header-brand-row {
    display: flex;
    align-items: center;
    gap: 18px;
    margin-bottom: 10px;
  }
  .header-company-name {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: .14em;
    color: rgba(255,255,255,0.58);
    font-weight: 600;
    margin-bottom: 3px;
  }
  header h1 {
    font-family: "Exo", sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: -.01em;
    margin-bottom: 0;
  }
  header p {
    font-size: 0.82rem;
    color: rgba(255,255,255,0.68);
    font-weight: 400;
    letter-spacing: .01em;
    margin-top: 6px;
  }

  /* ── Layout ──────────────────────────────────────────────────────────── */
  .container { max-width: 1280px; margin: 0 auto; padding: 28px 20px; }

  /* ── Search card ─────────────────────────────────────────────────────── */
  .search-box {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 22px 24px;
    margin-bottom: 20px;
    box-shadow: 0 1px 3px rgba(170,85,160,.08);
  }
  .row { display: flex; gap: 12px; flex-wrap: wrap; align-items: flex-end; }
  .field-group { display: flex; flex-direction: column; gap: 5px; }
  .field-group label {
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: .07em;
  }
  input[type=text], select {
    padding: 9px 13px;
    border: 1px solid var(--border);
    border-radius: 3px;
    font-size: 0.92rem;
    font-family: inherit;
    outline: none;
    color: var(--text);
    background: #fff;
    transition: border-color .15s, box-shadow .15s;
  }
  input[type=text]:focus, select:focus {
    border-color: var(--primary);
    box-shadow: 0 0 0 3px rgba(170,85,160,.15);
  }
  #query { min-width: 300px; }

  /* ── Buttons ─────────────────────────────────────────────────────────── */
  .btn {
    padding: 9px 22px;
    border: none;
    border-radius: 3px;
    cursor: pointer;
    font-size: 0.9rem;
    font-weight: 600;
    font-family: inherit;
    letter-spacing: .02em;
    transition: background .15s;
  }
  .btn-primary { background: var(--primary); color: white; }
  .btn-primary:hover { background: var(--primary-dark); }
  .btn-export { background: var(--teal); color: white; }
  .btn-export:hover:not(:disabled) { background: var(--teal-dark); }
  .btn:disabled { opacity: .45; cursor: not-allowed; }

  /* ── Status / summary bars ───────────────────────────────────────────── */
  @keyframes summaryBarIn {
    from { opacity: 0; transform: translateY(-6px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  .summary-bar {
    background: #fff3cd;
    border: 1px solid #ffc107;
    border-left: 6px solid #e6a500;
    border-radius: 4px;
    padding: 13px 18px;
    font-size: 0.93rem;
    font-weight: 500;
    margin-bottom: 20px;
    display: none;
    color: #4a3600;
    box-shadow: 0 2px 10px rgba(230, 165, 0, 0.22);
    animation: summaryBarIn 0.25s ease;
  }
  .ai-summary {
    background: #E8F4F8;
    border: 1px solid #A8D4E6;
    border-left: 4px solid var(--teal);
    border-radius: 3px;
    padding: 12px 16px;
    font-size: 0.88rem;
    margin-bottom: 16px;
    display: none;
  }
  .ai-summary strong { color: var(--teal); }

  /* ── Source tabs ─────────────────────────────────────────────────────── */
  .tabs {
    display: flex;
    gap: 0;
    border-bottom: 2px solid var(--border);
    margin-bottom: 16px;
    flex-wrap: wrap;
  }
  .tab-btn {
    padding: 9px 20px;
    border: 1px solid var(--border);
    border-bottom: none;
    background: var(--bg);
    cursor: pointer;
    font-size: 0.86rem;
    font-family: inherit;
    font-weight: 500;
    margin-bottom: -2px;
    border-radius: 3px 3px 0 0;
    color: var(--muted);
    transition: background .12s, color .12s;
  }
  .tab-btn.active {
    background: var(--card);
    border-bottom: 2px solid var(--card);
    font-weight: 700;
    color: var(--primary);
  }
  .tab-pane { display: none; }
  .tab-pane.active { display: block; }

  /* ── Source header badges ────────────────────────────────────────────── */
  .source-header { display: flex; align-items: center; gap: 10px; margin-bottom: 12px; }
  .badge {
    padding: 3px 11px;
    border-radius: 2px;
    color: white;
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: .04em;
    text-transform: uppercase;
  }
  .badge-DPD           { background: var(--badge-dpd); }
  .badge-GenericSubmissions { background: var(--badge-gen); }
  .badge-NOC           { background: var(--badge-noc); }
  .badge-PatentRegister { background: var(--badge-pr); }

  /* ── Status colors ───────────────────────────────────────────────────── */
  .status-ok           { color: var(--ok); font-weight: 600; }
  .status-no_results   { color: var(--muted); }
  .status-error, .status-timeout { color: var(--err); font-weight: 600; }
  .status-unsupported  { color: var(--warn); }

  /* ── Alert boxes ─────────────────────────────────────────────────────── */
  .error-box {
    background: #FDF3F2;
    border: 1px solid #E8B4B0;
    border-left: 4px solid var(--err);
    border-radius: 3px;
    padding: 12px 16px;
    font-size: 0.88rem;
    color: var(--err);
  }
  .info-box {
    background: #EBF0FA;
    border: 1px solid #B8C9F2;
    border-left: 4px solid var(--primary);
    border-radius: 3px;
    padding: 12px 16px;
    font-size: 0.88rem;
    color: var(--muted);
  }

  /* ── Data table ──────────────────────────────────────────────────────── */
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 4px;
    overflow: hidden;
  }
  thead tr { background: #F0F3FA; }
  th {
    padding: 10px 13px;
    text-align: left;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .06em;
    color: var(--primary);
    border-bottom: 2px solid var(--border);
    white-space: nowrap;
  }
  td {
    padding: 9px 13px;
    border-bottom: 1px solid #EBEBEB;
    vertical-align: top;
    color: var(--text);
  }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: #F5F7FD; }
  .record-link { color: var(--teal); text-decoration: none; font-size: 0.8rem; font-weight: 500; }
  .record-link:hover { text-decoration: underline; color: var(--teal-dark); }

  /* ── Spinner / loading ───────────────────────────────────────────────── */
  .spinner {
    display: inline-block; width: 18px; height: 18px;
    border: 3px solid rgba(170,85,160,.25);
    border-top-color: var(--primary);
    border-radius: 50%;
    animation: spin .8s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .loading-msg { display: flex; align-items: center; gap: 10px; color: var(--muted); padding: 28px 0; }

  /* ── Footer ──────────────────────────────────────────────────────────── */
  footer {
    text-align: center;
    color: var(--muted);
    font-size: 0.78rem;
    padding: 28px 24px;
    border-top: 1px solid var(--border);
    margin-top: 32px;
  }

  /* ── Form extras ─────────────────────────────────────────────────────── */
  .checkbox-label { display: flex; align-items: center; gap: 6px; font-size: 0.86rem; cursor: pointer; color: var(--text); }
  @media (max-width: 640px) { .row { flex-direction: column; } }

  /* ── Combination groups ──────────────────────────────────────────────── */
  .combo-group { margin-bottom: 8px; border: 1px solid var(--border); border-radius: 4px; overflow: hidden; }
  .combo-header {
    display: flex; align-items: center; gap: 12px;
    padding: 11px 16px; background: #F0F3FA;
    cursor: pointer; list-style: none; user-select: none; flex-wrap: wrap;
  }
  .combo-header::-webkit-details-marker { display: none; }
  .combo-header::before {
    content: '▶'; font-size: 0.65rem; color: var(--primary);
    transition: transform .15s; min-width: 10px; display: inline-block;
  }
  details[open] > .combo-header::before { transform: rotate(90deg); }
  .combo-label { font-weight: 700; font-size: 0.92rem; color: var(--primary); }
  .combo-stats { font-size: 0.78rem; color: var(--muted); white-space: nowrap; }
  .combo-companies { display: flex; gap: 4px; flex-wrap: wrap; }
  .company-chip {
    background: #DDE4F5; border-radius: 2px;
    padding: 2px 8px; font-size: 0.72rem; color: var(--primary);
  }
  .company-chip.more { background: transparent; border: 1px solid var(--border); color: var(--muted); }
  .combo-body table { border-radius: 0; border: none; border-top: 1px solid var(--border); }

  /* ── Export progress panel ───────────────────────────────────────────── */
  .export-panel {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 18px 20px;
    margin-bottom: 20px;
    display: none;
    box-shadow: 0 1px 3px rgba(170,85,160,.08);
  }
  .export-panel-header {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 12px;
  }
  .export-stage-label { font-weight: 700; font-size: 0.95rem; color: var(--primary); }
  .export-stats { font-size: 0.8rem; color: var(--muted); }
  .progress-track {
    height: 8px; background: #DDE4F5; border-radius: 4px;
    overflow: hidden; margin-bottom: 12px;
  }
  .progress-fill {
    height: 100%; background: var(--primary); border-radius: 4px;
    transition: width 0.4s ease; width: 0%;
  }
  .progress-fill.error { background: var(--err); }
  .export-log {
    height: 130px; overflow-y: auto;
    font-size: 0.73rem;
    font-family: "SF Mono", "Fira Code", "Consolas", monospace;
    background: #0D1829; color: #C8D4E8;
    padding: 10px 12px; border-radius: 3px; line-height: 1.55;
  }
  .export-log .log-ok { color: #5AC3BE; }
  .export-log .log-err { color: #E07070; }
  .export-log .log-dim { color: #666; }

  /* ── Dashboard panel ─────────────────────────────────────────────────── */
  .dashboard-panel { display: none; margin-top: 28px; }
  .dashboard-header {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 18px; flex-wrap: wrap; gap: 8px;
    border-bottom: 2px solid var(--primary); padding-bottom: 10px;
  }
  .dashboard-title {
    font-family: "Exo", sans-serif;
    font-size: 1.1rem; font-weight: 700; color: var(--primary);
  }

  /* KPI cards */
  .kpi-row { display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 22px; }
  .kpi-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-top: 3px solid var(--primary);
    border-radius: 4px;
    padding: 16px 20px;
    flex: 1; min-width: 140px;
    box-shadow: 0 1px 3px rgba(170,85,160,.08);
  }
  .kpi-value {
    font-family: "Exo", sans-serif;
    font-size: 2rem; font-weight: 700;
    color: var(--primary); line-height: 1;
  }
  .kpi-label {
    font-size: 0.72rem; color: var(--muted); margin-top: 5px;
    text-transform: uppercase; letter-spacing: .07em;
  }

  /* Dashboard table */
  .dash-table-wrap {
    overflow-x: auto;
    border: 1px solid var(--border);
    border-radius: 4px;
    box-shadow: 0 1px 3px rgba(170,85,160,.06);
  }
  .dash-table-wrap table { min-width: 800px; border: none; border-radius: 0; }

  /* Dashboard tabs */
  .dash-tab-bar { display: flex; gap: 0; margin-bottom: 14px; }
  .dash-tab {
    padding: 7px 16px;
    border: 1px solid var(--border); border-bottom: none;
    background: var(--bg); cursor: pointer;
    font-size: 0.84rem; font-family: inherit; font-weight: 500;
    border-radius: 3px 3px 0 0; color: var(--muted);
    transition: background .12s, color .12s;
  }
  .dash-tab.active {
    background: var(--card); border-bottom: 2px solid var(--card);
    font-weight: 700; color: var(--primary);
  }

  /* Canary / debug box */
  .canary-box {
    background: #E8F5F0; border: 1px solid #A8D5BE;
    border-left: 4px solid var(--ok);
    border-radius: 3px;
    padding: 10px 14px; font-size: 0.8rem;
    font-family: "SF Mono", monospace;
    margin-bottom: 14px; white-space: pre-wrap; color: #1A4D36;
  }
</style>
</head>
<body>
<nav class="site-nav" role="navigation" aria-label="Site navigation">
  <div class="site-nav-brand-wrap">
    <span class="site-nav-brand">Zydus</span>
    <span class="site-nav-sub">Dedicated To Life</span>
  </div>
  <span class="site-nav-divider"></span>
  <a class="site-nav-link" href="https://health-products.canada.ca/drug-product-database/" target="_blank" rel="noopener">DPD</a>
  <a class="site-nav-link" href="https://health-products.canada.ca/noc/" target="_blank" rel="noopener">NOC</a>
  <a class="site-nav-link" href="https://pr-rdb.hc-sc.gc.ca/pr-rdb/" target="_blank" rel="noopener">Patent Register</a>
</nav>
<header role="banner">
  <div class="header-brand-row">
    <div>
      <div class="header-company-name">Zydus &nbsp;&mdash;&nbsp; Dedicated To Life</div>
      <h1>Drug Intelligence Platform</h1>
    </div>
  </div>
  <p>Simultaneous search across DPD &middot; Generic Submissions Under Review &middot; Notice of Compliance &middot; Patent Register</p>
</header>
<div class="container">
  <div class="search-box">
    <div class="row">
      <div class="field-group" style="flex:1">
        <label for="query">Ingredients <span id="queryCount" style="font-weight:400;color:var(--muted);font-size:0.72rem">&mdash; one per line or comma-separated; export runs all</span></label>
        <textarea id="query" rows="3" style="resize:vertical;min-width:300px;font-family:inherit;padding:9px 13px;border:1px solid var(--border);border-radius:3px;font-size:0.92rem;color:var(--text);background:#fff;transition:border-color .15s,box-shadow .15s;outline:none" placeholder="alpelisib&#10;apremilast&#10;abrocitinib&#10;or: alpelisib, apremilast" oninput="updateQueryCount()"></textarea>
        <div id="queryHint" style="font-size:0.72rem;color:var(--muted);margin-top:3px"></div>
      </div>
      </div>
      <div class="field-group">
        <label for="field">Search By</label>
        <select id="field">
          <option value="ingredient">Active Ingredient (default)</option>
          <option value="brand">Product / Brand Name</option>
          <option value="company">Company Name</option>
          <option value="din">DIN</option>
        </select>
      </div>
      <div class="field-group" style="justify-content:flex-end; gap:6px; margin-top:12px;">
        <div style="display:flex; gap:8px;">
          <button class="btn btn-primary" id="searchBtn" onclick="doSearch()">Search</button>
          <button class="btn btn-export" id="exportBtn" onclick="doExport()" disabled>⬇ Export XLSX</button>
          <button class="btn" style="background:#888;color:white;font-size:0.8em;padding:4px 10px;" onclick="resetAllCaches()" title="Clear all cached data and force fresh fetches from all sources">Reset cache</button>
        </div>
      </div>
    </div>
  </div>

  <!-- Export progress panel (shown while job runs) -->
  <div class="export-panel" id="exportPanel">
    <div class="export-panel-header">
      <span class="export-stage-label" id="exportStageLabel">Starting…</span>
      <span class="export-stats" id="exportStats"></span>
    </div>
    <div class="progress-track">
      <div class="progress-fill" id="progressFill"></div>
    </div>
    <div class="export-log" id="exportLog"></div>
  </div>

  <div class="ai-summary" id="aiSummary"></div>
  <div id="results"></div>

  <!-- Dashboard panel: shown after export completes; renders exact XLSX dataset -->
  <div class="dashboard-panel" id="dashboardPanel">
    <div class="dashboard-header">
      <span class="dashboard-title">📊 Enriched Dashboard — same dataset as XLSX</span>
      <span style="font-size:0.8rem;color:var(--muted)" id="dashMeta"></span>
    </div>
    <div class="canary-box" id="canaryBox"></div>
    <div class="kpi-row" id="kpiRow"></div>
    <div class="dash-tab-bar">
      <button class="dash-tab active" id="dashTab1" onclick="switchDashTab(1)">DPD + NOC + Patents</button>
      <button class="dash-tab" id="dashTab2" onclick="switchDashTab(2)">Generic Submissions</button>
    </div>
    <div id="dashPane1"><div class="dash-table-wrap" id="dashSheet1"></div></div>
    <div id="dashPane2" style="display:none"><div class="dash-table-wrap" id="dashSheet2"></div></div>
  </div>
</div>
<footer role="contentinfo">
  <strong style="color:var(--primary);font-family:'Exo',sans-serif">Zydus</strong> <span style="color:var(--muted);font-size:0.76rem">— Dedicated To Life</span>
  <br/>
  Data sourced from Canadian government public databases (DPD, NOC, Patent Register, GSUR). &nbsp;|&nbsp;
  Accuracy relies on deterministic extraction — no AI-generated data fields.
</footer>

<script>
const SOURCE_LABELS = {
  DPD: 'Drug Product Database (DPD)',
  GenericSubmissions: 'Generic Submissions Under Review',
  NOC: 'Notice of Compliance',
  PatentRegister: 'Patent Register',
};

const SOURCE_COLS = {
  Combined: [
    { key: 'source', label: 'Source' },
    { key: 'ingredient', label: 'Ingredient(s)' },
    { key: 'brand_name', label: 'Brand' },
    { key: 'company', label: 'Company' },
    { key: 'din', label: 'DIN' },
    { key: 'strength', label: 'Strength' },
    { key: 'dosage_form', label: 'Form' },
    { key: 'status', label: 'Status' },
  ],
  DPD: [
    { key: 'brand_name', label: 'Brand Name' },
    { key: 'ingredient', label: 'Ingredient(s)' },
    { key: 'company', label: 'Company' },
    { key: 'din', label: 'DIN' },
    { key: 'strength', label: 'Strength' },
    { key: 'dosage_form', label: 'Form' },
    { key: 'route', label: 'Route' },
    { key: 'status', label: 'Status' },
  ],
  GenericSubmissions: [
    { key: 'ingredient', label: 'Medicinal Ingredient(s)' },
    { key: 'company', label: 'Company' },
    { key: '_therapeutic_area', label: 'Therapeutic Area' },
    { key: '_date_accepted', label: 'Date Accepted' },
    { key: 'status', label: 'Status' },
  ],
  NOC: [
    { key: 'brand_name', label: 'Product(s)' },
    { key: 'ingredient', label: 'Medicinal Ingredient(s)' },
    { key: 'company', label: 'Manufacturer' },
    { key: 'din', label: 'DIN(s)' },
    { key: '_noc_date', label: 'NOC Date' },
    { key: 'status', label: 'NOC Type' },
  ],
  PatentRegister: [
    { key: 'ingredient', label: 'Medicinal Ingredient' },
    { key: 'brand_name', label: 'Brand Name' },
    { key: 'strength', label: 'Strength' },
    { key: 'dosage_form', label: 'Dosage Form' },
    { key: 'din', label: 'DIN' },
    { key: '_patent_number', label: 'Patent' },
    { key: '_csp', label: 'CSP' },
  ],
};

let lastResponse = null;
let currentJobId = null;
let currentEventSource = null;

// ---- Multi-ingredient query parsing ----

function parseQueries() {
  const raw = document.getElementById('query').value || '';
  // Split on newlines, then commas within each segment
  const parts = raw.split(/[\\n\\r]+/).flatMap(line => line.split(','));
  const seen = new Set();
  const result = [];
  for (const p of parts) {
    const trimmed = p.trim();
    if (!trimmed) continue;
    const key = trimmed.toLowerCase();
    if (!seen.has(key)) {
      seen.add(key);
      result.push(trimmed);
    }
  }
  return result;
}

function updateQueryCount() {
  const qs = parseQueries();
  const hint = document.getElementById('queryHint');
  if (qs.length === 0) {
    hint.textContent = '';
  } else if (qs.length === 1) {
    hint.textContent = `1 ingredient — Search previews it; Export runs it.`;
  } else {
    hint.textContent = `${qs.length} ingredients: ${qs.map(q => '“' + q + '”').join(', ')} — Search previews the first; Export runs all side-by-side.`;
  }
}

function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function fieldVal(record, key) {
  if (key.startsWith('_')) return (record.source_specific || {})[key.slice(1)] || '';
  return record[key] || '';
}

// ---- Combination grouping helpers ----

function normalizeIngName(s) {
  return s.trim().toUpperCase().replace(/\s+/g, ' ');
}

function makeGroupKey(record) {
  let names;
  if (record.all_ingredients && record.all_ingredients.length > 0) {
    names = record.all_ingredients.map(normalizeIngName).filter(Boolean);
  } else if (record.ingredient) {
    names = record.ingredient.split(/\s*;\s*/).map(normalizeIngName).filter(Boolean);
  } else {
    return 'UNKNOWN / UNPARSED';
  }
  names = [...new Set(names)].sort();
  return names.length ? names.join(' + ') : 'UNKNOWN / UNPARSED';
}

function groupAndSort(records, searchedIngredient) {
  const map = {};
  for (const rec of records) {
    const key = makeGroupKey(rec);
    if (!map[key]) map[key] = [];
    map[key].push(rec);
  }
  const searchKey = searchedIngredient ? normalizeIngName(searchedIngredient) : null;
  return Object.entries(map)
    .map(([label, recs]) => ({ label, records: recs }))
    .sort((a, b) => {
      const ae = searchKey && a.label === searchKey ? 0 : 1;
      const be = searchKey && b.label === searchKey ? 0 : 1;
      if (ae !== be) return ae - be;
      if (b.records.length !== a.records.length) return b.records.length - a.records.length;
      return a.label.localeCompare(b.label);
    });
}

// ---- Table + grouped-view builders ----

function buildTable(source, records) {
  const cols = SOURCE_COLS[source] || SOURCE_COLS.Combined;
  let html = '<table><thead><tr>';
  cols.forEach(c => { html += `<th>${c.label}</th>`; });
  html += '<th>Link</th></tr></thead><tbody>';
  records.forEach(r => {
    html += '<tr>';
    cols.forEach(c => {
      let v;
      if (c.key === 'source') {
        v = `<span class="badge badge-${r.source}" style="font-size:.7rem">${r.source}</span>`;
      } else {
        v = fieldVal(r, c.key);
        v = v ? escHtml(v) : '<span style="color:#aaa">—</span>';
      }
      html += `<td>${v}</td>`;
    });
    const link = r.record_url
      ? `<a class="record-link" href="${escHtml(r.record_url)}" target="_blank" rel="noopener">View ↗</a>`
      : '—';
    html += `<td>${link}</td></tr>`;
  });
  html += '</tbody></table>';
  return html;
}

function buildGroupedView(source, records, searchedIngredient) {
  if (!records.length) return '<div class="info-box">No results.</div>';
  const groups = groupAndSort(records, searchedIngredient);
  return groups.map((g, i) => {
    const companies = [...new Set(g.records.map(r => r.company).filter(Boolean))].sort();
    const chips = companies.slice(0, 6).map(c => `<span class="company-chip">${escHtml(c)}</span>`).join('')
      + (companies.length > 6 ? `<span class="company-chip more">+${companies.length - 6} more</span>` : '');
    return `<details class="combo-group"${i === 0 ? ' open' : ''}>
<summary class="combo-header">
  <span class="combo-label">${escHtml(g.label)}</span>
  <span class="combo-stats">${g.records.length} product${g.records.length !== 1 ? 's' : ''} &middot; ${companies.length} compan${companies.length !== 1 ? 'ies' : 'y'}</span>
  <div class="combo-companies">${chips}</div>
</summary>
<div class="combo-body">${buildTable(source, g.records)}</div>
</details>`;
  }).join('');
}

function buildSourcePane(src, searchedIngredient) {
  const statusClass = `status-${src.status}`;
  let content = '';
  if (src.status === 'ok' && src.records.length) {
    content = buildGroupedView(src.source, src.records, searchedIngredient);
  } else if (src.status === 'no_results') {
    content = '<div class="info-box">No results found in this database for your query.</div>';
  } else if (src.status === 'error' || src.status === 'timeout') {
    content = `<div class="error-box"><strong>${src.status === 'timeout' ? 'Timeout' : 'Error'}:</strong> ${escHtml(src.error_message || 'Unknown error')}</div>`;
  } else if (src.status === 'unsupported') {
    content = `<div class="info-box">ℹ️ ${escHtml(src.error_message || 'This source does not support the selected search field.')}</div>`;
  }
  return `<div class="source-header">
    <span class="badge badge-${src.source}">${SOURCE_LABELS[src.source] || src.source}</span>
    <span class="${statusClass}">${src.count > 0 ? src.count + ' result' + (src.count !== 1 ? 's' : '') : src.status}</span>
  </div>${content}`;
}

function buildCombinedPane(sources, searchedIngredient) {
  const allRecords = sources.flatMap(s => s.records || []);
  if (!allRecords.length) return '<div class="info-box">No combined results.</div>';
  return buildGroupedView('Combined', allRecords, searchedIngredient);
}

function render(data) {
  lastResponse = data;
  const { metadata, sources, ai_summary } = data;
  const searchedIngredient = metadata.field === 'ingredient' ? metadata.query : null;

  // Build status bar with full inline styles — no separate CSS class needed
  const counts = sources.map(s => `${SOURCE_LABELS[s.source]||s.source}: <strong>${s.count}</strong>`).join(' &nbsp;|&nbsp; ');
  let barText = `Searched for <strong>&ldquo;${escHtml(metadata.query)}&rdquo;</strong> by <em>${escHtml(metadata.field)}</em> &nbsp;&middot;&nbsp; ${counts}`;
  if (metadata.normalized_terms?.length > 1) {
    barText += ` &nbsp;&middot;&nbsp; <em>Also searched: ${metadata.normalized_terms.slice(1).map(escHtml).join(', ')}</em>`;
  }
  const _barHtml = `<div style="background:#fff3cd;border:1px solid #ffc107;border-left:6px solid #e6a500;border-radius:4px;padding:13px 18px;margin-bottom:20px;font-size:0.93rem;font-weight:500;color:#4a3600;box-shadow:0 2px 8px rgba(230,165,0,.2)">${barText}</div>`;

  const aiDiv = document.getElementById('aiSummary');
  if (ai_summary) {
    aiDiv.innerHTML = `<strong>🤖 AI Summary</strong> <em style="font-size:.75rem;color:#666">(AI-generated, may be imprecise — verify against raw data)</em><br/>${ai_summary}`;
    aiDiv.style.display = 'block';
  } else {
    aiDiv.style.display = 'none';
  }

  const tabSources = ['Combined', ...sources.map(s => s.source)];
  const tabLabels = { Combined: 'Combined', ...Object.fromEntries(sources.map(s=>[s.source, SOURCE_LABELS[s.source]||s.source])) };

  let tabsHtml = '<div class="tabs">';
  tabSources.forEach((t, i) => {
    tabsHtml += `<button class="tab-btn${i===0?' active':''}" onclick="switchTab('${t}')" id="tab-${t}">${tabLabels[t]}</button>`;
  });
  tabsHtml += '</div>';

  let panesHtml = '';
  tabSources.forEach((t, i) => {
    const content = t === 'Combined'
      ? buildCombinedPane(sources, searchedIngredient)
      : buildSourcePane(sources.find(s=>s.source===t), searchedIngredient);
    panesHtml += `<div class="tab-pane${i===0?' active':''}" id="pane-${t}">${content}</div>`;
  });

  document.getElementById('results').innerHTML = _barHtml + tabsHtml + panesHtml;
  document.getElementById('exportBtn').disabled = false;
}

function switchTab(name) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
  document.getElementById(`tab-${name}`)?.classList.add('active');
  document.getElementById(`pane-${name}`)?.classList.add('active');
}

async function doSearch() {
  const queries = parseQueries();
  if (!queries.length) { alert('Please enter at least one ingredient.'); return; }
  // Search previews the FIRST ingredient; multi-product export runs all.
  const q = queries[0];
  const field = document.getElementById('field').value;
  if (queries.length > 1) {
    // Show a brief notice that only the first is being previewed
    document.getElementById('queryHint').innerHTML =
      `<span style="color:var(--warn)">Previewing <strong>"${escHtml(q)}"</strong> only &mdash; Export will run all ${queries.length} ingredients side-by-side.</span>`;
  }

  document.getElementById('searchBtn').disabled = true;
  document.getElementById('exportBtn').disabled = true;
  document.getElementById('aiSummary').style.display = 'none';
  document.getElementById('results').innerHTML = '<div class="loading-msg"><div class="spinner"></div>Querying all four databases concurrently…</div>';

  // Close any in-progress export
  _closeExport();

  try {
    const url = `/api/search?q=${encodeURIComponent(q)}&field=${field}&summary=false`;
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    render(data);
  } catch (e) {
    document.getElementById('results').innerHTML = `<div class="error-box">Search failed: ${e.message}</div>`;
  } finally {
    document.getElementById('searchBtn').disabled = false;
  }
}

// ---- Async export with SSE progress ----

function _closeExport() {
  if (currentEventSource) { currentEventSource.close(); currentEventSource = null; }
  currentJobId = null;
}

function _appendLog(line, cls) {
  const log = document.getElementById('exportLog');
  const div = document.createElement('div');
  div.className = cls || '';
  div.textContent = line;
  log.appendChild(div);
  // Keep last 50 lines
  while (log.children.length > 50) log.removeChild(log.firstChild);
  log.scrollTop = log.scrollHeight;
}

function _handleExportEvent(data) {
  const fill = document.getElementById('progressFill');
  const stageLabel = document.getElementById('exportStageLabel');
  const stats = document.getElementById('exportStats');

  if (data.status === 'complete') {
    fill.style.width = '100%';
    fill.classList.remove('error');
    stageLabel.textContent = `✓ Done in ${data.elapsed_s}s`;
    stats.textContent = '';
    _appendLog(`✓ ${data.log}`, 'log-ok');
    // Auto-trigger download
    const a = document.createElement('a');
    a.href = data.download_url;
    a.style.display = 'none';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    document.getElementById('exportBtn').disabled = false;
    // Load dashboard from the same job's snapshot — no re-scraping
    _loadDashboard(currentJobId);
    _closeExport();
    return;
  }

  if (data.status === 'error') {
    fill.classList.add('error');
    stageLabel.textContent = `✗ Error`;
    stats.textContent = data.elapsed_s ? `${data.elapsed_s}s elapsed` : '';
    _appendLog(`✗ ERROR: ${data.message}`, 'log-err');
    document.getElementById('exportBtn').disabled = false;
    _closeExport();
    return;
  }

  // Progress event
  const pct = Math.round((data.pct || 0) * 100);
  fill.style.width = `${pct}%`;
  stageLabel.textContent = `${data.stage}: ${data.done}/${data.total}`;
  const etaStr = data.eta_s != null ? ` · ETA ${data.eta_s}s` : '';
  stats.textContent = `${pct}% · ${data.elapsed_s}s elapsed${etaStr}`;
  if (data.log) _appendLog(`[${data.stage}] ${data.log}`);
}

async function resetAllCaches() {
  if (!confirm('Clear all cached data? The next search will re-fetch live data from all sources.')) return;
  const resp = await fetch('/api/reset-all-caches', {method: 'POST'});
  const data = await resp.json();
  alert(`Cache cleared — ${data.http_rows_cleared} search results, ${data.patent_rows_cleared} patent records, ${data.labeling_rows_cleared} labeling records removed.`);
}

async function doExport() {
  const queries = parseQueries();
  if (!queries.length) return;
  const field = document.getElementById('field').value;

  // Disable export button, show panel
  document.getElementById('exportBtn').disabled = true;
  const panel = document.getElementById('exportPanel');
  panel.style.display = 'block';
  document.getElementById('exportLog').innerHTML = '';
  document.getElementById('progressFill').style.width = '0%';
  document.getElementById('progressFill').classList.remove('error');
  const label = queries.length === 1
    ? `Exporting "${queries[0]}"…`
    : `Exporting ${queries.length} ingredients side-by-side…`;
  document.getElementById('exportStageLabel').textContent = label;
  document.getElementById('exportStats').textContent = '';

  _closeExport();

  // Start the job
  let jobId;
  try {
    const resp = await fetch('/export/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ queries, field, allow_partial: false, enable_ocr: true }),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${resp.status}`);
    }
    const data = await resp.json();
    jobId = data.job_id;
    currentJobId = jobId;
  } catch (e) {
    document.getElementById('exportStageLabel').textContent = `✗ Error: ${e.message}`;
    document.getElementById('progressFill').classList.add('error');
    document.getElementById('exportBtn').disabled = false;
    return;
  }

  // Open SSE stream
  const es = new EventSource(`/export/stream/${jobId}`);
  currentEventSource = es;

  es.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      _handleExportEvent(data);
    } catch (e) {
      _appendLog(`[parse error] ${event.data}`, 'log-err');
    }
  };

  es.onerror = () => {
    if (currentJobId !== jobId) return; // stale
    _appendLog('Connection lost — check server', 'log-err');
    document.getElementById('exportStageLabel').textContent = 'Connection error';
    document.getElementById('exportBtn').disabled = false;
    es.close();
  };
}

// Ctrl+Enter (or Cmd+Enter) submits search from the textarea
document.getElementById('query').addEventListener('keydown', e => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) doSearch();
});

// ---- Dashboard: consume exact XLSX dataset from job snapshot (no re-scraping) ----

function _dashTable(columns, records) {
  if (!records.length) return '<div class="info-box">No data.</div>';
  let html = '<table><thead><tr>';
  columns.forEach(c => { html += `<th>${escHtml(c)}</th>`; });
  html += '</tr></thead><tbody>';
  records.forEach(row => {
    html += '<tr>';
    columns.forEach(c => {
      const v = row[c];
      html += `<td>${v != null && v !== '' ? escHtml(String(v)) : '<span style="color:#aaa">—</span>'}</td>`;
    });
    html += '</tr>';
  });
  html += '</tbody></table>';
  return html;
}

function _kpiCards(sheet1Records, sheet1Cols) {
  const total = sheet1Records.length;
  const withPatents = sheet1Records.filter(r => r['patent_count'] && Number(r['patent_count']) > 0).length;
  const hasDP = sheet1Cols.includes('data_protection_ends');
  const underDP = hasDP
    ? sheet1Records.filter(r => r['data_protection_ends'] && String(r['data_protection_ends']).trim() !== '' && String(r['data_protection_ends']) !== 'None').length
    : 0;
  const hasPM = sheet1Records.filter(r => {
    const v = r['active_ingredient'];
    return v && v !== 'Not stated' && v !== 'No PM available' && v !== 'Not in PM' && String(v).trim() !== '';
  }).length;

  const cards = [
    { label: 'Total DINs', value: total },
    { label: 'With Patents', value: withPatents },
    { label: 'Under Data Protection', value: underDP },
    { label: 'With PM Labeling Data', value: hasPM },
  ];
  return cards.map(c =>
    `<div class="kpi-card"><div class="kpi-value">${c.value}</div><div class="kpi-label">${c.label}</div></div>`
  ).join('');
}

function switchDashTab(n) {
  document.getElementById('dashTab1').classList.toggle('active', n === 1);
  document.getElementById('dashTab2').classList.toggle('active', n === 2);
  document.getElementById('dashPane1').style.display = n === 1 ? '' : 'none';
  document.getElementById('dashPane2').style.display = n === 2 ? '' : 'none';
}

async function _loadDashboard(jobId) {
  if (!jobId) return;
  const panel = document.getElementById('dashboardPanel');
  panel.style.display = 'block';
  document.getElementById('dashSheet1').innerHTML = '<div class="loading-msg"><div class="spinner"></div>Loading dashboard from XLSX dataset…</div>';
  document.getElementById('kpiRow').innerHTML = '';
  document.getElementById('canaryBox').textContent = '';

  try {
    const resp = await fetch(`/api/export-data/${jobId}`);
    if (!resp.ok) { throw new Error(`HTTP ${resp.status}`); }
    const data = await resp.json();

    const s1 = data.sheet1;
    const s2 = data.sheet2;

    // KPI cards
    document.getElementById('kpiRow').innerHTML = _kpiCards(s1.records, s1.columns);

    // Canary comparison log
    const qs = data.queries || [data.query];
    const queryStr = qs.length === 1 ? `"${data.query}"` : `${qs.length} products: ${qs.map(q => '"' + q + '"').join(', ')}`;
    const canary = [
      `✓ Dashboard loaded from job snapshot — NO new outbound requests`,
      `  Sheet 1: ${s1.records.length} rows × ${s1.columns.length} columns`,
      `  Sheet 2: ${s2.records.length} rows × ${s2.columns.length} columns`,
      `  ${queryStr} by ${data.field}`,
      `  Columns: ${s1.columns.join(', ')}`,
    ].join('\\n');
    document.getElementById('canaryBox').textContent = canary;

    document.getElementById('dashMeta').textContent =
      `Job ${jobId.slice(0,8)}… · ${queryStr} by ${data.field}`;

    // Render tables
    document.getElementById('dashSheet1').innerHTML = _dashTable(s1.columns, s1.records);
    document.getElementById('dashSheet2').innerHTML = _dashTable(s2.columns, s2.records);

    console.log('[Dashboard canary]', {
      sheet1_rows: s1.records.length,
      sheet1_cols: s1.columns.length,
      sheet2_rows: s2.records.length,
      job_id: jobId,
      note: 'No new scraping — data read from job.sheet1_records (same as XLSX)',
    });
  } catch (e) {
    document.getElementById('dashSheet1').innerHTML =
      `<div class="error-box">Dashboard load failed: ${escHtml(e.message)}</div>`;
  }
}
</script>
</body>
</html>
"""
