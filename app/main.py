"""Zydus Drug Intelligence Platform — FastAPI main application."""
from __future__ import annotations

import asyncio
import json
import os
import pickle
import sys
import uuid
from datetime import datetime, timezone
from typing import Optional

# Windows consoles default to cp1252, which cannot encode characters such as
# "->"/check marks used in the export-pipeline progress prints. Force UTF-8 so
# those print() calls don't raise UnicodeEncodeError and crash an export job.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
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

# ── IQVIA upload store ────────────────────────────────────────────────────────
# Maps token → collapsed IQVIA DataFrame (one row per product group).
# "_persisted" is a special key that survives server restarts via disk pickle.
_IQVIA_STORE: dict[str, "pd.DataFrame"] = {}
_IQVIA_PERSIST_KEY = "_persisted"
_IQVIA_PERSIST_PATH = os.path.join(CACHE_DIR, "iqvia_collapsed.pkl")


@app.on_event("startup")
async def _load_persisted_iqvia() -> None:
    """Auto-load the last uploaded IQVIA file from disk so it survives server restarts."""
    if os.path.exists(_IQVIA_PERSIST_PATH):
        try:
            with open(_IQVIA_PERSIST_PATH, "rb") as fh:
                _IQVIA_STORE[_IQVIA_PERSIST_KEY] = pickle.load(fh)
        except Exception:
            pass  # corrupt pickle — ignore; user can re-upload

# CORS — lets Power BI Service, Fabric notebooks, and other browser clients call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Raise the upload body limit to 100 MB (default is 1 MB, too small for IQVIA Excel files).
_MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB

@app.middleware("http")
async def _limit_upload_size(request: Request, call_next):
    # /api/iqvia/compare carries TWO extracts, so it gets twice the per-file cap.
    _upload_paths = {"/api/iqvia/upload": _MAX_UPLOAD_BYTES,
                     "/api/iqvia/compare": 2 * _MAX_UPLOAD_BYTES}
    limit = _upload_paths.get(request.url.path) if request.method == "POST" else None
    if limit is not None:
        cl = request.headers.get("content-length")
        if cl and int(cl) > limit:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=413,
                content={"detail": f"Upload too large. The maximum total size is {limit // (1024 * 1024)} MB."},
            )
    return await call_next(request)

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
                f"Source(s) failed: {names}. Refusing to build a partial workbook. "
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
    """Clear all cached data: HTTP cache, patents, labeling, and full universe."""
    from app.enrichment.universe import reset_universe_cache
    http_rows = cache_clear_all()
    patent_rows = reset_patents_table()
    labeling_rows = reset_labeling_table()
    universe_cleared = reset_universe_cache()
    return {
        "status": "ok",
        "http_rows_cleared": http_rows,
        "patent_rows_cleared": patent_rows,
        "labeling_rows_cleared": labeling_rows,
        "universe_cleared": universe_cleared,
    }


@app.get("/api/dosage-forms")
async def dosage_forms() -> dict:
    """Base dosage-form list for the filter dropdown (used by BOTH tabs).

    Sourced from the cached full-universe build (base→raw map on the bundle), so it
    rides the same 4-hour freshness and reset-all-caches invalidation. The first
    call may trigger the allfiles.zip pull; subsequent calls return instantly.
    """
    from app.enrichment.universe import get_universe
    bundle = await get_universe()
    return {"base_forms": sorted(bundle.dosage_form_map.keys())}


# ── IQVIA upload ──────────────────────────────────────────────────────────────

@app.post("/api/iqvia/upload")
async def iqvia_upload(request: Request) -> dict:
    """Upload an IQVIA Canada Excel file and return a session token.

    The file is parsed immediately: the 'data' sheet is read, metric columns
    are detected by pattern, and rows are collapsed to one per (molecule,
    product, manufacturer, strength) by summing across channel/province/pack.

    The collapsed DataFrame is stored in-process under the returned token.
    Pass the token as ``iqvia_token`` in ``/export/start`` to attach IQVIA
    metrics to the exported workbook.

    Tokens are cleared on server restart.

    Note: reads form data directly with a raised max_part_size to bypass
    Starlette's default 1 MB multipart limit — IQVIA Excel files are 5–30 MB.
    """
    from app.enrichment.iqvia import parse_iqvia, collapse_iqvia, detect_metric_columns

    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" not in content_type:
        raise HTTPException(400, "Expected a multipart/form-data upload (select an .xlsx file)")

    form = await request.form(max_files=1, max_fields=0, max_part_size=_MAX_UPLOAD_BYTES)
    file_field = form.get("file")
    if file_field is None or not hasattr(file_field, "read"):
        raise HTTPException(400, "No file field in upload. The field must be named 'file'.")

    filename = getattr(file_field, "filename", "") or ""
    if not filename.lower().endswith((".xlsx", ".xls")):
        raise HTTPException(400, "File must be an Excel file (.xlsx or .xls)")

    content = await file_field.read()
    try:
        raw_df = parse_iqvia(content)
    except Exception as exc:
        raise HTTPException(422, f"Could not parse IQVIA file: {exc}") from exc

    metric_cols = detect_metric_columns(raw_df)
    if not metric_cols:
        raise HTTPException(
            422,
            "No metric columns found (expected 'Dollars MAT MM/YYYY', "
            "'Units MAT MM/YYYY', 'Ext Units MAT MM/YYYY'). "
            "Make sure you are uploading the 'data' sheet, not a Pivot sheet.",
        )

    collapsed = collapse_iqvia(raw_df)
    token = uuid.uuid4().hex
    _IQVIA_STORE[token] = collapsed
    _IQVIA_STORE[_IQVIA_PERSIST_KEY] = collapsed
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(_IQVIA_PERSIST_PATH, "wb") as fh:
            pickle.dump(collapsed, fh)
    except Exception:
        pass  # disk write failure is non-fatal; in-memory token still works

    molecules = sorted(collapsed["Combined Molecule"].dropna().unique().tolist()) if "Combined Molecule" in collapsed.columns else []
    return {
        "token": token,
        "raw_rows": len(raw_df),
        "collapsed_groups": len(collapsed),
        "metric_columns": metric_cols,
        "molecules": molecules,
        "status": "ok",
    }


# ── IQVIA quarter-over-quarter comparison ─────────────────────────────────────

async def _read_compare_file(form, field_name: str, label: str) -> bytes:
    """Pull one named file part from the compare upload form, or 400."""
    f = form.get(field_name)
    if f is None or not hasattr(f, "read"):
        raise HTTPException(400, f"Missing the {label} file (form field '{field_name}').")
    name = (getattr(f, "filename", "") or "").lower()
    if not name.endswith((".xlsx", ".xls", ".csv")):
        raise HTTPException(400, f"The {label} file must be .xlsx, .xls, or .csv (got '{name or 'unnamed'}').")
    return await f.read()


@app.post("/api/iqvia/compare")
async def iqvia_compare(request: Request) -> Response:
    """Compare an older and a newer IQVIA extract; return a changes-only XLSX.

    Multipart form with two file parts: ``old_file`` and ``new_file`` (each xlsx,
    xls, or csv).  The workbook has four sheets — Summary, New Entrants, Exits,
    Material Moves — at the platform's canonical product grain.  Read the body
    directly with a raised max_part_size so the two 5–30 MB extracts get through
    Starlette's default 1 MB multipart limit.
    """
    from app.enrichment.iqvia_diff import compare_iqvia, build_diff_workbook

    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" not in content_type:
        raise HTTPException(400, "Expected a multipart/form-data upload with 'old_file' and 'new_file'.")

    form = await request.form(max_files=2, max_fields=0, max_part_size=_MAX_UPLOAD_BYTES)
    old_bytes = await _read_compare_file(form, "old_file", "older")
    new_bytes = await _read_compare_file(form, "new_file", "newer")

    try:
        diff = compare_iqvia(old_bytes, new_bytes)
        xlsx = build_diff_workbook(diff)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(422, f"Could not compare IQVIA files: {exc}") from exc

    fname = f"iqvia_changes_{_period_tag(diff.old_period)}_to_{_period_tag(diff.new_period)}.xlsx"
    # Surface the auto-reorder decision + resolved periods to the browser so the
    # in-page UI can render the same older->newer notice the Excel Summary carries
    # (the workbook banner is unchanged). Header values are kept ASCII (latin-1) and
    # exposed for the cross-origin case; same-origin reads need no expose list.
    headers = {
        "Content-Disposition": f'attachment; filename="{fname}"',
        "X-IQVIA-Reordered": "true" if diff.reordered else "false",
        "X-IQVIA-Old-Period": _period_tag(diff.old_period),
        "X-IQVIA-New-Period": _period_tag(diff.new_period),
        "Access-Control-Expose-Headers": (
            "Content-Disposition, X-IQVIA-Reordered, X-IQVIA-Old-Period, "
            "X-IQVIA-New-Period, X-IQVIA-Warning"
        ),
    }
    if diff.warnings:
        headers["X-IQVIA-Warning"] = " | ".join(diff.warnings)
    return Response(
        content=xlsx,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers=headers,
    )


def _period_tag(period) -> str:
    """'YYYY-MM' tag for filenames, or 'NA'."""
    return f"{period[0]}-{period[1]:02d}" if period else "NA"


@app.get("/iqvia-compare", response_class=HTMLResponse)
async def iqvia_compare_page() -> HTMLResponse:
    return HTMLResponse(content=_IQVIA_COMPARE_HTML)


# ── Async export: start / stream / result ─────────────────────────────────────

class ExportStartRequest(BaseModel):
    q: str = ""                  # single-query backward compat
    queries: list[str] = []      # multi-product list (preferred)
    field: str = "ingredient"
    allow_partial: bool = False
    enable_ocr: bool = True
    iqvia_token: Optional[str] = None  # token from /api/iqvia/upload
    debug_iqvia_rows: bool = False     # append "IQVIA Source Rows (debug)" column to Sheet 1
    # Go/no-go screening criteria; each {metric, operator, value}. When non-empty,
    # the job ALSO produces a filtered Summary+Detail workbook over the built data.
    filter_criteria: list[dict] = []


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
        raise HTTPException(400, "No query provided. Set q or queries.")
    job_id = uuid.uuid4().hex
    job = create_job(
        job_id, qs[0], req.field, queries=qs,
        iqvia_token=req.iqvia_token,
        debug_iqvia_rows=req.debug_iqvia_rows,
        filter_criteria=req.filter_criteria,
    )
    asyncio.create_task(
        run_export_job(job, req.allow_partial, req.enable_ocr)
    )
    return {"job_id": job_id, "queries": qs}


# ── Full-universe tab (options 3 & 4) — additive, separate from the export path ─

class UniverseStartRequest(BaseModel):
    mode: str = "full"                 # "full" (option 3) | "filter_enrich" (option 4)
    enable_ocr: bool = True
    iqvia_token: Optional[str] = None
    debug_iqvia_rows: bool = False
    filter_criteria: list[dict] = []   # six-criteria; required for filter_enrich


@app.post("/universe/start")
async def universe_start(req: UniverseStartRequest) -> dict:
    """Start a Full-universe job (no-PDF universe, or filter-then-enrich survivors).

    Reuses the same job store + SSE stream + result endpoints as /export/start, so
    progress and download work identically.  This path never touches the
    single/multi-ingredient export pipeline.
    """
    from app.universe_job import run_universe_full_job, run_universe_filter_enrich_job

    if req.mode not in ("full", "filter_enrich"):
        raise HTTPException(400, "mode must be 'full' or 'filter_enrich'")
    if req.mode == "filter_enrich" and not req.filter_criteria:
        raise HTTPException(400, "filter_enrich requires at least one filter criterion")

    job_id = uuid.uuid4().hex
    job = create_job(
        job_id, "Full universe", "ingredient",
        iqvia_token=req.iqvia_token,
        debug_iqvia_rows=req.debug_iqvia_rows,
        filter_criteria=req.filter_criteria,
    )
    if req.mode == "full":
        asyncio.create_task(run_universe_full_job(job))
    else:
        asyncio.create_task(run_universe_filter_enrich_job(job, req.enable_ocr))
    return {"job_id": job_id, "mode": req.mode}


@app.get("/api/universe/status")
async def universe_status() -> dict:
    """Report whether a fresh (≤4 h) universe build is cached this session."""
    from app.enrichment.universe import universe_cache_status
    return universe_cache_status()


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


@app.get("/export/filtered-result/{job_id}")
async def export_filtered_result(job_id: str) -> FileResponse:
    """Download the finished filtered (go/no-go screened) XLSX."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job.status != "complete":
        raise HTTPException(409, f"Job not complete (status={job.status})")
    if not job.filtered_result_path:
        raise HTTPException(404, "No filtered workbook for this job (no criteria provided)")
    qs = job.queries or [job.query]
    if len(qs) == 1:
        filename = f"filtered_{qs[0].replace(' ', '_')}_{job.field}.xlsx"
    else:
        filename = f"filtered_{len(qs)}_products_{job.field}.xlsx"
    return FileResponse(
        path=job.filtered_result_path,
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
        raise HTTPException(409, "Job still running. Wait for the SSE complete event.")
    if job.status == "error":
        raise HTTPException(422, f"Job failed: {job.error}")
    result: dict = {
        "query": job.query,
        "queries": job.queries or [job.query],
        "field": job.field,
        "sheet1": {"columns": job.sheet1_columns, "records": job.sheet1_records},
        "sheet2": {"columns": job.sheet2_columns, "records": job.sheet2_records},
    }
    if job.recon_columns:
        result["reconciliation"] = {
            "columns": job.recon_columns,
            "records": job.recon_records,
        }
    if job.summary_columns:
        result["summary"] = {
            "columns": job.summary_columns,
            "records": job.summary_records,
        }
    return result


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse(content=_HTML_UI)


# ── Embedded IQVIA comparison page ─────────────────────────────────────────────
_IQVIA_COMPARE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>IQVIA Compare · Zydus Drug Intelligence Platform</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Exo:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet"/>
<style>
  :root {
    --primary:#aa55a0; --primary-dark:#7d3b75; --text:#2a2a33; --muted:#6b6b78;
    --border:#e2dce0; --bg:#faf7f9; --card:#fff; --ok:#2e9e6b; --warn:#c47f17;
  }
  * { box-sizing:border-box; }
  body { font-family:"Inter",system-ui,sans-serif; color:var(--text); background:var(--bg); margin:0; }
  .site-nav { display:flex; align-items:center; gap:14px; padding:10px 28px; background:var(--primary-dark); color:#fff; font-size:0.86rem; }
  .nav-zydus-logo { height:26px; width:auto; flex-shrink:0; filter:brightness(0) invert(1); opacity:0.92; }
  .site-nav a { color:#f4e8f2; text-decoration:none; }
  .site-nav a:hover { text-decoration:underline; }
  .site-nav-brand { font-family:"Exo",sans-serif; font-weight:700; font-size:1.05rem; }
  .site-nav-sub { font-size:0.7rem; opacity:.75; margin-left:6px; }
  .site-nav-divider { width:1px; height:18px; background:rgba(255,255,255,.3); }
  header { padding:26px 28px 10px; }
  header h1 { font-family:"Exo",sans-serif; color:var(--primary-dark); margin:2px 0 4px; font-size:1.7rem; }
  header p { color:var(--muted); margin:0; font-size:0.9rem; max-width:760px; }
  .container { max-width:880px; margin:18px auto 60px; padding:0 28px; }
  .card { background:var(--card); border:1px solid var(--border); border-top:3px solid var(--primary); border-radius:5px; padding:22px 24px; box-shadow:0 1px 3px rgba(170,85,160,.08); }
  .slots { display:flex; gap:20px; flex-wrap:wrap; }
  .slot { flex:1; min-width:260px; border:1px dashed var(--border); border-radius:5px; padding:16px; background:#fdfbfd; }
  .slot h3 { margin:0 0 4px; font-size:0.96rem; color:var(--primary-dark); font-family:"Exo",sans-serif; }
  .slot p { margin:0 0 11px; font-size:0.76rem; color:var(--muted); }
  .slot input[type=file] { font-size:0.8rem; border:1px solid var(--border); border-radius:3px; padding:6px; background:#fff; width:100%; cursor:pointer; }
  .actions { margin-top:20px; display:flex; align-items:center; gap:14px; flex-wrap:wrap; }
  .btn { font-family:inherit; font-size:0.92rem; font-weight:600; border:none; border-radius:4px; padding:11px 22px; cursor:pointer; }
  .btn-primary { background:var(--primary); color:#fff; }
  .btn-primary:disabled { background:#c9b7c6; cursor:not-allowed; }
  #status { font-size:0.84rem; color:var(--muted); min-height:1.2em; }
  #status.err { color:#b3261e; }
  #status.ok { color:var(--ok); }
  .note { margin-top:18px; font-size:0.78rem; color:var(--muted); line-height:1.55; border-top:1px solid var(--border); padding-top:14px; }
  .note b { color:var(--text); }
  .spinner { display:inline-block; width:14px; height:14px; border:2px solid var(--border); border-top-color:var(--primary); border-radius:50%; animation:spin .7s linear infinite; vertical-align:-2px; margin-right:6px; }
  @keyframes spin { to { transform:rotate(360deg); } }
</style>
</head>
<body>
<nav class="site-nav" role="navigation">
  <img class="nav-zydus-logo" src="/static/zydus_logo.png" alt="Zydus" />
  <span class="site-nav-brand">Zydus</span><span class="site-nav-sub">Dedicated To Life</span>
  <span class="site-nav-divider"></span>
  <a href="/">&larr; Search Platform</a>
</nav>
<header>
  <h1>IQVIA Quarter-over-Quarter Compare</h1>
  <p>Upload last quarter's IQVIA Canada file and this quarter's. You get an Excel of <b>only what changed</b>: new products, products that dropped off, and big moves in dollars or units. The two files can use different column names or date formats; we find the latest period inside each one. (.xlsx and .csv both work.)</p>
</header>
<div class="container">
  <div class="card">
    <div class="slots">
      <div class="slot">
        <h3>1 · Older extract</h3>
        <p>The previous pull (e.g. last quarter).</p>
        <input type="file" id="oldFile" accept=".xlsx,.xls,.csv" onchange="refresh()"/>
      </div>
      <div class="slot">
        <h3>2 · Newer extract</h3>
        <p>The current pull (e.g. this quarter).</p>
        <input type="file" id="newFile" accept=".xlsx,.xls,.csv" onchange="refresh()"/>
      </div>
    </div>
    <div class="actions">
      <button class="btn btn-primary" id="goBtn" onclick="runCompare()" disabled>⬇ Download Changes</button>
      <span id="status"></span>
    </div>
    <div class="note">
      <b>What counts as a change?</b> Almost every row moves a little each period, so listing every tiny difference is just noise.
      We call it a <b>material move</b> only when Dollars or Units shifts by both a dollar/unit amount and a percent.
      <b>New products</b> (just on market) and <b>exits</b> (now gone) are always shown. Each moved row shows old, new, and the change (as a number and a percent).
    </div>
  </div>
</div>
<script>
function refresh() {
  const ok = document.getElementById('oldFile').files.length && document.getElementById('newFile').files.length;
  document.getElementById('goBtn').disabled = !ok;
}
async function runCompare() {
  const oldF = document.getElementById('oldFile').files[0];
  const newF = document.getElementById('newFile').files[0];
  const btn = document.getElementById('goBtn');
  const status = document.getElementById('status');
  status.className = ''; status.innerHTML = '<span class="spinner"></span>Comparing… (large files take a moment)';
  btn.disabled = true;
  const fd = new FormData();
  fd.append('old_file', oldF);
  fd.append('new_file', newF);
  try {
    const resp = await fetch('/api/iqvia/compare', { method:'POST', body:fd });
    if (!resp.ok) {
      let msg = 'HTTP ' + resp.status;
      try { const j = await resp.json(); if (j.detail) msg = j.detail; } catch (e) {}
      throw new Error(msg);
    }
    const blob = await resp.blob();
    let fname = 'iqvia_changes.xlsx';
    const cd = resp.headers.get('Content-Disposition') || '';
    const m = cd.match(/filename="?([^"]+)"?/);
    if (m) fname = m[1];
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = fname; document.body.appendChild(a); a.click();
    a.remove(); URL.revokeObjectURL(url);
    status.className = 'ok'; status.textContent = '✓ Downloaded ' + fname;
  } catch (err) {
    status.className = 'err'; status.textContent = '✗ ' + err.message;
  } finally {
    btn.disabled = false; refresh();
  }
}
</script>
</body>
</html>"""


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
  .nav-zydus-logo {
    height: 28px;
    width: auto;
    flex-shrink: 0;
    filter: brightness(0) invert(1);
    opacity: 0.92;
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
  .header-logo-zydus {
    height: 40px;
    width: auto;
    flex-shrink: 0;
    filter: brightness(0) invert(1);
    opacity: 0.88;
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
    background: transparent;
    border: none;
    padding: 0;
    margin-bottom: 22px;
    font-size: 0.88rem;
    color: var(--primary);
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
  <a class="site-nav-link" href="https://health-products.canada.ca/dpd-bdpp/" target="_blank" rel="noopener">DPD</a>
  <a class="site-nav-link" href="https://www.canada.ca/en/health-canada/services/drug-health-product-review-approval/generic-submissions-under-review.html" target="_blank" rel="noopener">Generic Submissions</a>
  <a class="site-nav-link" href="https://health-products.canada.ca/noc-ac/" target="_blank" rel="noopener">NOC</a>
  <a class="site-nav-link" href="https://pr-rdb.hc-sc.gc.ca/pr-rdb/" target="_blank" rel="noopener">Patent Register</a>
</nav>
<header role="banner">
  <div class="header-brand-row">
    <img class="header-logo-zydus" src="/static/zydus_logo.png" alt="Zydus" />
    <div>
      <div class="header-company-name">Zydus &nbsp;&middot;&nbsp; Dedicated To Life</div>
      <h1>Drug Intelligence Platform</h1>
    </div>
  </div>
  <p>Type a drug ingredient and we check all four Canadian government drug databases at once: the Drug Product Database (DPD), Generic Submissions Under Review, Notice of Compliance (NOC), and the Patent Register.</p>
</header>
<div class="container">
  <div class="dash-tab-bar" style="margin-bottom:0;">
    <button class="dash-tab active" id="mainTabSearch" type="button" onclick="switchMainTab('search')">Search by ingredient(s)</button>
    <button class="dash-tab" id="mainTabUniverse" type="button" onclick="switchMainTab('universe')">Full universe</button>
    <button class="dash-tab" id="mainTabIqvia" type="button" onclick="switchMainTab('iqvia')">IQVIA Compare</button>
  </div>
  <div id="paneSearch" class="main-pane">
  <div class="search-box">
    <div class="info-box">
      <b>What this does.</b> Type one or more drug ingredients. We look them up in all four databases and show what we find. Press <b>Download Excel</b> for the full report, or <b>Download Filtered Excel</b> to keep only the products that pass the rules you set below.
    </div>
    <div class="row">
      <div class="field-group" style="flex:1">
        <label for="query">Ingredients <span id="queryCount" style="font-weight:400;color:var(--muted);font-size:0.72rem">(one per line or comma-separated; export runs all)</span></label>
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
        <div style="display:flex; gap:8px; flex-wrap:wrap; align-items:center;">
          <button class="btn btn-export" id="exportBtn" onclick="doExport(false)" title="Build the full two-tab enriched workbook (DPD+NOC+Patents and Generic Submissions)">⬇ Download Excel</button>
          <button class="btn btn-primary" id="filterBtn" onclick="doExport(true)" title="Screen products against the go/no-go criteria below, then export a filtered Summary + Detail workbook">⬇ Download Filtered Excel</button>
          <button class="btn" style="background:#888;color:white;" onclick="resetAllCaches()" title="Clear all cached data and force fresh fetches from all sources">Reset cache</button>
        </div>
        <div style="display:flex; align-items:center; gap:8px; margin-top:8px; flex-wrap:wrap;">
          <label style="font-size:0.8rem;color:var(--muted);white-space:nowrap;cursor:pointer;" for="iqviaFile">
            IQVIA file (optional):
          </label>
          <input type="file" id="iqviaFile" accept=".xlsx,.xls"
            style="font-size:0.78rem;border:1px solid var(--border);border-radius:3px;padding:3px 6px;background:#fff;cursor:pointer;max-width:220px;"
            onchange="uploadIqvia(this)">
          <span id="iqviaStatus" style="font-size:0.78rem;color:var(--muted)"></span>
        </div>
      </div>
    </div>

    <!-- Go/No-Go screening criteria — used only by the "Download Filtered Excel" action. -->
    <details class="filter-box" id="filterBox" ontoggle="if(this.open) populateDosageForms()" style="margin-top:14px;border:1px solid var(--border);border-radius:4px;padding:0 14px;background:#fff;">
      <summary style="cursor:pointer;font-weight:600;font-size:0.9rem;color:var(--primary-dark);padding:11px 2px;list-style:revert;">
        Go / No-Go Filter Criteria
        <span style="font-weight:400;color:var(--muted);font-size:0.78rem">&nbsp;(optional). Fill only the rows you care about. A product is kept only if it passes every row you filled in.</span>
      </summary>
      <div id="criteriaRows" style="padding:4px 0 6px;display:flex;flex-direction:column;gap:7px;"></div>
      <div id="iqviaCritNote" style="font-size:0.74rem;color:var(--warn);padding-bottom:11px;display:none;">
        ⚠ Value / Quantity rules need an IQVIA file. Upload one above to turn them on.
      </div>
      <div id="extraCriteria" style="padding:4px 0 11px;"></div>
    </details>
  </div><!-- /paneSearch -->

  <!-- Full universe tab (options 3 & 4) — new, separate from the frozen search path -->
  <div id="paneUniverse" class="main-pane" style="display:none">
    <div class="search-box">
      <div class="info-box">
        <b>See the whole market.</b> Download every product in the Drug Product Database, including older
        grandfathered (pre-NOC) products. It does not read the slow Product-Monograph PDFs up front. NOC,
        patent, and data-protection columns stay blank when there is no record. Add an IQVIA file to attach
        market size and a match-confidence check.
      </div>
      <div style="display:flex;align-items:center;gap:8px;margin:12px 0 4px;flex-wrap:wrap;">
        <label style="font-size:0.8rem;color:var(--muted);white-space:nowrap;cursor:pointer;" for="iqviaFileU">IQVIA file (optional):</label>
        <input type="file" id="iqviaFileU" accept=".xlsx,.xls"
          style="font-size:0.78rem;border:1px solid var(--border);border-radius:3px;padding:3px 6px;background:#fff;cursor:pointer;max-width:220px;"
          onchange="uploadIqviaU(this)">
        <span id="iqviaStatusU" style="font-size:0.78rem;color:var(--muted)"></span>
      </div>

      <div style="margin-top:14px;">
        <div style="font-weight:600;font-size:0.92rem;color:var(--primary-dark);">3 · Full universe sheet (no PDF data)</div>
        <p style="font-size:0.8rem;color:var(--muted);margin:4px 0 8px;">
          The whole list, fast. It skips reading the Product-Monograph PDFs, which would take hours for the full market. A note at the top of the file says so.
        </p>
        <button class="btn btn-export" id="universeFullBtn" type="button" onclick="doUniverse('full')"
          title="Build and download the full no-PDF universe workbook">⬇ Download full universe (no PDF)</button>
      </div>

      <details class="filter-box" id="filterBoxU" ontoggle="if(this.open) populateDosageForms()" style="margin-top:16px;border:1px solid var(--border);border-radius:4px;padding:0 14px;background:#fff;">
        <summary style="cursor:pointer;font-weight:600;font-size:0.9rem;color:var(--primary-dark);padding:11px 2px;list-style:revert;">
          4 · Filter &amp; enrich (Go / No-Go criteria)
          <span style="font-weight:400;color:var(--muted);font-size:0.78rem">&nbsp;Set your rules first. We filter the whole universe, then read PDF data only for the products that pass.</span>
        </summary>
        <div id="criteriaRowsU" style="padding:4px 0 6px;display:flex;flex-direction:column;gap:7px;"></div>
        <div id="iqviaCritNoteU" style="font-size:0.74rem;color:var(--warn);padding-bottom:11px;display:none;">
          ⚠ Value / Quantity rules need an IQVIA file. Upload one above to turn them on.
        </div>
        <div id="extraCriteriaU" style="padding:4px 0 11px;"></div>
      </details>
      <div style="margin-top:12px;">
        <button class="btn btn-primary" id="universeFilterBtn" type="button" onclick="doUniverse('filter_enrich')"
          title="Apply the six criteria across the whole universe, then enrich only the survivors with PDF data">⬇ Filter &amp; enrich (full PDF data)</button>
      </div>

      <div class="note" id="universeCacheNote" style="margin-top:16px;font-size:0.78rem;color:var(--muted);line-height:1.5;border-top:1px solid var(--border);padding-top:12px;">
        The full-universe dataset is cached for 4 hours. Click <b>Reset cache</b> (on the Search tab) to force a fresh pull from DPD.
      </div>
    </div>
  </div><!-- /paneUniverse -->

  <!-- IQVIA quarter-over-quarter compare tab -->
  <div id="paneIqvia" class="main-pane" style="display:none">
    <div class="search-box">
      <div class="info-box">
        <b>Quarter-over-quarter change report.</b> Upload the previous quarter's IQVIA Canada extract alongside
        the current quarter's. The tool produces an Excel report containing <b>only the changes</b>: newly
        introduced products, products that have exited the market, and material movements in dollars or units.
        The two files may use different column names or date formats; the latest reporting period is resolved
        independently within each file. Supported formats: .xlsx, .xls, and .csv.
      </div>

      <div class="row" style="margin-top:14px;">
        <div class="field-group" style="flex:1;min-width:260px;">
          <label for="cmpOldFile">1 · Previous-quarter extract <span style="font-weight:400;color:var(--muted);text-transform:none;letter-spacing:0;">(prior reporting period)</span></label>
          <input type="file" id="cmpOldFile" accept=".xlsx,.xls,.csv" onchange="refreshCompare()"
            style="font-size:0.82rem;border:1px solid var(--border);border-radius:3px;padding:6px;background:#fff;cursor:pointer;">
        </div>
        <div class="field-group" style="flex:1;min-width:260px;">
          <label for="cmpNewFile">2 · Current-quarter extract <span style="font-weight:400;color:var(--muted);text-transform:none;letter-spacing:0;">(current reporting period)</span></label>
          <input type="file" id="cmpNewFile" accept=".xlsx,.xls,.csv" onchange="refreshCompare()"
            style="font-size:0.82rem;border:1px solid var(--border);border-radius:3px;padding:6px;background:#fff;cursor:pointer;">
        </div>
      </div>

      <div style="margin-top:16px;display:flex;align-items:center;gap:14px;flex-wrap:wrap;">
        <button class="btn btn-export" id="cmpGoBtn" type="button" onclick="runCompare()" disabled
          title="Compare the two extracts and download an Excel report of the changes">⬇ Download Change Report</button>
        <span id="cmpStatus" style="font-size:0.84rem;color:var(--muted)"></span>
      </div>

      <!-- Reorder / same-period notice, mirrored from the Excel Summary banner. -->
      <div id="cmpNotice" style="display:none;margin-top:14px;border-radius:4px;padding:12px 16px;font-size:0.88rem;line-height:1.5;"></div>

      <div class="note">
        <b>Methodology.</b> MAT totals fluctuate marginally on nearly every row in each period, so a direct
        difference would report predominantly noise. A product is classified as a <b>material move</b> only when
        its Dollars or Units shift exceeds both an absolute and a percentage threshold. Market <b>entrants</b>
        (newly available) and <b>exits</b> (discontinued) are always reported, irrespective of magnitude. Each
        moved row presents the prior value&nbsp;&rarr;&nbsp;current value&nbsp;&rarr;&nbsp;change&nbsp;(absolute and %).
      </div>
    </div>
  </div><!-- /paneIqvia -->

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

  <!-- Dashboard panel: shown after export completes; renders exact XLSX dataset -->
  <div class="dashboard-panel" id="dashboardPanel">
    <div class="dashboard-header">
      <span class="dashboard-title">📊 Enriched Dashboard (same dataset as XLSX)</span>
      <span style="font-size:0.8rem;color:var(--muted)" id="dashMeta"></span>
    </div>
    <div class="canary-box" id="canaryBox"></div>
    <div class="kpi-row" id="kpiRow"></div>
    <div class="dash-tab-bar">
      <button class="dash-tab active" id="dashTab1" onclick="switchDashTab(1)">DPD + NOC + Patents</button>
      <button class="dash-tab" id="dashTab2" onclick="switchDashTab(2)">Generic Submissions</button>
      <button class="dash-tab" id="dashTab3" onclick="switchDashTab(3)" style="display:none">IQVIA Reconciliation</button>
    </div>
    <div id="dashPane1"><div class="dash-table-wrap" id="dashSheet1"></div></div>
    <div id="dashPane2" style="display:none"><div class="dash-table-wrap" id="dashSheet2"></div></div>
    <div id="dashPane3" style="display:none"><div class="dash-table-wrap" id="dashSheet3"></div></div>
  </div>
</div>
<footer role="contentinfo">
  <strong style="color:var(--primary);font-family:'Exo',sans-serif">Zydus</strong> <span style="color:var(--muted);font-size:0.76rem">· Dedicated To Life</span>
  <br/>
  All data comes from public Canadian government databases (DPD, Generic Submissions Under Review, NOC, Patent Register). &nbsp;|&nbsp;
  Every value is copied straight from the source. Nothing is guessed or made up.
</footer>

<script>
let currentJobId = null;
let currentEventSource = null;
let _exportMode = 'full';  // 'full' | 'filtered' — which download to trigger on complete

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
    hint.textContent = `1 ingredient, exported on its own.`;
  } else {
    hint.textContent = `${qs.length} ingredients: ${qs.map(q => '"' + q + '"').join(', ')}, exported side by side.`;
  }
}

function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// ---- Go/No-Go filter criteria form ----

// The six screening criteria. iqvia=true rows are disabled until an IQVIA file
// is loaded (their sums require IQVIA metric columns).
const CRITERIA_DEFS = [
  { metric: 'competitors',  label: 'Number of Competitors',     iqvia: false },
  { metric: 'filings',      label: 'Number of Filings',          iqvia: false },
  { metric: 'approvals',    label: 'Number of Approvals',        iqvia: false },
  { metric: 'value',        label: 'Value ($)',                  iqvia: true  },
  { metric: 'quantity',     label: 'Quantity (Units)',           iqvia: true  },
  { metric: 'quantity_ext', label: 'Quantity Ext (Units)',       iqvia: true  },
];

function buildCriteriaRows() {
  const wrap = document.getElementById('criteriaRows');
  wrap.innerHTML = CRITERIA_DEFS.map(d => `
    <div class="crit-row" data-metric="${d.metric}" data-iqvia="${d.iqvia}"
         style="display:flex;align-items:center;gap:9px;flex-wrap:wrap;">
      <label style="display:flex;align-items:center;gap:7px;min-width:230px;font-size:0.86rem;cursor:pointer;">
        <input type="checkbox" id="crit_${d.metric}_on">
        <span>${d.label}</span>
      </label>
      <select id="crit_${d.metric}_op" style="padding:4px 8px;border:1px solid var(--border);border-radius:3px;font-size:0.84rem;">
        <option value="above">above</option>
        <option value="below">below</option>
        <option value="exactly">exactly</option>
      </select>
      <input type="number" id="crit_${d.metric}_val" step="any" placeholder="value"
        style="width:130px;padding:4px 8px;border:1px solid var(--border);border-radius:3px;font-size:0.84rem;">
    </div>`).join('');
  setIqviaCriteriaEnabled(!!_iqviaToken);
}

// Enable/disable the IQVIA-dependent criteria (Value/Quantity/Quantity Ext).
function setIqviaCriteriaEnabled(enabled) {
  document.querySelectorAll('.crit-row[data-iqvia="true"]').forEach(row => {
    row.style.opacity = enabled ? '1' : '0.45';
    row.querySelectorAll('input, select').forEach(el => { el.disabled = !enabled; });
    if (!enabled) {
      const cb = row.querySelector('input[type="checkbox"]');
      if (cb) cb.checked = false;
    }
  });
  const note = document.getElementById('iqviaCritNote');
  if (note) note.style.display = enabled ? 'none' : 'block';
}

// Collect the enabled, filled-in criteria as request dicts. Throws (message) if a
// checked row has no value.
function collectCriteria() {
  const out = [];
  for (const d of CRITERIA_DEFS) {
    const cb = document.getElementById(`crit_${d.metric}_on`);
    if (!cb || !cb.checked || cb.disabled) continue;
    const op = document.getElementById(`crit_${d.metric}_op`).value;
    const raw = document.getElementById(`crit_${d.metric}_val`).value;
    if (raw === '' || raw == null) {
      throw new Error(`Enter a value for "${d.label}" or uncheck it.`);
    }
    const value = Number(raw);
    if (!isFinite(value)) throw new Error(`"${d.label}" value must be a number.`);
    out.push({ metric: d.metric, operator: op, value });
  }
  collectExtraCriteria('s', out);
  return out;
}

// ---- Two additive filters: dosage form + six-year no-file date (BOTH tabs) ----

// no-file-date operators (distinct from the numeric above/below/exactly set).
const NO_FILE_OPS = [
  { value: 'less',              label: 'before (<)'      },
  { value: 'greater',           label: 'after (>)'       },
  { value: 'greater_or_equal',  label: 'on or after (≥)' },
  { value: 'equal',             label: 'on (=)'          },
];

// Render the dosage-form multi-select + no-file-date row for one tab (suffix
// 's' = Search, 'u' = Universe). Identical option set on both tabs.
function buildExtraCriteria(suffix, containerId) {
  const wrap = document.getElementById(containerId);
  if (!wrap) return;
  const opOpts = ['<option value="">(choose)</option>']
    .concat(NO_FILE_OPS.map(o => `<option value="${o.value}">${o.label}</option>`)).join('');
  wrap.innerHTML = `
    <div class="crit-row" style="display:flex;align-items:flex-start;gap:9px;flex-wrap:wrap;margin-top:2px;">
      <label style="min-width:230px;font-size:0.86rem;padding-top:4px;display:flex;align-items:flex-start;gap:7px;cursor:pointer;">
        <input type="checkbox" id="dform_${suffix}_on" style="margin-top:3px;">
        <span>Dosage form
          <span style="display:block;font-weight:400;color:var(--muted);font-size:0.72rem;">base form; matches every sub-form (e.g. TABLET also matches TABLET (EXTENDED-RELEASE))</span>
        </span>
      </label>
      <div id="dform_${suffix}"
        style="min-width:240px;max-height:160px;overflow-y:auto;padding:6px 8px;border:1px solid var(--border);border-radius:3px;font-size:0.84rem;display:flex;flex-direction:column;gap:4px;">
        <span style="color:var(--muted);">open this panel to load…</span>
      </div>
    </div>
    <div class="crit-row" style="display:flex;align-items:flex-start;gap:9px;flex-wrap:wrap;margin-top:9px;">
      <label style="min-width:230px;font-size:0.86rem;padding-top:4px;display:flex;align-items:flex-start;gap:7px;cursor:pointer;">
        <input type="checkbox" id="nofile_${suffix}_on" style="margin-top:3px;">
        <span>Six-year no-file date
          <span style="display:block;font-weight:400;color:var(--muted);font-size:0.72rem;">Month / Day / Year; future dates only</span>
        </span>
      </label>
      <select id="nofileop_${suffix}" style="padding:4px 8px;border:1px solid var(--border);border-radius:3px;font-size:0.84rem;">${opOpts}</select>
      <input type="text" id="nofileval_${suffix}" placeholder="MM/DD/YYYY" maxlength="10"
        style="width:130px;padding:4px 8px;border:1px solid var(--border);border-radius:3px;font-size:0.84rem;">
    </div>`;
}

// Populate BOTH tabs' dosage-form dropdowns from the cached universe list. Fetched
// once, lazily, when a filter panel is first opened (avoids forcing the universe
// pull on plain Search use).
let _dosageFormsLoaded = false;
async function populateDosageForms() {
  const sels = ['dform_s', 'dform_u'].map(id => document.getElementById(id)).filter(Boolean);
  if (!sels.length || _dosageFormsLoaded) return;
  sels.forEach(s => { s.innerHTML = '<span style="color:var(--muted);">Loading dosage forms…</span>'; });
  try {
    const resp = await fetch('/api/dosage-forms');
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    const data = await resp.json();
    const forms = data.base_forms || [];
    const opts = forms.map(f => `
      <label style="display:flex;align-items:center;gap:6px;cursor:pointer;font-weight:400;">
        <input type="checkbox" class="dform-opt" value="${escHtml(f)}">
        <span>${escHtml(f)}</span>
      </label>`).join('') || '<span style="color:var(--muted);">No dosage forms found.</span>';
    sels.forEach(s => { s.innerHTML = opts; });
    _dosageFormsLoaded = true;
  } catch (e) {
    sels.forEach(s => { s.innerHTML = '<span style="color:var(--err);">Failed to load. Reopen to retry</span>'; });
  }
}

// Validate a user MM/DD/YYYY string: well-formed calendar date AND in the future.
function _checkFutureMdy(s) {
  const m = new RegExp('^([0-9]{2})/([0-9]{2})/([0-9]{4})$').exec(s);
  if (!m) throw new Error(`Six-year no-file date must be MM/DD/YYYY (Month/Day/Year): ${s}`);
  const mm = +m[1], dd = +m[2], yyyy = +m[3];
  const d = new Date(yyyy, mm - 1, dd);
  if (d.getFullYear() !== yyyy || d.getMonth() !== mm - 1 || d.getDate() !== dd)
    throw new Error(`Invalid calendar date (MM/DD/YYYY): ${s}`);
  const today = new Date(); today.setHours(0, 0, 0, 0);
  if (d <= today) throw new Error(`Six-year no-file date must be in the future: ${s}`);
}

// Append the two additive filter entries (if set) to a criteria list. Throws on
// a partially-filled / invalid date so input errors abort early, like the six.
function collectExtraCriteria(suffix, out) {
  const dformOn = document.getElementById(`dform_${suffix}_on`);
  if (dformOn && dformOn.checked) {
    const dsel = document.getElementById(`dform_${suffix}`);
    const picked = dsel ? Array.from(dsel.querySelectorAll('input.dform-opt:checked')).map(o => o.value).filter(Boolean) : [];
    if (!picked.length) throw new Error('Pick at least one dosage form, or uncheck "Dosage form".');
    out.push({ metric: 'dosage_form', value: picked });
  }
  const nofileOn = document.getElementById(`nofile_${suffix}_on`);
  if (nofileOn && nofileOn.checked) {
    const op = (document.getElementById(`nofileop_${suffix}`) || {}).value || '';
    const val = ((document.getElementById(`nofileval_${suffix}`) || {}).value || '').trim();
    if (!op) throw new Error('Pick an operator for the six-year no-file date, or uncheck it.');
    if (!val) throw new Error('Enter a six-year no-file date (MM/DD/YYYY), or uncheck it.');
    _checkFutureMdy(val);
    out.push({ metric: 'no_file_date', operator: op, value: val });
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
    // Auto-trigger download of the file the user asked for (full vs filtered).
    const url = (_exportMode === 'filtered' && data.filtered_download_url)
      ? data.filtered_download_url
      : data.download_url;
    const a = document.createElement('a');
    a.href = url;
    a.style.display = 'none';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    _setExportButtonsDisabled(false);
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
    _setExportButtonsDisabled(false);
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
  alert(`Cache cleared: ${data.http_rows_cleared} search results, ${data.patent_rows_cleared} patent records, ${data.labeling_rows_cleared} labeling records removed.`);
}

function _setExportButtonsDisabled(disabled) {
  document.getElementById('exportBtn').disabled = disabled;
  document.getElementById('filterBtn').disabled = disabled;
}

// filtered=false → full workbook; filtered=true → screened Summary+Detail workbook.
async function doExport(filtered) {
  const queries = parseQueries();
  if (!queries.length) { alert('Please enter at least one ingredient.'); return; }
  const field = document.getElementById('field').value;

  // For the filtered export, gather criteria up front so input errors abort early.
  let criteria = [];
  if (filtered) {
    try {
      criteria = collectCriteria();
    } catch (e) {
      alert(e.message);
      document.getElementById('filterBox').open = true;
      return;
    }
    if (!criteria.length) {
      alert('Tick at least one filter criterion (and give it a value) to download a filtered Excel.');
      document.getElementById('filterBox').open = true;
      return;
    }
  }
  _exportMode = filtered ? 'filtered' : 'full';

  _setExportButtonsDisabled(true);
  const panel = document.getElementById('exportPanel');
  panel.style.display = 'block';
  document.getElementById('exportLog').innerHTML = '';
  document.getElementById('progressFill').style.width = '0%';
  document.getElementById('progressFill').classList.remove('error');
  const what = filtered ? 'filtered export' : 'export';
  const label = queries.length === 1
    ? `Running ${what} for "${queries[0]}"…`
    : `Running ${what} for ${queries.length} ingredients side-by-side…`;
  document.getElementById('exportStageLabel').textContent = label;
  document.getElementById('exportStats').textContent = '';

  _closeExport();

  // Start the job
  let jobId;
  try {
    const body = { queries, field, allow_partial: false, enable_ocr: true };
    if (_iqviaToken) body.iqvia_token = _iqviaToken;
    if (filtered) body.filter_criteria = criteria;
    const resp = await fetch('/export/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
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
    _setExportButtonsDisabled(false);
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
    _appendLog('Connection lost. Check server.', 'log-err');
    document.getElementById('exportStageLabel').textContent = 'Connection error';
    _setExportButtonsDisabled(false);
    es.close();
  };
}

// Ctrl+Enter (or Cmd+Enter) runs the full export from the textarea.
document.getElementById('query').addEventListener('keydown', e => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) doExport(false);
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
      html += `<td>${v != null && v !== '' ? escHtml(String(v)) : '<span style="color:#aaa">-</span>'}</td>`;
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
  document.getElementById('dashTab3').classList.toggle('active', n === 3);
  document.getElementById('dashPane1').style.display = n === 1 ? '' : 'none';
  document.getElementById('dashPane2').style.display = n === 2 ? '' : 'none';
  document.getElementById('dashPane3').style.display = n === 3 ? '' : 'none';
}

// ---- IQVIA upload ----
let _iqviaToken = null;

async function uploadIqvia(input) {
  const file = input.files[0];
  if (!file) return;
  const status = document.getElementById('iqviaStatus');
  status.style.color = 'var(--muted)';
  status.textContent = 'Uploading…';
  _iqviaToken = null;

  try {
    const fd = new FormData();
    fd.append('file', file);
    const resp = await fetch('/api/iqvia/upload', { method: 'POST', body: fd });
    let data;
    try { data = await resp.json(); } catch (_) { data = {}; }
    if (!resp.ok) throw new Error(data.detail || `HTTP ${resp.status}: the server rejected the file (check it is a valid .xlsx)`);
    _iqviaToken = data.token;
    status.style.color = 'var(--muted)';
    status.textContent = `✓ Loaded`;
    setIqviaCriteriaEnabled(true);  // unlock Value / Quantity criteria
  } catch (e) {
    status.style.color = 'var(--err)';
    status.textContent = `✗ ${e.message}`;
    _iqviaToken = null;
    setIqviaCriteriaEnabled(false);
  }
}

async function _loadDashboard(jobId) {
  if (!jobId) return;
  const panel = document.getElementById('dashboardPanel');
  panel.style.display = 'block';
  document.getElementById('dashSheet1').innerHTML = '<div class="loading-msg"><div class="spinner"></div>Loading dashboard from XLSX dataset…</div>';
  document.getElementById('kpiRow').innerHTML = '';
  document.getElementById('canaryBox').textContent = '';
  // Hide reconciliation tab until we know if data exists
  document.getElementById('dashTab3').style.display = 'none';
  document.getElementById('dashPane3').style.display = 'none';

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
    const hasIqvia = !!data.reconciliation;
    const canary = [
      `✓ Dashboard loaded from the saved job results. No new outbound requests.`,
      `  Sheet 1: ${s1.records.length} rows × ${s1.columns.length} columns`,
      `  Sheet 2: ${s2.records.length} rows × ${s2.columns.length} columns`,
      hasIqvia ? `  IQVIA reconciliation: ${data.reconciliation.records.length} entries` : '',
      `  ${queryStr} by ${data.field}`,
      `  Columns: ${s1.columns.join(', ')}`,
    ].filter(Boolean).join('\\n');
    document.getElementById('canaryBox').textContent = canary;

    document.getElementById('dashMeta').textContent =
      `Job ${jobId.slice(0,8)}… · ${queryStr} by ${data.field}`;

    // Render tables
    document.getElementById('dashSheet1').innerHTML = _dashTable(s1.columns, s1.records);
    document.getElementById('dashSheet2').innerHTML = _dashTable(s2.columns, s2.records);

    // IQVIA reconciliation tab (shown only when data is available)
    if (hasIqvia && data.reconciliation.records.length > 0) {
      document.getElementById('dashTab3').style.display = '';
      document.getElementById('dashSheet3').innerHTML = _dashTable(
        data.reconciliation.columns,
        data.reconciliation.records
      );
    }

    console.log('[Dashboard canary]', {
      sheet1_rows: s1.records.length,
      sheet1_cols: s1.columns.length,
      sheet2_rows: s2.records.length,
      job_id: jobId,
      note: 'No new scraping; data read from job.sheet1_records (same as XLSX)',
    });
  } catch (e) {
    document.getElementById('dashSheet1').innerHTML =
      `<div class="error-box">Dashboard load failed: ${escHtml(e.message)}</div>`;
  }
}

// ---- Full universe tab (options 3 & 4) ----

function switchMainTab(name) {
  document.getElementById('mainTabSearch').classList.toggle('active', name === 'search');
  document.getElementById('mainTabUniverse').classList.toggle('active', name === 'universe');
  document.getElementById('mainTabIqvia').classList.toggle('active', name === 'iqvia');
  document.getElementById('paneSearch').style.display = name === 'search' ? '' : 'none';
  document.getElementById('paneUniverse').style.display = name === 'universe' ? '' : 'none';
  document.getElementById('paneIqvia').style.display = name === 'iqvia' ? '' : 'none';
  if (name === 'universe') {
    setUniverseIqviaEnabled(!!_iqviaToken);
    refreshUniverseCacheNote();
  }
}

// ---- IQVIA quarter-over-quarter compare tab ----
function refreshCompare() {
  const ok = document.getElementById('cmpOldFile').files.length && document.getElementById('cmpNewFile').files.length;
  document.getElementById('cmpGoBtn').disabled = !ok;
}

async function runCompare() {
  const oldF = document.getElementById('cmpOldFile').files[0];
  const newF = document.getElementById('cmpNewFile').files[0];
  const btn = document.getElementById('cmpGoBtn');
  const status = document.getElementById('cmpStatus');
  const notice = document.getElementById('cmpNotice');
  status.style.color = 'var(--muted)';
  status.textContent = 'Comparing… (large files take a moment)';
  if (notice) notice.style.display = 'none';
  btn.disabled = true;
  const fd = new FormData();
  fd.append('old_file', oldF);
  fd.append('new_file', newF);
  try {
    const resp = await fetch('/api/iqvia/compare', { method: 'POST', body: fd });
    if (!resp.ok) {
      let msg = 'HTTP ' + resp.status;
      try { const j = await resp.json(); if (j.detail) msg = j.detail; } catch (e) {}
      throw new Error(msg);
    }
    const blob = await resp.blob();
    let fname = 'iqvia_changes.xlsx';
    const cd = resp.headers.get('Content-Disposition') || '';
    const m = cd.match(/filename="?([^"]+)"?/);
    if (m) fname = m[1];
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = fname; document.body.appendChild(a); a.click();
    a.remove(); URL.revokeObjectURL(url);
    status.style.color = 'var(--ok)';
    status.textContent = '✓ Downloaded ' + fname;
    _showCompareNotice(resp);
  } catch (err) {
    status.style.color = 'var(--err)';
    status.textContent = '✗ ' + err.message;
  } finally {
    btn.disabled = false; refreshCompare();
  }
}

// Render the older->newer reorder banner (or same-period info note) from the
// compare response headers. Mirrors the Excel Summary banner so the on-screen
// user sees the same correction the workbook records.
function _showCompareNotice(resp) {
  const notice = document.getElementById('cmpNotice');
  if (!notice) return;
  const reordered = (resp.headers.get('X-IQVIA-Reordered') || '') === 'true';
  const oldP = resp.headers.get('X-IQVIA-Old-Period') || '—';
  const newP = resp.headers.get('X-IQVIA-New-Period') || '—';
  const warn = resp.headers.get('X-IQVIA-Warning') || '';
  if (reordered) {
    notice.style.background = '#fff3cd';
    notice.style.border = '1px solid #ffc107';
    notice.style.borderLeft = '6px solid #e6a500';
    notice.style.color = '#4a3600';
    notice.innerHTML = '<b>⚠ Files were uploaded in reverse order.</b> Slot 1 was newer ('
      + escHtml(newP) + ') than slot 2 (' + escHtml(oldP) + '), so they were automatically '
      + 'compared older&nbsp;&rarr;&nbsp;newer. In the report, <b>Old</b> = your slot-2 file and '
      + '<b>New</b> = your slot-1 file.';
    notice.style.display = 'block';
  } else if (warn) {
    notice.style.background = '#E8F4F8';
    notice.style.border = '1px solid #A8D4E6';
    notice.style.borderLeft = '4px solid var(--teal)';
    notice.style.color = 'var(--text)';
    notice.innerHTML = 'ℹ ' + escHtml(warn);
    notice.style.display = 'block';
  } else {
    notice.style.display = 'none';
  }
}

// Tab B renders its own copy of the six criteria (ids prefixed ucrit_) so the
// frozen Search-tab form is left exactly as-is.
function buildUniverseCriteriaRows() {
  const wrap = document.getElementById('criteriaRowsU');
  if (!wrap) return;
  wrap.innerHTML = CRITERIA_DEFS.map(d => `
    <div class="crit-row" data-metric="${d.metric}" data-iqvia="${d.iqvia}"
         style="display:flex;align-items:center;gap:9px;flex-wrap:wrap;">
      <label style="display:flex;align-items:center;gap:7px;min-width:230px;font-size:0.86rem;cursor:pointer;">
        <input type="checkbox" id="ucrit_${d.metric}_on">
        <span>${d.label}</span>
      </label>
      <select id="ucrit_${d.metric}_op" style="padding:4px 8px;border:1px solid var(--border);border-radius:3px;font-size:0.84rem;">
        <option value="above">above</option>
        <option value="below">below</option>
        <option value="exactly">exactly</option>
      </select>
      <input type="number" id="ucrit_${d.metric}_val" step="any" placeholder="value"
        style="width:130px;padding:4px 8px;border:1px solid var(--border);border-radius:3px;font-size:0.84rem;">
    </div>`).join('');
  setUniverseIqviaEnabled(!!_iqviaToken);
}

function setUniverseIqviaEnabled(enabled) {
  document.querySelectorAll('#criteriaRowsU .crit-row[data-iqvia="true"]').forEach(row => {
    row.style.opacity = enabled ? '1' : '0.45';
    row.querySelectorAll('input, select').forEach(el => { el.disabled = !enabled; });
    if (!enabled) {
      const cb = row.querySelector('input[type="checkbox"]');
      if (cb) cb.checked = false;
    }
  });
  const note = document.getElementById('iqviaCritNoteU');
  if (note) note.style.display = enabled ? 'none' : 'block';
}

function collectUniverseCriteria() {
  const out = [];
  for (const d of CRITERIA_DEFS) {
    const cb = document.getElementById(`ucrit_${d.metric}_on`);
    if (!cb || !cb.checked || cb.disabled) continue;
    const op = document.getElementById(`ucrit_${d.metric}_op`).value;
    const raw = document.getElementById(`ucrit_${d.metric}_val`).value;
    if (raw === '' || raw == null) throw new Error(`Enter a value for "${d.label}" or uncheck it.`);
    const value = Number(raw);
    if (!isFinite(value)) throw new Error(`"${d.label}" value must be a number.`);
    out.push({ metric: d.metric, operator: op, value });
  }
  collectExtraCriteria('u', out);
  return out;
}

// Tab B IQVIA upload — shares the global _iqviaToken with the Search tab.
async function uploadIqviaU(input) {
  const file = input.files[0];
  if (!file) return;
  const status = document.getElementById('iqviaStatusU');
  status.style.color = 'var(--muted)';
  status.textContent = 'Uploading…';
  _iqviaToken = null;
  try {
    const fd = new FormData();
    fd.append('file', file);
    const resp = await fetch('/api/iqvia/upload', { method: 'POST', body: fd });
    let data;
    try { data = await resp.json(); } catch (_) { data = {}; }
    if (!resp.ok) throw new Error(data.detail || `HTTP ${resp.status}`);
    _iqviaToken = data.token;
    status.style.color = 'var(--muted)';
    status.textContent = '✓ Loaded';
    setUniverseIqviaEnabled(true);
    setIqviaCriteriaEnabled(true);  // keep the Search tab's IQVIA criteria in sync
  } catch (e) {
    status.style.color = 'var(--err)';
    status.textContent = `✗ ${e.message}`;
    _iqviaToken = null;
    setUniverseIqviaEnabled(false);
  }
}

async function refreshUniverseCacheNote() {
  const note = document.getElementById('universeCacheNote');
  if (!note) return;
  try {
    const resp = await fetch('/api/universe/status');
    const s = await resp.json();
    const base = 'The full-universe dataset is cached for 4 hours. Click <b>Reset cache</b> (on the Search tab) to force a fresh pull from DPD.';
    if (s && s.cached && s.fresh) {
      const mins = Math.max(0, Math.round((s.expires_in_seconds || 0) / 60));
      note.innerHTML = `Cached universe is fresh (${s.dpd_records} products, expires in ~${mins} min). ` + base;
    } else {
      note.innerHTML = base;
    }
  } catch (e) { /* leave default note */ }
}

function _setUniverseButtonsDisabled(disabled) {
  const a = document.getElementById('universeFullBtn');
  const b = document.getElementById('universeFilterBtn');
  if (a) a.disabled = disabled;
  if (b) b.disabled = disabled;
}

// mode: 'full' (option 3) | 'filter_enrich' (option 4)
async function doUniverse(mode) {
  let criteria = [];
  if (mode === 'filter_enrich') {
    try {
      criteria = collectUniverseCriteria();
    } catch (e) {
      alert(e.message);
      document.getElementById('filterBoxU').open = true;
      return;
    }
    if (!criteria.length) {
      alert('Tick at least one filter criterion (and give it a value) to filter & enrich.');
      document.getElementById('filterBoxU').open = true;
      return;
    }
  }
  _exportMode = 'full';  // universe jobs expose a single download_url

  _setUniverseButtonsDisabled(true);
  const panel = document.getElementById('exportPanel');
  panel.style.display = 'block';
  document.getElementById('exportLog').innerHTML = '';
  document.getElementById('progressFill').style.width = '0%';
  document.getElementById('progressFill').classList.remove('error');
  document.getElementById('exportStageLabel').textContent =
    mode === 'full' ? 'Building full universe (no PDF)…' : 'Filtering universe, then enriching survivors…';
  document.getElementById('exportStats').textContent = '';

  _closeExport();

  let jobId;
  try {
    const body = { mode, enable_ocr: true };
    if (_iqviaToken) body.iqvia_token = _iqviaToken;
    if (mode === 'filter_enrich') body.filter_criteria = criteria;
    const resp = await fetch('/universe/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
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
    _setUniverseButtonsDisabled(false);
    return;
  }

  const es = new EventSource(`/export/stream/${jobId}`);
  currentEventSource = es;
  es.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      _handleExportEvent(data);
      if (data.status === 'complete' || data.status === 'error') {
        _setUniverseButtonsDisabled(false);
        refreshUniverseCacheNote();
      }
    } catch (e) {
      _appendLog(`[parse error] ${event.data}`, 'log-err');
    }
  };
  es.onerror = () => {
    if (currentJobId !== jobId) return;
    _appendLog('Connection lost. Check server.', 'log-err');
    document.getElementById('exportStageLabel').textContent = 'Connection error';
    _setUniverseButtonsDisabled(false);
    es.close();
  };
}

// ---- Init ----
buildCriteriaRows();
buildUniverseCriteriaRows();
buildExtraCriteria('s', 'extraCriteria');
buildExtraCriteria('u', 'extraCriteriaU');
</script>
</body>
</html>
"""
