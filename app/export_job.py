"""Background export job with per-stage SSE progress reporting.

Runs all enrichment steps concurrently (bounded by semaphores), emitting
structured progress events into the job's event list. The SSE stream endpoint
reads from that list.

Multi-product support: job.queries may contain N ingredient names.  Each
product is searched sequentially (one at a time) so SSE progress is clear.
Total concurrency across all products is bounded by the same per-host
semaphores (DPD_SEMAPHORE etc.) as in single-product runs — the semaphores
are module-level, so no product can exceed its host's cap regardless of N.

Whole-run shared resources fetched ONCE:
  Patent.zip          — cached per-process inside enrich_patents; the first
                        call downloads it; subsequent calls within the same
                        process reuse it.  All DINs from all products are
                        collected and enriched in a single enrich_patents call.
  Data Protection     — fetch_data_protection_table() is called once and the
                        result passed to every product's workbook block.

Cross-product DIN deduplication: patents are enriched for each unique DIN
across all products exactly once.  Labeling is also per-unique-DIN.

Concurrency limits (all configurable via env vars):
  LABELING_SEMAPHORE  — concurrent PDF downloads + labeling extractions (default 8)
  DPD_SEMAPHORE       — already set in config.py (default 10)
  _DETAIL_SEM         — CPD patent-detail fetches, set in patents.py (default 3)
"""
from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import time
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from app.config import ENABLE_OCR, LABELING_STORE_TTL, SOURCE_TIMEOUT
from app.consistency import check_cross_source_consistency
from app.enrichment.data_protection import fetch_data_protection_table
from app.enrichment.labeling import enrich_labeling_batch_fast
from app.enrichment.patents import enrich_patents
from app.enrichment.store import is_labeling_stale
from app.enrichment.workbook import _is_excluded_din, build_workbook_multiproduct
from app.jobs import JobState, emit
from app.models import SearchMetadata, SearchResponse, SourceResult
from app.normalize import normalize_query
from app.sources.dpd import search_dpd
from app.sources.generic_submissions import search_generic_submissions
from app.sources.noc import search_noc
from app.sources.patent_register import search_patent_register

logger = logging.getLogger(__name__)

_LABEL_SEM_SIZE = int(os.getenv("LABELING_SEMAPHORE", "8"))

# Stage bounds: (overall_pct_start, overall_pct_end)
# Search is now weighted higher because N products are searched sequentially.
_STAGE_BOUNDS: dict[str, tuple[float, float]] = {
    "Search":         (0.00, 0.20),
    "Patents":        (0.20, 0.50),
    "Labeling":       (0.50, 0.85),
    "DataProtection": (0.85, 0.90),
    "Workbook":       (0.90, 1.00),
}


def _overall_pct(stage: str, done: int, total: int) -> float:
    lo, hi = _STAGE_BOUNDS[stage]
    frac = (done / total) if total > 0 else 1.0
    return round(lo + frac * (hi - lo), 3)


def _eta_s(stage_start: float, done: int, total: int) -> Optional[float]:
    if done == 0 or total == 0:
        return None
    elapsed = time.time() - stage_start
    rate = done / elapsed
    return round((total - done) / rate, 1) if rate > 0 else None


async def _search_one_product(
    job: JobState,
    query: str,
    product_idx: int,
    n_products: int,
    t0: float,
) -> tuple[SearchResponse, dict[str, Optional[str]]]:
    """Run all four sources for one ingredient, emit per-source SSE progress.

    Returns (SearchResponse, error_sources_dict).
    """
    def elapsed() -> float:
        return round(time.time() - t0, 1)

    prefix = f"[{query} ({product_idx+1}/{n_products})]"

    canonical, extra_terms = await normalize_query(query, "ingredient")

    search_done = 0
    sources: list[SourceResult] = []
    lock = asyncio.Lock()

    async def _timed_source(coro, source_name: str) -> SourceResult:
        try:
            return await asyncio.wait_for(coro, timeout=SOURCE_TIMEOUT)
        except asyncio.TimeoutError:
            return SourceResult(
                source=source_name, status="timeout",
                error_message=f"Timed out after {SOURCE_TIMEOUT}s",
            )
        except Exception as exc:
            return SourceResult(
                source=source_name, status="error", error_message=str(exc)
            )

    async def _do_source(name: str, coro) -> SourceResult:
        nonlocal search_done
        result = await _timed_source(coro, name)
        async with lock:
            sources.append(result)
            search_done += 1
            sd = search_done
        # Compute position in global search total: product_idx * 4 + sd
        global_done = product_idx * 4 + sd
        global_total = n_products * 4
        await emit(job, {
            "stage": "Search",
            "done": global_done,
            "total": global_total,
            "pct": _overall_pct("Search", global_done, global_total),
            "elapsed_s": elapsed(), "eta_s": None,
            "log": f"{prefix} {name}: {result.status} ({result.count} records)",
        })
        return result

    await asyncio.gather(
        _do_source("DPD",               search_dpd(canonical, "ingredient", extra_terms)),
        _do_source("GenericSubmissions", search_generic_submissions(canonical, "ingredient", extra_terms)),
        _do_source("NOC",               search_noc(canonical, "ingredient", extra_terms)),
        _do_source("PatentRegister",    search_patent_register(canonical, "ingredient", extra_terms)),
    )

    check_cross_source_consistency([r for s in sources for r in s.records])

    error_sources: dict[str, Optional[str]] = {
        s.source: s.error_message for s in sources if s.status == "error"
    }

    response = SearchResponse(
        metadata=SearchMetadata(
            query=query,
            field="ingredient",
            timestamp=datetime.now(timezone.utc).isoformat(),
            normalized_terms=[canonical] + extra_terms,
            per_source_status={s.source: s.status for s in sources},
        ),
        sources=sources,
    )
    return response, error_sources


async def run_export_job(
    job: JobState,
    allow_partial: bool,
    enable_ocr: bool,
) -> None:
    """Run the full export pipeline in the background, emitting progress events.

    Handles 1..N products via job.queries.  A single-product run is a
    degenerate case of N=1 — the output is a one-block side-by-side workbook.
    """
    t0 = time.time()

    def elapsed() -> float:
        return round(time.time() - t0, 1)

    queries = job.queries or [job.query]
    n = len(queries)

    try:
        # ── Stage 1: Search (all products, sequential) ────────────────────────
        await emit(job, {
            "stage": "Search", "done": 0, "total": n * 4,
            "pct": _overall_pct("Search", 0, n * 4),
            "elapsed_s": elapsed(), "eta_s": None,
            "log": (
                f"Searching {n} ingredient{'s' if n > 1 else ''} "
                f"({', '.join(repr(q) for q in queries)}) across 4 sources each…"
            ),
        })

        product_results: list[tuple[str, SearchResponse]] = []
        all_error_sources: dict[str, Optional[str]] = {}

        for i, query in enumerate(queries):
            await emit(job, {
                "stage": "Search", "done": i * 4, "total": n * 4,
                "pct": _overall_pct("Search", i * 4, n * 4),
                "elapsed_s": elapsed(), "eta_s": None,
                "log": f"Searching product {i+1}/{n}: {query!r}…",
            })
            response, err = await _search_one_product(job, query, i, n, t0)
            product_results.append((query, response))
            all_error_sources.update(err)

        if all_error_sources and not allow_partial:
            names = ", ".join(all_error_sources.keys())
            raise RuntimeError(
                f"Source(s) failed: {names}. Pass allow_partial=true to override."
            )

        await emit(job, {
            "stage": "Search", "done": n * 4, "total": n * 4,
            "pct": _overall_pct("Search", n * 4, n * 4),
            "elapsed_s": elapsed(), "eta_s": 0,
            "log": f"All {n} product search{'es' if n > 1 else ''} complete.",
        })

        # ── Enrichment scope: DINs that will actually appear in Sheet 1 ───────
        # Sheet 1 keeps only DINs present in BOTH DPD and NOC (see workbook
        # build_sheet1).  Enriching DPD-only or NOC-only DINs is wasted work — they
        # are dropped from the output.  Compute the DPD∩NOC intersection once and
        # use it to scope BOTH patent and labeling enrichment.  For metformin this
        # cuts the labeling set from every DPD DIN (hundreds) down to only the rows
        # the user will see.
        dpd_dins: set[str] = set()
        noc_dins: set[str] = set()
        for _, response in product_results:
            for s in response.sources:
                if s.source not in ("DPD", "NOC"):
                    continue
                for r in s.records:
                    if _is_excluded_din(r.din):
                        continue
                    (dpd_dins if s.source == "DPD" else noc_dins).add(r.din.strip())
        sheet_dins = dpd_dins & noc_dins

        # ── Stage 2: Patents (DPD∩NOC DINs across all products, ONE call) ─────
        # Collect unique DINs from DPD records, scoped to sheet_dins (order preserved).
        seen_dins: set[str] = set()
        all_valid_dins: list[str] = []
        for _, response in product_results:
            for s in response.sources:
                if s.source != "DPD":
                    continue
                for r in s.records:
                    din = r.din.strip() if r.din else None
                    if din and din in sheet_dins and din not in seen_dins:
                        seen_dins.add(din)
                        all_valid_dins.append(din)

        patent_total = len(all_valid_dins)

        if all_valid_dins:
            t_patent = time.time()

            await emit(job, {
                "stage": "Patents", "done": 0, "total": patent_total,
                "pct": _overall_pct("Patents", 0, patent_total),
                "elapsed_s": elapsed(), "eta_s": None,
                "log": (
                    f"Enriching patents for {patent_total} unique DINs "
                    f"across all {n} products (Patent.zip fetched once)…"
                ),
            })

            async def _on_patent_progress(done: int, total: int, log_line: str) -> None:
                await emit(job, {
                    "stage": "Patents", "done": done, "total": total,
                    "pct": _overall_pct("Patents", done, total),
                    "elapsed_s": elapsed(),
                    "eta_s": _eta_s(t_patent, done, total),
                    "log": log_line,
                })

            await enrich_patents(all_valid_dins, on_progress=_on_patent_progress)

            await emit(job, {
                "stage": "Patents", "done": patent_total, "total": patent_total,
                "pct": _overall_pct("Patents", patent_total, patent_total),
                "elapsed_s": elapsed(), "eta_s": 0,
                "log": f"Patents done ({patent_total} unique DINs across {n} products)",
            })
        else:
            await emit(job, {
                "stage": "Patents", "done": 0, "total": 0,
                "pct": _overall_pct("Patents", 1, 1),
                "elapsed_s": elapsed(), "eta_s": 0,
                "log": "No valid DINs — patents skipped",
            })

        # ── Stage 3: Labeling (per-unique-DIN, all products combined) ─────────
        # Collect unique DINs needing labeling across all products.
        din_map: dict[str, tuple[int, Optional[str]]] = {}
        for _, response in product_results:
            for s in response.sources:
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
                        if din_key not in sheet_dins:
                            continue  # not in Sheet 1 (DPD-only) — don't label it
                        if din_key not in din_map and is_labeling_stale(din_key, LABELING_STORE_TTL):
                            din_map[din_key] = (int(drug_code_raw), r.strength)
                    except (ValueError, TypeError):
                        pass

        dpd_dins_total = sum(
            1 for _, response in product_results
            for s in response.sources if s.source == "DPD"
            for r in s.records
            if not _is_excluded_din(r.din)
            and r.din.strip() in sheet_dins
            and r.source_specific.get("drug_code") is not None
        )
        label_cache_hits = dpd_dins_total - len(din_map)
        label_total = len(din_map)
        t_label = time.time()
        unique_drug_codes_count = len({dc for dc, _ in din_map.values()})

        await emit(job, {
            "stage": "Labeling", "done": 0, "total": max(label_total, 1),
            "pct": _overall_pct("Labeling", 0, max(label_total, 1)),
            "elapsed_s": elapsed(), "eta_s": None,
            "log": (
                f"Labeling: {label_total} new DINs to fetch "
                f"({label_cache_hits} store cache hits, "
                f"{unique_drug_codes_count} unique drug codes across {n} products); "
                f"ocr={'on' if enable_ocr else 'off'}, "
                f"concurrency={_LABEL_SEM_SIZE}"
            ),
        })

        if din_map:
            async def _on_label_progress(done: int, _total: int, din: str) -> None:
                await emit(job, {
                    "stage": "Labeling", "done": done, "total": label_total,
                    "pct": _overall_pct("Labeling", done, max(label_total, 1)),
                    "elapsed_s": elapsed(),
                    "eta_s": _eta_s(t_label, done, label_total),
                    "log": f"DIN {din} labeling complete",
                })

            await enrich_labeling_batch_fast(
                din_map,
                enable_ocr=enable_ocr,
                concurrency=_LABEL_SEM_SIZE,
                on_progress=_on_label_progress,
            )

        await emit(job, {
            "stage": "Labeling", "done": label_total, "total": max(label_total, 1),
            "pct": _overall_pct("Labeling", label_total, max(label_total, 1)),
            "elapsed_s": elapsed(), "eta_s": 0,
            "log": (
                f"Labeling done — {label_total} fetched, "
                f"{label_cache_hits} reused from store cache"
            ),
        })

        # ── Stage 4: Data Protection (fetched ONCE for all products) ──────────
        await emit(job, {
            "stage": "DataProtection", "done": 0, "total": 1,
            "pct": _overall_pct("DataProtection", 0, 1),
            "elapsed_s": elapsed(), "eta_s": None,
            "log": "Fetching Register of Innovative Drugs (shared across all products)…",
        })
        dp_table = await fetch_data_protection_table()
        await emit(job, {
            "stage": "DataProtection", "done": 1, "total": 1,
            "pct": _overall_pct("DataProtection", 1, 1),
            "elapsed_s": elapsed(), "eta_s": 0,
            "log": f"Data protection: {len(dp_table)} active entries",
        })

        # ── Stage 5: Workbook ─────────────────────────────────────────────────
        await emit(job, {
            "stage": "Workbook", "done": 0, "total": 1,
            "pct": _overall_pct("Workbook", 0, 1),
            "elapsed_s": elapsed(), "eta_s": None,
            "log": (
                f"Assembling {'multi-product ' if n > 1 else ''}two-tab workbook "
                f"({n} ingredient{'s' if n > 1 else ''}, vertical layout)…"
            ),
        })

        # Load IQVIA data from the server-side store.
        # Falls back to the persisted key ("_persisted") so IQVIA data survives server restarts.
        iqvia_df = None
        from app.main import _IQVIA_STORE, _IQVIA_PERSIST_KEY  # type: ignore[attr-defined]
        if job.iqvia_token:
            iqvia_df = _IQVIA_STORE.get(job.iqvia_token)
            if iqvia_df is None:
                # Session token expired (e.g. server reloaded) — try persisted upload
                iqvia_df = _IQVIA_STORE.get(_IQVIA_PERSIST_KEY)
        else:
            # No token sent; use persisted upload if one exists
            iqvia_df = _IQVIA_STORE.get(_IQVIA_PERSIST_KEY)
        if iqvia_df is not None:
            await emit(job, {
                "stage": "Workbook", "done": 0, "total": 1,
                "pct": _overall_pct("Workbook", 0, 1),
                "elapsed_s": elapsed(), "eta_s": None,
                "log": (
                    f"IQVIA data loaded ({len(iqvia_df)} collapsed groups) — "
                    "will match to DINs and append metric columns"
                ),
            })

        xlsx_bytes, sheet1_df, sheet2_df, _recon_df = build_workbook_multiproduct(
            product_results,
            source_errors=all_error_sources if allow_partial and all_error_sources else None,
            dp_table=dp_table,
            iqvia_df=iqvia_df,
        )

        # Snapshot combined flat DataFrames so the dashboard can display them.
        # These have a leading 'product' column when n > 1.
        job.sheet1_columns = list(sheet1_df.columns)
        job.sheet1_records = (
            sheet1_df.where(pd.notna(sheet1_df), None).to_dict("records")
        )
        job.sheet2_columns = list(sheet2_df.columns)
        job.sheet2_records = (
            sheet2_df.where(pd.notna(sheet2_df), None).to_dict("records")
        )

        fd, path = tempfile.mkstemp(suffix=".xlsx", prefix="cdn_drugs_")
        with os.fdopen(fd, "wb") as fh:
            fh.write(xlsx_bytes)
        job.result_path = path

        # ── Optional Stage 6: go/no-go screening (filtered export) ────────────
        # Runs over the just-built workbook data (sheet1_df/sheet2_df) — no extra
        # scraping. Produces a two-tab Summary + Detail filtered workbook.
        filtered_bytes: Optional[bytes] = None
        if job.filter_criteria:
            from app.enrichment.screen import build_filtered_workbook, parse_criteria
            try:
                criteria = parse_criteria(job.filter_criteria)
                filtered_bytes, summary_out, _detail_out, screen_warnings = (
                    build_filtered_workbook(sheet1_df, sheet2_df, criteria)
                )
                for label in screen_warnings:
                    await emit(job, {
                        "stage": "Workbook", "done": 1, "total": 1,
                        "pct": 1.0, "elapsed_s": elapsed(), "eta_s": 0,
                        "log": (
                            f"⚠ Product {label!r} has a blank dosage form — "
                            "grouped on its own; please verify the source data."
                        ),
                    })
                job.summary_columns = list(summary_out.columns)
                job.summary_records = (
                    summary_out.where(pd.notna(summary_out), None).to_dict("records")
                )
                fd_f, path_f = tempfile.mkstemp(suffix=".xlsx", prefix="cdn_drugs_filtered_")
                with os.fdopen(fd_f, "wb") as fh:
                    fh.write(filtered_bytes)
                job.filtered_result_path = path_f
                await emit(job, {
                    "stage": "Workbook", "done": 1, "total": 1,
                    "pct": 1.0, "elapsed_s": elapsed(), "eta_s": 0,
                    "log": (
                        f"Filtered workbook ready — {len(summary_out)} qualifying "
                        f"product(s), {len(filtered_bytes):,} bytes"
                    ),
                })
            except Exception as exc:
                logger.exception("Filtered screening failed for job %s", job.job_id)
                raise RuntimeError(f"Filtered screening failed: {exc}") from exc

        total_elapsed = elapsed()
        await emit(job, {
            "stage": "Workbook", "done": 1, "total": 1,
            "pct": 1.0, "elapsed_s": total_elapsed, "eta_s": 0,
            "log": f"Workbook ready — {len(xlsx_bytes):,} bytes",
        })

        job.status = "complete"
        complete_event = {
            "status": "complete",
            "download_url": f"/export/result/{job.job_id}",
            "elapsed_s": total_elapsed,
            "log": f"Done in {total_elapsed}s",
        }
        if filtered_bytes is not None:
            complete_event["filtered_download_url"] = f"/export/filtered-result/{job.job_id}"
        await emit(job, complete_event)

    except Exception as exc:
        logger.exception("Export job %s failed", job.job_id)
        job.status = "error"
        job.error = str(exc)
        await emit(job, {
            "status": "error",
            "message": str(exc),
            "elapsed_s": elapsed(),
        })
