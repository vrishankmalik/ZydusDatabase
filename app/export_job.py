"""Background export job with per-stage SSE progress reporting.

Runs all enrichment steps concurrently (bounded by semaphores), emitting
structured progress events into the job's event list. The SSE stream endpoint
reads from that list.

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

from app.config import ENABLE_OCR, SOURCE_TIMEOUT
from app.consistency import check_cross_source_consistency
from app.enrichment.data_protection import fetch_data_protection_table
from app.enrichment.labeling import enrich_labeling
from app.enrichment.patents import enrich_patents
from app.enrichment.store import get_labeling_for_din
from app.enrichment.workbook import _is_excluded_din, build_workbook
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
_STAGE_BOUNDS: dict[str, tuple[float, float]] = {
    "Search":         (0.00, 0.10),
    "Patents":        (0.10, 0.40),
    "Labeling":       (0.40, 0.85),
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


async def run_export_job(
    job: JobState,
    allow_partial: bool,
    enable_ocr: bool,
    enable_llm: bool,
) -> None:
    """Run the full export pipeline in the background, emitting progress events."""
    t0 = time.time()

    def elapsed() -> float:
        return round(time.time() - t0, 1)

    try:
        # ── Stage 1: Search ───────────────────────────────────────────────────
        await emit(job, {
            "stage": "Search", "done": 0, "total": 4,
            "pct": _overall_pct("Search", 0, 4),
            "elapsed_s": elapsed(), "eta_s": None,
            "log": f"Querying all four sources for {job.query!r} by {job.field}…",
        })

        canonical, extra_terms = await normalize_query(job.query, job.field)

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

        search_done = 0
        sources: list[SourceResult] = []
        lock = asyncio.Lock()

        async def _do_source(name: str, coro) -> SourceResult:
            nonlocal search_done
            result = await _timed_source(coro, name)
            async with lock:
                sources.append(result)
                search_done += 1
                sd = search_done
            await emit(job, {
                "stage": "Search", "done": sd, "total": 4,
                "pct": _overall_pct("Search", sd, 4),
                "elapsed_s": elapsed(), "eta_s": None,
                "log": f"{name}: {result.status} ({result.count} records)",
            })
            return result

        await asyncio.gather(
            _do_source("DPD",                search_dpd(canonical, job.field, extra_terms)),
            _do_source("GenericSubmissions",  search_generic_submissions(canonical, job.field, extra_terms)),
            _do_source("NOC",                 search_noc(canonical, job.field, extra_terms)),
            _do_source("PatentRegister",      search_patent_register(canonical, job.field, extra_terms)),
        )

        check_cross_source_consistency([r for s in sources for r in s.records])

        error_sources: dict[str, Optional[str]] = {
            s.source: s.error_message for s in sources if s.status == "error"
        }
        if error_sources and not allow_partial:
            names = ", ".join(error_sources.keys())
            raise RuntimeError(
                f"Source(s) failed: {names}. Pass allow_partial=true to override."
            )

        result = SearchResponse(
            metadata=SearchMetadata(
                query=job.query,
                field=job.field,
                timestamp=datetime.now(timezone.utc).isoformat(),
                normalized_terms=[canonical] + extra_terms,
                per_source_status={s.source: s.status for s in sources},
            ),
            sources=sources,
        )

        # ── Stage 2: Patents ──────────────────────────────────────────────────
        all_valid_dins = list(dict.fromkeys(
            r.din for s in sources for r in s.records
            if not _is_excluded_din(r.din)
        ))
        patent_total = len(all_valid_dins)

        if all_valid_dins:
            t_patent = time.time()
            patent_done_ref = [0]

            await emit(job, {
                "stage": "Patents", "done": 0, "total": patent_total,
                "pct": _overall_pct("Patents", 0, patent_total),
                "elapsed_s": elapsed(), "eta_s": None,
                "log": f"Enriching patents for {patent_total} DINs (Patent.zip + CPD)…",
            })

            async def _on_patent_progress(done: int, total: int, log_line: str) -> None:
                patent_done_ref[0] = done
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
                "log": f"Patents done ({patent_total} DINs processed)",
            })
        else:
            await emit(job, {
                "stage": "Patents", "done": 0, "total": 0,
                "pct": _overall_pct("Patents", 1, 1),
                "elapsed_s": elapsed(), "eta_s": 0,
                "log": "No valid DINs — patents skipped",
            })

        # ── Stage 3: Labeling ─────────────────────────────────────────────────
        # Collect only DPD DINs not yet in the labeling store
        din_map: dict[str, tuple[int, Optional[str]]] = {}
        for s in sources:
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

        # DINs already in store count as cache hits
        dpd_dins_total = sum(
            1 for s in sources if s.source == "DPD"
            for r in s.records
            if not _is_excluded_din(r.din) and r.source_specific.get("drug_code") is not None
        )
        label_cache_hits = dpd_dins_total - len(din_map)
        label_total = len(din_map)
        label_done_ref = [0]
        t_label = time.time()
        label_sem = asyncio.Semaphore(_LABEL_SEM_SIZE)
        label_lock = asyncio.Lock()

        await emit(job, {
            "stage": "Labeling", "done": 0, "total": label_total,
            "pct": _overall_pct("Labeling", 0, max(label_total, 1)),
            "elapsed_s": elapsed(), "eta_s": None,
            "log": (
                f"Labeling: {label_total} new DINs to fetch "
                f"({label_cache_hits} cache hits); "
                f"ocr={'on' if enable_ocr else 'off'}, "
                f"llm={'on' if enable_llm else 'off'}, "
                f"concurrency={_LABEL_SEM_SIZE}"
            ),
        })

        async def _enrich_one_label(din: str, drug_code: int, strength: Optional[str]) -> None:
            async with label_sem:
                await enrich_labeling(
                    din, drug_code, strength,
                    enable_ocr=enable_ocr,
                    enable_llm=enable_llm,
                )
            async with label_lock:
                label_done_ref[0] += 1
                done = label_done_ref[0]
            await emit(job, {
                "stage": "Labeling", "done": done, "total": label_total,
                "pct": _overall_pct("Labeling", done, max(label_total, 1)),
                "elapsed_s": elapsed(),
                "eta_s": _eta_s(t_label, done, label_total),
                "log": f"DIN {din} labeling complete",
            })

        if din_map:
            await asyncio.gather(*[
                _enrich_one_label(din, dc, st)
                for din, (dc, st) in din_map.items()
            ])

        await emit(job, {
            "stage": "Labeling", "done": label_total, "total": label_total,
            "pct": _overall_pct("Labeling", label_total, max(label_total, 1)),
            "elapsed_s": elapsed(), "eta_s": 0,
            "log": (
                f"Labeling done — {label_total} fetched, "
                f"{label_cache_hits} reused from cache"
            ),
        })

        # ── Stage 4: Data Protection ──────────────────────────────────────────
        await emit(job, {
            "stage": "DataProtection", "done": 0, "total": 1,
            "pct": _overall_pct("DataProtection", 0, 1),
            "elapsed_s": elapsed(), "eta_s": None,
            "log": "Fetching Register of Innovative Drugs…",
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
            "log": "Assembling two-tab workbook…",
        })

        xlsx_bytes = build_workbook(
            result,
            source_errors=error_sources if allow_partial and error_sources else None,
            dp_table=dp_table,
        )

        # Write to temp file — result endpoint streams it from disk
        fd, path = tempfile.mkstemp(suffix=".xlsx", prefix="cdn_drugs_")
        with os.fdopen(fd, "wb") as fh:
            fh.write(xlsx_bytes)
        job.result_path = path

        total_elapsed = elapsed()
        await emit(job, {
            "stage": "Workbook", "done": 1, "total": 1,
            "pct": 1.0, "elapsed_s": total_elapsed, "eta_s": 0,
            "log": f"Workbook ready — {len(xlsx_bytes):,} bytes",
        })

        job.status = "complete"
        await emit(job, {
            "status": "complete",
            "download_url": f"/export/result/{job.job_id}",
            "elapsed_s": total_elapsed,
            "log": f"Done in {total_elapsed}s",
        })

    except Exception as exc:
        logger.exception("Export job %s failed", job.job_id)
        job.status = "error"
        job.error = str(exc)
        await emit(job, {
            "status": "error",
            "message": str(exc),
            "elapsed_s": elapsed(),
        })
