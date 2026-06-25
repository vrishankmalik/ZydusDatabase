# Railway Deployment Checklist — Zydus Drug Intelligence Platform

Part 2 of the pre-launch gate. Automation covers what it can; this checklist is the
human sign-off for what it can't. Work top to bottom **after** you create the
Railway service, then run the live test suite against the deployed URL.

> Measured on this machine 2026-06-25: full DPD universe = **13,550 products**,
> **55** dosage-form bases, **308** active data-protection register rows; full
> universe + IQVIA build **peak RSS ≈ 359 MB** (fits the 512 MB free tier after the
> streaming-workbook fix below; was 529 MB before).

---

## How to run the live tests after deploy

```powershell
$env:BASE_URL = "https://<your-app>.up.railway.app"
# pandas-WMI shim only needed on this dev host, NOT on Railway:
$env:PYTHONPATH = "C:\Users\vmalik\AppData\Local\Temp\pyshim;<repo-root>"
<python> -m pytest tests/deploy -m integration -v
# Optional heavy PDF-enrichment case (minutes): add  $env:RUN_HEAVY = "1"
```

The **static** readiness tests (`tests/deploy/test_deploy_static_readiness.py`) run
in the normal offline suite and need neither BASE_URL nor network.

---

## 1. `$PORT` binding & start command — **risk R1** *(was a real gap; fixed)*

- [x] A `Procfile` exists with `web: uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1 --timeout-keep-alive 75`.
      *(`run_dev.py` hardcodes :8000 and is a local-only launcher — it must NOT be the deploy entrypoint.)*
- [ ] Railway "Start Command" is empty (so it uses the Procfile) **or** set to the same line.
- [ ] After deploy: `GET /` returns 200 → `test_health_root_responds`.
- [ ] Health check path in Railway settings set to `/` (returns 200, no heavy work).

## 2. Memory ceiling — **risk R7** *(was a blocker; fixed — now fits free tier)*

- [x] **Fits the 512 MB free tier.** Measured peak ≈ **359 MB** (baseline ≈ 110 MB),
      ~150 MB headroom. *Before the fix it was 529 MB and would OOM-kill on 512 MB.*
- [x] Fix: the universe workbook writer now streams via openpyxl **write_only** mode
      (`build_universe_workbook` in `app/enrichment/universe.py`) instead of holding
      the whole 13.5k-row styled cell tree in RAM. Output is byte-for-byte identical.
      The remaining ~359 MB peak is the IQVIA xlsx *read*, which happens at upload time
      (a separate request from the universe build), not stacked with the workbook write.
- [x] Regression guard: `tests/deploy/test_deploy_memory_local.py` asserts peak <
      `MEM_CEILING_MB` (**default 512**) on a full build.
- [ ] Concurrent full builds stack memory — see §6; keep to one build at a time
      (single instance already enforces this).

## 3. Single worker / single instance — **risk R2 (highest prod-break likelihood)**

The app keeps **in-process state**: the job store (`app/jobs.py`) and the IQVIA
upload-token store (`_IQVIA_STORE` in `app/main.py`). These are NOT shared across
workers or instances.

- [x] Start command uses `--workers 1` → `test_start_command_is_single_worker`.
- [ ] Railway **horizontal scaling = 1 instance** (no autoscale / replicas).
      With >1 instance, a job started on instance A can't be polled/downloaded from B,
      and an IQVIA token uploaded to A is invisible to B.
- [ ] Accept that a restart mid-job loses the job and its temp workbook (consistent —
      both vanish together). Long jobs should be re-run after a redeploy.

## 4. Ephemeral / non-shared filesystem — **risk R2**

Audited every disk write; all are best-effort caches or process-lifetime temp files
rooted at `CACHE_DIR` or the OS tempdir, and every one re-creates / re-pulls on miss:

| Write site | Path | On wipe |
|---|---|---|
| HTTP cache (`cache.py`) | `$CACHE_DIR/cache.db` | re-fetched |
| Enrichment store (`store.py`) | `$CACHE_DIR/enrichment.db` | schema recreated, re-fetched |
| Universe extract (`universe.py`) | `$CACHE_DIR/universe/*.txt` | re-downloaded (`_extract_is_fresh`→False) |
| IQVIA persist (`main.py`) | `$CACHE_DIR/iqvia_collapsed.pkl` | best-effort (try/except); in-mem token still works |
| Exclusion CSV (`main.py`) | `$CACHE_DIR/<q>_excluded.csv` | read-after-write across requests; graceful 404 if gone |
| Result workbooks (`export_job.py`, `universe_job.py`) | OS tempdir via `mkstemp` | process-lifetime; gone on restart with its job |

- [ ] Set `CACHE_DIR` to a writable container path (e.g. `/tmp/zydus_cache`, or a
      mounted **Railway Volume** at `/data/zydus_cache` if you want caches to survive
      restarts and skip the cold re-pull). A Volume is optional, not required.
- [x] Reset tolerates a missing dir; store self-heals on a fresh FS → `test_reset_universe_cache_tolerates_missing_dir`, `test_enrichment_store_recreates_on_fresh_filesystem`.
- [ ] After deploy: `test_reset_all_caches_then_rebuild` (reset → rebuild from wiped FS).

## 5. Long-request / proxy timeout — **risk R6**

- [x] SSE stream sends a `: keepalive` every 15 s and `X-Accel-Buffering: no`.
- [x] Procfile sets `--timeout-keep-alive 75` (keeps idle keep-alive sockets open).
- [ ] After deploy: `test_full_universe_job_streams_to_completion_and_downloads`
      (full build completes over SSE + downloads) and `test_sse_stream_survives_idle_keepalive_window`.
- [ ] Worst-case PDF run: `RUN_HEAVY=1 pytest -k worst_case_option4` — confirm the
      multi-minute option-4 job is not cut by the proxy. If it IS cut, raise the
      Railway proxy/request timeout or move long jobs to a background worker.
- [ ] Confirm Railway does not impose a hard max request duration shorter than your
      heaviest job (option-4 PDF enrichment can run many minutes).

## 6. Cold start & concurrency — **risk R7**

- [ ] First request triggers the allfiles.zip pull (cold start, ~30–60 s) — acceptable.
- [ ] After deploy: `test_two_concurrent_universe_requests_single_coherent_build`
      (two simultaneous builds converge on one coherent cache, no divergent output).
- [ ] Watch memory during concurrent builds (see §2) — two full builds can stack.

## 7. Outbound egress — **risk R6**

Railway's network (TLS, proxies, rate limits, egress allow-list) differs from your
laptop. The app fetches: DPD REST API, NOC JSON API, GSUR HTML, Patent Register
(`verify=False` — intentional, untrusted CA chain), the Register of Innovative Drugs
HTML, and `allfiles.zip`.

- [ ] After deploy: `test_dosage_forms_proves_outbound_egress` (allfiles.zip) and
      `test_search_egress_abrocitinib_anchor` (DPD + NOC).
- [ ] Confirm outbound HTTPS is not blocked and the Patent Register `verify=False`
      path is not broken by a TLS-inspecting proxy.

## 8. Env / secrets / config — **risk R5**

- [x] `CORS_ALLOWED_ORIGINS` is env-driven → `test_cors_origins_are_env_driven`.
- [ ] Set `CORS_ALLOWED_ORIGINS=https://app.powerbi.com,https://*.fabric.microsoft.com`
      (and your own domain) for production instead of the default `*`.
- [ ] `ENABLE_OCR=0` unless poppler + tesseract are in the image (they are not by default).
- [ ] `LLM_PROVIDER` unset (NullProvider) unless Azure OpenAI is configured.
- [x] No app/ module makes a localhost self-call → `test_app_code_has_no_runtime_localhost_dependency`.
- [ ] Python ≥ 3.11 in the build image (the codebase uses `list[str]` / `X | None` syntax).
- [ ] `requirements.txt` installs cleanly in the Railway build (fastapi, uvicorn, httpx,
      beautifulsoup4, openpyxl, pandas, pydantic, python-multipart, pdfplumber).

---

### One-line go/no-go after deploy
```
BASE_URL=<url> pytest tests/deploy -m integration -v   # all green = ship
```
