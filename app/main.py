"""Canadian Drug Database Aggregator — FastAPI main application."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from app.config import SOURCE_TIMEOUT
from app.consistency import check_cross_source_consistency
from app.enrichment.data_protection import fetch_data_protection_table
from app.enrichment.labeling import enrich_labeling_batch
from app.enrichment.patents import enrich_patents
from app.enrichment.store import get_labeling_for_din
from app.enrichment.workbook import _is_excluded_din, build_workbook
from app.match import generate_summary
from app.models import SearchMetadata, SearchResponse, SourceResult
from app.normalize import normalize_query
from app.sources.dpd import search_dpd
from app.sources.generic_submissions import search_generic_submissions
from app.sources.noc import search_noc
from app.sources.patent_register import search_patent_register

app = FastAPI(title="Canadian Drug Database Aggregator", version="1.0.0")


async def _timed_source(
    coro,
    source_name: str,
) -> SourceResult:
    """Run a source coroutine with a global timeout."""
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

    # Normalize / synonym-expand
    canonical, extra_terms = await normalize_query(q, field)

    # Fan out to all four sources concurrently
    dpd_task = _timed_source(
        search_dpd(canonical, field, extra_terms), "DPD"
    )
    gen_task = _timed_source(
        search_generic_submissions(canonical, field, extra_terms), "GenericSubmissions"
    )
    noc_task = _timed_source(
        search_noc(canonical, field, extra_terms), "NOC"
    )
    pr_task = _timed_source(
        search_patent_register(canonical, field, extra_terms), "PatentRegister"
    )

    dpd_result, gen_result, noc_result, pr_result = await asyncio.gather(
        dpd_task, gen_task, noc_task, pr_task
    )

    sources = [dpd_result, gen_result, noc_result, pr_result]

    # Cross-source consistency warnings (logged; never fail the request)
    all_records = [rec for s in sources for rec in s.records]
    check_cross_source_consistency(all_records)

    # Optional AI summary
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
    allow_partial: bool = Query(
        False,
        description=(
            "Build even if a source failed (adds '⚠ Source Status' warning sheet). "
            "By default the endpoint refuses with HTTP 409 when any source is in error."
        ),
    ),
) -> Response:
    """Two-sheet enriched workbook download. Patent and labeling enrichment run automatically.

    Sheet 1 — 'DPD + NOC + Patents': one row per DIN, NOC N/A rows excluded,
    patent dates and labeling fields populated from the enrichment store.
    Sheet 2 — 'Generic Submissions': GSUR filtered to the queried ingredient.

    If any source returns status='error' and allow_partial=False (default), the
    endpoint returns HTTP 409 naming the failed source(s).
    A no_results source is truthful and does not block the build.
    """
    result = await search(q=q, field=field, summary=False)

    # Guard: refuse to silently build a partial workbook when a source failed.
    # "error" = fetch failed; "no_results" = genuine empty (allowed to proceed).
    error_sources: dict[str, Optional[str]] = {
        s.source: s.error_message
        for s in result.sources
        if s.status == "error"
    }
    if error_sources and not allow_partial:
        names = ", ".join(error_sources.keys())
        details = "; ".join(
            f"{k}: {v or 'unknown error'}" for k, v in error_sources.items()
        )
        raise HTTPException(
            status_code=409,
            detail=(
                f"Source(s) failed: {names} — refusing to build a partial workbook. "
                f"Pass allow_partial=true to override. Details: {details}"
            ),
        )

    # Collect valid DINs for enrichment
    all_valid_dins = [
        r.din
        for s in result.sources
        for r in s.records
        if not _is_excluded_din(r.din)
    ]

    # Patent enrichment — always runs; the module skips DINs already stored
    if all_valid_dins:
        await enrich_patents(all_valid_dins)

    # Labeling enrichment — only for DPD records (need drug_code + strength);
    # skip DINs already in the store so repeat exports are fast.
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
    filename = f"canadian_drugs_{q.replace(' ', '_')}_{field}.xlsx"
    return Response(
        content=xlsx_bytes,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse(content=_HTML_UI)


# --- Embedded single-page UI -----------------------------------------------
_HTML_UI = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Canadian Drug Database Aggregator</title>
<style>
  :root {
    --red: #cc0000;
    --maple: #d62828;
    --bg: #f8f9fa;
    --card: #ffffff;
    --border: #dee2e6;
    --text: #212529;
    --muted: #6c757d;
    --ok: #198754;
    --warn: #e67e22;
    --err: #dc3545;
    --badge-dpd: #1565C0;
    --badge-gen: #6A1B9A;
    --badge-noc: #2E7D32;
    --badge-pr: #C62828;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: var(--bg); color: var(--text); }
  header { background: var(--maple); color: white; padding: 16px 24px; }
  header h1 { font-size: 1.4rem; font-weight: 700; }
  header p { font-size: 0.85rem; opacity: 0.85; margin-top: 2px; }
  .container { max-width: 1200px; margin: 0 auto; padding: 24px 16px; }
  .search-box { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 20px; margin-bottom: 20px; }
  .row { display: flex; gap: 12px; flex-wrap: wrap; align-items: flex-end; }
  .field-group { display: flex; flex-direction: column; gap: 4px; }
  .field-group label { font-size: 0.8rem; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }
  input[type=text], select { padding: 8px 12px; border: 1px solid var(--border); border-radius: 6px; font-size: 0.95rem; outline: none; }
  input[type=text]:focus, select:focus { border-color: var(--maple); box-shadow: 0 0 0 3px rgba(214,40,40,.12); }
  #query { min-width: 280px; }
  .btn { padding: 9px 20px; border: none; border-radius: 6px; cursor: pointer; font-size: 0.95rem; font-weight: 600; }
  .btn-primary { background: var(--maple); color: white; }
  .btn-primary:hover { background: #b52222; }
  .btn-export { background: #1a7340; color: white; margin-left: 8px; }
  .btn-export:hover { background: #155a33; }
  .btn:disabled { opacity: .5; cursor: not-allowed; }
  .summary-bar { background: #fff3cd; border: 1px solid #ffc107; border-radius: 6px; padding: 10px 16px; font-size: 0.9rem; margin-bottom: 16px; display: none; }
  .ai-summary { background: #e8f4fd; border: 1px solid #90caf9; border-radius: 6px; padding: 12px 16px; font-size: 0.9rem; margin-bottom: 16px; display: none; }
  .ai-summary strong { color: #1565C0; }
  .tabs { display: flex; gap: 0; border-bottom: 2px solid var(--border); margin-bottom: 16px; flex-wrap: wrap; }
  .tab-btn { padding: 8px 18px; border: 1px solid var(--border); border-bottom: none; background: var(--bg); cursor: pointer; font-size: 0.9rem; margin-bottom: -2px; border-radius: 6px 6px 0 0; font-weight: 500; }
  .tab-btn.active { background: var(--card); border-bottom: 2px solid var(--card); font-weight: 700; }
  .tab-pane { display: none; }
  .tab-pane.active { display: block; }
  .source-header { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
  .badge { padding: 3px 10px; border-radius: 20px; color: white; font-size: 0.78rem; font-weight: 700; }
  .badge-DPD { background: var(--badge-dpd); }
  .badge-GenericSubmissions { background: var(--badge-gen); }
  .badge-NOC { background: var(--badge-noc); }
  .badge-PatentRegister { background: var(--badge-pr); }
  .status-ok { color: var(--ok); font-weight: 600; }
  .status-no_results { color: var(--muted); }
  .status-error, .status-timeout { color: var(--err); }
  .status-unsupported { color: var(--warn); }
  .error-box { background: #fff5f5; border: 1px solid #fc8181; border-radius: 6px; padding: 12px; font-size: 0.88rem; color: var(--err); }
  .info-box { background: #f0f4ff; border: 1px solid #c3dafe; border-radius: 6px; padding: 12px; font-size: 0.88rem; color: var(--muted); }
  table { width: 100%; border-collapse: collapse; font-size: 0.88rem; background: var(--card); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
  thead tr { background: var(--bg); }
  th { padding: 10px 12px; text-align: left; font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); border-bottom: 2px solid var(--border); white-space: nowrap; }
  td { padding: 9px 12px; border-bottom: 1px solid var(--border); vertical-align: top; }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: #f5f5f5; }
  .record-link { color: var(--maple); text-decoration: none; font-size: 0.8rem; }
  .record-link:hover { text-decoration: underline; }
  .spinner { display: inline-block; width: 18px; height: 18px; border: 3px solid rgba(214,40,40,.3); border-top-color: var(--maple); border-radius: 50%; animation: spin .8s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .loading-msg { display: flex; align-items: center; gap: 10px; color: var(--muted); padding: 24px 0; }
  footer { text-align: center; color: var(--muted); font-size: 0.8rem; padding: 24px; }
  .checkbox-label { display: flex; align-items: center; gap: 6px; font-size: 0.88rem; cursor: pointer; }
  @media (max-width: 600px) { .row { flex-direction: column; } }
  /* Combination groups */
  .combo-group { margin-bottom: 8px; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
  .combo-header { display: flex; align-items: center; gap: 12px; padding: 12px 16px; background: var(--bg); cursor: pointer; list-style: none; user-select: none; flex-wrap: wrap; }
  .combo-header::-webkit-details-marker { display: none; }
  .combo-header::before { content: '▶'; font-size: 0.7rem; color: var(--muted); transition: transform .15s; min-width: 10px; display: inline-block; }
  details[open] > .combo-header::before { transform: rotate(90deg); }
  .combo-label { font-weight: 700; font-size: 0.93rem; }
  .combo-stats { font-size: 0.8rem; color: var(--muted); white-space: nowrap; }
  .combo-companies { display: flex; gap: 4px; flex-wrap: wrap; }
  .company-chip { background: #e9ecef; border-radius: 12px; padding: 2px 8px; font-size: 0.74rem; color: var(--muted); }
  .company-chip.more { background: transparent; border: 1px solid var(--border); }
  .combo-body table { border-radius: 0; border: none; border-top: 1px solid var(--border); }
</style>
</head>
<body>
<header>
  <h1>🍁 Canadian Drug Database Aggregator</h1>
  <p>Search DPD · Generic Submissions Under Review · Notice of Compliance · Patent Register</p>
</header>
<div class="container">
  <div class="search-box">
    <div class="row">
      <div class="field-group" style="flex:1">
        <label for="query">Search Term</label>
        <input type="text" id="query" placeholder="e.g. metformin, acetaminophen, Lipitor…" autocomplete="off"/>
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
      <div class="field-group" style="justify-content:flex-end">
        <label class="checkbox-label">
          <input type="checkbox" id="summary"/> AI Summary (requires Ollama)
        </label>
      </div>
      <div class="field-group" style="justify-content:flex-end">
        <button class="btn btn-primary" id="searchBtn" onclick="doSearch()">Search</button>
        <button class="btn btn-export" id="exportBtn" onclick="doExport()" disabled>⬇ Download XLSX</button>
      </div>
    </div>
  </div>
  <div class="summary-bar" id="summaryBar"></div>
  <div class="ai-summary" id="aiSummary"></div>
  <div id="results"></div>
</div>
<footer>Data sourced from Health Canada public databases. Accuracy relies on deterministic extraction — no AI-generated data fields.</footer>

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

function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function fieldVal(record, key) {
  if (key.startsWith('_')) {
    return (record.source_specific || {})[key.slice(1)] || '';
  }
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

  // Summary bar
  const bar = document.getElementById('summaryBar');
  const counts = sources.map(s => `${SOURCE_LABELS[s.source]||s.source}: <strong>${s.count}</strong>`).join(' &nbsp;|&nbsp; ');
  bar.innerHTML = `Searched for <strong>"${escHtml(metadata.query)}"</strong> by <em>${escHtml(metadata.field)}</em> &nbsp;·&nbsp; ${counts}`;
  if (metadata.normalized_terms?.length > 1) {
    bar.innerHTML += ` &nbsp;·&nbsp; <em>Also searched: ${metadata.normalized_terms.slice(1).map(escHtml).join(', ')}</em>`;
  }
  bar.style.display = 'block';

  // AI Summary
  const aiDiv = document.getElementById('aiSummary');
  if (ai_summary) {
    aiDiv.innerHTML = `<strong>🤖 AI Summary</strong> <em style="font-size:.75rem;color:#666">(AI-generated, may be imprecise — verify against raw data)</em><br/>${ai_summary}`;
    aiDiv.style.display = 'block';
  } else {
    aiDiv.style.display = 'none';
  }

  // Build tabs
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

  document.getElementById('results').innerHTML = tabsHtml + panesHtml;
  document.getElementById('exportBtn').disabled = false;
}

function switchTab(name) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
  document.getElementById(`tab-${name}`)?.classList.add('active');
  document.getElementById(`pane-${name}`)?.classList.add('active');
}

async function doSearch() {
  const q = document.getElementById('query').value.trim();
  const field = document.getElementById('field').value;
  const summary = document.getElementById('summary').checked;
  if (!q) { alert('Please enter a search term.'); return; }

  document.getElementById('searchBtn').disabled = true;
  document.getElementById('exportBtn').disabled = true;
  document.getElementById('summaryBar').style.display = 'none';
  document.getElementById('aiSummary').style.display = 'none';
  document.getElementById('results').innerHTML = '<div class="loading-msg"><div class="spinner"></div>Querying all four databases concurrently…</div>';

  try {
    const url = `/api/search?q=${encodeURIComponent(q)}&field=${field}&summary=${summary}`;
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

function doExport() {
  const q = document.getElementById('query').value.trim();
  const field = document.getElementById('field').value;
  if (!q) return;
  window.location = `/api/export?q=${encodeURIComponent(q)}&field=${field}`;
}

document.getElementById('query').addEventListener('keydown', e => {
  if (e.key === 'Enter') doSearch();
});
</script>
</body>
</html>
"""
