# Canadian Drug Database Aggregator

A local web application that searches four Canadian health-product databases simultaneously and returns consolidated results viewable in a web UI or downloadable as XLSX.

## Architecture Overview

```
app/
  main.py                # FastAPI app, routes, HTML UI embedded
  config.py              # All configuration (URLs, timeouts, model name, TTLs)
  consistency.py         # Cross-source DIN consistency checker (warnings, not errors)
  sources/
    dpd.py               # Drug Product Database — official REST API only (no scraping)
    generic_submissions.py  # Static HTML table, httpx + BeautifulSoup
    noc.py               # NOC — CSRF-protected form POST, session cookie handling
    patent_register.py   # Patent Register — JSP form POST, SSL workaround
  normalize.py           # Static synonym map + optional Ollama llama3 expansion
  match.py               # Optional llama3 AI summary generation
  cache.py               # SQLite disk cache with TTL
  models.py              # Shared Pydantic result schema
  enrichment/
    store.py             # SQLite enrichment store (patents + labeling tables, $CACHE_DIR/enrichment.db)
    patents.py           # enrich-patents: fetch patent dates + cross-check Patent.zip bulk data
    labeling.py          # enrich-labeling: per-strength PDF field extraction (cite-or-blank)
    workbook.py          # build-workbook: two-tab enriched Excel export
tests/
  reconciliation/        # Completeness tests against DPD bulk extract (integration, slow)
    downloader.py        # Download + cache allfiles.zip with freshness check
    dpd_parser.py        # Parse drug.txt/ingred.txt, build DIN sets
    test_reconciliation.py  # Hard-fail if extract − pipeline > 0.5%
  test_cross_source_consistency.py  # DIN-keyed ingredient/brand agreement across sources
  test_fuzzy_precision.py           # Precision ≥ 0.95 on labeled fuzzy_pairs.csv
  test_enrich_patents.py            # Patent detail parsing + discrepancy resolution tests
  test_enrich_labeling.py           # Alpelisib PIQRAY golden test + no-fabrication assertion
  test_build_workbook.py            # Two-tab schema: NOC N/A exclusion, DIN sort, GSUR standalone
  fixtures/
    fuzzy_pairs.csv                 # Hand-labeled (query, option, expected_match) benchmark
    labeling/piqray_pages.json      # PIQRAY monograph fixture for golden labeling tests
    patent_register/detail_2709025.html  # Patent detail page fixture
```

## Running

```bash
cd /Users/vmalik/canadian-drug-db
python3 -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
# then open http://localhost:8000
```

## Configuration (environment variables)

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint |
| `OLLAMA_MODEL` | `llama3` | Model for normalization/summary |
| `DPD_SEMAPHORE` | `5` | Max concurrent DPD per-drug-code requests |
| `SOURCE_TIMEOUT` | `30.0` | Seconds before a source is marked timed-out |
| `CACHE_DIR` | `/tmp/canadian_drug_db_cache` | Disk cache location |
| `CACHE_TTL` | `14400` | Cache TTL in seconds (4h default) |

## Data Sources

### 1. Drug Product Database (DPD)
- **Method:** Official REST API — no scraping
- **Base URL:** `https://health-products.canada.ca/api/drug/`
- **Key API behaviour:** `/drugproduct/?id=<code>` returns a **dict**, not a list. `/status/?id=<code>` also returns a dict. All other enrichment endpoints (`/form/`, `/route/`, `/schedule/`) return lists.
- **Supports:** ingredient, brand, company, DIN
- **Rate limiting:** semaphore-capped at 5 concurrent requests

### 2. Generic Submissions Under Review
- **Method:** httpx + BeautifulSoup table parse
- **URL:** `https://www.canada.ca/en/health-canada/services/drug-health-product-review-approval/generic-submissions-under-review.html`
- **Table columns:** Medicinal Ingredient(s) | Company Name | Therapeutic Area | Year/Month Accepted
- **Note:** Table uses `wb-tables` class. Company "Not available" for pre-April 2024 entries.
- **Supports:** ingredient, company only (brand/DIN → unsupported)

### 3. Notice of Compliance (NOC)
- **Method:** Official JSON API — no form posts, no CSRF
- **API base:** `https://health-products.canada.ca/api/notice-of-compliance/`
- **Ingredient search flow:**
  1. `GET /medicinalingredient/?type=json&lang=en` — full list (~93 k rows, cached per TTL)
  2. Filter in memory where `noc_pi_medic_ingr_name` contains the queried term
  3. Expand each matched product to capture all its co-ingredients (combination products)
  4. For each unique `noc_number` (capped at 200): concurrently fetch
     - `GET /drugproduct/?id=<n>&type=json&lang=en` → `(noc_br_product_id, noc_br_din, noc_br_brandname)`
     - `GET /noticeofcompliancemain/?id=<n>&type=json&lang=en` → `noc_date, noc_manufacturer_name, noc_submission_class`
  5. Join `noc_pi_din_product_id == noc_br_product_id` to attach DIN to each product row
  6. Emit one `DrugRecord` per `(noc_number, product_id)`; `record_url = /noc-ac/nocInfo?id=<noc_number>`
- **Supports:** ingredient only — brand/company/DIN return `status="unsupported"` (the old CSRF scraper returned "too many records" for broad ingredient searches; the JSON API has no such limit)

### 4. Patent Register (PR-RDB)
- **Method:** Session cookie (JSESSIONID), POST to `/pr-rdb/search`
- **URL:** `https://pr-rdb.hc-sc.gc.ca/pr-rdb/`
- **SSL note:** Server certificate does not chain to trusted CA — `verify=False` is intentional
- **Ingredient field:** dropdown select — values must exactly match listed options. Our code fetches the dropdown and does substring + fuzzy matching.
- **Table columns:** Medicinal ingredient | Brand name | Strength | Dosage | DIN | Patent | CSP
- **Supports:** ingredient, brand, DIN (not company)

## LLM Usage (Ollama)

LLM is **never used for data extraction**. It is used only for:

1. **Ingredient synonym expansion** (before querying) — `normalize.py`
   - Primary: static synonym map (`_STATIC_SYNONYMS`)
   - Secondary: Ollama llama3 (disabled gracefully if Ollama is offline)
   - Searches run for the original term + all synonyms in parallel

2. **Plain-language summary** — `match.py`, only if `?summary=true`
   - Labeled clearly as AI-generated in the UI
   - Skipped entirely if Ollama is offline

### Ollama setup
```bash
# Install Ollama: https://ollama.com/
ollama pull llama3
ollama serve  # runs on http://localhost:11434
```

## Caching

SQLite cache at `$CACHE_DIR/cache.db`. Keyed by `sha256(source:query)`. Default TTL 4 hours. Cache is per-source-and-endpoint so partial results are cached independently.

To clear: delete `/tmp/canadian_drug_db_cache/cache.db`.

## Tests

```bash
# Offline suite (fast, no network — always run this in CI)
make test

# Integration suite (live government sites)
make test-live

# Completeness reconciliation against DPD nightly bulk extract (slow, nightly)
make reconcile
```

Three additional accuracy checks beyond fixture-based tests:
1. **Completeness reconciliation** (`make reconcile`) — downloads the DPD `allfiles.zip` bulk extract, builds extract DIN set per ingredient, hard-fails if pipeline misses >0.5% of extract DINs.
2. **Cross-source consistency** (`test_cross_source_consistency.py`) — for every DIN in ≥2 sources, asserts ingredient sets and brand names agree. Runs as warnings on every live search; also has offline fixture tests.
3. **Fuzzy precision/recall** (`test_fuzzy_precision.py`) — uses `tests/fixtures/fuzzy_pairs.csv` (hand-labeled) to assert Patent Register ingredient matcher precision ≥ 0.95. Fuzzy cutoff raised to 0.75 to eliminate false-positive drug matches.

See `tests/README.md` for full tier descriptions and tolerance rationale.

## Enrichment Pipeline (three chained commands)

All three commands share the SQLite enrichment store at `$CACHE_DIR/enrichment.db`. Run them in order after a search to build a fully enriched workbook.

### `enrich-patents`

`python -m app.enrichment.patents --dins DIN1 DIN2 ...`  
Or via API: `GET /api/export-enriched?q=alpelisib&field=ingredient` (runs enrich-patents automatically).

- Looks up each DIN's patent numbers from the Patent Register live search.
- For each patent, fetches the detail page for filing/grant/expiry dates.
- Downloads `Patent.zip` bulk extract and cross-checks every date field.
- **On discrepancy: uses the website value** and logs (DIN, patent_number, field, website_value, zip_value) to the `patent_discrepancies` table.
- Stores rows in `patents(din, patent_number, filing_date, grant_date, expiry_date)`.
- A DIN with no patents, or a patent with no dates, is recorded cleanly.

### `enrich-labeling` (accuracy-critical)

`python -m app.enrichment.labeling --drug-code CODE --din DIN --strength "50 mg"`

**Cite-or-blank rule:** every extracted value stores the page number it came from. If a field is not in the document, store exactly `"Not stated"` — never infer. Non-stated values have `_page = NULL`.

**Per-strength matching (required):** the DIN's strength (from DPD) is normalised and used to select only the matching Description block in §6. One PDF → one row per DIN with that DIN's physical descriptors.

**Sections read:**
- §6 Dosage Forms/Composition/Packaging: active_ingredient, excipients (core + coating separately), preservatives, pack_size, pack_style, colour, shape, size_mm, weight.
- §13 Pharmaceutical Information: pH. If only a pH-dependent solubility table is present, stores `"Not stated (pH-dependent solubility only)"`.

**Scanned PDFs:** if the PDF has no selectable text, every field → `"needs OCR / manual check"`, `needs_ocr=1`.

**Golden accuracy fixture:** `tests/fixtures/labeling/piqray_pages.json` contains the alpelisib/PIQRAY monograph text (human-verified). `TestPiqrayGolden` in `test_enrich_labeling.py` asserts exact match on all expected values and enforces the no-fabrication rule (every non-Not-stated field must cite a page present in the fixture).

### `build-workbook`

`python -m app.enrichment.workbook --q alpelisib --field ingredient`  
Or via the **single export API**: `GET /api/export?q=alpelisib&field=ingredient`

**There is one export. It is the two-sheet enriched workbook. Enrichment (patents + labeling) runs automatically — no flags needed.** The old multi-sheet format (`export.py`, per-source sheets, Combined, By Combination) has been deleted.

Produces a **two-tab XLSX**:

**Sheet 1 — "DPD + NOC + Patents"** (one row per DIN, sorted ascending):
- DPD fields: brand, company, ingredient, strength, form, route, status, record_url.
- NOC fields: noc_brand_name, noc_company, noc_date, noc_submission_type (joined by DIN).
- **NOC rows whose DIN is blank / "Not Applicable" / "N/A" are excluded entirely.**
- Patent block per DIN: patent_count, patent_numbers, earliest_filing_date, earliest_grant_date, latest_expiry_date, all_patents_detail.
- Labeling block per DIN: all 11 label fields + `_page` citation columns.

**Sheet 2 — "Generic Submissions"** (standalone, never joined to Sheet 1):
- GSUR records filtered to the queried ingredient (substring match, same normalisation).
- No DIN column. Completely separate from Sheet 1.

**Export API behaviour:**
- `GET /api/export?q=<term>&field=ingredient` — always produces the two-sheet workbook.
- Patent enrichment runs automatically before workbook assembly; already-stored patents are reused.
- Labeling enrichment runs automatically for DPD DINs not yet in the store; cached data is reused.
- `allow_partial=true` — build even when a source is in error (adds a `⚠ Source Status` warning sheet). Default is HTTP 409 on error.
- `no_results` from a source is truthful and never blocks the export.

### Enrichment store schema

`$CACHE_DIR/enrichment.db` (separate from the HTTP cache `cache.db`):

```sql
patents(din, patent_number, filing_date, grant_date, expiry_date, detail_url, fetched_at)
patent_discrepancies(din, patent_number, field, website_value, zip_value, logged_at)
labeling(din, drug_code, pdf_url, active_ingredient, active_ingredient_page, ..., needs_ocr, has_unverified, fetched_at)
```

## Dependencies

```
fastapi uvicorn httpx beautifulsoup4 openpyxl pandas pydantic python-multipart pdfplumber
```

Optional for tests:
```
pytest pytest-asyncio
```

## Ingredient-Combination Grouping

Results are grouped by each product's **full active-ingredient combination**, not by the single searched ingredient. Implemented in `app/grouping.py`.

### Grouping key definition

For every `DrugRecord`:

1. Use `all_ingredients` (a `list[str]` on the model) if non-empty; otherwise fall back to parsing the `ingredient` string on `;`.
2. Normalize each name: `strip()`, uppercase, collapse internal whitespace. Salt forms are kept as-is by default (e.g. `DIPHENHYDRAMINE HCL` is not further split).
3. Deduplicate and sort alphabetically → the sorted tuple is the **group key**. `{A, B}` and `{B, A}` hash to the same group.
4. **Group label**: join sorted names with `COMBINATION_SEPARATOR` (` + ` by default, defined in `config.py`).

### Salt-form normalization (config flag)

`NORMALIZE_SALT_FORMS` env var (default `0` / off). When enabled, salt forms would be stripped before matching. Currently implemented as a config flag; the normalization pass itself can be added to `_normalize_name()` in `grouping.py` when needed.

### Where all_ingredients is populated

- **DPD**: `_fetch_ingredients_by_code(drug_code)` fetches all active-ingredient rows for the product via `/activeingredient/?id=<code>`. This is always called (not just for ingredient searches) so every record has the complete combination, even for brand/company searches.
- **NOC**: split `ingredients` field on `;`.
- **Generic Submissions**: split on `;` if present; otherwise treat the whole string as one ingredient.
- **Patent Register**: split on `;`; typically a single ingredient per row.

### Group ordering (default)

1. The group whose label exactly equals the searched ingredient (single-ingredient exact match) — first.
2. Remaining groups by descending product count, ties broken alphabetically by label.
3. When no `searched_ingredient` is provided (brand/company/DIN search), sort purely by descending count then alphabetically.

### UI rendering

Combined view and each per-source tab render results as **collapsible `<details>` groups**. Group header shows: combination label, product count, company count, and company chips. First group is expanded by default.

### XLSX export

- Every data row in every tab has a `combination` column (the group label). Rows are sorted by combination within each sheet.
- A `By Combination` summary tab lists each combination with product count, company count, and a comma-separated company list.

## Known Limitations / Gotchas

- **NOC broad searches fail:** The NOC site returns an error for ingredient names that match too many records (>500). The UI surfaces this as an error with a helpful message.
- **Patent Register SSL:** The PR-RDB server has a certificate the standard CA bundle won't verify. We disable verification (`verify=False`) explicitly — this is equivalent to a user clicking "Proceed anyway" in a browser.
- **Patent Register ingredient matching:** The ingredient field is a dropdown of 469 specific values (exact salt forms). Fuzzy substring matching is used to find closest options, but niche ingredients may not appear in the Patent Register at all.
- **Patent Register no_results for generics:** Long-off-patent generics (e.g. metformin) return `no_results` from the Patent Register — this is truthful, not a bug.
- **Generic Submissions company names:** Pre-April 2024 entries show "Not available" as the company name — this is accurate, not a bug.
- **DPD concurrency cap:** With 242 drug codes for "metformin", all 242 are queried concurrently behind a semaphore of 5. Full results may take 5–15 seconds on first load (cached after).
- **NOC brand/company/DIN searches unsupported:** The JSON API migration only exposes ingredient search. Brand, company, and DIN searches return `status="unsupported"`. The old HTML/CSRF scraper would have handled these but was removed.
- **error vs no_results distinction:** `status="error"` means the source fetch failed (network/parse error). `status="no_results"` means the source responded successfully but returned no matching records. These must never be conflated. The `/api/export-enriched` endpoint refuses to build a workbook (HTTP 409) when any source is in `error` by default; pass `allow_partial=true` to override (adds a `⚠ Source Status` warning sheet). A `no_results` source does not block the build.
