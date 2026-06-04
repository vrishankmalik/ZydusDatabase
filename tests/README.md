# Test Suite — Canadian Drug DB Aggregator

## Two-tier model

| Tier | Marker | Network? | When to run |
|------|--------|----------|-------------|
| **Unit / offline** | `unit` (default) | No | Always — CI, pre-commit, local |
| **Integration / live** | `integration` | Yes | `make test-live` — nightly or on demand |

`make test` runs only the offline suite.  Tests without an explicit marker run by default.

---

## Tier descriptions

### Tier 1 — Connectivity smoke (`test_tier1_connectivity.py`)
One test per source.  Hits live endpoints, verifies HTTP 200 + parseable payload.
Skipped unless you run with `-m integration`.

### Tier 2 — Schema / contract (`test_tier2_schema.py`)
Validates raw API/HTML responses against explicit field sets (offline, using fixture files).
Also contains a **schema-drift canary** (integration): compares live field sets against
the committed schema — fails with a clear diff if Health Canada renames or removes a field.

**Fields asserted:**
- DPD `activeingredient`: `drug_code`, `ingredient_name`, `strength`, `strength_unit`
- DPD `drugproduct`: `drug_code`, `brand_name`, `drug_identification_number`, `company_name`,
  `ai_group_no`, `number_of_ais`, `class_name`, `last_update_date`
- NOC parsed row: `products`, `manufacturer`, `noc_date`, `ingredients`, `dins`,
  `record_url`, `noc_with_conditions`
- Patent Register: `ingredient`, `brand`, `strength`, `dosage`, `din`, `patent`, `csp`
- GSUR: `ingredient`, `company`, `therapeutic_area`, `date_accepted`

### Tier 3 — Golden values (`test_tier3_golden.py`)
Uses stable discontinued/historical records whose DINs and brand names never change:

| DIN | Expected brand | Why stable |
|-----|----------------|------------|
| `00326925` | `SINEQUAN` | Cancelled post-market; won't change |
| `00000019` | `PLACIDYL CAP 200MG` | Cancelled pre-market; won't change |
| NOC `3369` | `NORINYL 1/50 21DAY` | Historical 1984 record |

Guards against: field-mapping mistakes (wrong column index, renamed key).

### Tier 4 — Critical regressions (`test_tier4_regression.py`)

| Test | Bug guarded |
|------|-------------|
| `test_dpd_acetaminophen_no_silent_cap` | Prior 150-row truncation hid thousands of products |
| `test_noc_din_attachment_rate` | NOC scraper was reading the wrong HTML column for DINs |
| `test_multi_din_split_three` | Multi-DIN strings like `02535742,; 02535750` must explode |
| `test_dpd_nonsense_query_*` | Empty result must be `no_results`, not an exception |
| `test_*_no_html_leakage` | Parser must strip tags before storing field values |

### Tier 5 — Normalization & join units (`test_tier5_normalization.py`)
Pure function tests, no I/O:
- DIN normalization: padding, stripping, splitting multi-value strings
- Ingredient synonym map
- Combination key: `{A,B}` and `{B,A}` hash identically; sorted; deduped
- `join_by_din`: DIN-level merge across DPD + NOC, one-to-many aggregation,
  DIN-less rows kept as standalone rows with `match_method="no_din"`
- Fuzzy ingredient matching (Patent Register dropdown)

### Tier 6 — Robustness (`test_tier6_robustness.py`)
Mocked HTTP failures — each source must return a structured `SourceResult`
(status `"error"` or `"no_results"`), never raise an exception:
- 500 / 503 responses
- Connection timeouts
- Truncated / invalid JSON or HTML
- Unsupported search fields

### Tier 7 — Data quality (`test_tier7_quality.py`)
Format rules enforced on fixture data (always) and live data (integration):
- Every DIN matches `^\d{8}$`
- All dates parse to ISO `YYYY-MM-DD`
- No critical column (brand, company, ingredient) is 100% null for a non-empty result
- Per-source record counts logged for trend monitoring

### Cache & determinism (`test_cache_determinism.py`)
- SQLite cache round-trips values correctly and respects TTL
- A cache hit avoids re-fetching (HTTP client is never called)
- Same offline query → byte-identical serialized output

---

## Running

```bash
# Offline suite (CI default, fast, no network)
make test

# Integration suite (live network — nightly / on demand)
make test-live

# Both tiers combined
make test-all

# With coverage report
make coverage
```

## Refreshing fixtures

Fixtures in `tests/fixtures/` are hand-curated snapshots.  To re-record them
from live government endpoints:

```bash
make refresh-fixtures
```

This calls `tests/scripts/refresh_fixtures.py`, which fetches real responses
and overwrites the committed fixture files.  Review the diff before committing
to make sure schema changes are expected.

## Fixture layout

```
tests/fixtures/
  dpd/
    activeingredient_metformin.json       # /activeingredient/?ingredientname=metformin
    drugproduct_code_<code>.json          # /drugproduct/?id=<code>
    form_<code>.json / route_<code>.json  # enrichment endpoints
    status_<code>.json / schedule_<code>.json
    activeingredient_code_<code>.json     # /activeingredient/?id=<code>
  noc/
    csrf_page.html                        # GET /noc-ac/?lang=eng (CSRF form)
    results_norinyl.html                  # POST /noc-ac/doSearch (NORINYL brand)
    results_glucophage.html               # POST /noc-ac/doSearch (Glucophage brand)
    results_too_many.html                 # "too many records" error page
    results_no_results.html               # empty results page
    results_multi_din.html                # multi-DIN regression fixture
  generic_submissions/
    page.html                             # full GSUR HTML page
  patent_register/
    index.html                            # index page with ingredient/brand dropdowns
    results_metformin.html                # results for METFORMIN HYDROCHLORIDE
    results_no_results.html               # empty results page
```

---

## Accuracy tests (beyond fixture coverage)

### Completeness reconciliation (`tests/reconciliation/`)

**What it guards against:** Result caps, missing pages, source-drift bugs that cause the
pipeline to silently return a subset of available products.  No fixture-based test can catch
a 150-vs-250 DIN gap because fixtures are hand-curated at the number you see.

**How it works:** Downloads the Health Canada DPD nightly bulk extract (`allfiles.zip`,
~1.4 MB, cached 12 h).  Applies the same case-insensitive substring rule the live API uses
to build the extract's DIN set for a given ingredient.  Compares against the pipeline's DIN
set.  Hard-fails if `extract − pipeline` exceeds 0.5% (≈1 in 200 DINs) — a tolerance chosen
to absorb one-day nightly-refresh timing skew without masking real bugs.

**Tolerance rationale:** 0.5% is tight enough to catch any result-cap regression (a 150/250
gap = 37% miss rate, 74× the tolerance) while being loose enough to pass when the bulk
extract and live API differ by one product due to same-day updates.

**Sample ingredients tested:** `acetaminophen` (high-volume, ~250 DINs),
`ibuprofen` (mid, ~68), `azithromycin` (rare, ~24),
`cetirizine hydrochloride` (salt/multi-word, ~29).

**Run:** `make reconcile`  (requires live network; slow on first run, fast from cache)

**NOC note:** Health Canada does not publish a downloadable bulk export for the NOC database.
The NOC reconciliation stub is included in the code and will activate if/when those files
become publicly available.

### Cross-source consistency (`test_cross_source_consistency.py`)

**What it guards against:** Column-index shifts and mis-joins that look correct in isolation
but produce wrong values when the same DIN is seen from two sources.  E.g., a NOC parser
reading the wrong HTML column would pass all single-source tests but disagree with DPD on
ingredient name for the same DIN.

**How it works:** For every DIN present in ≥2 sources, asserts that the normalized ingredient
sets and case-normalized brand names agree.  Disagreements are emitted as `WARNING` log
entries during live runs (never raise exceptions) and are caught as hard assertion failures
in the fixture test (which deliberately injects a mismatched record).

**Wired into:** `app/main.py` — runs on every search response; results only go to the server
log.

### Fuzzy matcher precision / recall (`test_fuzzy_precision.py`)

**What it guards against:** The Patent Register ingredient dropdown matcher silently linking
a user query to the *wrong* dropdown option, corrupting patent-linkage decisions.  False
positives ("aspirin" matching "atorvastatin") are more dangerous than false negatives.

**How it works:** Loads `tests/fixtures/fuzzy_pairs.csv` (25 hand-labeled
query/option/expected_match triples), runs `_find_matching_options` from `patent_register.py`
against the fixture dropdown, and computes precision and recall.  **Asserts precision ≥ 0.95.**
Recall is printed but not hard-asserted (a missed link is safer than a wrong one).

**Threshold tuning:** The fuzzy cutoff was raised from 0.6 → 0.75 after the benchmark
revealed that `"canagliflozn"` (typo) matched `"EMPAGLIFLOZIN"` (different drug) at 0.6.
At 0.75 that false positive is eliminated; precision = 1.0 on the benchmark.

---

## Adding a new test

1. Decide the tier (schema? golden? regression? normalization?).
2. Add to the appropriate `test_tier*.py` file.
3. If the test is offline, keep it network-free; inject the `mock_dpd` / `mock_noc`
   / `mock_gsur` / `mock_patent_register` fixture as needed.
4. If the test hits live endpoints, add `@pytest.mark.integration`.
5. If you need a new fixture value, add a fixture file and update the dispatcher
   in `conftest.py` if needed.

## Coverage target

`make coverage` targets ≥90% line coverage on:
- `app/sources/dpd.py`
- `app/sources/noc.py`
- `app/sources/generic_submissions.py`
- `app/sources/patent_register.py`
- `app/normalize.py`
- `app/grouping.py`
- `app/din_utils.py`
- `app/join.py`
- `app/cache.py`
