"""Patent enrichment — Stage 1: dates from the Canadian Patents Database (CPD).

Flow:
  1. For each DIN, POST a DIN search to the Patent Register to find patent numbers.
     Log if zero patents found — confirms the PR-RDB linkage isn't silently failing.
  2. For each patent number, scrape the CPD summary page for filed/issued/expiry dates:
       https://brevets-patents.ic.gc.ca/opic-cipo/cpd/eng/patent/{N}/summary.html
     (strip any "CA" prefix, commas, spaces from the number first)
  3. Expiry: prefer an explicit "expiry" row in the Event History table.
     If absent: filing >= 1989-10-01 → +20 years; earlier → flag "verify: pre-1989".
  4. Download Patent.zip bulk extract and cross-check every date field.
     On discrepancy: use the CPD (website) value; log to patent_discrepancies.
  5. Aggregate across all patents for the DIN:
       earliest_filing_date = min(filing dates)
       earliest_grant_date  = min(issue dates)
       latest_expiry_date   = max(expiry dates)

This stage touches NO PDF — pure HTML scraping + date math.

CLI:
  python -m app.enrichment.patents --dins 02498014 02498022
"""
from __future__ import annotations

import asyncio
import csv
import io
import logging
import re
import zipfile
from datetime import datetime
from typing import Callable, Optional

import httpx
from bs4 import BeautifulSoup

from app.cache import cache_get, cache_set
from app.config import HTTP_TIMEOUT, PATENT_BASE, USER_AGENT
from app.enrichment.store import get_patents_for_din, log_discrepancy, upsert_patent
from app.sources.patent_register import (
    _get_dropdown_options as _pr_get_session,
    _parse_results_table as _pr_parse_table,
    _post_search as _pr_post_search,
)

logger = logging.getLogger(__name__)

_BASE_HOST = "https://pr-rdb.hc-sc.gc.ca"
# Patent.zip lives under /patent/, not /pr-rdb/ — separate from the PATENT_BASE search URL
_PATENT_ZIP_URL = f"{_BASE_HOST}/patent/Patent.zip"
_PR_DETAIL_URL = f"{_BASE_HOST}/pr-rdb/patentDetails"

# Canadian Patents Database — authoritative source for filing/grant/expiry dates
_CPD_BASE = "https://brevets-patents.ic.gc.ca/opic-cipo/cpd/eng/patent"

_DATE_FIELDS = ("filing_date", "grant_date", "expiry_date")

# CPD is currently broken (instant 308 redirect to bare IP) so fetches return
# immediately.  Raise to 20 so the fast-fail path doesn't serialise 222 patents.
_DETAIL_SEM = asyncio.Semaphore(20)

# In-flight dedup: many DINs share the same patent number.  Without this,
# 222 DINs with the same patent would fire 222 concurrent CPD fetches.
# Each waiter shares the first caller's result via a Future.
_PATENT_DETAIL_INFLIGHT: dict[str, "asyncio.Future[dict]"] = {}
_PATENT_DETAIL_INFLIGHT_LOCK = asyncio.Lock()

# Earliest date of 20-year patent term rule in Canada
_TWENTY_YEAR_CUTOFF = datetime(1989, 10, 1)

# Month abbreviation map for CPD date parsing
_MONTHS: dict[str, str] = {
    "january": "01", "february": "02", "march": "03", "april": "04",
    "may": "05", "june": "06", "july": "07", "august": "08",
    "september": "09", "october": "10", "november": "11", "december": "12",
    "jan": "01", "feb": "02", "mar": "03", "apr": "04", "jun": "06",
    "jul": "07", "aug": "08", "sep": "09", "oct": "10", "nov": "11", "dec": "12",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _split_merged_patent_number(token: str) -> list[str]:
    """Defensive split: a 14-digit string is almost certainly two 7-digit patents merged.

    e.g. "26458103022097" → ["2645810", "3022097"]
    Returns a list with one element for a normal-length patent number.
    """
    clean = _clean_patent_number(token)
    if re.match(r"^\d{14}$", clean):
        a, b = clean[:7], clean[7:]
        logger.warning("Merged 14-digit patent token %r split → %s, %s", clean, a, b)
        return [a, b]
    return [clean] if clean else []


def _parse_detail_page(html: str) -> dict[str, Optional[str]]:
    """Parse a PR-RDB patent detail page for filing, grant, and expiry dates.

    Expected format (from fixture tests/fixtures/patent_register/detail_2709025.html):
        <tr><th>Filing Date</th><td>2008-12-10</td></tr>
        <tr><th>Date Granted</th><td>2014-08-26</td></tr>
        <tr><th>Expiry Date</th><td>2028-12-10</td></tr>
    """
    soup = BeautifulSoup(html, "html.parser")
    result: dict[str, Optional[str]] = {
        "filing_date": None,
        "grant_date": None,
        "expiry_date": None,
    }
    for row in soup.find_all("tr"):
        cells = row.find_all(["th", "td"])
        if len(cells) < 2:
            continue
        label = cells[0].get_text(strip=True).lower()
        val = cells[-1].get_text(strip=True) or None
        if "filing" in label:
            result["filing_date"] = val
        elif "grant" in label or "issued" in label:
            result["grant_date"] = val
        elif "expir" in label:
            result["expiry_date"] = val
    return result


def _clean_patent_number(pn: str) -> str:
    """Strip leading 'CA', commas, spaces from a raw patent number string.

    '2,709,025' → '2709025'
    'CA 2709025' → '2709025'
    """
    pn = re.sub(r"(?i)^\s*ca\s*", "", pn.strip())
    pn = re.sub(r"[,\s]+", "", pn)
    return pn.strip()


def _parse_cpd_date(raw: str) -> Optional[str]:
    """Parse a variety of CPD date formats to YYYY-MM-DD.

    Handles:
      "1999-03-15"  →  ISO passthrough
      "1999 03 15"  →  space-separated
      "March 15, 1999"  →  Month Day, Year
      "15 March 1999"   →  Day Month Year
      "19990315"        →  compact
    Returns None if unparseable.
    """
    if not raw:
        return None
    raw = raw.strip()
    if not raw:
        return None

    # ISO: YYYY-MM-DD
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", raw)
    if m:
        return raw

    # Space-separated: YYYY MM DD
    m = re.match(r"^(\d{4})\s+(\d{1,2})\s+(\d{1,2})$", raw)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"

    # Compact: YYYYMMDD
    m = re.match(r"^(\d{4})(\d{2})(\d{2})$", raw)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    # "March 15, 1999" or "March 15 1999"
    m = re.match(r"^(\w+)\s+(\d{1,2}),?\s+(\d{4})$", raw)
    if m and m.group(1).lower() in _MONTHS:
        return f"{m.group(3)}-{_MONTHS[m.group(1).lower()]}-{int(m.group(2)):02d}"

    # "15 March 1999"
    m = re.match(r"^(\d{1,2})\s+(\w+)\s+(\d{4})$", raw)
    if m and m.group(2).lower() in _MONTHS:
        return f"{m.group(3)}-{_MONTHS[m.group(2).lower()]}-{int(m.group(1)):02d}"

    # Generic: first YYYY-MM-DD-like substring
    m = re.search(r"(\d{4})[/\-](\d{1,2})[/\-](\d{1,2})", raw)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"

    logger.debug("CPD date unparseable: %r", raw)
    return None


def _compute_expiry(filing_date_str: str, patent_number: str) -> Optional[str]:
    """Compute expiry from filing date when Event History has no explicit expiry row."""
    try:
        fd = datetime.strptime(filing_date_str[:10], "%Y-%m-%d")
    except ValueError:
        return None
    if fd >= _TWENTY_YEAR_CUTOFF:
        # 20-year term from filing date
        try:
            expiry = fd.replace(year=fd.year + 20)
        except ValueError:
            # Feb 29 edge case
            expiry = fd.replace(year=fd.year + 20, day=28)
        computed = expiry.strftime("%Y-%m-%d")
        logger.debug("Patent %s: no explicit expiry → computed 20yr from filing → %s", patent_number, computed)
        return computed
    else:
        flag = "verify: pre-1989 filing (17yr-from-issue vs 20yr-from-filing)"
        logger.info("Patent %s filed %s (pre-1989): cannot auto-compute expiry → %s", patent_number, filing_date_str, flag)
        return flag


# ── Stage 1: CPD scraping ─────────────────────────────────────────────────────

async def _fetch_cpd_dates(patent_number: str) -> dict[str, Optional[str]]:
    """Scrape the CPD summary page for filing, grant, and expiry dates.

    URL: https://brevets-patents.ic.gc.ca/opic-cipo/cpd/eng/patent/{N}/summary.html

    Strategy 1: scan all <tr> elements for (22)/(45) labels in th/td pairs.
    Strategy 2: full-text regex scan as fallback.
    Expiry: prefer Event History table row whose description contains 'expir';
            fall back to computing from filing date.
    """
    clean_pn = _clean_patent_number(patent_number)
    url = f"{_CPD_BASE}/{clean_pn}/summary.html"
    result: dict[str, Optional[str]] = {
        "filing_date": None, "grant_date": None, "expiry_date": None,
        "detail_url": url,
    }

    async with _DETAIL_SEM:
        try:
            async with httpx.AsyncClient(follow_redirects=False) as client:
                r = await client.get(
                    url,
                    headers={"User-Agent": USER_AGENT, "Accept": "text/html"},
                    timeout=5.0,  # CPD is broken (timeout/redirect) — fail fast
                )
            # The CPD server currently issues a 308 redirect to a bare IP address
            # (https://<IP>/) with no path — following it causes a 20-second timeout
            # per patent.  Detect any 3xx immediately and skip; ZIP data is the
            # authoritative fallback.
            if r.is_redirect:
                logger.warning(
                    "CPD patent %s: server redirected to %r (broken server config) — "
                    "skipping CPD; ZIP bulk data will be used as fallback",
                    patent_number, r.headers.get("location", "?"),
                )
                return result
            if r.status_code == 404:
                logger.warning("CPD page not found for patent %s (clean: %s) → %s",
                               patent_number, clean_pn, url)
                return result
            if r.status_code != 200:
                logger.warning("CPD page HTTP %d for patent %s", r.status_code, patent_number)
                return result
            soup = BeautifulSoup(r.text, "html.parser")
        except Exception as exc:
            logger.warning("CPD fetch failed for %s: %s", patent_number, exc)
            return result

    # Strategy 1: labeled table rows
    # CPD uses patterns like:
    #   <th scope="row">(22) Filed:</th> <td>1999-03-15</td>
    #   <th scope="row">(45) Issued:</th> <td>2005-08-23</td>
    for row in soup.find_all("tr"):
        cells = row.find_all(["th", "td"])
        if len(cells) < 2:
            continue
        label = cells[0].get_text(" ", strip=True)
        value = cells[-1].get_text(" ", strip=True)
        if re.search(r"\(22\)|filed\s+date|date\s+filed", label, re.IGNORECASE):
            result["filing_date"] = _parse_cpd_date(value)
        elif re.search(r"\(45\)|issued|granted|date\s+of\s+grant", label, re.IGNORECASE):
            result["grant_date"] = _parse_cpd_date(value)

    # Strategy 2: regex fallback over full page text
    if not result["filing_date"] or not result["grant_date"]:
        text = soup.get_text(" ", strip=True)
        if not result["filing_date"]:
            m = re.search(
                r"\(22\)[^:]*:\s*(\d{4}[-\s/]\d{1,2}[-\s/]\d{1,2}|\w+\s+\d{1,2},?\s+\d{4})",
                text, re.IGNORECASE,
            )
            if m:
                result["filing_date"] = _parse_cpd_date(m.group(1))
        if not result["grant_date"]:
            m = re.search(
                r"\(45\)[^:]*:\s*(\d{4}[-\s/]\d{1,2}[-\s/]\d{1,2}|\w+\s+\d{1,2},?\s+\d{4})",
                text, re.IGNORECASE,
            )
            if m:
                result["grant_date"] = _parse_cpd_date(m.group(1))

    # Expiry from Event History table: find any row whose description says "expir"
    # The table typically has Date | Code | Description columns.
    for row in soup.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 2:
            continue
        row_text = " ".join(c.get_text(" ", strip=True) for c in cells)
        if re.search(r"expir", row_text, re.IGNORECASE):
            # Date should be in the first or last cell — try both
            for candidate_cell in (cells[0], cells[-1]):
                date_str = candidate_cell.get_text(strip=True)
                parsed = _parse_cpd_date(date_str)
                if parsed:
                    result["expiry_date"] = parsed
                    logger.debug("Patent %s: explicit expiry from Event History → %s",
                                 patent_number, parsed)
                    break
            if result["expiry_date"]:
                break

    # Compute expiry if still missing
    if result["expiry_date"] is None and result["filing_date"]:
        result["expiry_date"] = _compute_expiry(result["filing_date"], patent_number)

    return result


# ── PR-RDB fallback (kept for non-CPD patents / legacy) ──────────────────────

async def _fetch_pr_detail_dates(patent_number: str, session_id: str) -> dict[str, Optional[str]]:
    """Fetch dates from the PR-RDB detail page as a fallback when CPD has nothing."""
    result: dict[str, Optional[str]] = {"filing_date": None, "grant_date": None, "expiry_date": None}
    try:
        async with httpx.AsyncClient(verify=False) as client:  # noqa: S501
            r = await client.get(
                _PR_DETAIL_URL,
                params={"patentNumber": patent_number},
                cookies={"JSESSIONID": session_id} if session_id else {},
                headers={"User-Agent": USER_AGENT, "Accept": "text/html"},
                timeout=HTTP_TIMEOUT,
                follow_redirects=True,
            )
        if r.status_code == 200 and r.text.strip():
            soup = BeautifulSoup(r.text, "html.parser")
            for row in soup.find_all("tr"):
                cells = row.find_all(["th", "td"])
                if len(cells) < 2:
                    continue
                label = cells[0].get_text(strip=True).lower()
                val = cells[1].get_text(strip=True) or None
                if "filing" in label:
                    result["filing_date"] = val
                elif "grant" in label or "issued" in label:
                    result["grant_date"] = val
                elif "expir" in label:
                    result["expiry_date"] = val
    except Exception as exc:
        logger.debug("PR-RDB detail fallback failed for %s: %s", patent_number, exc)
    return result


async def fetch_patent_detail(
    patent_number: str,
    session_id: str = "",
) -> dict[str, Optional[str]]:
    """Return {filing_date, grant_date, expiry_date, detail_url}.

    Primary source: CPD summary page.
    Fallback: PR-RDB detail page (legacy, dates often absent there).
    Result is cached by patent_number.

    In-flight dedup: many DINs share the same patent; without this each DIN fires
    a separate CPD fetch for the same patent number, serialised by _DETAIL_SEM(3).
    For 222 DINs with the same LIPITOR patent at 20s CPD timeout each = 25 minutes.
    With dedup: one fetch per unique patent_number; all other callers wait on the Future.
    """
    # v2: busts stale all-None entries cached when CPD was timing out
    cache_key = f"cpd_detail_v2:{patent_number}"
    cached = cache_get("patent_detail", cache_key)
    if cached is not None:
        return cached

    # In-flight dedup — if another coroutine is already fetching this patent, wait for it.
    async with _PATENT_DETAIL_INFLIGHT_LOCK:
        if patent_number in _PATENT_DETAIL_INFLIGHT:
            fut: asyncio.Future[dict] = _PATENT_DETAIL_INFLIGHT[patent_number]
            is_leader = False
        else:
            loop = asyncio.get_event_loop()
            fut = loop.create_future()
            _PATENT_DETAIL_INFLIGHT[patent_number] = fut
            is_leader = True

    if not is_leader:
        return await asyncio.shield(fut)

    try:
        # Primary: CPD
        result = await _fetch_cpd_dates(patent_number)

        # Fallback: if CPD returned nothing, try PR-RDB
        if all(result.get(k) is None for k in _DATE_FIELDS):
            pr_dates = await _fetch_pr_detail_dates(patent_number, session_id)
            for field in _DATE_FIELDS:
                if pr_dates.get(field):
                    result[field] = pr_dates[field]
            if any(result.get(k) for k in _DATE_FIELDS):
                logger.info("Patent %s: CPD had no dates; used PR-RDB fallback", patent_number)

        # Only cache when we have at least one real date — a transient failure (CPD
        # redirect, network error) must not poison the cache with all-None results.
        if any(result.get(k) for k in _DATE_FIELDS):
            cache_set("patent_detail", cache_key, result)

        fut.set_result(result)
        return result
    except Exception as exc:
        fut.set_exception(exc)
        raise
    finally:
        async with _PATENT_DETAIL_INFLIGHT_LOCK:
            _PATENT_DETAIL_INFLIGHT.pop(patent_number, None)


# ── Patent.zip bulk data ──────────────────────────────────────────────────────

async def load_patent_zip() -> dict[str, dict[str, Optional[str]]]:
    """Download and parse the Patent.zip bulk extract.

    Returns dict[patent_number → {filing_date, grant_date, expiry_date}].
    Cached for 24 h.
    """
    # v2: busts stale cache from prior run that stored only 1 entry
    cached = cache_get("patent_zip", "bulk_v2")
    if cached is not None:
        return cached

    try:
        async with httpx.AsyncClient(verify=False) as client:  # noqa: S501
            r = await client.get(
                _PATENT_ZIP_URL,
                headers={"User-Agent": USER_AGENT},
                timeout=120.0,
                follow_redirects=True,
            )
            r.raise_for_status()
            zip_bytes = r.content
    except Exception as exc:
        logger.warning("Patent.zip download failed: %s", exc)
        return {}

    return _parse_patent_zip(zip_bytes)


def _parse_zip_date(raw: Optional[str]) -> Optional[str]:
    """Parse a Patent.zip date (MM/DD/YYYY) to ISO YYYY-MM-DD.

    Dates in patent-service_e.txt are formatted as MM/DD/YYYY (US-style).
    """
    if not raw or not raw.strip():
        return None
    raw = raw.strip()
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", raw)
    if m:
        return f"{m.group(3)}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"
    return _parse_cpd_date(raw)


def _parse_patent_zip(zip_bytes: bytes) -> dict[str, dict[str, Optional[str]]]:
    """Parse Patent.zip and return {patent_number → {filing_date, grant_date, expiry_date}}.

    Uses patent-service_e.txt (PATENT_NUMBER, FILING_DATE, DATE_GRANTED, EXPIRATION_DATE).
    Dates are MM/DD/YYYY in the bulk extract; converted to ISO YYYY-MM-DD.
    """
    result: dict[str, dict[str, Optional[str]]] = {}
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            names_lower = {n.lower(): n for n in zf.namelist()}
            patent_file = names_lower.get("patent-service_e.txt")
            if not patent_file:
                logger.warning("Patent.zip: patent-service_e.txt not found; files: %s", zf.namelist())
                return {}
            with zf.open(patent_file) as raw_f:
                f = io.TextIOWrapper(raw_f, encoding="utf-8-sig", errors="replace")
                reader = csv.DictReader(f)
                for row in reader:
                    pn = (row.get("PATENT_NUMBER") or "").strip()
                    if not pn:
                        continue
                    clean_pn = _clean_patent_number(pn)
                    if not clean_pn:
                        continue
                    result[clean_pn] = {
                        "filing_date": _parse_zip_date(row.get("FILING_DATE")),
                        "grant_date": _parse_zip_date(row.get("DATE_GRANTED")),
                        "expiry_date": _parse_zip_date(row.get("EXPIRATION_DATE")),
                    }
    except Exception as exc:
        logger.warning("Patent.zip parse failed: %s", exc)
        return {}

    cache_set("patent_zip", "bulk_v2", result, ttl=60 * 60 * 24)
    return result


def _parse_patent_zip_by_din(zip_bytes: bytes) -> dict[str, list[str]]:
    """Parse Patent.zip and return {DIN (8-digit zero-padded) → [patent_numbers]}.

    The ZIP uses a two-file join:
      drugs_e.txt          — DRUG_ID → DIN mapping
      patent-service_e.txt — DRUG_ID → PATENT_NUMBER (one row per DIN-patent pair)

    The two files are joined on DRUG_ID to produce the DIN → patent list.
    """
    result: dict[str, list[str]] = {}
    if not zip_bytes:
        return result
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            names_lower = {n.lower(): n for n in zf.namelist()}

            # Step 1: build DRUG_ID → DIN from drugs_e.txt
            drug_id_to_din: dict[str, str] = {}
            drugs_file = names_lower.get("drugs_e.txt")
            if not drugs_file:
                logger.warning("Patent.zip: drugs_e.txt not found; files: %s", zf.namelist())
                return result
            with zf.open(drugs_file) as raw_f:
                f = io.TextIOWrapper(raw_f, encoding="utf-8-sig", errors="replace")
                reader = csv.DictReader(f)
                for row in reader:
                    did = (row.get("DRUG_ID") or "").strip()
                    raw_din = (row.get("DIN") or "").strip()
                    if not did or not raw_din:
                        continue
                    try:
                        din_norm = str(int(raw_din)).zfill(8)
                    except ValueError:
                        din_norm = re.sub(r"[^0-9]", "", raw_din).zfill(8)
                    if not din_norm.strip("0"):
                        continue
                    drug_id_to_din[did] = din_norm
            logger.debug("Patent.zip: built DRUG_ID→DIN map with %d entries", len(drug_id_to_din))

            # Step 2: build DIN → [patent_numbers] from patent-service_e.txt via DRUG_ID join
            patent_file = names_lower.get("patent-service_e.txt")
            if not patent_file:
                logger.warning("Patent.zip: patent-service_e.txt not found; files: %s", zf.namelist())
                return result
            with zf.open(patent_file) as raw_f:
                f = io.TextIOWrapper(raw_f, encoding="utf-8-sig", errors="replace")
                reader = csv.DictReader(f)
                for row in reader:
                    did = (row.get("DRUG_ID") or "").strip()
                    raw_pn = (row.get("PATENT_NUMBER") or "").strip()
                    if not did or not raw_pn:
                        continue
                    din = drug_id_to_din.get(did)
                    if not din:
                        continue
                    for pn in _split_merged_patent_number(raw_pn):
                        bucket = result.setdefault(din, [])
                        if pn not in bucket:
                            bucket.append(pn)
    except Exception as exc:
        logger.warning("Patent.zip by-DIN parse failed: %s", exc)
    return result


async def load_patent_zip_din_map() -> dict[str, list[str]]:
    """Download Patent.zip and return {DIN → [patent_numbers]}, cached 24 h.

    Uses the same bulk extract as load_patent_zip() but keyed by DIN rather
    than patent number, giving an accurate per-DIN patent list with no
    string-concatenation artefacts from HTML scraping.
    """
    cached = cache_get("patent_zip_din_map", "v1")
    if cached is not None:
        return cached

    try:
        async with httpx.AsyncClient(verify=False) as client:  # noqa: S501
            r = await client.get(
                _PATENT_ZIP_URL,
                headers={"User-Agent": USER_AGENT},
                timeout=120.0,
                follow_redirects=True,
            )
            r.raise_for_status()
            zip_bytes = r.content
    except Exception as exc:
        logger.warning("Patent.zip download failed (by-DIN): %s", exc)
        return {}

    result = _parse_patent_zip_by_din(zip_bytes)
    cache_set("patent_zip_din_map", "v1", result, ttl=60 * 60 * 24)
    return result


async def load_patent_zip_both() -> tuple[dict[str, dict], dict[str, list[str]]]:
    """Download Patent.zip ONCE and return (by_patent_number, by_din) maps.

    Both load_patent_zip() and load_patent_zip_din_map() previously downloaded
    the same ~20 MB ZIP file independently on a cold cache — this function
    checks both caches first and issues at most one HTTP request.
    """
    by_patent: Optional[dict] = cache_get("patent_zip", "bulk_v2")
    by_din: Optional[dict] = cache_get("patent_zip_din_map", "v1")
    if by_patent is not None and by_din is not None:
        logger.debug("Patent.zip: both caches warm — no download needed")
        return by_patent, by_din

    missing = [k for k, v in [("bulk_v2", by_patent), ("v1", by_din)] if v is None]
    logger.info("Patent.zip: downloading once for missing cache key(s): %s", missing)
    try:
        async with httpx.AsyncClient(verify=False) as client:  # noqa: S501
            r = await client.get(
                _PATENT_ZIP_URL,
                headers={"User-Agent": USER_AGENT},
                timeout=120.0,
                follow_redirects=True,
            )
            r.raise_for_status()
            zip_bytes = r.content
    except Exception as exc:
        logger.warning("Patent.zip combined download failed: %s", exc)
        return by_patent or {}, by_din or {}

    if by_patent is None:
        by_patent = _parse_patent_zip(zip_bytes)  # cache_set done inside _parse_patent_zip

    if by_din is None:
        by_din = _parse_patent_zip_by_din(zip_bytes)
        cache_set("patent_zip_din_map", "v1", by_din, ttl=60 * 60 * 24)

    return by_patent, by_din


# ── DIN → patent number mapping ───────────────────────────────────────────────

async def _din_to_patent_numbers(
    din: str,
    session_id: str,
    zip_by_din: Optional[dict[str, list[str]]] = None,
) -> list[str]:
    """Return list of patent numbers for a DIN.

    Primary source: Patent.zip DIN map (if provided) — avoids HTML concat artefacts.
    Fallback: PR-RDB live search, with defensive split applied to every token.
    """
    # Patent.zip is the preferred source: one row per DIN-patent pair, no concat bugs.
    if zip_by_din is not None:
        zip_patents = zip_by_din.get(din)
        if zip_patents is not None:
            logger.debug("DIN %s → %d patent(s) from Patent.zip: %s", din, len(zip_patents), zip_patents)
            return zip_patents

    # Fall back to PR-RDB live search
    cached = cache_get("pr_din_patents", din)
    if cached is not None:
        return cached

    try:
        html = await _pr_post_search(din, "din", session_id)
        if "query has no results" in html.lower():
            cache_set("pr_din_patents", din, [])
            logger.info("DIN %s: Patent Register returned no results (patent_count=0 is correct)", din)
            return []
        rows = _pr_parse_table(html)
        raw_tokens = list({r["patent"] for r in rows if r.get("patent")})
        # Apply defensive split — in case any cell contains merged patent numbers
        patents: list[str] = []
        seen: set[str] = set()
        for tok in raw_tokens:
            for part in _split_merged_patent_number(tok):
                if part not in seen:
                    seen.add(part)
                    patents.append(part)
        cache_set("pr_din_patents", din, patents)
        if not patents:
            logger.info("DIN %s: PR-RDB table parsed but no patent column values found", din)
        else:
            logger.debug("DIN %s → %d patent(s): %s", din, len(patents), patents)
        return patents
    except Exception as exc:
        logger.warning("DIN patent lookup failed for %s: %s", din, exc)
        return []


# ── Main enrichment entry point ───────────────────────────────────────────────

async def enrich_patents(
    dins: list[str],
    on_progress: Optional[Callable] = None,
) -> dict[str, list[dict]]:
    """Enrich a list of DINs with patent dates from the CPD.

    Stores results in the patents and patent_discrepancies tables.
    Returns {din → [patent rows]} for the caller.

    DIN→patent mapping: Patent.zip primary (no HTML concat bugs), PR-RDB fallback.
    Date cross-check: live CPD dates vs Patent.zip dates; website wins on discrepancy.
    """
    if not dins:
        return {}

    # Skip DINs that already have at least one patent row with a non-null date.
    # A row with all-null dates means the prior enrichment got nothing (e.g. cache
    # poisoned by a test or CPD/ZIP both unavailable) — treat it as unenriched so
    # it gets retried on the next export.
    def _has_dates(din: str) -> bool:
        return any(
            r.get("filing_date") or r.get("grant_date") or r.get("expiry_date")
            for r in get_patents_for_din(din)
        )

    unenriched = [d for d in dins if not _has_dates(d)]
    already_enriched = [d for d in dins if d not in set(unenriched)]
    if already_enriched:
        logger.debug(
            "enrich_patents: skipping %d already-stored DINs: %s",
            len(already_enriched), already_enriched[:10],
        )
    if not unenriched:
        return {din: get_patents_for_din(din) for din in dins}

    session_id = ""
    try:
        _, _, session_id = await _pr_get_session()
    except Exception as exc:
        logger.warning("Could not obtain Patent Register session: %s", exc)

    # Download Patent.zip ONCE for both uses: by-patent (date cross-check) and by-DIN (mapping).
    # load_patent_zip_both() issues at most one HTTP request even on a cold cache, replacing the
    # previous two independent download tasks (zip_task + zip_din_task) that each fetched ~20 MB.
    zip_data, zip_by_din = await load_patent_zip_both()

    # DIN → patent numbers (Patent.zip primary, PR-RDB fallback)
    din_patent_map: dict[str, list[str]] = {}
    for din, patents in zip(
        unenriched,
        await asyncio.gather(*[_din_to_patent_numbers(d, session_id, zip_by_din or None) for d in unenriched]),
    ):
        if patents:
            din_patent_map[din] = patents

    zero_patent_dins = [d for d in unenriched if d not in din_patent_map]
    if zero_patent_dins:
        logger.info(
            "DINs with 0 patents (generic / off-patent / linkage check needed): %s",
            zero_patent_dins,
        )

    # Enrich each (DIN, patent_number) pair using CPD dates
    pairs = [(din, pn) for din, pns in din_patent_map.items() for pn in pns]
    total_pairs = len(pairs)
    done_count = 0

    async def _enrich_tracked(din: str, pn: str) -> None:
        nonlocal done_count
        await _enrich_one(din, pn, session_id, zip_data)
        done_count += 1
        if on_progress is not None:
            cb = on_progress(done_count, total_pairs, f"Patent {pn} (DIN {din})")
            if asyncio.iscoroutine(cb):
                await cb

    await asyncio.gather(*[_enrich_tracked(din, pn) for din, pn in pairs])

    return {din: get_patents_for_din(din) for din in dins}


async def _enrich_one(
    din: str,
    patent_number: str,
    session_id: str,
    zip_data: dict[str, dict],
) -> None:
    """Fetch CPD dates, cross-check zip, resolve discrepancies, store."""
    live = await fetch_patent_detail(patent_number, session_id)
    # zip_data keys are cleaned patent numbers
    clean_pn = _clean_patent_number(patent_number)
    zip_entry: dict = zip_data.get(clean_pn, zip_data.get(patent_number, {}))

    final: dict[str, Optional[str]] = {}
    for field in _DATE_FIELDS:
        web_val = live.get(field) or None
        zip_val = (zip_entry.get(field) or None)

        if web_val and zip_val and web_val != zip_val:
            log_discrepancy(din, patent_number, field, web_val, zip_val)
            logger.info(
                "Patent discrepancy DIN=%s patent=%s field=%s website=%r zip=%r → using website",
                din, patent_number, field, web_val, zip_val,
            )
        # Prefer CPD/website; fall back to zip
        final[field] = web_val if web_val is not None else zip_val

    upsert_patent(
        din=din,
        patent_number=patent_number,
        filing_date=final.get("filing_date"),
        grant_date=final.get("grant_date"),
        expiry_date=final.get("expiry_date"),
        detail_url=live.get("detail_url"),
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Enrich patents for a list of DINs.")
    parser.add_argument("--dins", nargs="+", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    results = asyncio.run(enrich_patents(args.dins))
    for din, rows in results.items():
        print(f"\nDIN {din}: {len(rows)} patent(s)")
        for row in rows:
            print(f"  {row['patent_number']}: "
                  f"filing={row['filing_date']} "
                  f"grant={row['grant_date']} "
                  f"expiry={row['expiry_date']} "
                  f"url={row.get('detail_url','')}")
