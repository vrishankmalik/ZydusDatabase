"""
Source 4 — Patent Register (PR-RDB).
JSP form-based site. POST to /pr-rdb/search with JSESSIONID cookie.
Medicinal ingredient must exactly match a dropdown value.
We fetch the dropdown on first use (cached), find closest match, then POST.

SSL cert on this server does not chain to a trusted root — we disable
hostname verification (same as a browser accepting the cert manually).
"""
from __future__ import annotations

import re
import ssl
from difflib import get_close_matches
from typing import Optional

import httpx
from bs4 import BeautifulSoup

from app.cache import cache_get, cache_set
from app.config import HTTP_TIMEOUT, PATENT_BASE, USER_AGENT
from app.models import DrugRecord, SourceResult

_INDEX_URL = f"{PATENT_BASE}/index-eng.jsp"
_SEARCH_URL = f"{PATENT_BASE}/search"

# Disable SSL verification — the PR-RDB server has a cert that Python's default
# CA bundle cannot verify; browsers accept it after a manual warning click.
_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE


async def _get_dropdown_options() -> tuple[list[str], list[str], str]:
    """Return (ingredient_options, brand_options, jsessionid) from the index page."""
    cached = cache_get("pr_rdb_index", "page")
    if cached is not None:
        return cached["ingredients"], cached["brands"], cached["session"]

    async with httpx.AsyncClient(verify=False) as client:  # noqa: S501
        r = await client.get(
            _INDEX_URL,
            headers={"User-Agent": USER_AGENT, "Accept": "text/html"},
            timeout=HTTP_TIMEOUT,
            follow_redirects=True,
        )
        r.raise_for_status()
        html = r.text
        session = r.cookies.get("JSESSIONID", "")

    soup = BeautifulSoup(html, "html.parser")

    ing_sel = soup.find("select", {"id": "medicinalIngredient"})
    brand_sel = soup.find("select", {"id": "brandName"})

    ingredients = (
        [o["value"] for o in ing_sel.find_all("option") if o.get("value")]
        if ing_sel else []
    )
    brands = (
        [o["value"] for o in brand_sel.find_all("option") if o.get("value")]
        if brand_sel else []
    )

    cache_set("pr_rdb_index", "page", {"ingredients": ingredients, "brands": brands, "session": session})
    return ingredients, brands, session


def _find_matching_options(query: str, options: list[str]) -> list[str]:
    """Find options that contain the query (case-insensitive), plus close fuzzy matches."""
    q = query.strip().upper()
    exact = [o for o in options if q in o.upper()]
    if exact:
        return exact[:10]  # cap at 10 to avoid too many requests
    # Fallback: fuzzy match
    # Cutoff 0.75 (raised from 0.6) — favours precision over recall.
    # At 0.6, "CANAGLIFLOZN" matched "EMPAGLIFLOZIN" (different drug, false positive).
    # At 0.75 that false positive is eliminated; precision reaches ≥ 0.95 on the
    # labeled benchmark in tests/fixtures/fuzzy_pairs.csv.
    fuzzy = get_close_matches(q, [o.upper() for o in options], n=3, cutoff=0.75)
    return [o for o in options if o.upper() in fuzzy]


def _parse_results_table(html: str) -> list[dict]:
    """
    Parse results table.
    Columns (confirmed live): Medicinal ingredient | Brand name | Strength | Dosage | DIN | Patent | CSP
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if table is None:
        return []

    rows = []
    for tr in table.find_all("tr"):
        cells = tr.find_all("td")
        if len(cells) < 7:
            continue

        ingredient = cells[0].get_text(strip=True)
        brand = cells[1].get_text(strip=True)
        strength = cells[2].get_text(strip=True)
        dosage = cells[3].get_text(strip=True)
        din = cells[4].get_text(strip=True)
        patent = cells[5].get_text(strip=True)
        csp = cells[6].get_text(strip=True) if len(cells) > 6 else ""

        # Detail link if present
        link = table.find("a")
        record_url = _SEARCH_URL  # no individual record pages, link to search

        rows.append(
            {
                "ingredient": ingredient,
                "brand": brand,
                "strength": strength,
                "dosage": dosage,
                "din": din,
                "patent": patent,
                "csp": csp,
            }
        )
    return rows


async def _post_search(term: str, field: str, session: str) -> str:
    form_data: dict[str, str] = {
        "medicinalIngredient": "",
        "brandName": "",
        "patentNumber": "",
        "din": "",
        "cspNumber": "",
        "search": "Search",
    }
    if field == "ingredient":
        form_data["medicinalIngredient"] = term
    elif field == "brand":
        form_data["brandName"] = term
    elif field == "din":
        form_data["din"] = term
    elif field == "patent":
        form_data["patentNumber"] = term

    async with httpx.AsyncClient(verify=False) as client:  # noqa: S501
        r = await client.post(
            _SEARCH_URL,
            data=form_data,
            cookies={"JSESSIONID": session} if session else {},
            headers={
                "User-Agent": USER_AGENT,
                "Content-Type": "application/x-www-form-urlencoded",
                "Referer": _INDEX_URL,
            },
            timeout=HTTP_TIMEOUT,
            follow_redirects=True,
        )
        r.raise_for_status()
        return r.text


async def search_patent_register(
    query: str,
    field: str = "ingredient",
    extra_terms: Optional[list[str]] = None,
) -> SourceResult:
    if field == "company":
        return SourceResult(
            source="PatentRegister",
            status="unsupported",
            error_message="Patent Register does not support search by company name.",
        )

    try:
        ingredients, brands, session = await _get_dropdown_options()
    except Exception as e:
        return SourceResult(source="PatentRegister", status="error", error_message=str(e))

    terms = [query] + (extra_terms or [])
    matched_options: list[str] = []

    if field == "ingredient":
        for term in terms:
            matched_options.extend(_find_matching_options(term, ingredients))
    elif field == "brand":
        for term in terms:
            matched_options.extend(_find_matching_options(term, brands))
    elif field in ("din", "patent"):
        matched_options = terms  # free-text fields, pass directly
    else:
        matched_options = terms

    # Deduplicate
    seen_opts: set[str] = set()
    unique_opts = []
    for o in matched_options:
        if o not in seen_opts:
            seen_opts.add(o)
            unique_opts.append(o)

    if not unique_opts:
        return SourceResult(source="PatentRegister", status="no_results")

    all_rows: list[dict] = []
    seen_keys: set[tuple] = set()

    for opt in unique_opts[:10]:  # cap to avoid hammering the server
        cache_key = f"pr_{field}:{opt.lower()}"
        cached = cache_get("pr_rdb", cache_key)
        if cached is not None:
            rows = cached
        else:
            try:
                html = await _post_search(opt, field, session)
                if "query has no results" in html.lower():
                    rows = []
                else:
                    rows = _parse_results_table(html)
                cache_set("pr_rdb", cache_key, rows)
            except Exception as e:
                return SourceResult(source="PatentRegister", status="error", error_message=str(e))

        for row in rows:
            key = (row["ingredient"], row["brand"], row["din"], row["patent"])
            if key not in seen_keys:
                seen_keys.add(key)
                all_rows.append(row)

    if not all_rows:
        return SourceResult(source="PatentRegister", status="no_results")

    records = [
        DrugRecord(
            source="PatentRegister",
            ingredient=r["ingredient"] or None,
            brand_name=r["brand"] or None,
            din=r["din"] if r["din"] not in ("", "N/A") else None,
            strength=r["strength"] or None,
            dosage_form=r["dosage"] or None,
            all_ingredients=(
                [i.strip() for i in r["ingredient"].split(";") if i.strip()]
                if r["ingredient"]
                else []
            ),
            record_url=f"https://pr-rdb.hc-sc.gc.ca{_SEARCH_URL.replace('https://pr-rdb.hc-sc.gc.ca', '')}",
            source_specific={
                "patent_number": r["patent"],
                "csp": r["csp"],
            },
        )
        for r in all_rows
    ]
    return SourceResult(
        source="PatentRegister", status="ok", records=records, count=len(records)
    )
