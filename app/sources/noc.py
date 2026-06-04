"""
Source 3 — Notice of Compliance (NOC / NOC with conditions).
Official JSON API: health-products.canada.ca/api/notice-of-compliance/

Ingredient search:
1. GET /medicinalingredient/?type=json&lang=en  →  full list (~93 k rows, cached)
2. Filter where noc_pi_medic_ingr_name contains the queried term (case-insensitive)
3. Group by noc_number; cap at NOC_MAX_NOC_NUMBERS to bound request volume
4. For each noc_number, concurrently:
   a. GET /drugproduct/?id=<n>&type=json&lang=en  →  product_id, DIN, brand
   b. GET /noticeofcompliancemain/?id=<n>&type=json&lang=en  →  date, manufacturer, class
5. Join: noc_pi_din_product_id == noc_br_product_id  →  attach DIN per ingredient row
6. Emit one DrugRecord per (noc_number, noc_br_product_id)

Only ingredient searches are supported; brand/company/DIN return "unsupported".
"""
from __future__ import annotations

import asyncio
from typing import Optional

import httpx

from app.cache import cache_get, cache_set
from app.config import HTTP_TIMEOUT, USER_AGENT
from app.models import DrugRecord, SourceResult

_API_BASE = "https://health-products.canada.ca/api/notice-of-compliance"
_INGREDIENT_URL = f"{_API_BASE}/medicinalingredient/?type=json&lang=en"

NOC_MAX_NOC_NUMBERS = 200  # cap per-noc-number API calls
_NOC_CONCURRENCY = 20       # max simultaneous HTTP requests


# ── Low-level fetchers ────────────────────────────────────────────────────────

async def _fetch_all_ingredients() -> list[dict]:
    """Return the full /medicinalingredient list, fetched once and cached."""
    cached = cache_get("noc_api", "all_ingredients")
    if cached is not None:
        return cached  # type: ignore[return-value]
    async with httpx.AsyncClient() as client:
        r = await client.get(
            _INGREDIENT_URL,
            headers={"User-Agent": USER_AGENT},
            timeout=HTTP_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
    result: list[dict] = data if isinstance(data, list) else []
    cache_set("noc_api", "all_ingredients", result)
    return result


async def _fetch_drugproduct(
    client: httpx.AsyncClient,
    noc_number: int,
    sem: asyncio.Semaphore,
) -> list[dict]:
    cached = cache_get("noc_api_dp", str(noc_number))
    if cached is not None:
        return cached  # type: ignore[return-value]
    async with sem:
        r = await client.get(
            f"{_API_BASE}/drugproduct/?id={noc_number}&type=json&lang=en",
            headers={"User-Agent": USER_AGENT},
            timeout=HTTP_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
    result: list[dict] = data if isinstance(data, list) else []
    cache_set("noc_api_dp", str(noc_number), result)
    return result


async def _fetch_main(
    client: httpx.AsyncClient,
    noc_number: int,
    sem: asyncio.Semaphore,
) -> dict:
    cached = cache_get("noc_api_mn", str(noc_number))
    if cached is not None:
        return cached  # type: ignore[return-value]
    async with sem:
        r = await client.get(
            f"{_API_BASE}/noticeofcompliancemain/?id={noc_number}&type=json&lang=en",
            headers={"User-Agent": USER_AGENT},
            timeout=HTTP_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
    result: dict = data if isinstance(data, dict) else {}
    cache_set("noc_api_mn", str(noc_number), result)
    return result


# ── Ingredient-search helpers ─────────────────────────────────────────────────

def _filter_ingredient_records(all_ingr: list[dict], query: str) -> list[dict]:
    q = query.strip().upper()
    return [r for r in all_ingr if q in r.get("noc_pi_medic_ingr_name", "").upper()]


def _build_pi_map(pi_records: list[dict]) -> dict[int, dict[int, list[str]]]:
    """Map noc_number → {product_id → [ingredient_name, ...]}."""
    pi_map: dict[int, dict[int, list[str]]] = {}
    for rec in pi_records:
        nn: int = rec["noc_number"]
        pid: int = rec["noc_pi_din_product_id"]
        name: str = rec.get("noc_pi_medic_ingr_name", "")
        pi_map.setdefault(nn, {}).setdefault(pid, [])
        if name and name not in pi_map[nn][pid]:
            pi_map[nn][pid].append(name)
    return pi_map


def _normalize_din(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    if raw.strip().upper() in ("N/A", "NA", "NOT APPLICABLE", ""):
        return None
    return raw.strip()


# ── Main search function ──────────────────────────────────────────────────────

async def search_noc(
    query: str,
    field: str = "ingredient",
    extra_terms: Optional[list[str]] = None,
) -> SourceResult:
    if field != "ingredient":
        return SourceResult(
            source="NOC",
            status="unsupported",
            error_message=(
                f"NOC JSON API supports ingredient searches only. "
                f"Received field='{field}'."
            ),
        )

    terms = [query] + (extra_terms or [])

    try:
        all_ingr = await _fetch_all_ingredients()
    except Exception as e:
        return SourceResult(source="NOC", status="error", error_message=str(e))

    # Filter ingredient records for every synonym/term, deduplicate by (noc_number, product_id)
    pi_records: list[dict] = []
    seen_pi: set[tuple[int, int]] = set()
    for term in terms:
        for rec in _filter_ingredient_records(all_ingr, term):
            key = (rec["noc_number"], rec["noc_pi_din_product_id"])
            if key not in seen_pi:
                seen_pi.add(key)
                pi_records.append(rec)

    if not pi_records:
        return SourceResult(source="NOC", status="no_results")

    # Preserve order, cap number of noc_numbers to fetch
    noc_order: dict[int, None] = {}
    for rec in pi_records:
        noc_order[rec["noc_number"]] = None
    noc_numbers = list(noc_order)[:NOC_MAX_NOC_NUMBERS]
    noc_number_set = set(noc_numbers)

    # For every (noc_number, product_id) pair that matched, collect ALL ingredient records
    # from the full list for those pairs — captures co-ingredients in combination products.
    matched_pid_pairs = {
        (r["noc_number"], r["noc_pi_din_product_id"])
        for r in pi_records
        if r["noc_number"] in noc_number_set
    }
    full_pi_for_matched = [
        r for r in all_ingr
        if (r["noc_number"], r["noc_pi_din_product_id"]) in matched_pid_pairs
    ]
    pi_map = _build_pi_map(full_pi_for_matched)

    # Interleaved task list: (kind, noc_number) — even=dp, odd=mn
    sem = asyncio.Semaphore(_NOC_CONCURRENCY)
    all_tasks: list[tuple[str, int]] = []
    for nn in noc_numbers:
        all_tasks.append(("dp", nn))
        all_tasks.append(("mn", nn))

    async with httpx.AsyncClient() as client:
        coros = [
            _fetch_drugproduct(client, nn, sem) if kind == "dp"
            else _fetch_main(client, nn, sem)
            for kind, nn in all_tasks
        ]
        raw_results = await asyncio.gather(*coros, return_exceptions=True)

    dp_by_nn: dict[int, list[dict]] = {}
    mn_by_nn: dict[int, dict] = {}
    for i, (kind, nn) in enumerate(all_tasks):
        res = raw_results[i]
        if isinstance(res, Exception):
            res = [] if kind == "dp" else {}
        if kind == "dp":
            dp_by_nn[nn] = res  # type: ignore[assignment]
        else:
            mn_by_nn[nn] = res  # type: ignore[assignment]

    # Assemble one DrugRecord per (noc_number, product_id)
    records: list[DrugRecord] = []
    seen_records: set[tuple[int, int]] = set()

    for nn in noc_numbers:
        products = dp_by_nn.get(nn, [])
        main = mn_by_nn.get(nn, {})
        products_by_pid = {
            p["noc_br_product_id"]: p
            for p in products
            if "noc_br_product_id" in p
        }
        status_with_cond = (main.get("noc_status_with_conditions") or "N") == "Y"

        for pid, ingr_names in pi_map.get(nn, {}).items():
            product = products_by_pid.get(pid)
            if product is None:
                continue

            key = (nn, pid)
            if key in seen_records:
                continue
            seen_records.add(key)

            sorted_names = sorted(set(ingr_names))
            records.append(
                DrugRecord(
                    source="NOC",
                    ingredient="; ".join(sorted_names) or None,
                    brand_name=product.get("noc_br_brandname") or None,
                    company=main.get("noc_manufacturer_name") or None,
                    din=_normalize_din(product.get("noc_br_din")),
                    all_ingredients=sorted_names,
                    status="NOC/c" if status_with_cond else "NOC",
                    record_url=f"https://health-products.canada.ca/noc-ac/nocInfo?id={nn}",
                    source_specific={
                        "noc_date": main.get("noc_date") or None,
                        "submission_type": main.get("noc_on_submission_type"),
                        "therapeutic_class": main.get("noc_therapeutic_class"),
                        "noc_number": nn,
                        "submission_class": main.get("noc_submission_class"),
                    },
                )
            )

    if not records:
        return SourceResult(source="NOC", status="no_results")
    return SourceResult(source="NOC", status="ok", records=records, count=len(records))
