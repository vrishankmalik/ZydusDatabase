"""
Drug Product Database (DPD) — official REST API client.
No scraping. All data from health-products.canada.ca/api/drug/.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Optional

import httpx

from app.cache import cache_get, cache_set
from app.config import DPD_BASE, DPD_MAX_RESULTS, DPD_SEMAPHORE, HTTP_TIMEOUT, USER_AGENT
from app.models import DrugRecord, SourceResult

_HEADERS = {"User-Agent": USER_AGENT, "Accept": "application/json"}


async def _get_json(client: httpx.AsyncClient, url: str, params: dict) -> Any:
    r = await client.get(url, params=params, headers=_HEADERS, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()


async def _fetch_drug_codes_by_ingredient(
    client: httpx.AsyncClient, ingredient: str
) -> list[dict]:
    cache_key = f"dpd_ingredient:{ingredient.lower()}"
    cached = cache_get("dpd_ingredient", ingredient.lower())
    if cached is not None:
        return cached
    data = await _get_json(
        client,
        f"{DPD_BASE}/activeingredient/",
        {"ingredientname": ingredient, "lang": "en", "type": "json"},
    )
    result = data if isinstance(data, list) else []
    cache_set("dpd_ingredient", ingredient.lower(), result)
    return result


async def _fetch_drug_codes_by_brand(
    client: httpx.AsyncClient, brand: str
) -> list[dict]:
    cached = cache_get("dpd_brand", brand.lower())
    if cached is not None:
        return cached
    data = await _get_json(
        client,
        f"{DPD_BASE}/drugproduct/",
        {"brandname": brand, "lang": "en", "type": "json"},
    )
    result = data if isinstance(data, list) else []
    cache_set("dpd_brand", brand.lower(), result)
    return result


async def _fetch_drug_codes_by_company(
    client: httpx.AsyncClient, company: str
) -> list[dict]:
    cached = cache_get("dpd_company", company.lower())
    if cached is not None:
        return cached
    data = await _get_json(
        client,
        f"{DPD_BASE}/drugproduct/",
        {"companyname": company, "lang": "en", "type": "json"},
    )
    result = data if isinstance(data, list) else []
    cache_set("dpd_company", company.lower(), result)
    return result


async def _fetch_drugproduct(
    client: httpx.AsyncClient, sem: asyncio.Semaphore, drug_code: int
) -> Optional[dict]:
    cached = cache_get("dpd_product", str(drug_code))
    if cached is not None:
        return cached
    async with sem:
        data = await _get_json(
            client,
            f"{DPD_BASE}/drugproduct/",
            {"id": drug_code, "lang": "en", "type": "json"},
        )
    # API returns a single dict for ?id= queries, not a list
    if isinstance(data, dict) and data:
        result = data
    elif isinstance(data, list) and data:
        result = data[0]
    else:
        result = None
    if result:
        cache_set("dpd_product", str(drug_code), result)
    return result


async def _fetch_form(
    client: httpx.AsyncClient, sem: asyncio.Semaphore, drug_code: int
) -> list[dict]:
    cached = cache_get("dpd_form", str(drug_code))
    if cached is not None:
        return cached
    async with sem:
        try:
            data = await _get_json(
                client,
                f"{DPD_BASE}/form/",
                {"id": drug_code, "lang": "en", "type": "json"},
            )
        except Exception:
            return []
    result = data if isinstance(data, list) else []
    cache_set("dpd_form", str(drug_code), result)
    return result


async def _fetch_route(
    client: httpx.AsyncClient, sem: asyncio.Semaphore, drug_code: int
) -> list[dict]:
    cached = cache_get("dpd_route", str(drug_code))
    if cached is not None:
        return cached
    async with sem:
        try:
            data = await _get_json(
                client,
                f"{DPD_BASE}/route/",
                {"id": drug_code, "lang": "en", "type": "json"},
            )
        except Exception:
            return []
    result = data if isinstance(data, list) else []
    cache_set("dpd_route", str(drug_code), result)
    return result


async def _fetch_status(
    client: httpx.AsyncClient, sem: asyncio.Semaphore, drug_code: int
) -> list[dict]:
    cached = cache_get("dpd_status", str(drug_code))
    if cached is not None:
        return cached
    async with sem:
        try:
            data = await _get_json(
                client,
                f"{DPD_BASE}/status/",
                {"id": drug_code, "lang": "en", "type": "json"},
            )
        except Exception:
            return []
    # /status/ returns a single dict, not a list
    if isinstance(data, dict) and data:
        result = [data]
    elif isinstance(data, list):
        result = data
    else:
        result = []
    cache_set("dpd_status", str(drug_code), result)
    return result


async def _fetch_schedule(
    client: httpx.AsyncClient, sem: asyncio.Semaphore, drug_code: int
) -> list[dict]:
    cached = cache_get("dpd_schedule", str(drug_code))
    if cached is not None:
        return cached
    async with sem:
        try:
            data = await _get_json(
                client,
                f"{DPD_BASE}/schedule/",
                {"id": drug_code, "lang": "en", "type": "json"},
            )
        except Exception:
            return []
    result = data if isinstance(data, list) else []
    cache_set("dpd_schedule", str(drug_code), result)
    return result


async def _fetch_ingredients_by_code(
    client: httpx.AsyncClient, sem: asyncio.Semaphore, drug_code: int
) -> list[dict]:
    """Fetch ALL active ingredients for a specific drug_code (for grouping)."""
    cached = cache_get("dpd_ings_by_code", str(drug_code))
    if cached is not None:
        return cached
    async with sem:
        try:
            data = await _get_json(
                client,
                f"{DPD_BASE}/activeingredient/",
                {"id": drug_code, "lang": "en", "type": "json"},
            )
        except Exception:
            return []
    result = data if isinstance(data, list) else []
    cache_set("dpd_ings_by_code", str(drug_code), result)
    return result


async def _build_record_for_code(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    drug_code: int,
    ingredient_rows: list[dict],
) -> Optional[DrugRecord]:
    """Assemble a DrugRecord from product + enrichment endpoints."""
    product, forms, routes, statuses, schedules, all_ing_rows = await asyncio.gather(
        _fetch_drugproduct(client, sem, drug_code),
        _fetch_form(client, sem, drug_code),
        _fetch_route(client, sem, drug_code),
        _fetch_status(client, sem, drug_code),
        _fetch_schedule(client, sem, drug_code),
        _fetch_ingredients_by_code(client, sem, drug_code),
    )
    if product is None:
        return None

    # Full ingredient list from the per-code API; fall back to the search-filtered rows
    code_ingredients = all_ing_rows or [
        r for r in ingredient_rows if r.get("drug_code") == drug_code
    ]
    # Sort ingredients alphabetically by name so combination products present in a
    # stable, canonical order. ingredient_str (SKU Name), strength, and
    # all_ingredients are all derived from this same ordered list, so each strength
    # stays bound to its own ingredient (e.g. "AMLODIPINE ... 10 MG; PERINDOPRIL ...
    # 14 MG"). This removes the 10/14-vs-14/10 ambiguity when comparing across DINs.
    code_ingredients = sorted(
        code_ingredients,
        key=lambda r: (r.get("ingredient_name") or "").strip().upper(),
    )
    ingredient_str = "; ".join(
        _format_ingredient(r) for r in code_ingredients
    ) or None
    all_ingredient_names = [
        r.get("ingredient_name", "").strip()
        for r in code_ingredients
        if r.get("ingredient_name", "").strip()
    ]

    dosage_form = "; ".join(f.get("pharmaceutical_form_name", "") for f in forms) or None
    route_str = "; ".join(r.get("route_of_administration_name", "") for r in routes) or None
    status_str = "; ".join(s.get("status", "") for s in statuses) or None
    schedule_str = "; ".join(s.get("schedule_name", "") for s in schedules) or None

    din = str(product.get("drug_identification_number", "")).strip() or None
    brand = product.get("brand_name", "").strip() or None
    company = product.get("company_name", "").strip() or None

    record_url = (
        f"https://health-products.canada.ca/dpd-bdpp/info?lang=eng&code={drug_code}"
    )

    return DrugRecord(
        source="DPD",
        ingredient=ingredient_str,
        brand_name=brand,
        company=company,
        din=din,
        all_ingredients=all_ingredient_names,
        strength=_strength_from_ingredients(code_ingredients),
        dosage_form=dosage_form,
        route=route_str,
        status=status_str,
        record_url=record_url,
        source_specific={
            "drug_code": drug_code,
            "class_name": product.get("class_name"),
            "number_of_ais": product.get("number_of_ais"),
            "last_update_date": product.get("last_update_date"),
            "schedule": schedule_str,
        },
    )


def _format_ingredient(row: dict) -> str:
    parts = [row.get("ingredient_name", "")]
    strength = row.get("strength", "")
    strength_unit = row.get("strength_unit", "")
    if strength and strength_unit:
        parts.append(f"{strength} {strength_unit}")
    return " ".join(p for p in parts if p).strip()


def _strength_from_ingredients(rows: list[dict]) -> Optional[str]:
    parts = []
    for r in rows:
        s = r.get("strength", "")
        u = r.get("strength_unit", "")
        if s:
            parts.append(f"{s} {u}".strip())
    return "; ".join(parts) if parts else None


async def search_dpd(
    query: str,
    field: str = "ingredient",
    extra_terms: Optional[list[str]] = None,
) -> SourceResult:
    """
    Search DPD API.
    field: "ingredient" | "brand" | "company" | "din"
    extra_terms: additional normalized synonyms to include
    """
    if field == "din":
        # DIN search: look up by drug_identification_number directly
        return await _search_by_din(query)

    all_ingredient_rows: list[dict] = []
    all_drug_codes: set[int] = set()

    terms = [query] + (extra_terms or [])

    async with httpx.AsyncClient() as client:
        sem = asyncio.Semaphore(DPD_SEMAPHORE)

        async def _fetch_by_term(term: str) -> None:
            try:
                if field == "ingredient":
                    rows = await _fetch_drug_codes_by_ingredient(client, term)
                    for r in rows:
                        dc = r.get("drug_code")
                        if dc:
                            all_drug_codes.add(int(dc))
                    all_ingredient_rows.extend(rows)
                elif field == "brand":
                    products = await _fetch_drug_codes_by_brand(client, term)
                    for p in products:
                        dc = p.get("drug_code")
                        if dc:
                            all_drug_codes.add(int(dc))
                elif field == "company":
                    products = await _fetch_drug_codes_by_company(client, term)
                    for p in products:
                        dc = p.get("drug_code")
                        if dc:
                            all_drug_codes.add(int(dc))
            except Exception:
                pass

        await asyncio.gather(*[_fetch_by_term(t) for t in terms])

        if not all_drug_codes:
            return SourceResult(source="DPD", status="no_results", records=[])

        # Cap results to avoid overwhelming the API and timing out on very broad queries
        codes_to_fetch = list(all_drug_codes)[:DPD_MAX_RESULTS]
        capped = len(all_drug_codes) > DPD_MAX_RESULTS

        # For brand/company searches, we still need ingredient rows for display
        # Fetch them lazily via the product endpoint which has brand/company info
        records_tasks = [
            _build_record_for_code(client, sem, dc, all_ingredient_rows)
            for dc in codes_to_fetch
        ]
        results = await asyncio.gather(*records_tasks, return_exceptions=True)

    records: list[DrugRecord] = []
    for r in results:
        if isinstance(r, DrugRecord):
            records.append(r)

    if not records:
        return SourceResult(source="DPD", status="no_results", records=[])

    result = SourceResult(source="DPD", status="ok", records=records, count=len(records))
    if capped:
        result.total_matches = len(all_drug_codes)
        result.error_message = (
            f"Showing first {DPD_MAX_RESULTS} of {len(all_drug_codes)} matching products. "
            f"Use a more specific term to see all results."
        )
    return result


async def _search_by_din(din: str) -> SourceResult:
    """Lookup a single product by DIN."""
    cached = cache_get("dpd_din", din)
    if cached is not None:
        data = cached
    else:
        async with httpx.AsyncClient() as client:
            try:
                data = await _get_json(
                    client,
                    f"{DPD_BASE}/drugproduct/",
                    {"din": din, "lang": "en", "type": "json"},
                )
                cache_set("dpd_din", din, data)
            except Exception as e:
                return SourceResult(
                    source="DPD", status="error", error_message=str(e)
                )

    # API may return a dict or list for DIN queries
    if isinstance(data, dict) and data:
        product = data
    elif isinstance(data, list) and data:
        product = data[0]
    else:
        return SourceResult(source="DPD", status="no_results")

    drug_code = product.get("drug_code")
    if not drug_code:
        return SourceResult(source="DPD", status="no_results")

    async with httpx.AsyncClient() as client:
        sem = asyncio.Semaphore(DPD_SEMAPHORE)
        record = await _build_record_for_code(client, sem, int(drug_code), [])

    if record is None:
        return SourceResult(source="DPD", status="no_results")
    return SourceResult(source="DPD", status="ok", records=[record], count=1)
