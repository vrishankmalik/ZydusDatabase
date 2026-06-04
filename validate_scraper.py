#!/usr/bin/env python3
"""Three-stage scraper validation.

Tests Stage 1 (patent dates from CPD), Stage 2 (DPD API packaging + info page),
and Stage 3 (PDF label extraction) for four known DINs.

Usage:
    python validate_scraper.py
    python validate_scraper.py --verbose
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
from typing import Optional

import httpx

# ---------------------------------------------------------------------------
# Argument parsing (must happen before project imports so logging is set up)
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--verbose", "-v", action="store_true")
args = parser.parse_args()

logging.basicConfig(
    level=logging.DEBUG if args.verbose else logging.INFO,
    format="%(levelname)-8s %(name)s: %(message)s",
)

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, ".")

from app.config import DPD_BASE, USER_AGENT, HTTP_TIMEOUT
from app.enrichment.patents import (
    _clean_patent_number,
    _fetch_cpd_dates,
    _din_to_patent_numbers,
)
from app.sources.patent_register import _get_dropdown_options as _pr_get_session
from app.enrichment.labeling import (
    _fetch_packaging_api,
    _scrape_dpd_info_page,
    fetch_stage2_data,
    enrich_labeling,
    NOT_IN_PM,
)

# ---------------------------------------------------------------------------
# DINs under test
# ---------------------------------------------------------------------------
TEST_CASES = [
    ("02370050", "GSK IV infusion 80mg/mL 5mL vial"),
    ("00723894", "TYLENOL RS (PM 00076531)"),
    ("02230436", "ACET 325 supp (PM 00067036)"),
    ("02242468", "RIVACOCET (PM 00048982) — known misparse case"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HEADERS = {"User-Agent": USER_AGENT, "Accept": "application/json"}


async def lookup_drug_code(din: str) -> Optional[int]:
    """Fetch the DPD drug_code for a DIN via the DPD API.

    Tries /drugproduct/?f_din=<DIN> then /drugproduct/?din=<DIN>.
    """
    for param in ("din", "f_din"):
        try:
            async with httpx.AsyncClient() as client:
                r = await client.get(
                    f"{DPD_BASE}/drugproduct/",
                    params={param: din, "lang": "en", "type": "json"},
                    headers=_HEADERS,
                    timeout=HTTP_TIMEOUT,
                )
            if r.status_code != 200:
                continue
            data = r.json()
            entries = data if isinstance(data, list) else ([data] if isinstance(data, dict) and data else [])
            for e in entries:
                if isinstance(e, dict):
                    code = e.get("drug_code") or e.get("drugCode")
                    if code:
                        return int(code)
        except Exception as exc:
            logging.debug("drug_code lookup via %s failed for %s: %s", param, din, exc)

    # Fallback: search by DIN as brandname query (unlikely to work, but worth trying)
    logging.warning("Could not determine drug_code for DIN %s via API", din)
    return None


async def lookup_strength(drug_code: int) -> Optional[str]:
    """Return the first active ingredient strength for a drug_code."""
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(
                f"{DPD_BASE}/activeingredient/",
                params={"id": drug_code, "lang": "en", "type": "json"},
                headers=_HEADERS,
                timeout=HTTP_TIMEOUT,
            )
        if r.status_code != 200:
            return None
        data = r.json()
        entries = data if isinstance(data, list) else ([data] if isinstance(data, dict) else [])
        for e in entries:
            if isinstance(e, dict):
                strength = str(e.get("strength") or "").strip()
                unit = str(e.get("strength_unit") or "").strip()
                if strength:
                    return f"{strength} {unit}".strip() if unit else strength
    except Exception as exc:
        logging.debug("strength lookup failed for drug_code=%s: %s", drug_code, exc)
    return None


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

async def run_stage1(din: str, patent_with_dates: Optional[str] = None) -> dict:
    """Test Stage 1: patent numbers from PR-RDB + dates from CPD."""
    result: dict = {"din": din, "patent_numbers": [], "patents": []}

    session_id = ""
    try:
        _, _, session_id = await _pr_get_session()
    except Exception as exc:
        result["session_error"] = str(exc)

    patent_numbers = await _din_to_patent_numbers(din, session_id)
    result["patent_numbers"] = patent_numbers
    result["patent_count"] = len(patent_numbers)

    if not patent_numbers:
        result["note"] = "zero patents from PR-RDB (generic / off-patent or lookup failure)"
        return result

    # Scrape CPD dates for each patent
    for pn in patent_numbers[:5]:  # cap to avoid hammering CPD
        cpd = await _fetch_cpd_dates(pn)
        result["patents"].append({
            "patent_number": pn,
            "clean_number": _clean_patent_number(pn),
            "cpd_url": cpd.get("detail_url"),
            "filing_date": cpd.get("filing_date"),
            "grant_date": cpd.get("grant_date"),
            "expiry_date": cpd.get("expiry_date"),
        })

    # If a specific patent was nominated for side-by-side comparison, highlight it
    if patent_with_dates:
        for p in result["patents"]:
            if patent_with_dates in (p["patent_number"], p["clean_number"]):
                result["highlighted"] = p

    return result


async def run_stage2(din: str, drug_code: int) -> dict:
    """Test Stage 2: DPD API packaging + info page scrape."""
    stage2 = await fetch_stage2_data(drug_code)
    info = await _scrape_dpd_info_page(drug_code)

    return {
        "din": din,
        "drug_code": drug_code,
        "active_ingredient": stage2.get("active_ingredient"),
        "pack_size": stage2.get("pack_size"),
        "pack_style": stage2.get("pack_style"),
        "pdf_url": stage2.get("pdf_url"),
        "pdf_date": stage2.get("pdf_date"),
        "description": (stage2.get("description") or "")[:120],
    }


async def run_stage3(din: str, drug_code: int, strength: Optional[str]) -> dict:
    """Test Stage 3: full enrich_labeling (Stage 2 + PDF)."""
    row = await enrich_labeling(din, drug_code, strength)
    if row is None:
        return {"din": din, "error": "enrich_labeling returned None"}
    return {"din": din, **{k: v for k, v in row.items() if not k.startswith("_")}}


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------

def run_assertions(stage2_results: list[dict], stage3_results: list[dict]) -> list[str]:
    failures: list[str] = []

    poison_pat = re.compile(
        r"administration\s+strength|strength\s+and\s+dosage|recommended\s+dose",
        re.IGNORECASE,
    )
    date_pat = re.compile(
        r"^\d{4}-\d{2}-\d{2}$|"
        r"^verify:|"
        r"^None$|"
        r"^$"
    )

    for r in stage2_results:
        din = r.get("din", "?")
        pack_size = r.get("pack_size") or ""
        if poison_pat.search(pack_size):
            failures.append(
                f"FAIL: DIN {din} pack_size contains dosing instruction: {pack_size!r}"
            )

    for r in stage3_results:
        din = r.get("din", "?")
        for field in ("excipients_core", "excipients_coating", "preservatives"):
            val = str(r.get(field) or "")
            if poison_pat.search(val):
                failures.append(
                    f"FAIL: DIN {din} {field} contains dosing instruction: {val!r}"
                )
        for date_field in ("earliest_filing_date", "earliest_grant_date", "latest_expiry_date"):
            val = str(r.get(date_field) or "")
            if val and not date_pat.match(val):
                failures.append(
                    f"FAIL: DIN {din} {date_field} is not a valid date or null: {val!r}"
                )

    return failures


# ---------------------------------------------------------------------------
# Coverage table
# ---------------------------------------------------------------------------

def print_coverage(all_rows: list[dict]) -> None:
    fields = (
        "active_ingredient", "pack_size", "pack_style",
        "excipients_core", "excipients_coating", "preservatives",
        "ph", "colour", "shape", "size_mm", "weight",
    )
    header = f"{'Field':<25} {'Real value':>10} {'Not in PM':>10} {'Blank':>7}"
    print("\n" + "=" * 60)
    print("COVERAGE TABLE")
    print("=" * 60)
    print(header)
    print("-" * 60)
    for field in fields:
        real_v = sum(1 for r in all_rows if r.get(field) and r[field] not in ("", NOT_IN_PM))
        not_in = sum(1 for r in all_rows if r.get(field) == NOT_IN_PM)
        blank  = sum(1 for r in all_rows if not r.get(field) or r[field] == "")
        print(f"  {field:<23} {real_v:>10} {not_in:>10} {blank:>7}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> int:
    print("\n" + "=" * 70)
    print("VALIDATOR — three-stage DPD/Patent/Labeling scraper")
    print("=" * 70)

    # Resolve drug_codes for all test DINs
    print("\n[setup] Resolving drug_code for each DIN ...")
    din_info: list[tuple[str, str, Optional[int], Optional[str]]] = []
    for din, label in TEST_CASES:
        drug_code = await lookup_drug_code(din)
        strength = await lookup_strength(drug_code) if drug_code else None
        print(f"  DIN {din} ({label}): drug_code={drug_code}, strength={strength!r}")
        din_info.append((din, label, drug_code, strength))

    print()

    # -------------------------------------------------------------------
    # Stage 1: Patents
    # -------------------------------------------------------------------
    print("=" * 70)
    print("STAGE 1 — PATENT DATES (from CPD)")
    print("=" * 70)
    stage1_results = []
    for din, label, drug_code, _ in din_info:
        print(f"\n  DIN {din} — {label}")
        r = await run_stage1(din)
        stage1_results.append(r)
        print(f"    patent_count (PR-RDB): {r['patent_count']}")
        if r.get("note"):
            print(f"    note: {r['note']}")
        for p in r.get("patents", []):
            print(f"    patent {p['patent_number']} (clean: {p['clean_number']})")
            print(f"      CPD URL   : {p['cpd_url']}")
            print(f"      filed     : {p['filing_date']}")
            print(f"      granted   : {p['grant_date']}")
            print(f"      expiry    : {p['expiry_date']}")

    # Pick the first DIN that actually has patents for the side-by-side demo
    patent_demo = next(
        (r for r in stage1_results if r.get("patents")),
        None,
    )
    if patent_demo and patent_demo["patents"]:
        p = patent_demo["patents"][0]
        print(f"\n  SIDE-BY-SIDE for patent {p['patent_number']} (DIN {patent_demo['din']}):")
        print(f"    CPD filed  : {p['filing_date']}")
        print(f"    CPD granted: {p['grant_date']}")
        print(f"    CPD expiry : {p['expiry_date']}")
    else:
        print("\n  (no DIN with patents found — Stage 1 CPD comparison not shown)")

    # -------------------------------------------------------------------
    # Stage 2: DPD API
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STAGE 2 — DPD API + INFO PAGE")
    print("=" * 70)
    stage2_results = []
    for din, label, drug_code, strength in din_info:
        if drug_code is None:
            print(f"\n  DIN {din}: drug_code unknown — skipping Stage 2")
            stage2_results.append({"din": din, "error": "no drug_code"})
            continue
        print(f"\n  DIN {din} — {label} (drug_code={drug_code})")
        r = await run_stage2(din, drug_code)
        stage2_results.append(r)
        print(f"    active_ingredient : {r.get('active_ingredient')!r}")
        print(f"    pack_size (API)   : {r.get('pack_size')!r}   ← Stage 2")
        print(f"    pack_style (API)  : {r.get('pack_style')!r}  ← Stage 2")
        print(f"    pdf_url           : {r.get('pdf_url')!r}")
        print(f"    pdf_date          : {r.get('pdf_date')!r}")
        if r.get("description"):
            print(f"    description       : {r.get('description')!r}")

    # -------------------------------------------------------------------
    # Stage 3: PDF
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STAGE 3 — LABELLING PDF (Ollama or regex fallback)")
    print("=" * 70)
    stage3_results = []
    for din, label, drug_code, strength in din_info:
        if drug_code is None:
            print(f"\n  DIN {din}: drug_code unknown — skipping Stage 3")
            stage3_results.append({"din": din, "error": "no drug_code"})
            continue
        print(f"\n  DIN {din} — {label} (strength={strength!r})")
        r = await run_stage3(din, drug_code, strength)
        stage3_results.append(r)

        if r.get("error"):
            print(f"    ERROR: {r['error']}")
            continue

        stage3_fields = (
            "excipients_core", "excipients_coating", "preservatives",
            "ph", "colour", "shape", "size_mm", "weight",
        )
        for field in stage3_fields:
            val = r.get(field, "")
            page = r.get(f"{field}_page")
            src = f"(p.{page})" if page else ""
            flag = ""
            if val == NOT_IN_PM:
                flag = " ← not in PM"
            elif not val:
                flag = " ← no PM / not extracted"
            print(f"    {field:<22} : {val!r:40} {src}{flag}")
        if r.get("needs_ocr"):
            print("    *** PDF is scanned — needs OCR ***")

    # -------------------------------------------------------------------
    # RIVACOCET-specific assertion
    # -------------------------------------------------------------------
    rivacocet = next((r for r in stage3_results if r.get("din") == "02242468"), None)
    print("\n" + "=" * 70)
    print("RIVACOCET ASSERTIONS (DIN 02242468)")
    print("=" * 70)
    if rivacocet and not rivacocet.get("error"):
        coating = rivacocet.get("excipients_coating", "")
        pack    = rivacocet.get("pack_size", "")
        print(f"  excipients_coating : {coating!r}")
        print(f"  pack_size          : {pack!r}")
        bad_coating = bool(re.search(r"administration\s+strength", coating or "", re.IGNORECASE))
        bad_psize   = bool(re.search(r"administration\s+strength", pack or "", re.IGNORECASE))
        print(f"  coating contains 'Administration Strength': {bad_coating}  (want: False)")
        print(f"  pack_size contains 'Administration Strength': {bad_psize}  (want: False)")
    else:
        print("  (RIVACOCET not enriched — cannot assert)")

    # -------------------------------------------------------------------
    # Coverage table
    # -------------------------------------------------------------------
    combined = [
        {**s2, **s3}
        for s2 in stage2_results
        for s3 in stage3_results
        if s2.get("din") == s3.get("din")
    ]
    print_coverage(combined)

    # -------------------------------------------------------------------
    # Assertions
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ASSERTIONS")
    print("=" * 70)
    failures = run_assertions(stage2_results, stage3_results)
    if failures:
        for f in failures:
            print(f"  {f}")
        print(f"\n  {len(failures)} assertion(s) FAILED")
    else:
        print("  All assertions PASSED")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
