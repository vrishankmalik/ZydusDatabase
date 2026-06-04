"""Download and locally cache the Health Canada DPD bulk extract files.

DPD allfiles.zip bundles all status sets (marketed + approved + cancelled +
dormant) into a single download.  We cache it under RECONCILE_CACHE_DIR and
refresh when the cached copy is older than RECONCILE_FRESHNESS_HOURS.

Column layout (confirmed from live data 2026-06-01, no header row):

  drug.txt  — comma-separated, quoted
    col 0  DRUG_CODE
    col 1  PRODUCT_CATEGORIZATION
    col 2  CLASS
    col 3  DRUG_IDENTIFICATION_NUMBER  ← DIN
    col 4  BRAND_NAME
    (13 columns total)

  ingred.txt — comma-separated, quoted
    col 0  DRUG_CODE
    col 1  ACTIVE_INGREDIENT_CODE
    col 2  INGREDIENT                  ← ingredient name
    col 3  INGREDIENT_SUPPLIED_IND
    col 4  STRENGTH
    col 5  STRENGTH_UNIT
    (14-15 columns total)

Reference: https://www.canada.ca/en/health-canada/services/drugs-health-products/
           drug-products/drug-product-database/what-data-extract-drug-product-database.html
"""
from __future__ import annotations

import io
import os
import time
import zipfile
from pathlib import Path

import httpx

DPD_ALLFILES_URL = (
    "https://www.canada.ca/content/dam/hc-sc/documents/services/"
    "drug-product-database/allfiles.zip"
)

RECONCILE_CACHE_DIR = Path(
    os.getenv("RECONCILE_CACHE_DIR", "/tmp/canadian_drug_reconcile_cache")
)
RECONCILE_FRESHNESS_HOURS = int(os.getenv("RECONCILE_FRESHNESS_HOURS", "12"))

# Exact column positions (0-indexed) — verified against live extract.
DPD_DRUG_COL_CODE = 0
DPD_DRUG_COL_DIN = 3
DPD_DRUG_COL_BRAND = 4

DPD_INGRED_COL_CODE = 0
DPD_INGRED_COL_NAME = 2


def _cache_is_fresh(path: Path, max_age_hours: int) -> bool:
    if not path.exists():
        return False
    age_seconds = time.time() - path.stat().st_mtime
    return age_seconds < max_age_hours * 3600


def ensure_dpd_extract(
    *,
    cache_dir: Path = RECONCILE_CACHE_DIR,
    freshness_hours: int = RECONCILE_FRESHNESS_HOURS,
) -> Path:
    """Return the local directory containing drug.txt and ingred.txt.

    Downloads allfiles.zip if the cached copy is missing or stale, then
    extracts the two files we need.  Returns the directory path.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    drug_path = cache_dir / "drug.txt"
    ingred_path = cache_dir / "ingred.txt"
    zip_path = cache_dir / "allfiles.zip"

    if _cache_is_fresh(drug_path, freshness_hours) and _cache_is_fresh(
        ingred_path, freshness_hours
    ):
        return cache_dir

    # Download the ZIP
    resp = httpx.get(DPD_ALLFILES_URL, follow_redirects=True, timeout=120.0)
    resp.raise_for_status()

    zip_bytes = resp.content

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for name in ("drug.txt", "ingred.txt"):
            data = zf.read(name)
            (cache_dir / name).write_bytes(data)

    # Optionally keep the ZIP for debugging
    zip_path.write_bytes(zip_bytes)

    _verify_columns(cache_dir)
    return cache_dir


def _verify_columns(cache_dir: Path) -> None:
    """Spot-check that columns are where we expect using a known DIN (GLUCOPHAGE=02099233).

    Raises AssertionError with a clear message if columns have shifted.
    """
    import csv

    din_to_find = "02099233"
    drug_code_found: str | None = None

    with open(cache_dir / "drug.txt", encoding="latin-1") as f:
        for row in csv.reader(f):
            if len(row) > DPD_DRUG_COL_DIN and row[DPD_DRUG_COL_DIN] == din_to_find:
                drug_code_found = row[DPD_DRUG_COL_CODE]
                brand = row[DPD_DRUG_COL_BRAND] if len(row) > DPD_DRUG_COL_BRAND else ""
                assert "GLUCOPHAGE" in brand.upper(), (
                    f"Schema drift: col {DPD_DRUG_COL_BRAND} expected BRAND_NAME, "
                    f"got '{brand}' for DIN {din_to_find}"
                )
                break

    assert drug_code_found is not None, (
        f"Schema verification failed: DIN {din_to_find} not found in drug.txt. "
        f"The extract format may have changed."
    )

    with open(cache_dir / "ingred.txt", encoding="latin-1") as f:
        for row in csv.reader(f):
            if (
                len(row) > DPD_INGRED_COL_NAME
                and row[DPD_INGRED_COL_CODE] == drug_code_found
            ):
                ing = row[DPD_INGRED_COL_NAME].upper()
                assert "METFORMIN" in ing, (
                    f"Schema drift: col {DPD_INGRED_COL_NAME} expected INGREDIENT, "
                    f"got '{row[DPD_INGRED_COL_NAME]}' for drug_code {drug_code_found}"
                )
                return

    assert False, (
        f"Schema verification failed: drug_code {drug_code_found} (DIN {din_to_find}) "
        f"not found in ingred.txt."
    )
