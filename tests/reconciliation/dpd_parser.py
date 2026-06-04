"""Parse DPD bulk-extract flat files to build DIN sets per ingredient.

Exposes one public function:
    build_extract_din_set(ingredient, extract_dir) -> set[str]

Applies the same substring matching rule the DPD REST API uses:
case-insensitive substring of the INGREDIENT column in ingred.txt.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

from tests.reconciliation.downloader import (
    DPD_DRUG_COL_BRAND,
    DPD_DRUG_COL_CODE,
    DPD_DRUG_COL_DIN,
    DPD_INGRED_COL_CODE,
    DPD_INGRED_COL_NAME,
)


def _normalize_din(raw: str) -> str | None:
    digits = re.sub(r"\D", "", raw.strip())
    if not digits:
        return None
    return digits.zfill(8)


def _ingredient_matches(extract_name: str, query: str) -> bool:
    """Mirror the DPD API's case-insensitive substring match."""
    return query.upper() in extract_name.upper()


def build_extract_din_set(ingredient: str, extract_dir: Path) -> set[str]:
    """Return the set of 8-digit DINs whose products contain *ingredient*.

    Replicates the pipeline's matching rule (case-insensitive substring of
    the ingredient name column) so the comparison isolates completeness, not
    matching differences.
    """
    ingred_path = extract_dir / "ingred.txt"
    drug_path = extract_dir / "drug.txt"

    # Phase 1: collect drug_codes whose ingredient column contains the query
    matching_codes: set[str] = set()
    with open(ingred_path, encoding="latin-1") as f:
        for row in csv.reader(f):
            if (
                len(row) > DPD_INGRED_COL_NAME
                and _ingredient_matches(row[DPD_INGRED_COL_NAME], ingredient)
            ):
                matching_codes.add(row[DPD_INGRED_COL_CODE])

    if not matching_codes:
        return set()

    # Phase 2: map drug_codes → DINs via drug.txt
    dins: set[str] = set()
    with open(drug_path, encoding="latin-1") as f:
        for row in csv.reader(f):
            if (
                len(row) > DPD_DRUG_COL_DIN
                and row[DPD_DRUG_COL_CODE] in matching_codes
            ):
                din = _normalize_din(row[DPD_DRUG_COL_DIN])
                if din:
                    dins.add(din)

    return dins


def build_extract_brand_map(extract_dir: Path) -> dict[str, str]:
    """Return a drug_code → brand_name mapping for spot-checks."""
    brand_map: dict[str, str] = {}
    drug_path = extract_dir / "drug.txt"
    with open(drug_path, encoding="latin-1") as f:
        for row in csv.reader(f):
            if len(row) > DPD_DRUG_COL_BRAND:
                brand_map[row[DPD_DRUG_COL_CODE]] = row[DPD_DRUG_COL_BRAND]
    return brand_map
