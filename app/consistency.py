"""Cross-source consistency checks for DIN-keyed data.

For each DIN present in two or more sources, the sources must agree on:
  1. Normalized ingredient set (same active ingredients after uppercasing/stripping)
  2. Brand name (case-insensitive equality)

Disagreements are data-quality issues surfaced as warnings, never errors.
They indicate mis-joins, parse bugs, or real discrepancies between databases.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

from app.din_utils import normalize_din
from app.models import DrugRecord

log = logging.getLogger(__name__)


@dataclass
class ConsistencyWarning:
    din: str
    field: str           # "ingredient" | "brand"
    source_a: str
    value_a: Optional[str]
    source_b: str
    value_b: Optional[str]
    detail: str

    def __str__(self) -> str:
        return (
            f"[DIN {self.din}] {self.field} mismatch: "
            f"{self.source_a}={self.value_a!r} vs {self.source_b}={self.value_b!r} — {self.detail}"
        )


def _normalized_ingredient_set(record: DrugRecord) -> frozenset[str]:
    """Return the set of normalized ingredient names for a record."""
    if record.all_ingredients:
        names = record.all_ingredients
    elif record.ingredient:
        names = re.split(r"\s*;\s*", record.ingredient.strip())
    else:
        return frozenset()
    return frozenset(re.sub(r"\s+", " ", n.strip()).upper() for n in names if n.strip())


def _normalized_brand(record: DrugRecord) -> Optional[str]:
    if not record.brand_name:
        return None
    return re.sub(r"\s+", " ", record.brand_name.strip()).upper()


def check_cross_source_consistency(
    records: list[DrugRecord],
) -> list[ConsistencyWarning]:
    """Compare ingredient sets and brand names across sources for each shared DIN.

    Returns a (possibly empty) list of ConsistencyWarning objects.
    Logs each warning at WARNING level so it appears in live-run output.
    """
    # Group records by normalized DIN; skip DIN-less records
    by_din: dict[str, list[DrugRecord]] = {}
    for rec in records:
        if not rec.din:
            continue
        nd = normalize_din(rec.din)
        if nd:
            by_din.setdefault(nd, []).append(rec)

    warnings: list[ConsistencyWarning] = []

    for din, group in by_din.items():
        if len(group) < 2:
            continue

        # Only compare records that actually carry ingredient or brand data
        records_with_data = [
            r for r in group if r.ingredient or r.all_ingredients or r.brand_name
        ]
        if len(records_with_data) < 2:
            continue

        # Build per-source ingredient sets and brand names
        # When a source contributes multiple records for the same DIN, union them.
        source_ingredients: dict[str, frozenset[str]] = {}
        source_brands: dict[str, str | None] = {}

        for rec in records_with_data:
            ings = _normalized_ingredient_set(rec)
            brand = _normalized_brand(rec)

            if ings:
                existing = source_ingredients.get(rec.source, frozenset())
                source_ingredients[rec.source] = existing | ings

            if brand and rec.source not in source_brands:
                source_brands[rec.source] = brand

        sources_with_ings = list(source_ingredients.keys())
        sources_with_brands = list(source_brands.keys())

        # Check ingredient sets pairwise
        for i in range(len(sources_with_ings)):
            for j in range(i + 1, len(sources_with_ings)):
                sa, sb = sources_with_ings[i], sources_with_ings[j]
                ings_a, ings_b = source_ingredients[sa], source_ingredients[sb]
                if ings_a and ings_b and ings_a != ings_b:
                    w = ConsistencyWarning(
                        din=din,
                        field="ingredient",
                        source_a=sa,
                        value_a="; ".join(sorted(ings_a)),
                        source_b=sb,
                        value_b="; ".join(sorted(ings_b)),
                        detail=(
                            f"only_in_{sa}={sorted(ings_a - ings_b)} "
                            f"only_in_{sb}={sorted(ings_b - ings_a)}"
                        ),
                    )
                    warnings.append(w)
                    log.warning(str(w))

        # Check brand names pairwise
        for i in range(len(sources_with_brands)):
            for j in range(i + 1, len(sources_with_brands)):
                sa, sb = sources_with_brands[i], sources_with_brands[j]
                brand_a, brand_b = source_brands[sa], source_brands[sb]
                if brand_a and brand_b and brand_a != brand_b:
                    w = ConsistencyWarning(
                        din=din,
                        field="brand",
                        source_a=sa,
                        value_a=brand_a,
                        source_b=sb,
                        value_b=brand_b,
                        detail="case-insensitive brand mismatch",
                    )
                    warnings.append(w)
                    log.warning(str(w))

    return warnings
