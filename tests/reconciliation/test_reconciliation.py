"""Completeness reconciliation: pipeline DIN set vs. Health Canada bulk extract.

Tests that the pipeline returns *every* DIN the authoritative extract has for a
given ingredient.  A gap beyond the tolerance means a result cap, a missing page,
or source drift — exactly the class of bugs no fixture-based test can catch.

Tolerance rationale
-------------------
0.5 % (≤ 1 in 200 DINs) is chosen to absorb:
  • One-day nightly-refresh timing skew between the bulk extract and the live API.
  • Rare edge cases where the extract and live API disagree on a single product.
Anything larger (e.g., 150 out of 250 acetaminophen DINs) indicates a real cap
or pagination bug and must fail loudly.

NOC bulk extract
----------------
Health Canada does not currently publish a downloadable bulk export for the NOC
database (noc_brand / noc_ingredient flat files).  The NOC reconciliation block
is included as a stub and is skipped automatically when the files are absent.
If the NOC bulk export becomes available, set RECONCILE_NOC_DIR to its path.

Run
---
    # Offline: never (requires live download + live API)
    make reconcile

    # Or directly:
    pytest tests/reconciliation/ -v -m integration --tb=short \
        -p no:randomly  # deterministic order

Environment variables
---------------------
RECONCILE_CACHE_DIR      — local cache directory (default /tmp/canadian_drug_reconcile_cache)
RECONCILE_FRESHNESS_HOURS — how old the cached extract can be (default 12)
RECONCILE_TOLERANCE      — max fraction of missed DINs before hard-fail (default 0.005)
DPD_MAX_RESULTS          — **must be unset or very large** for a valid completeness check.
                           The test overrides it to 9999 automatically.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.reconciliation.downloader import ensure_dpd_extract
from tests.reconciliation.dpd_parser import build_extract_din_set

log = logging.getLogger(__name__)

RECONCILE_TOLERANCE = float(os.getenv("RECONCILE_TOLERANCE", "0.005"))

# Sample ingredients: high-volume, mid-volume, rare, salt/multi-word
SAMPLE_INGREDIENTS = [
    "acetaminophen",         # high-volume: ~250 extract DINs
    "ibuprofen",             # mid-volume:  ~68 extract DINs
    "azithromycin",          # rare:        ~24 extract DINs
    "cetirizine hydrochloride",  # salt/multi-word: ~29 extract DINs
]


@pytest.fixture(scope="module")
def dpd_extract_dir():
    """Download (or load from cache) the DPD bulk extract.  Module-scoped so
    we only download once per test session even with multiple test cases."""
    return ensure_dpd_extract()


@pytest.fixture(autouse=True)
def lift_dpd_cap(monkeypatch):
    """Remove the DPD_MAX_RESULTS cap for the duration of each test.

    The cap exists to protect the live API from overload; for a completeness
    check we need every result, so we override it here.
    """
    monkeypatch.setenv("DPD_MAX_RESULTS", "9999")
    # Also patch the config value that was already imported
    import app.config as cfg
    import app.sources.dpd as dpd_mod
    monkeypatch.setattr(cfg, "DPD_MAX_RESULTS", 9999)
    monkeypatch.setattr(dpd_mod, "DPD_MAX_RESULTS", 9999)


def _get_pipeline_dins(ingredient: str) -> set[str]:
    """Run the live DPD pipeline and collect all DINs it returns."""
    from app.din_utils import normalize_din
    from app.sources.dpd import search_dpd

    result = asyncio.run(search_dpd(ingredient, field="ingredient", extra_terms=[]))
    dins: set[str] = set()
    for record in result.records:
        if record.din:
            nd = normalize_din(record.din)
            if nd:
                dins.add(nd)
    return dins


@pytest.mark.integration
@pytest.mark.parametrize("ingredient", SAMPLE_INGREDIENTS)
def test_dpd_completeness(ingredient: str, dpd_extract_dir: Path) -> None:
    """Pipeline DIN set must cover the extract DIN set within tolerance.

    Hard fails on:  extract_dins − pipeline_dins  > tolerance  (pipeline missed DINs)
    Warns only on:  pipeline_dins − extract_dins  > 0           (pipeline has extras)
    """
    extract_dins = build_extract_din_set(ingredient, dpd_extract_dir)
    assert extract_dins, (
        f"No DINs found in extract for '{ingredient}'. "
        "Check that the extract was downloaded correctly."
    )

    pipeline_dins = _get_pipeline_dins(ingredient)

    missed = extract_dins - pipeline_dins
    extra = pipeline_dins - extract_dins

    missed_rate = len(missed) / len(extract_dins) if extract_dins else 0.0

    log.info(
        "ingredient=%r  extract=%d  pipeline=%d  missed=%d (%.1f%%)  extra=%d",
        ingredient,
        len(extract_dins),
        len(pipeline_dins),
        len(missed),
        missed_rate * 100,
        len(extra),
    )

    if extra:
        sample = sorted(extra)[:5]
        log.warning(
            "ingredient=%r: pipeline has %d DINs not in extract (timing/scope skew). "
            "Sample: %s",
            ingredient,
            len(extra),
            sample,
        )

    if missed:
        sample_missed = sorted(missed)[:10]

    assert missed_rate <= RECONCILE_TOLERANCE, (
        f"COMPLETENESS FAILURE for '{ingredient}': pipeline missed "
        f"{len(missed)}/{len(extract_dins)} extract DINs "
        f"({missed_rate*100:.1f}% > tolerance {RECONCILE_TOLERANCE*100:.1f}%). "
        f"Sample missed DINs: {sorted(missed)[:10]}. "
        f"This indicates a result cap, missing page, filter bug, or source drift. "
        f"Check DPD_MAX_RESULTS and the pipeline's ingredient search logic."
    )
