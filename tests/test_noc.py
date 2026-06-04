"""Tests for NOC source — mix of offline (API fixture) and live network (integration-marked).

Offline tests use the mock_noc fixture (respx-backed JSON API stubs).
Live tests are marked @pytest.mark.integration and run with: make test-live
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from app.sources.noc import search_noc, _normalize_din


# ── Offline (fixture-based) tests ─────────────────────────────────────────────

async def test_noc_ingredient_search_ok(mock_noc):
    """Ingredient search returns ok with at least one record."""
    result = await search_noc("METFORMIN HYDROCHLORIDE", field="ingredient")
    assert result.status == "ok", f"{result.status}: {result.error_message}"
    assert result.count > 0


async def test_noc_din_join_attaches_din(mock_noc):
    """Every record must carry a DIN — validates the product_id→DIN join."""
    result = await search_noc("METFORMIN HYDROCHLORIDE", field="ingredient")
    assert result.status == "ok"
    for r in result.records:
        assert r.din is not None and r.din.strip(), (
            f"Record {r.brand_name!r} has no DIN — product_id→DIN join broken"
        )


async def test_noc_record_has_noc_date(mock_noc):
    """source_specific must contain noc_date from the noticeofcompliancemain endpoint."""
    result = await search_noc("METFORMIN HYDROCHLORIDE", field="ingredient")
    assert result.status == "ok"
    for r in result.records:
        assert "noc_date" in r.source_specific, "noc_date missing from source_specific"
        assert r.source_specific["noc_date"], "noc_date is empty"


async def test_noc_all_ingredients_populated(mock_noc):
    """all_ingredients must be non-empty for ingredient search results."""
    result = await search_noc("METFORMIN HYDROCHLORIDE", field="ingredient")
    assert result.status == "ok"
    for r in result.records:
        assert r.all_ingredients, f"all_ingredients empty for {r.brand_name!r}"


async def test_noc_no_results_for_unknown_ingredient(mock_noc):
    """A query that matches nothing in the ingredient list returns no_results."""
    result = await search_noc("ZZZZNOTADRUG12345", field="ingredient")
    assert result.status in ("no_results", "ok")
    if result.status == "ok":
        assert result.count == 0


async def test_noc_brand_field_unsupported(mock_noc):
    """Brand search returns unsupported — the JSON API only exposes ingredient lookup."""
    result = await search_noc("Glucophage", field="brand")
    assert result.status == "unsupported", (
        f"Expected unsupported for field=brand, got {result.status}"
    )


async def test_noc_company_field_unsupported(mock_noc):
    result = await search_noc("Merck", field="company")
    assert result.status == "unsupported"


async def test_noc_din_field_unsupported(mock_noc):
    result = await search_noc("02229895", field="din")
    assert result.status == "unsupported"


async def test_noc_records_source_tagged(mock_noc):
    result = await search_noc("METFORMIN HYDROCHLORIDE", field="ingredient")
    assert result.status == "ok"
    for r in result.records:
        assert r.source == "NOC"


# ── DIN normalisation unit tests ──────────────────────────────────────────────

def test_normalize_din_na():
    assert _normalize_din("N/A") is None
    assert _normalize_din("NA") is None
    assert _normalize_din("Not Applicable") is None
    assert _normalize_din("") is None
    assert _normalize_din(None) is None


def test_normalize_din_valid():
    assert _normalize_din("02242974") == "02242974"
    assert _normalize_din("  02242974  ") == "02242974"


# ── Live integration tests ─────────────────────────────────────────────────────

@pytest.mark.integration
async def test_noc_live_metformin_ingredient():
    """Live: metformin ingredient search must return ok with >0 records and ≥95% DIN rate."""
    result = await search_noc("METFORMIN HYDROCHLORIDE", field="ingredient")
    assert result.status == "ok", f"{result.status}: {result.error_message}"
    assert result.count > 0

    records_with_din = [r for r in result.records if r.din]
    rate = len(records_with_din) / result.count
    assert rate >= 0.95, f"Only {rate*100:.1f}% of NOC records carry a DIN; expected ≥95%"
    print(f"\n[live] NOC METFORMIN HYDROCHLORIDE count={result.count}, DIN rate={rate*100:.1f}%")


@pytest.mark.integration
async def test_noc_live_broad_metformin():
    """Live: broad 'metformin' ingredient search also returns ok (no 'too many records' error)."""
    result = await search_noc("metformin", field="ingredient")
    assert result.status in ("ok", "no_results"), (
        f"Expected ok/no_results for broad metformin; got {result.status}: {result.error_message}"
    )


@pytest.mark.integration
async def test_noc_live_brand_unsupported():
    result = await search_noc("Glucophage", field="brand")
    assert result.status == "unsupported"


@pytest.mark.integration
async def test_noc_live_no_results():
    result = await search_noc("xyznonexistentdrugabc123", field="ingredient")
    assert result.status in ("no_results", "ok")
