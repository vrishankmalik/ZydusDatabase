"""Tests for app/enrichment/patents.py.

Tier 1 (unit, offline): discrepancy resolution, no-patents path, detail-page parsing.
Tier 2 (integration, live): dates match the live Patent Register page.
"""
from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


# ── helpers ───────────────────────────────────────────────────────────────────

def _reset_store(tmp_path):
    """Point the enrichment store at a fresh temp DB."""
    import app.enrichment.store as store_mod
    store_mod.reset_for_testing(str(tmp_path / "enrich.db"))


# ── unit: detail page parsing ─────────────────────────────────────────────────

def test_parse_detail_page_extracts_dates():
    from app.enrichment.patents import _parse_detail_page

    html = (FIXTURES / "patent_register" / "detail_2709025.html").read_text()
    result = _parse_detail_page(html)

    assert result["filing_date"] == "2008-12-10"
    assert result["grant_date"] == "2014-08-26"
    assert result["expiry_date"] == "2028-12-10"


def test_parse_detail_page_missing_fields_returns_none():
    from app.enrichment.patents import _parse_detail_page

    result = _parse_detail_page("<html><body>No dates here.</body></html>")
    assert result.get("filing_date") is None
    assert result.get("grant_date") is None
    assert result.get("expiry_date") is None


# ── unit: Patent.zip parsing ──────────────────────────────────────────────────

def test_parse_patent_zip_reads_csv():
    from app.enrichment.patents import _parse_patent_zip
    import io, zipfile, csv

    # Build a minimal in-memory zip with a CSV
    csv_content = (
        "PATENT_NO,FILING_DATE,DATE_GRANTED,EXPIRY_DATE\n"
        "2709025,2008-12-10,2014-08-26,2028-12-10\n"
        "9999999,2001-01-01,,2021-01-01\n"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("Patent.csv", csv_content)
    zip_bytes = buf.getvalue()

    result = _parse_patent_zip(zip_bytes)

    assert "2709025" in result
    assert result["2709025"]["filing_date"] == "2008-12-10"
    assert result["2709025"]["grant_date"] == "2014-08-26"
    assert result["2709025"]["expiry_date"] == "2028-12-10"
    assert result["9999999"]["grant_date"] is None or result["9999999"]["grant_date"] == ""


def test_parse_patent_zip_empty_returns_empty():
    from app.enrichment.patents import _parse_patent_zip
    import io, zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("README.txt", "no csv here")
    result = _parse_patent_zip(buf.getvalue())
    assert result == {}


# ── unit: discrepancy resolution ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_discrepancy_resolved_to_website_value(tmp_path):
    """When live dates differ from zip dates, store website value and log discrepancy."""
    _reset_store(tmp_path)
    import app.enrichment.store as store_mod

    from app.enrichment.patents import _enrich_one

    live_dates = {
        "filing_date": "2008-12-10",
        "grant_date": "2014-08-26",
        "expiry_date": "2028-12-10",
        "detail_url": "https://example.com/patent/2709025",
    }
    # Zip has a different expiry date
    zip_data = {
        "2709025": {
            "filing_date": "2008-12-10",
            "grant_date": "2014-08-26",
            "expiry_date": "2029-01-01",  # differs from live
        }
    }

    with patch(
        "app.enrichment.patents.fetch_patent_detail",
        new=AsyncMock(return_value=live_dates),
    ):
        await _enrich_one("02498014", "2709025", "", zip_data)

    patents = store_mod.get_patents_for_din("02498014")
    assert len(patents) == 1
    assert patents[0]["expiry_date"] == "2028-12-10", "Should use website value on discrepancy"

    discrepancies = store_mod.get_discrepancies()
    assert len(discrepancies) == 1
    assert discrepancies[0]["field"] == "expiry_date"
    assert discrepancies[0]["website_value"] == "2028-12-10"
    assert discrepancies[0]["zip_value"] == "2029-01-01"


@pytest.mark.asyncio
async def test_no_discrepancy_when_dates_match(tmp_path):
    _reset_store(tmp_path)
    import app.enrichment.store as store_mod

    from app.enrichment.patents import _enrich_one

    dates = {
        "filing_date": "2008-12-10",
        "grant_date": "2014-08-26",
        "expiry_date": "2028-12-10",
        "detail_url": None,
    }
    zip_data = {"2709025": {k: dates[k] for k in ("filing_date", "grant_date", "expiry_date")}}

    with patch(
        "app.enrichment.patents.fetch_patent_detail",
        new=AsyncMock(return_value=dates),
    ):
        await _enrich_one("02498014", "2709025", "", zip_data)

    assert store_mod.get_discrepancies() == []


# ── unit: DIN with no patents is clean ────────────────────────────────────────

@pytest.mark.asyncio
async def test_din_with_no_patents_is_clean(tmp_path):
    _reset_store(tmp_path)
    import app.enrichment.store as store_mod

    from app.enrichment.patents import enrich_patents

    with (
        patch("app.enrichment.patents._pr_get_session", new=AsyncMock(return_value=([], [], "sess"))),
        patch("app.enrichment.patents._din_to_patent_numbers", new=AsyncMock(return_value=[])),
        patch("app.enrichment.patents.load_patent_zip", new=AsyncMock(return_value={})),
        patch("app.enrichment.patents.load_patent_zip_din_map", new=AsyncMock(return_value={})),
    ):
        result = await enrich_patents(["99999999"])

    assert result == {"99999999": []}
    assert store_mod.get_patents_for_din("99999999") == []


# ── unit: zip-only dates used when live page has nothing ─────────────────────

@pytest.mark.asyncio
async def test_zip_only_used_when_live_page_empty(tmp_path):
    _reset_store(tmp_path)
    import app.enrichment.store as store_mod

    from app.enrichment.patents import _enrich_one

    live_dates = {"filing_date": None, "grant_date": None, "expiry_date": None, "detail_url": None}
    zip_data = {"9999999": {"filing_date": "2001-01-01", "grant_date": None, "expiry_date": "2021-01-01"}}

    with patch("app.enrichment.patents.fetch_patent_detail", new=AsyncMock(return_value=live_dates)):
        await _enrich_one("12345678", "9999999", "", zip_data)

    patents = store_mod.get_patents_for_din("12345678")
    assert patents[0]["filing_date"] == "2001-01-01"
    assert patents[0]["grant_date"] is None


# ── integration (live) ────────────────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_patent_detail_has_dates():
    """Fetch a known patent from the live Patent Register and assert dates are present."""
    from app.enrichment.patents import fetch_patent_detail

    # Patent 2709025 is a real alpelisib patent
    result = await fetch_patent_detail("2709025")

    # We don't hard-code the exact dates in case the site updates them,
    # but at minimum the expiry date should be present and look like a date.
    expiry = result.get("expiry_date")
    assert expiry is not None, "Expected expiry_date from live Patent Register"
    import re
    assert re.match(r"\d{4}[-/]\d{2}[-/]\d{2}", expiry), f"Unexpected date format: {expiry}"
