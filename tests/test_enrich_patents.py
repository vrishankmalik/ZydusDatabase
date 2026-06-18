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

def _make_patent_zip(drug_rows: list, patent_rows: list) -> bytes:
    """Build a minimal Patent.zip in the real two-file format.

    drug_rows:   list of (DRUG_ID, DIN)
    patent_rows: list of (DRUG_ID, PATENT_NUMBER, FILING_DATE, DATE_GRANTED, EXPIRATION_DATE)
    """
    import io, zipfile

    drugs_header = "DRUG_ID,MEDICINAL_INGREDIENT_E,BRAND_NAME_E,ROUTE_OF_ADMINISTRATION_E,STRENGTH_PER_UNIT_E,HUMAN_OR_VET_E,THERAPEUTIC_CLASS,DOSAGE_FORM_E,DIN\n"
    drugs_csv = drugs_header + "".join(
        "%s,TestIngredient,TestBrand,Oral,100mg,Human,Test,Tablet,%s\n" % (did, din)
        for did, din in drug_rows
    )
    patent_header = "DRUG_ID,FORM_ID,PATENT_NUMBER,CATEGORY,FILING_DATE,DATE_GRANTED,EXPIRATION_DATE,SERVICE_COMPANY_NAME_E,FIRST_NAME,LAST_NAME,POSITION_TITLE,ADDRESS,CITY_NAME_E,PROVINCE_NAME_E,POSTAL_CODE\n"
    patent_csv = patent_header + "".join(
        "%s,999,%s,C,%s,%s,%s,TestCo,,,,,Toronto,ONTARIO,M5V1A1\n" % (did, pn, fd, gd, ed)
        for did, pn, fd, gd, ed in patent_rows
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("drugs_e.txt", drugs_csv)
        zf.writestr("patent-service_e.txt", patent_csv)
    return buf.getvalue()


def test_parse_patent_zip_reads_correct_files(monkeypatch):
    """_parse_patent_zip uses patent-service_e.txt with MM/DD/YYYY dates."""
    import app.enrichment.patents as patents_mod
    monkeypatch.setattr(patents_mod, "cache_set", lambda *a, **kw: None)

    from app.enrichment.patents import _parse_patent_zip

    zip_bytes = _make_patent_zip(
        drug_rows=[("100", "02709025")],
        patent_rows=[("100", "2709025", "12/10/2008", "08/26/2014", "12/10/2028")],
    )
    result = _parse_patent_zip(zip_bytes)

    assert "2709025" in result
    assert result["2709025"]["filing_date"] == "2008-12-10"
    assert result["2709025"]["grant_date"] == "2014-08-26"
    assert result["2709025"]["expiry_date"] == "2028-12-10"


def test_parse_patent_zip_by_din_joins_correctly(monkeypatch):
    """_parse_patent_zip_by_din joins drugs_e.txt → patent-service_e.txt by DRUG_ID."""
    import app.enrichment.patents as patents_mod
    monkeypatch.setattr(patents_mod, "cache_set", lambda *a, **kw: None)

    from app.enrichment.patents import _parse_patent_zip_by_din

    zip_bytes = _make_patent_zip(
        drug_rows=[("100", "02709025"), ("101", "09999999")],
        patent_rows=[
            ("100", "2709025", "12/10/2008", "08/26/2014", "12/10/2028"),
            ("101", "9999999", "01/01/2001", "", "01/01/2021"),
        ],
    )
    result = _parse_patent_zip_by_din(zip_bytes)

    assert "02709025" in result
    assert "2709025" in result["02709025"]
    assert "09999999" in result
    assert "9999999" in result["09999999"]


def test_parse_patent_zip_empty_returns_empty(monkeypatch):
    import app.enrichment.patents as patents_mod
    monkeypatch.setattr(patents_mod, "cache_set", lambda *a, **kw: None)

    from app.enrichment.patents import _parse_patent_zip
    import io, zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("README.txt", "no patent-service_e.txt here")
    result = _parse_patent_zip(buf.getvalue())
    assert result == {}


# ── unit: discrepancy resolution ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_zip_dates_used_without_website_fetch(tmp_path):
    """ZIP-first: when Patent.zip has dates, store them and never hit the website.

    While CPD is broken the bulk file is authoritative, so a patent present in the
    ZIP skips the per-patent live fetch entirely (the dominant cost). fetch_patent_detail
    must NOT be called, and the stored dates are the ZIP values.
    """
    _reset_store(tmp_path)
    import app.enrichment.store as store_mod

    from app.enrichment.patents import _enrich_one

    zip_data = {
        "2709025": {
            "filing_date": "2008-12-10",
            "grant_date": "2014-08-26",
            "expiry_date": "2029-01-01",
        }
    }

    fetch_mock = AsyncMock(return_value={
        "filing_date": "2008-12-10", "grant_date": "2014-08-26",
        "expiry_date": "2028-12-10", "detail_url": "https://example.com/patent/2709025",
    })
    with patch("app.enrichment.patents.fetch_patent_detail", new=fetch_mock):
        await _enrich_one("02498014", "2709025", "", zip_data)

    fetch_mock.assert_not_awaited()  # website must not be consulted when ZIP has dates

    patents = store_mod.get_patents_for_din("02498014")
    assert len(patents) == 1
    assert patents[0]["expiry_date"] == "2029-01-01", "Should use ZIP value (no website fetch)"
    # No website call → no website-vs-zip comparison → no discrepancy logged.
    assert store_mod.get_discrepancies() == []


@pytest.mark.asyncio
async def test_website_fetch_only_when_patent_absent_from_zip(tmp_path):
    """Fallback: a patent missing from the bulk file still attempts the live fetch."""
    _reset_store(tmp_path)
    import app.enrichment.store as store_mod

    from app.enrichment.patents import _enrich_one

    live_dates = {
        "filing_date": "2008-12-10",
        "grant_date": "2014-08-26",
        "expiry_date": "2028-12-10",
        "detail_url": "https://example.com/patent/2709025",
    }
    zip_data: dict = {}  # patent not in the bulk file

    fetch_mock = AsyncMock(return_value=live_dates)
    with patch("app.enrichment.patents.fetch_patent_detail", new=fetch_mock):
        await _enrich_one("02498014", "2709025", "", zip_data)

    fetch_mock.assert_awaited_once()  # website IS consulted when ZIP lacks the patent

    patents = store_mod.get_patents_for_din("02498014")
    assert len(patents) == 1
    assert patents[0]["expiry_date"] == "2028-12-10", "Should use website value (ZIP empty)"


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


# ── regression: Bug 1 — Patent.zip two-file join ─────────────────────────────

def test_patent_zip_url_is_correct():
    """Patent.zip URL must use /patent/ path, not /pr-rdb/ (was 404)."""
    from app.enrichment.patents import _PATENT_ZIP_URL
    assert "/patent/Patent.zip" in _PATENT_ZIP_URL, (
        f"Patent.zip URL should contain /patent/Patent.zip, got: {_PATENT_ZIP_URL}"
    )


def test_parse_zip_date_handles_us_format():
    """Patent.zip dates are MM/DD/YYYY; must parse to YYYY-MM-DD."""
    from app.enrichment.patents import _parse_zip_date
    assert _parse_zip_date("03/23/2007") == "2007-03-23"
    assert _parse_zip_date("12/10/2028") == "2028-12-10"
    assert _parse_zip_date("") is None
    assert _parse_zip_date(None) is None


def test_parse_patent_zip_dates_from_patent_service_file(monkeypatch):
    """_parse_patent_zip reads PATENT_NUMBER+dates from patent-service_e.txt."""
    import app.enrichment.patents as patents_mod
    monkeypatch.setattr(patents_mod, "cache_set", lambda *a, **kw: None)

    from app.enrichment.patents import _parse_patent_zip

    zip_bytes = _make_patent_zip(
        drug_rows=[("5556", "02562383")],
        patent_rows=[("5556", "2630344", "03/23/2007", "04/28/2015", "03/23/2027")],
    )
    result = _parse_patent_zip(zip_bytes)
    assert "2630344" in result
    assert result["2630344"]["filing_date"] == "2007-03-23"
    assert result["2630344"]["grant_date"] == "2015-04-28"
    assert result["2630344"]["expiry_date"] == "2027-03-23"


def test_parse_patent_zip_by_din_uses_drug_id_join(monkeypatch):
    """DIN→patent mapping requires the DRUG_ID join; DIN alone in wrong file is insufficient."""
    import app.enrichment.patents as patents_mod
    monkeypatch.setattr(patents_mod, "cache_set", lambda *a, **kw: None)

    from app.enrichment.patents import _parse_patent_zip_by_din

    # Lecanemab: DRUG_ID=5556, DIN=02562383, patent=2630344
    zip_bytes = _make_patent_zip(
        drug_rows=[("5556", "02562383")],
        patent_rows=[("5556", "2630344", "03/23/2007", "04/28/2015", "03/23/2027")],
    )
    result = _parse_patent_zip_by_din(zip_bytes)
    assert "02562383" in result
    assert "2630344" in result["02562383"]


# ── regression: CPD redirect bug — dates must come from ZIP when CPD broken ───

def test_fetch_cpd_dates_detects_redirect_and_returns_none():
    """When CPD returns a 3xx redirect (broken server config), _fetch_cpd_dates
    must return None for all date fields immediately — no 20-second timeout."""
    import asyncio
    from unittest.mock import MagicMock, patch

    # Simulate the real broken CPD redirect: 308 → bare IP
    mock_response = MagicMock()
    mock_response.is_redirect = True
    mock_response.headers = {"location": "https://50.195.125.245/"}
    mock_response.status_code = 308

    async def _mock_get(*args, **kwargs):
        return mock_response

    async def run():
        from app.enrichment.patents import _fetch_cpd_dates
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = _mock_get
            mock_client_cls.return_value = mock_client
            return await _fetch_cpd_dates("2630344")

    from unittest.mock import AsyncMock
    result = asyncio.run(run())
    assert result["filing_date"] is None
    assert result["grant_date"] is None
    assert result["expiry_date"] is None


def test_cpd_fixture_parses_correctly():
    """Offline fixture test: _fetch_cpd_dates parse logic produces correct dates
    when the CPD page is reachable (regression guard against parse-logic breakage)."""
    import asyncio
    from unittest.mock import MagicMock, AsyncMock, patch

    html = (FIXTURES / "cpd" / "summary_2630344.html").read_text()

    mock_response = MagicMock()
    mock_response.is_redirect = False
    mock_response.status_code = 200
    mock_response.text = html

    async def _mock_get(*args, **kwargs):
        return mock_response

    async def run():
        from app.enrichment.patents import _fetch_cpd_dates
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = _mock_get
            mock_client_cls.return_value = mock_client
            return await _fetch_cpd_dates("2630344")

    result = asyncio.run(run())
    assert result["filing_date"] == "2007-03-23", f"filing_date wrong: {result['filing_date']}"
    assert result["grant_date"] == "2015-04-28", f"grant_date wrong: {result['grant_date']}"
    assert result["expiry_date"] == "2027-03-23", f"expiry_date wrong: {result['expiry_date']}"


@pytest.mark.asyncio
async def test_zip_fallback_when_cpd_redirects(tmp_path):
    """End-to-end: CPD broken (308 redirect) → ZIP data provides all 3 dates."""
    _reset_store(tmp_path)
    import app.enrichment.store as store_mod

    from app.enrichment.patents import _enrich_one

    # CPD returns all-None because of the redirect
    cpd_none = {"filing_date": None, "grant_date": None, "expiry_date": None,
                "detail_url": "https://brevets-patents.ic.gc.ca/opic-cipo/cpd/eng/patent/2630344/summary.html"}

    zip_data = {
        "2630344": {"filing_date": "2007-03-23", "grant_date": "2015-04-28", "expiry_date": "2027-03-23"}
    }

    with patch("app.enrichment.patents.fetch_patent_detail", new=AsyncMock(return_value=cpd_none)):
        await _enrich_one("02562383", "2630344", "", zip_data)

    patents = store_mod.get_patents_for_din("02562383")
    assert len(patents) == 1
    assert patents[0]["filing_date"] == "2007-03-23", "Filing date must come from ZIP when CPD returns None"
    assert patents[0]["grant_date"] == "2015-04-28", "Grant date must come from ZIP when CPD returns None"
    assert patents[0]["expiry_date"] == "2027-03-23", "Expiry date must come from ZIP when CPD returns None"
