"""Tier 7 — Data-quality assertions.

These run on fixture data (always) and can be wired as runtime assertions
on live pulls.  They catch format violations that would silently corrupt
downstream consumers.
"""
import re
import pytest

from tests.conftest import load_json, load_html  # noqa: F401  (load_html kept for future use)
from app.din_utils import is_valid_din
from app.sources.dpd import search_dpd
from app.sources.noc import search_noc
from app.sources.generic_submissions import search_generic_submissions, _parse_table as gsur_parse_table
from app.sources.patent_register import search_patent_register, _parse_results_table as pr_parse_table


_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DIN_RE = re.compile(r"^\d{8}$")


# ── DIN format ────────────────────────────────────────────────────────────────

class TestDINFormat:
    async def test_dpd_dins_are_8_digits(self, mock_dpd):
        result = await search_dpd("metformin", field="ingredient")
        assert result.status == "ok"
        for r in result.records:
            if r.din:
                assert _DIN_RE.fullmatch(r.din), (
                    f"DPD DIN {r.din!r} for {r.brand_name!r} is not 8 digits"
                )

    async def test_noc_dins_are_8_digits(self, mock_noc):
        result = await search_noc("METFORMIN HYDROCHLORIDE", field="ingredient")
        assert result.status == "ok"
        for r in result.records:
            if r.din:
                assert _DIN_RE.fullmatch(r.din), (
                    f"NOC DIN {r.din!r} for {r.brand_name!r} is not 8 digits"
                )

    def test_patent_register_din_8_digits(self):
        html = load_html("patent_register/results_metformin.html")
        rows = pr_parse_table(html)
        for row in rows:
            if row["din"] and row["din"] not in ("N/A", ""):
                assert _DIN_RE.fullmatch(row["din"]), (
                    f"Patent Register DIN {row['din']!r} is not 8 digits"
                )


# ── Date format ───────────────────────────────────────────────────────────────

class TestDateFormat:
    def test_noc_dates_are_iso(self):
        from tests.conftest import load_json
        main = load_json("noc/api_main_99001.json")
        date_str = main.get("noc_date", "")
        assert date_str, "api_main_99001.json must have a noc_date field"
        assert _ISO_DATE_RE.fullmatch(date_str), (
            f"NOC API date {date_str!r} is not ISO YYYY-MM-DD"
        )

    async def test_dpd_last_update_date_is_iso(self, mock_dpd):
        result = await search_dpd("metformin", field="ingredient")
        assert result.status == "ok"
        for r in result.records:
            date_str = r.source_specific.get("last_update_date", "")
            if date_str:
                assert _ISO_DATE_RE.fullmatch(date_str), (
                    f"DPD last_update_date {date_str!r} is not ISO"
                )


# ── No 100% null columns ──────────────────────────────────────────────────────

class TestNoNullColumns:
    async def test_dpd_brand_name_not_all_null(self, mock_dpd):
        result = await search_dpd("metformin", field="ingredient")
        assert result.status == "ok"
        brands = [r.brand_name for r in result.records if r.brand_name]
        assert len(brands) > 0, "brand_name is 100% null for DPD result — field mapping broken"

    async def test_dpd_company_not_all_null(self, mock_dpd):
        result = await search_dpd("metformin", field="ingredient")
        assert result.status == "ok"
        companies = [r.company for r in result.records if r.company]
        assert len(companies) > 0, "company is 100% null for DPD result"

    async def test_noc_ingredient_not_all_null(self, mock_noc):
        result = await search_noc("METFORMIN HYDROCHLORIDE", field="ingredient")
        assert result.status == "ok"
        ingredients = [r.ingredient for r in result.records if r.ingredient]
        assert len(ingredients) > 0, "ingredient is 100% null for NOC result"

    def test_gsur_ingredient_not_all_null(self):
        html = load_html("generic_submissions/page.html")
        rows = gsur_parse_table(html)
        ingredients = [r["ingredient"] for r in rows if r.get("ingredient")]
        assert len(ingredients) > 0

    def test_gsur_company_not_all_null(self):
        html = load_html("generic_submissions/page.html")
        rows = gsur_parse_table(html)
        companies = [r["company"] for r in rows if r.get("company") and r["company"] != "Not available"]
        assert len(companies) > 0, "All GSUR companies are null or 'Not available'"


# ── Source field completeness (integration) ───────────────────────────────────

@pytest.mark.integration
async def test_dpd_live_record_counts_logged():
    """Per-source counts must be > 0 for a known ingredient; a drop to near-zero is visible."""
    result = await search_dpd("metformin hydrochloride", field="ingredient")
    assert result.status == "ok"
    # Record count logged to stdout for trend monitoring.
    print(f"\n[quality] DPD metformin_hcl count={result.count}")
    assert result.count > 0


@pytest.mark.integration
async def test_noc_live_record_counts_logged():
    result = await search_noc("METFORMIN HYDROCHLORIDE", field="ingredient")
    assert result.status in ("ok", "no_results"), f"unexpected: {result.status}: {result.error_message}"
    if result.status == "ok":
        print(f"\n[quality] NOC METFORMIN HYDROCHLORIDE count={result.count}")
        assert result.count > 0


# ── Per-record source tagging ─────────────────────────────────────────────────

class TestSourceTagging:
    async def test_dpd_records_tagged(self, mock_dpd):
        result = await search_dpd("metformin", field="ingredient")
        for r in result.records:
            assert r.source == "DPD"

    async def test_noc_records_tagged(self, mock_noc):
        result = await search_noc("METFORMIN HYDROCHLORIDE", field="ingredient")
        assert result.status == "ok"
        for r in result.records:
            assert r.source == "NOC"

    async def test_gsur_records_tagged(self, mock_gsur):
        result = await search_generic_submissions("metformin", field="ingredient")
        for r in result.records:
            assert r.source == "GenericSubmissions"

    async def test_patent_register_records_tagged(self, mock_patent_register):
        result = await search_patent_register("metformin", field="ingredient")
        if result.status == "ok":
            for r in result.records:
                assert r.source == "PatentRegister"
