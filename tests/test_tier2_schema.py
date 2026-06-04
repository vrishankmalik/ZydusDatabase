"""Tier 2 — Schema / contract tests (offline, against fixtures).

Every parsed record is validated against an explicit field set.
A renamed or removed field fails the test here — before it silently
slips through to the UI.
"""
import pytest

from tests.conftest import FIXTURES_DIR, load_json, load_html
from app.sources.dpd import search_dpd
from app.sources.noc import search_noc
from app.sources.patent_register import _parse_results_table as pr_parse_table
from app.sources.generic_submissions import _parse_table as gsur_parse_table


# ── DPD raw API schema ────────────────────────────────────────────────────────

_ACTIVEINGREDIENT_REQUIRED = {"drug_code", "ingredient_name", "strength", "strength_unit"}
_DRUGPRODUCT_REQUIRED = {
    "drug_code",
    "brand_name",
    "drug_identification_number",
    "company_name",
    "ai_group_no",
    "number_of_ais",
    "class_name",
    "last_update_date",
}


class TestDPDSchema:
    def test_activeingredient_fixture_fields(self):
        rows = load_json("dpd/activeingredient_metformin.json")
        assert isinstance(rows, list) and len(rows) > 0
        for row in rows:
            missing = _ACTIVEINGREDIENT_REQUIRED - set(row)
            assert not missing, f"activeingredient row missing fields: {missing}"

    def test_drugproduct_fixture_fields(self):
        product = load_json("dpd/drugproduct_code_99999.json")
        missing = _DRUGPRODUCT_REQUIRED - set(product)
        assert not missing, f"drugproduct missing fields: {missing}"

    def test_activeingredient_types(self):
        rows = load_json("dpd/activeingredient_metformin.json")
        for row in rows:
            assert isinstance(row["drug_code"], int), "drug_code must be int"
            assert isinstance(row["ingredient_name"], str), "ingredient_name must be str"
            assert isinstance(row["strength"], str), "strength must be str"
            assert isinstance(row["strength_unit"], str), "strength_unit must be str"

    def test_drugproduct_types(self):
        product = load_json("dpd/drugproduct_code_99999.json")
        assert isinstance(product["drug_code"], int)
        assert isinstance(product["brand_name"], str)
        assert isinstance(product["drug_identification_number"], str)
        assert isinstance(product["company_name"], str)
        assert isinstance(product["number_of_ais"], int)
        assert isinstance(product["class_name"], str)
        assert isinstance(product["last_update_date"], str)

    async def test_dpd_record_schema(self, mock_dpd):
        """DrugRecord produced from DPD fixture has all required model fields."""
        result = await search_dpd("metformin", field="ingredient")
        assert result.status == "ok"
        for r in result.records:
            assert r.source == "DPD"
            assert r.record_url is not None
            assert isinstance(r.all_ingredients, list)
            assert isinstance(r.source_specific, dict)
            assert "drug_code" in r.source_specific


# ── NOC JSON API schema ───────────────────────────────────────────────────────

class TestNOCSchema:
    def test_noc_api_ingredient_fixture_fields(self):
        rows = load_json("noc/api_medicinalingredient.json")
        assert isinstance(rows, list) and len(rows) > 0
        required = {"noc_number", "noc_pi_din_product_id", "noc_pi_medic_ingr_name"}
        for row in rows:
            missing = required - set(row)
            assert not missing, f"ingredient record missing fields: {missing}"

    def test_noc_api_drugproduct_fixture_fields(self):
        rows = load_json("noc/api_drugproduct_99001.json")
        assert isinstance(rows, list) and len(rows) > 0
        required = {"noc_number", "noc_br_product_id", "noc_br_brandname", "noc_br_din"}
        for row in rows:
            missing = required - set(row)
            assert not missing, f"drugproduct record missing fields: {missing}"

    def test_noc_api_main_fixture_fields(self):
        main = load_json("noc/api_main_99001.json")
        assert isinstance(main, dict)
        required = {"noc_number", "noc_date", "noc_manufacturer_name", "noc_status_with_conditions"}
        missing = required - set(main)
        assert not missing, f"noticeofcompliancemain missing fields: {missing}"

    async def test_noc_api_record_schema(self, mock_noc):
        """DrugRecord produced from NOC JSON API fixture has all required model fields."""
        result = await search_noc("METFORMIN HYDROCHLORIDE", field="ingredient")
        assert result.status == "ok"
        for r in result.records:
            assert r.source == "NOC"
            assert isinstance(r.all_ingredients, list)
            assert isinstance(r.source_specific, dict)
            assert "noc_date" in r.source_specific
            assert "noc_number" in r.source_specific


# ── Patent Register parsed-table schema ───────────────────────────────────────

_PR_ROW_REQUIRED = {"ingredient", "brand", "strength", "dosage", "din", "patent", "csp"}


class TestPatentRegisterSchema:
    def test_pr_results_table_fields(self):
        html = load_html("patent_register/results_metformin.html")
        rows = pr_parse_table(html)
        assert len(rows) > 0
        for row in rows:
            missing = _PR_ROW_REQUIRED - set(row)
            assert not missing, f"Patent Register row missing fields: {missing}"

    def test_pr_row_types(self):
        html = load_html("patent_register/results_metformin.html")
        rows = pr_parse_table(html)
        r = rows[0]
        assert isinstance(r["ingredient"], str)
        assert isinstance(r["brand"], str)
        assert isinstance(r["din"], str)
        assert isinstance(r["patent"], str)

    def test_pr_empty_result_returns_empty_list(self):
        html = load_html("patent_register/results_no_results.html")
        rows = pr_parse_table(html)
        assert rows == []


# ── Generic Submissions parsed-table schema ───────────────────────────────────

_GSUR_ROW_REQUIRED = {"ingredient", "company", "therapeutic_area", "date_accepted"}


class TestGSURSchema:
    def test_gsur_table_fields(self):
        html = load_html("generic_submissions/page.html")
        rows = gsur_parse_table(html)
        assert len(rows) > 0
        for row in rows:
            missing = _GSUR_ROW_REQUIRED - set(row)
            assert not missing, f"GSUR row missing fields: {missing}"

    def test_gsur_row_types(self):
        html = load_html("generic_submissions/page.html")
        rows = gsur_parse_table(html)
        r = rows[0]
        assert isinstance(r["ingredient"], str)
        assert isinstance(r["company"], str)
        assert isinstance(r["therapeutic_area"], str)
        assert isinstance(r["date_accepted"], str)

    def test_gsur_row_count_reasonable(self):
        html = load_html("generic_submissions/page.html")
        rows = gsur_parse_table(html)
        assert len(rows) >= 5, "Fixture should have ≥5 rows"


# ── Schema-drift canary (integration) ────────────────────────────────────────

@pytest.mark.integration
async def test_dpd_live_activeingredient_schema_drift():
    """Fetch live activeingredient and assert required fields still present.
    Fails with a clear diff if Health Canada renames or removes a field.
    """
    import httpx as _httpx
    from app.config import DPD_BASE, USER_AGENT, HTTP_TIMEOUT
    async with _httpx.AsyncClient() as client:
        r = await client.get(
            f"{DPD_BASE}/activeingredient/",
            params={"ingredientname": "metformin", "lang": "en", "type": "json"},
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            timeout=HTTP_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()

    assert isinstance(data, list) and len(data) > 0, "Expected a non-empty list"
    live_fields = set(data[0].keys())
    missing = _ACTIVEINGREDIENT_REQUIRED - live_fields
    added = live_fields - (_ACTIVEINGREDIENT_REQUIRED | {
        "dosage_value", "dosage_unit", "base", "nature", "active_ingredient_code",
    })
    assert not missing, (
        f"SCHEMA DRIFT — fields removed from activeingredient API: {missing}\n"
        f"Live fields: {sorted(live_fields)}"
    )


@pytest.mark.integration
async def test_dpd_live_drugproduct_schema_drift():
    """Fetch live drugproduct by code and assert required fields still present."""
    import httpx as _httpx
    from app.config import DPD_BASE, USER_AGENT, HTTP_TIMEOUT
    async with _httpx.AsyncClient() as client:
        # Drug code 99999 is fictional; use a real stable code via DIN lookup first.
        r = await client.get(
            f"{DPD_BASE}/drugproduct/",
            params={"din": "02229895", "lang": "en", "type": "json"},
            headers={"User-Agent": USER_AGENT},
            timeout=HTTP_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()

    product = data if isinstance(data, dict) else (data[0] if data else {})
    assert product, "Expected non-empty drugproduct response"
    live_fields = set(product.keys())
    missing = _DRUGPRODUCT_REQUIRED - live_fields
    assert not missing, (
        f"SCHEMA DRIFT — fields removed from drugproduct API: {missing}\n"
        f"Live fields: {sorted(live_fields)}"
    )
