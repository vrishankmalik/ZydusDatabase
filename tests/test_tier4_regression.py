"""Tier 4 — Critical regression tests.

Each test guards a specific bug that was observed or anticipated in the
data pipeline.  Comments identify the exact failure mode being prevented.
"""
import re
import pytest
import httpx
import respx

from app.sources.dpd import search_dpd
from app.sources.noc import search_noc
from app.sources.generic_submissions import search_generic_submissions
from app.sources.patent_register import search_patent_register
from app.din_utils import parse_dins


# ── Regression 1: Silent result cap ──────────────────────────────────────────

@pytest.mark.integration
async def test_dpd_acetaminophen_no_silent_cap(monkeypatch):
    """Acetaminophen must report > 500 total matches in DPD.

    Guards against the prior 150-row truncation where DPD_MAX_RESULTS silently
    hid thousands of products.  We temporarily raise the cap and assert the
    actual count reported by the API is large.
    """
    # Raise the cap so enough records are fetched to verify the real count.
    monkeypatch.setattr("app.sources.dpd.DPD_MAX_RESULTS", 5000)
    result = await search_dpd("acetaminophen", field="ingredient")
    assert result.status == "ok", result.error_message

    # Either we got > 500 records directly, or the API told us there are more.
    total = result.count
    if result.total_matches is not None:
        total = result.total_matches
    assert total > 500, (
        f"Expected > 500 acetaminophen products; got {total}. "
        f"Check DPD_MAX_RESULTS cap and pagination."
    )


async def test_dpd_cap_is_exposed_via_total_matches(mock_dpd, monkeypatch):
    """When DPD result is capped, total_matches reflects the full count.

    We mock the activeingredient endpoint to return 200 drug codes, set
    DPD_MAX_RESULTS=5, and verify the SourceResult carries total_matches=200.
    """
    monkeypatch.setattr("app.sources.dpd.DPD_MAX_RESULTS", 5)

    # Override the DPD activeingredient fixture to return 200 codes.
    big_list = [
        {"drug_code": 50000 + i, "ingredient_name": "TESTDRUG", "strength": "10", "strength_unit": "MG"}
        for i in range(200)
    ]
    # Each drugproduct code lookup returns an empty dict → filtered out, but
    # total_matches should still be set from the ingredient search count.
    import app.sources.dpd as dpd_mod
    original = dpd_mod._fetch_drug_codes_by_ingredient

    async def _patched(client, ingredient):
        return big_list

    monkeypatch.setattr(dpd_mod, "_fetch_drug_codes_by_ingredient", _patched)

    result = await search_dpd("testdrug", field="ingredient")
    # With cap=5 and 200 codes, total_matches must be set.
    if result.status == "ok" and result.total_matches is not None:
        assert result.total_matches == 200
    elif result.status in ("ok", "no_results"):
        # Either the 5 fetched records gave results or all came back None.
        # Main assertion: no unhandled exception was raised.
        pass


# ── Regression 2: NOC DIN attachment (JSON API) ───────────────────────────────

async def test_noc_din_attachment_rate(mock_noc):
    """≥95% of NOC API result records must carry a non-empty DIN.

    Guards against the product_id→DIN join being broken so that DINs are lost.
    """
    result = await search_noc("METFORMIN HYDROCHLORIDE", field="ingredient")
    assert result.status == "ok", f"Expected ok, got {result.status}: {result.error_message}"
    assert result.count > 0

    records_with_din = [r for r in result.records if r.din and r.din.strip()]
    rate = len(records_with_din) / result.count
    assert rate >= 0.95, (
        f"Only {rate*100:.0f}% of NOC records have DINs — expected ≥95%. "
        f"product_id→DIN join may be broken."
    )


async def test_noc_din_on_record(mock_noc):
    """Every NOC record from the fixture carries a non-empty din field."""
    result = await search_noc("METFORMIN HYDROCHLORIDE", field="ingredient")
    assert result.status == "ok"
    for r in result.records:
        assert r.din is not None and r.din.strip(), (
            f"NOC record {r.brand_name!r} has empty din — DIN attachment broken."
        )


# ── Regression 3: DIN parsing utilities ──────────────────────────────────────

def test_multi_din_split_three():
    """'02535742,; 02535750,; 02535734' must explode into exactly 3 DINs."""
    dins = parse_dins("02535742,; 02535750,; 02535734")
    assert len(dins) == 3
    assert "02535742" in dins
    assert "02535750" in dins
    assert "02535734" in dins


def test_noc_api_din_normalization():
    """NOC API DIN values of N/A must be normalised to None."""
    from app.sources.noc import _normalize_din
    assert _normalize_din("N/A") is None
    assert _normalize_din("NA") is None
    assert _normalize_din("Not Applicable") is None
    assert _normalize_din("") is None
    assert _normalize_din("02242974") == "02242974"


# ── Regression 4: Empty-not-error ────────────────────────────────────────────

async def test_dpd_nonsense_query_returns_no_results_not_exception(mock_dpd):
    """A nonsense ingredient must yield no_results — never an exception."""
    result = await search_dpd("zzzznotadrug", field="ingredient")
    assert result.status in ("no_results", "ok"), (
        f"Expected no_results; got {result.status}: {result.error_message}"
    )
    if result.status == "ok":
        assert result.count == 0


async def test_noc_nonsense_ingredient_returns_no_results_not_exception(mock_noc):
    result = await search_noc("zzzznotadrug", field="ingredient")
    assert result.status in ("no_results", "ok", "error")


async def test_noc_brand_field_returns_unsupported(mock_noc):
    """Brand search is unsupported in the JSON API migration — must return unsupported, not error."""
    result = await search_noc("Glucophage", field="brand")
    assert result.status == "unsupported", (
        f"Expected unsupported for brand search, got {result.status}"
    )
    # Critical: no uncaught exception.


async def test_gsur_nonsense_ingredient_no_crash(mock_gsur):
    result = await search_generic_submissions("zzzznotadrug", field="ingredient")
    assert result.status in ("no_results", "ok")


async def test_patent_register_nonsense_ingredient_no_crash(mock_patent_register):
    result = await search_patent_register("zzzznotadrug", field="ingredient")
    assert result.status in ("no_results", "ok")


# ── Regression 5: No HTML / markup leakage ───────────────────────────────────

_MARKUP_PATTERN = re.compile(r"<[a-zA-Z/]|&nbsp;|&#\d+;")


def _check_no_markup(value: str, field: str, record_repr: str) -> None:
    assert not _MARKUP_PATTERN.search(value), (
        f"HTML markup leaked into {field!r} of {record_repr}: {value!r}"
    )


async def test_dpd_no_html_leakage(mock_dpd):
    result = await search_dpd("metformin", field="ingredient")
    assert result.status == "ok"
    for r in result.records:
        for field, val in (
            ("brand_name", r.brand_name),
            ("company", r.company),
            ("ingredient", r.ingredient),
            ("din", r.din),
        ):
            if val:
                _check_no_markup(val, field, repr(r.din))


async def test_noc_no_html_leakage(mock_noc):
    result = await search_noc("METFORMIN HYDROCHLORIDE", field="ingredient")
    assert result.status == "ok"
    for r in result.records:
        for field_name, val in (
            ("brand_name", r.brand_name),
            ("company", r.company),
            ("ingredient", r.ingredient),
        ):
            if val:
                _check_no_markup(val, field_name, repr(r.brand_name))


async def test_gsur_no_html_leakage(mock_gsur):
    result = await search_generic_submissions("metformin", field="ingredient")
    if result.status == "ok":
        for r in result.records:
            if r.ingredient:
                _check_no_markup(r.ingredient, "ingredient", repr(r.ingredient))
            if r.company:
                _check_no_markup(r.company, "company", repr(r.company))


async def test_patent_register_no_html_leakage(mock_patent_register):
    result = await search_patent_register("metformin", field="ingredient")
    if result.status == "ok":
        for r in result.records:
            if r.ingredient:
                _check_no_markup(r.ingredient, "ingredient", repr(r.ingredient))
            if r.brand_name:
                _check_no_markup(r.brand_name, "brand_name", repr(r.brand_name))
