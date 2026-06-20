"""Tests for DPD API source — live network (integration-marked).

All tests in this file hit the real DPD REST API at health-products.canada.ca.
Run with: make test-live
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from app.sources.dpd import search_dpd


@pytest.mark.integration
async def test_metformin_ingredient_search():
    """Metformin HCl is in DPD with many products; at minimum 1 result expected."""
    result = await search_dpd("metformin", field="ingredient")
    assert result.status == "ok", f"Expected ok, got {result.status}: {result.error_message}"
    assert result.count > 0
    for r in result.records:
        assert r.source == "DPD"
    found = any("METFORMIN" in (r.ingredient or "").upper() for r in result.records)
    assert found, "No record has METFORMIN in the ingredient field"


@pytest.mark.integration
async def test_dpd_record_has_record_url():
    """Each DPD record must have a non-empty record_url (provenance)."""
    result = await search_dpd("metformin", field="ingredient")
    assert result.status == "ok"
    for r in result.records[:5]:
        assert r.record_url and r.record_url.startswith("http"), \
            f"Missing or invalid record_url: {r.record_url}"


@pytest.mark.integration
async def test_dpd_brand_search():
    """Brand name search should return results for 'Glucophage'."""
    result = await search_dpd("Glucophage", field="brand")
    assert result.status in ("ok", "no_results")


@pytest.mark.integration
async def test_dpd_unknown_ingredient():
    """An unknown ingredient should return no_results, not an error."""
    result = await search_dpd("xyznonexistentdrugabc123", field="ingredient")
    assert result.status == "no_results"
    assert result.count == 0


@pytest.mark.integration
async def test_dpd_din_search():
    """DIN search: 02229895 is metformin (Glucophage)."""
    result = await search_dpd("02229895", field="din")
    assert result.status in ("ok", "no_results")


@pytest.mark.integration
async def test_dpd_dosage_form_golden():
    """dosage_form is captured per DIN from the already-fetched /form/ endpoint.

    Golden values were observed from an actual DPD run on 2026-06-18; they are
    pinned here, never hand-entered.  Every anchor resolves to a non-blank form.

        02567709 ORTHOXICAM     → Solution
        02334992 ENDOMETRIN     → Vaginal Tablet, Effervescent
        02505223 BIJUVA         → Capsule
        02515504 INPROSUB       → Solution
        00030848 DEPO-PROVERA   → Suspension
    """
    golden = {
        "02567709": "Solution",
        "02334992": "Vaginal Tablet, Effervescent",
        "02505223": "Capsule",
        "02515504": "Solution",
        "00030848": "Suspension",
    }
    for din, expected_form in golden.items():
        result = await search_dpd(din, field="din")
        assert result.status == "ok", (
            f"{din}: expected ok, got {result.status}: {result.error_message}"
        )
        assert result.records, f"{din}: no records returned"
        form = result.records[0].dosage_form
        assert form and form.strip(), f"{din}: dosage_form must be non-blank, got {form!r}"
        assert form == expected_form, (
            f"{din}: dosage_form changed — expected {expected_form!r}, got {form!r}"
        )


def test_sync_wrapper():
    """Smoke test that the async function is importable and callable."""
    from app.sources.dpd import search_dpd as fn
    assert callable(fn)
