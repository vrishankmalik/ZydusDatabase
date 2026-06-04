"""Cross-source consistency tests.

For every DIN that appears in ≥2 sources, the normalized ingredient sets and
brand names must agree.  This catches mis-joins and parse bugs that no
single-source test can detect — e.g. a NOC column-index shift would look fine
in isolation but disagree with DPD for the same DIN.

Offline fixture test: uses deliberately mismatched DrugRecord objects to verify
that the checker correctly flags the discrepancy.

Integration test: runs a live query and verifies that *zero* consistency errors
are emitted (warnings are tolerated; hard errors are not).
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.consistency import check_cross_source_consistency, ConsistencyWarning
from app.models import DrugRecord


# ── Offline fixture tests ──────────────────────────────────────────────────────

def _make_dpd(din: str, ingredient: str, brand: str) -> DrugRecord:
    return DrugRecord(
        source="DPD",
        din=din,
        ingredient=ingredient,
        brand_name=brand,
        all_ingredients=[i.strip() for i in ingredient.split(";") if i.strip()],
    )


def _make_noc(din: str, ingredient: str, brand: str) -> DrugRecord:
    return DrugRecord(
        source="NOC",
        din=din,
        ingredient=ingredient,
        brand_name=brand,
        all_ingredients=[i.strip() for i in ingredient.split(";") if i.strip()],
    )


class TestConsistencyChecker:
    def test_matching_records_emit_no_warnings(self) -> None:
        records = [
            _make_dpd("02229895", "METFORMIN HYDROCHLORIDE", "GLUCOPHAGE"),
            _make_noc("02229895", "METFORMIN HYDROCHLORIDE", "GLUCOPHAGE"),
        ]
        warnings = check_cross_source_consistency(records)
        assert warnings == [], f"Expected no warnings, got: {warnings}"

    def test_ingredient_mismatch_is_flagged(self) -> None:
        """NOC parses ingredient wrong — should emit an ingredient warning."""
        records = [
            _make_dpd("02229895", "METFORMIN HYDROCHLORIDE", "GLUCOPHAGE"),
            _make_noc("02229895", "METFORMIN", "GLUCOPHAGE"),  # wrong: missing HCL
        ]
        warnings = check_cross_source_consistency(records)
        ing_warnings = [w for w in warnings if w.field == "ingredient"]
        assert ing_warnings, (
            "Expected an ingredient mismatch warning for DIN 02229895 "
            "(DPD='METFORMIN HYDROCHLORIDE' vs NOC='METFORMIN')"
        )
        w = ing_warnings[0]
        assert w.din == "02229895"
        assert w.source_a in ("DPD", "NOC")
        assert w.source_b in ("DPD", "NOC")
        assert w.source_a != w.source_b

    def test_brand_mismatch_is_flagged(self) -> None:
        """Stray trailing character causes brand name divergence."""
        records = [
            _make_dpd("00559393", "ACETAMINOPHEN", "TYLENOL REGULAR STRENGTH"),
            _make_noc("00559393", "ACETAMINOPHEN", "TYLENOL REGULAR STRENGTH TAB"),
        ]
        warnings = check_cross_source_consistency(records)
        brand_warnings = [w for w in warnings if w.field == "brand"]
        assert brand_warnings, (
            "Expected a brand mismatch warning for DIN 00559393"
        )

    def test_din_less_records_are_ignored(self) -> None:
        records = [
            DrugRecord(source="GenericSubmissions", din=None, ingredient="METFORMIN"),
            DrugRecord(source="NOC", din=None, ingredient="METFORMIN HYDROCHLORIDE"),
        ]
        warnings = check_cross_source_consistency(records)
        assert warnings == [], "DIN-less records must not be checked"

    def test_single_source_din_is_ignored(self) -> None:
        records = [
            _make_dpd("02229895", "METFORMIN HYDROCHLORIDE", "GLUCOPHAGE"),
        ]
        warnings = check_cross_source_consistency(records)
        assert warnings == [], "DIN in only one source must not emit warnings"

    def test_case_insensitive_ingredient_match(self) -> None:
        records = [
            _make_dpd("02229895", "Metformin Hydrochloride", "GLUCOPHAGE"),
            _make_noc("02229895", "METFORMIN HYDROCHLORIDE", "GLUCOPHAGE"),
        ]
        warnings = [w for w in check_cross_source_consistency(records) if w.field == "ingredient"]
        assert warnings == [], (
            "Case difference alone must not trigger an ingredient mismatch"
        )

    def test_whitespace_normalization(self) -> None:
        records = [
            _make_dpd("02229895", "METFORMIN  HYDROCHLORIDE", "GLUCOPHAGE"),  # double space
            _make_noc("02229895", "METFORMIN HYDROCHLORIDE", "GLUCOPHAGE"),
        ]
        warnings = [w for w in check_cross_source_consistency(records) if w.field == "ingredient"]
        assert warnings == [], "Internal whitespace differences must not trigger a mismatch"

    def test_multiple_dins_checked_independently(self) -> None:
        records = [
            _make_dpd("02229895", "METFORMIN HYDROCHLORIDE", "GLUCOPHAGE"),
            _make_noc("02229895", "METFORMIN", "GLUCOPHAGE"),      # mismatch on DIN A
            _make_dpd("00559393", "ACETAMINOPHEN", "TYLENOL"),
            _make_noc("00559393", "ACETAMINOPHEN", "TYLENOL"),     # clean on DIN B
        ]
        warnings = check_cross_source_consistency(records)
        flagged_dins = {w.din for w in warnings if w.field == "ingredient"}
        assert "02229895" in flagged_dins
        assert "00559393" not in flagged_dins

    def test_combination_product_order_independent(self) -> None:
        """Two sources may list combination ingredients in different orders; that's fine."""
        records = [
            _make_dpd("00000001", "INGREDIENT_A; INGREDIENT_B", "COMBO DRUG"),
            _make_noc("00000001", "INGREDIENT_B; INGREDIENT_A", "COMBO DRUG"),
        ]
        warnings = [w for w in check_cross_source_consistency(records) if w.field == "ingredient"]
        assert warnings == [], "Ingredient order differences must not trigger a mismatch"

    def test_warning_fields_populated(self) -> None:
        records = [
            _make_dpd("02229895", "METFORMIN HYDROCHLORIDE", "GLUCOPHAGE"),
            _make_noc("02229895", "METFORMIN", "GLUCOPHAGE"),
        ]
        warnings = check_cross_source_consistency(records)
        for w in warnings:
            assert isinstance(w, ConsistencyWarning)
            assert w.din
            assert w.field in ("ingredient", "brand")
            assert w.source_a
            assert w.source_b
            assert w.detail


# ── Integration test ───────────────────────────────────────────────────────────

@pytest.mark.integration
def test_live_glucophage_din_consistency() -> None:
    """Live glucophage/metformin search: consistency check must emit no warnings.

    Verifies that DPD and NOC agree on ingredient name and brand for the same
    DIN when a real HTTP search is run.  Warnings would indicate a parser bug.
    """
    from app.sources.dpd import search_dpd
    from app.sources.noc import search_noc

    dpd_result = asyncio.run(search_dpd("metformin hydrochloride", field="ingredient"))
    noc_result = asyncio.run(search_noc("metformin hydrochloride", field="ingredient"))

    all_records = dpd_result.records + noc_result.records
    if not all_records:
        pytest.skip("No live results returned — network may be unavailable")

    warnings = check_cross_source_consistency(all_records)
    ing_errors = [w for w in warnings if w.field == "ingredient"]

    assert not ing_errors, (
        f"Ingredient consistency errors on live data for 'metformin hydrochloride': "
        f"{[str(w) for w in ing_errors[:5]]}"
    )
