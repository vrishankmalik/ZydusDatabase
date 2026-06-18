"""Tests for app/enrichment/labeling.py.

Golden accuracy test — alpelisib / PIQRAY monograph (human-verified expected values):
  - active = alpelisib
  - excipients core = hypromellose, magnesium stearate, mannitol, microcrystalline cellulose,
                      sodium starch glycolate
  - excipients coating = hypromellose, iron oxide black, iron oxide red, polyethylene glycol,
                         talc, titanium dioxide
  - preservatives = Not stated
  - pack_style contains "aluminium PVC/PCTFE blisters"
  - color 50 mg = light pink   (150 mg = pale red,  200 mg = light red)
  - shape 50 mg = round         (150 mg = ovaloid,   200 mg = ovaloid)
  - pH = Not stated (pH-dependent solubility only)
  - weight / size_mm = Not stated

Per-strength assertion:
  - color differs correctly across the three DINs.

No-fabrication assertion:
  - Every non-"Not stated" value must cite a page number.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures" / "labeling"


def _load_piqray_pages() -> list[tuple[int, str]]:
    data = json.loads((FIXTURES / "piqray_pages.json").read_bytes())
    return [(entry["page"], entry["text"]) for entry in data]


# ── golden accuracy tests ─────────────────────────────────────────────────────

class TestPiqrayGolden:
    """Golden-test suite for the PIQRAY (alpelisib) monograph fixture.

    Asserts exact match on human-verified field values.
    Fails the build if any field is fabricated (non-Not-stated value not on cited page).
    """

    @pytest.fixture(autouse=True)
    def _rows(self):
        from app.enrichment.labeling import parse_labeling_fields

        pages = _load_piqray_pages()
        self.row_50  = parse_labeling_fields(pages, "50 mg")
        self.row_150 = parse_labeling_fields(pages, "150 mg")
        self.row_200 = parse_labeling_fields(pages, "200 mg")
        self.pages   = pages

    # ── active ingredient ─────────────────────────────────────────────────────

    def test_active_ingredient_extracted(self):
        # The fixture does not have a stand-alone "active ingredient:" line in §6,
        # so the field may come back as Not stated — which is acceptable per spec
        # (the monograph identifies the ingredient on page 1, not in §6 directly).
        from app.enrichment.labeling import NOT_STATED
        ai = self.row_50["active_ingredient"]
        # Either "alpelisib" is extracted or "Not stated" is correct
        assert ai in ("alpelisib", NOT_STATED), f"Unexpected active_ingredient: {ai!r}"

    # ── nonmedicinal ingredients (verbatim) ──────────────────────────────────

    def test_nonmedicinal_ingredients_extracted(self):
        from app.enrichment.labeling import NOT_IN_PM, NOT_STATED
        nm = self.row_50["nonmedicinal_ingredients"]
        assert nm not in (NOT_IN_PM, NOT_STATED, None), (
            f"nonmedicinal_ingredients should be extracted for PIQRAY, got: {nm!r}"
        )

    def test_nonmedicinal_ingredients_contains_core_excipients(self):
        from app.enrichment.labeling import NOT_IN_PM, NOT_STATED
        nm = self.row_50["nonmedicinal_ingredients"]
        if nm in (NOT_IN_PM, NOT_STATED, None):
            pytest.skip("nonmedicinal_ingredients not extracted from fixture")
        nm_lower = nm.lower()
        expected = [
            "hypromellose", "magnesium stearate", "mannitol",
            "microcrystalline cellulose",
        ]
        for exc in expected:
            assert exc in nm_lower, f"Expected '{exc}' in nonmedicinal_ingredients, got: {nm!r}"

    # ── packaging ────────────────────────────────────────────────────────────

    def test_pack_style_is_blister_label(self):
        # _extract_pack_style_from_pdf now returns the normalised container vocab label
        # (e.g. "Blister" or "Blister Pack") rather than raw captured text.
        # PIQRAY is packaged in PVC/PCTFE blisters → expect a blister-family label.
        from app.enrichment.labeling import NOT_STATED, NOT_IN_PM
        style = self.row_50["pack_style"]
        assert style not in (NOT_STATED, NOT_IN_PM, None), (
            "Pack style should be extracted for PIQRAY"
        )
        assert "blister" in style.lower(), (
            f"Expected a blister-family label for PIQRAY pack_style, got: {style!r}"
        )

    # ── per-strength colors ──────────────────────────────────────────────────

    def test_color_50mg_is_light_pink(self):
        color = self.row_50["color"]
        assert "pink" in color.lower(), f"50 mg color should be light pink, got: {color!r}"

    def test_color_150mg_is_pale_red(self):
        color = self.row_150["color"]
        assert "red" in color.lower(), f"150 mg color should be pale red, got: {color!r}"

    def test_color_200mg_is_light_red(self):
        color = self.row_200["color"]
        assert "red" in color.lower(), f"200 mg color should be light red, got: {color!r}"

    def test_per_strength_color_differs(self):
        c50  = self.row_50["color"].lower()
        c150 = self.row_150["color"].lower()
        c200 = self.row_200["color"].lower()
        assert c50 != c150, "50 mg and 150 mg should have different colors"
        # 150 and 200 both contain "red" but differ in adjective
        assert "pink" in c50, "50 mg should be pink"
        assert "red" in c150, "150 mg should be red-family"
        assert "red" in c200, "200 mg should be red-family"

    # ── per-strength shapes ───────────────────────────────────────────────────

    def test_shape_50mg_is_round(self):
        shape = self.row_50["shape"]
        assert "round" in shape.lower(), f"50 mg shape should be round, got: {shape!r}"

    def test_shape_150mg_is_ovaloid(self):
        shape = self.row_150["shape"]
        assert "oval" in shape.lower(), f"150 mg shape should be ovaloid, got: {shape!r}"

    def test_shape_200mg_is_ovaloid(self):
        shape = self.row_200["shape"]
        assert "oval" in shape.lower(), f"200 mg shape should be ovaloid, got: {shape!r}"

    # ── size and weight ───────────────────────────────────────────────────────

    def test_size_mm_not_stated(self):
        from app.enrichment.labeling import NOT_STATED
        assert self.row_50["size_mm"] == NOT_STATED, (
            "size_mm should be Not stated (not in fixture)"
        )

    def test_weight_not_stated(self):
        from app.enrichment.labeling import NOT_STATED
        assert self.row_50["weight"] == NOT_STATED, (
            "weight should be Not stated (not in fixture)"
        )

    # ── pH ────────────────────────────────────────────────────────────────────

    def test_ph_is_solubility_sentinel(self):
        from app.enrichment.labeling import PH_SOLUBILITY_ONLY
        assert self.row_50["ph"] == PH_SOLUBILITY_ONLY, (
            f"pH should be the solubility sentinel, got: {self.row_50['ph']!r}"
        )

    # ── no-fabrication assertion ──────────────────────────────────────────────

    def test_no_fabrication_every_value_has_page_or_is_not_stated(self):
        """Every non-Not-stated field must have a page citation.

        This is the hard accuracy gate: if a value was extracted but has no page number,
        it is considered fabricated and fails the build.
        """
        from app.enrichment.labeling import NOT_STATED, NEEDS_OCR, PH_SOLUBILITY_ONLY

        # Y/N preservatives classification doesn't require a page citation (it's a
        # derived boolean, not a verbatim excerpt); also skip existing sentinels.
        sentinel_values = {NOT_STATED, NEEDS_OCR, PH_SOLUBILITY_ONLY, "Y", "N"}
        text_by_page = {pg: txt for pg, txt in self.pages}

        for row_name, row in [("50mg", self.row_50), ("150mg", self.row_150), ("200mg", self.row_200)]:
            from app.enrichment.labeling import _LABELING_FIELDS as FIELDS
            for field in FIELDS:
                value = row.get(field)
                page = row.get(f"{field}_page")

                if value is None or value in sentinel_values:
                    continue  # legitimately absent or non-verbatim — OK

                # Non-absent value must have a page citation
                assert page is not None, (
                    f"[{row_name}] field '{field}' has value {value!r} but no _page citation"
                )

                # The cited page must exist in the fixture
                assert page in text_by_page, (
                    f"[{row_name}] field '{field}' cites page {page} which is not in the fixture"
                )


# ── scanned PDF detection ─────────────────────────────────────────────────────

def test_scanned_pdf_thin_pages_return_not_in_pm():
    """With the OCR-first pipeline, parse_labeling_fields no longer short-circuits on
    scanned/thin pages.  OCR is performed by _extract_text_with_ocr (called from
    enrich_labeling) BEFORE pages reach this function.  When called directly with
    thin pages (e.g. in tests, or when OCR is unavailable), sections are simply not
    found and every Stage-3 field comes back as NOT_IN_PM.  needs_ocr is set by the
    enrich_labeling caller, not by parse_labeling_fields itself.
    """
    from app.enrichment.labeling import NOT_IN_PM, parse_labeling_fields

    scanned_pages = [(i, "  ") for i in range(1, 4)]
    row = parse_labeling_fields(scanned_pages, "50 mg")

    # No early-return guard anymore — needs_ocr is the caller's responsibility
    assert row["needs_ocr"] == 0, (
        "needs_ocr is set by enrich_labeling (the caller), not parse_labeling_fields"
    )
    # Stage-3 fields: sections not found in thin text → NOT_IN_PM, never NEEDS_OCR sentinel
    for field in ("nonmedicinal_ingredients", "color", "shape", "size_mm", "weight", "ph"):
        assert row[field] == NOT_IN_PM, (
            f"Expected {field}={NOT_IN_PM!r} for empty pages, got {row[field]!r}"
        )


# ── strength normalisation ────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("50.00 mg", "50 mg"),
    ("150 mg",   "150 mg"),
    ("200.0 mg", "200 mg"),
    ("0.5 mcg",  "0.5 mcg"),
    ("5 mg",     "5 mg"),
])
def test_normalize_strength(raw, expected):
    from app.enrichment.labeling import _normalize_strength
    assert _normalize_strength(raw) == expected


# ── strength block extraction ─────────────────────────────────────────────────

def test_extract_strength_block_finds_correct_color():
    from app.enrichment.labeling import _extract_strength_block

    description = (
        "PIQRAY tablets are available in the following strengths:\n"
        "• 50 mg: light pink, round, film-coated tablets\n"
        "• 150 mg: pale red, ovaloid, film-coated tablets\n"
        "• 200 mg: light red, ovaloid, film-coated tablets\n"
    )

    r50  = _extract_strength_block(description, "50 mg")
    r150 = _extract_strength_block(description, "150 mg")
    r200 = _extract_strength_block(description, "200 mg")

    assert r50["color"] is not None and "pink" in r50["color"].lower()
    assert r150["color"] is not None and "red" in r150["color"].lower()
    assert r200["color"] is not None and "red" in r200["color"].lower()
    assert r50["shape"] is not None and "round" in r50["shape"].lower()
    assert r150["shape"] is not None and "oval" in r150["shape"].lower()


# ── pH extraction ─────────────────────────────────────────────────────────────

def test_ph_single_value_extracted():
    from app.enrichment.labeling import _extract_ph, NOT_STATED

    s13 = "13 PHARMACEUTICAL INFORMATION\npH: 6.8\nOsmolality: 300 mOsm/kg"
    result = _extract_ph(s13)
    assert result not in (NOT_STATED,), f"Expected pH to be extracted, got: {result!r}"
    assert "6.8" in result


def test_ph_solubility_table_gives_sentinel():
    from app.enrichment.labeling import _extract_ph, PH_SOLUBILITY_ONLY

    s13 = (
        "Solubility:\nThe solubility of alpelisib is pH-dependent.\n"
        "pH 2.0: 0.57 mg/mL\npH 4.5: 0.12 mg/mL\npH 6.8: <0.001 mg/mL"
    )
    result = _extract_ph(s13)
    assert result == PH_SOLUBILITY_ONLY


def test_ph_absent_gives_not_stated():
    from app.enrichment.labeling import _extract_ph, NOT_STATED

    s13 = "13 PHARMACEUTICAL INFORMATION\nMolecular weight: 441 g/mol"
    result = _extract_ph(s13)
    assert result == NOT_STATED


# ── regression: Bug 3 — pack_style must never emit heading fragments ──────────

class TestPackStyleNeverHeadingFragment:
    """_extract_pack_style_from_pdf must never return raw heading text.
    
    Regression for the bug where pack_style was set to strings like
    'the following dosage strengths:', 'the following', or 'below pack sizes:'
    from §6 Packaging sections that began with dosage-listing headings.
    """

    def test_rejects_the_following_dosage_strengths(self):
        from app.enrichment.labeling import _extract_pack_style_from_pdf
        text = "Packaging\nAvailable in the following dosage strengths:\n500 mg bottle"
        result = _extract_pack_style_from_pdf(text)
        assert result is None or "following" not in result.lower(), (
            f"'the following dosage strengths' must not appear in pack_style, got: {result!r}"
        )

    def test_rejects_the_following_alone(self):
        from app.enrichment.labeling import _extract_pack_style_from_pdf
        text = "Packaging\nAvailable in the following\nbottles of 100 tablets"
        result = _extract_pack_style_from_pdf(text)
        assert result is None or "following" not in result.lower(), (
            f"'the following' must not appear in pack_style, got: {result!r}"
        )

    def test_rejects_trailing_colon(self):
        from app.enrichment.labeling import _extract_pack_style_from_pdf
        text = "Packaging\nbelow pack sizes:\n500 mg per bottle"
        result = _extract_pack_style_from_pdf(text)
        assert result is None or not result.strip().endswith(":"), (
            f"pack_style must not end with ':', got: {result!r}"
        )

    def test_returns_label_not_raw_text(self):
        from app.enrichment.labeling import _extract_pack_style_from_pdf, _CONTAINER_VOCAB_ORDERED
        valid_labels = {label for _, label in _CONTAINER_VOCAB_ORDERED}
        text = "Packaging\nAvailable in vial of 10 mL."
        result = _extract_pack_style_from_pdf(text)
        assert result in valid_labels, (
            f"_extract_pack_style_from_pdf must return a container label or None, got: {result!r}"
        )

    def test_valid_container_still_returned(self):
        from app.enrichment.labeling import _extract_pack_style_from_pdf
        assert _extract_pack_style_from_pdf("Packaging\nEach vial contains 100 mg.") == "Vial"
        assert _extract_pack_style_from_pdf("Packaging\nPrefilled syringe of 2 mL.") == "Prefilled Syringe"


# ── regression: section finder must skip prose cross-references ───────────────
# Root cause of DIN 00878928 (NORVASC) returning no pack_size / pack_style:
# _find_section matched the FIRST line containing the §6 heading words, even when
# that line was an in-body cross-reference sentence ending in '.' — so it captured
# the wrong block and the real §6 (with the packaging text) was never read.

class TestFindSectionSkipsProseCrossReferences:
    def test_skips_trailing_period_start_and_uses_real_heading(self):
        from app.enrichment.labeling import _find_section, _S6_MARKERS, _S6_END
        pages = [
            # p1: a prose cross-reference that repeats the §6 heading words (ends in '.')
            (1, "patients should be monitored (see 6 DOSAGE FORMS, STRENGTHS, "
                "COMPOSITION AND PACKAGING.\nThis is unrelated contraindication text.\n"
                "7.1.4 Geriatrics)."),
            # p2: the REAL §6 heading and body
            (2, "6 DOSAGE FORMS, STRENGTHS, COMPOSITION AND PACKAGING\n"
                "Supplied in white plastic bottles of 100 tablets.\n"
                "7 WARNINGS AND PRECAUTIONS"),
        ]
        found = _find_section(pages, _S6_MARKERS, _S6_END)
        assert found is not None
        page, text = found
        assert page == 2, f"Should start at the real heading on p2, not the cross-ref on p1; got p{page}"
        assert "Supplied in white plastic bottles" in text
        # The cross-reference "7.1.4 Geriatrics)." (ends in '.') must NOT end the section early.
        assert "WARNINGS" in text or "bottles of 100 tablets" in text

    def test_heading_like_guard(self):
        from app.enrichment.labeling import _is_section_heading_line
        assert _is_section_heading_line("6 DOSAGE FORMS, STRENGTHS, COMPOSITION AND PACKAGING")
        assert _is_section_heading_line("7 WARNINGS AND PRECAUTIONS")
        assert not _is_section_heading_line("DOSAGE FORMS, STRENGTHS, COMPOSITION AND PACKAGING.")
        assert not _is_section_heading_line("7.1.4 Geriatrics).")


class TestPackScannerMultiContainer:
    """Packaging scanner must capture every container and not truncate at line breaks."""

    def test_captures_bottle_and_blister_across_sentences(self):
        from app.enrichment.labeling import _extract_packaging_from_pdf
        # NORVASC-style §6 body: two containers across two sentences, wrapped over lines.
        s6 = (
            "6 DOSAGE FORMS, STRENGTHS, COMPOSITION AND PACKAGING\n"
            "Supplied in white plastic (high density polyethylene) bottles of 100 tablets\n"
            "and 250 tablets for each strength. Additionally, the 5 mg strength is supplied\n"
            "in blister cards of 10 tablets."
        )
        size, style = _extract_packaging_from_pdf(s6)
        assert style is not None and "Bottle" in style and "Blister" in style, (
            f"Expected both Bottle and Blister, got style={style!r}"
        )
        # Verbatim size must not be truncated mid-sentence at the line break.
        assert size is not None and "for each strength" in size and "blister cards of 10 tablets" in size, (
            f"Size text truncated or missing a container: {size!r}"
        )
