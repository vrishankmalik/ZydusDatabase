"""Tests for IQVIA parse / collapse / match pipeline.

Verification anchors (computed from the real IQVIA_SAMPLE_progesterone.xlsx):
  DIN 02516187 (SANIS / PROGESTERONE / 100MG):
    Units MAT 12/2025 = 218591, Dollars MAT 12/2025 = 21215081
  DIN 02493578 (AURO / AURO-PROGESTERONE / 100MG):
    Units MAT 12/2025 = 233159, Dollars MAT 12/2025 = 13005865
  DIN 00585092 (PFIZER / DEPO-PROVERA / 150MG -> 150MG/ML in IQVIA):
    Units MAT 12/2025 = 262834, Dollars MAT 12/2025 = 8853659

Ambiguous case:
  IQVIA "PROVERA / PFIZER / 5MG" → two candidate DINs (00030937 PROVERA 5MG
  and 02010739 PROVERA PAK 5MG) → neither DIN should receive data.

Unmatched IQVIA groups:
  "PROGESTERONE / CYTEX / 50MG/ML" and "PROGESTERONE / HIKMA / 50MG/ML"
  have no DIN in the sample Sheet 1 → status = no_din_match.
"""
import io
import os
from pathlib import Path

import pytest
import pandas as pd

from app.enrichment.iqvia import (
    parse_iqvia,
    collapse_iqvia,
    detect_metric_columns,
    match_iqvia_to_sheet1,
    _norm_strength,
    _norm_company,
    _norm_brand,
)

_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "iqvia"


def _resolve_sample(filename: str, env_var: str) -> str:
    """Locate an IQVIA sample workbook, returning "" if none is found.

    These workbooks contain real IQVIA data and are NOT checked into the repo.
    Search order: explicit env var → tests/fixtures/iqvia/<filename> → the legacy
    ~/Downloads location.  Tests that depend on the data skip when it is absent
    (rather than erroring), so the offline suite stays green without the file.
    """
    candidates = []
    env_val = os.getenv(env_var)
    if env_val:
        candidates.append(env_val)
    candidates.append(str(_FIXTURE_DIR / filename))
    candidates.append(str(Path.home() / "Downloads" / filename))
    for c in candidates:
        if os.path.isfile(c):
            return c
    return ""


SAMPLE_PATH = _resolve_sample("IQVIA_SAMPLE_progesterone.xlsx", "IQVIA_SAMPLE_PATH")
COMBINATIONS_PATH = _resolve_sample("IQVIA_SAMPLE_combinations.xlsx", "IQVIA_COMBINATIONS_PATH")

_MISSING_SAMPLE_MSG = (
    "IQVIA sample workbook not found — set IQVIA_SAMPLE_PATH / "
    "IQVIA_COMBINATIONS_PATH, or drop the .xlsx into tests/fixtures/iqvia/"
)


@pytest.fixture(scope="module")
def iqvia_raw():
    if not SAMPLE_PATH:
        pytest.skip(_MISSING_SAMPLE_MSG)
    with open(SAMPLE_PATH, "rb") as fh:
        raw = parse_iqvia(fh.read())
    return raw


@pytest.fixture(scope="module")
def iqvia_collapsed(iqvia_raw):
    return collapse_iqvia(iqvia_raw)


# ── Unit tests: normalisation helpers ────────────────────────────────────────

class TestNormStrength:
    def test_simple(self):
        assert _norm_strength("100 MG") == frozenset({"100MG"})

    def test_strips_space_in_percent(self):
        assert _norm_strength("8 %") == frozenset({"8%"})

    def test_dpd_semicolon_combo(self):
        assert _norm_strength("1 MG; 100 MG") == frozenset({"1MG", "100MG"})

    def test_iqvia_slash_combo(self):
        assert _norm_strength("100MG/1MG") == frozenset({"100MG", "1MG"})

    def test_combo_order_irrelevant(self):
        assert _norm_strength("1MG/100MG") == _norm_strength("100MG/1MG")

    def test_concentration_drops_ml(self):
        # DEPO-PROVERA: IQVIA "150MG/ML" should match DPD "150 MG"
        assert _norm_strength("150MG/ML") == frozenset({"150MG"})
        assert _norm_strength("150MG/ML") == _norm_strength("150 MG")

    def test_concentration_mg_per_g_converts_to_percent(self):
        # 50 MG/G = 5 % w/w; code converts MG/G to % before dropping the denominator.
        assert _norm_strength("50MG/G") == frozenset({"5%"})

    def test_empty(self):
        assert _norm_strength("") == frozenset()
        assert _norm_strength(None) == frozenset()


class TestNormCompany:
    def test_strips_ulc(self):
        assert _norm_company("PFIZER CANADA ULC") == "pfizer"

    def test_strips_inc(self):
        # "pharma" is also stripped, leaving just "auro"
        assert _norm_company("AURO PHARMA INC") == "auro"

    def test_strips_ltd(self):
        assert _norm_company("TEVA CANADA LTD") == "teva"

    def test_pfizer_bare(self):
        assert _norm_company("PFIZER") == "pfizer"

    def test_knight(self):
        # "THERAPEUTICS" is NOT in the strip list — that's fine, sim still works.
        norm = _norm_company("KNIGHT THERAPEUTICS INC.")
        assert "knight" in norm

    def test_same_after_strip(self):
        assert _norm_company("PFIZER CANADA ULC") == _norm_company("PFIZER")


class TestNormBrand:
    def test_strips_trailing_strength(self):
        assert _norm_brand("PROVERA 5MG TABLETS") == "provera"

    def test_leaves_pak(self):
        # After stripping "5MG", what's left?
        # "PROVERA PAK 5MG" → trailing "5MG" stripped → "PROVERA PAK"
        assert _norm_brand("PROVERA PAK 5MG") == "provera pak"

    def test_lowercase(self):
        assert _norm_brand("DEPO-PROVERA") == "depo-provera"

    def test_strips_bare_tablets(self):
        # DPD brand "APO-ABACAVIR-LAMIVUDINE TABLETS" — no digit before "TABLETS"
        # so the old digit-requiring pattern left "tablets" in place.
        assert _norm_brand("APO-ABACAVIR-LAMIVUDINE TABLETS") == "apo-abacavir-lamivudine"

    def test_strips_bare_capsules(self):
        assert _norm_brand("JAMP-SOMETHINGCAPS CAPSULES") == "jamp-somethingcaps"

    def test_does_not_strip_mid_word(self):
        # "tablets" mid-name must not be removed
        assert _norm_brand("TABLET-X DRUG") == "tablet-x drug"


class TestApprovedDinExclusion:
    """DINs with DPD status 'Approved' (never marketed) must not appear as IQVIA
    candidates. A never-launched DIN has no sales history and including it creates
    false near-ties against the correctly marketed sibling DIN.

    Regression anchor: DIN 02518287 (APO-ABACAVIR-LAMIVUDINE TABLETS, Approved)
    was scoring 92.6 against the IQVIA group for APO-ABACAVIR-LAMIVUDINE, creating
    a 7.4-point gap vs. the correct DIN 02399539 (Marketed, score 100). The
    TIE_MARGIN of 15 flagged this as ambiguous, so DIN 02399539 received no IQVIA
    data despite being the only marketed match.
    """

    def _abacavir_sheet1(self):
        return pd.DataFrame([
            # DIN 02399539 — APO-ABACAVIR-LAMIVUDINE, APOTEX, Marketed 2016-03-15
            {
                "din": "02399539",
                "ingredient": "ABACAVIR SULFATE; LAMIVUDINE",
                "brand_name": "APO-ABACAVIR-LAMIVUDINE",
                "company": "APOTEX INC",
                "strength": "600 MG; 300 MG",
                "status": "Marketed",
            },
            # DIN 02518287 — APO-ABACAVIR-LAMIVUDINE TABLETS, APOTEX, Approved (never marketed)
            {
                "din": "02518287",
                "ingredient": "ABACAVIR SULFATE; LAMIVUDINE",
                "brand_name": "APO-ABACAVIR-LAMIVUDINE TABLETS",
                "company": "APOTEX INC",
                "strength": "600 MG; 300 MG",
                "status": "Approved",
            },
            # DIN 02454513 — AURO-ABACAVIR/LAMIVUDINE, AURO PHARMA, Marketed
            {
                "din": "02454513",
                "ingredient": "ABACAVIR SULFATE; LAMIVUDINE",
                "brand_name": "AURO-ABACAVIR/LAMIVUDINE",
                "company": "AURO PHARMA INC",
                "strength": "600 MG; 300 MG",
                "status": "Marketed",
            },
        ])

    def _abacavir_iqvia(self):
        """Minimal collapsed IQVIA DataFrame for the APO-ABACAVIR group."""
        return pd.DataFrame([{
            "Combined Molecule": "ABACAVIR/LAMIVUDINE",
            "Product": "APO-ABACAVIR-LAMIVUDINE",
            "Manufacturer": "APOTEX INC",
            "Strength": "0.6GM/300MG",
            "Units MAT 12/2025": 5000,
            "Dollars MAT 12/2025": 249814,
        }])

    def test_marketed_din_gets_iqvia_data(self):
        """DIN 02399539 (Marketed) must receive the $249,814 — not left ambiguous."""
        enriched, _ = match_iqvia_to_sheet1(self._abacavir_sheet1(), self._abacavir_iqvia())
        row = enriched[enriched["din"] == "02399539"]
        assert len(row) == 1
        assert int(row["Dollars MAT 12/2025"].iloc[0]) == 249814

    def test_approved_din_gets_no_iqvia_data(self):
        """DIN 02518287 (Approved, never marketed) must receive no IQVIA data."""
        enriched, _ = match_iqvia_to_sheet1(self._abacavir_sheet1(), self._abacavir_iqvia())
        row = enriched[enriched["din"] == "02518287"]
        assert len(row) == 1
        val = row["Dollars MAT 12/2025"].iloc[0]
        assert val is None or pd.isna(val), (
            f"Approved DIN 02518287 must not receive IQVIA data; got {val!r}"
        )

    def test_approved_din_not_flagged_ambiguous_in_recon(self):
        """The IQVIA group must be matched (not ambiguous) once the Approved DIN is excluded."""
        _, recon = match_iqvia_to_sheet1(self._abacavir_sheet1(), self._abacavir_iqvia())
        abacavir_rows = recon[recon["iqvia_product"] == "APO-ABACAVIR-LAMIVUDINE"]
        assert len(abacavir_rows) == 1
        assert abacavir_rows["status"].iloc[0] == "matched", (
            f"Expected 'matched' but got {abacavir_rows['status'].iloc[0]!r} — "
            "Approved DIN 02518287 must be excluded from candidates"
        )


class TestLifecycleStatusTier:
    """Discontinued DINs (Cancelled Post Market, Dormant, …) must not create
    false near-ties that blank the one currently-marketed seller.

    IQVIA reports a recent Moving Annual Total, so its sales belong to the
    marketed product.  When any candidate for an IQVIA group is marketed, only
    marketed candidates are considered; once-sold-but-discontinued DINs are used
    only as a fallback when no marketed DIN matches.

    Regression anchor (real data): IQVIA 'JAMP-METFORMIN / JAMP PHARMA / 0.5GM'.
    DIN 02380196 (JAMP METFORMIN, Marketed) scored 96.4 but DIN 02380722
    (JAMP-METFORMIN BLACKBERRY, JAMP PHARMA, Cancelled Post Market) scored 85.9 —
    gap 11 < TIE_MARGIN=15 → both wrongly flagged ambiguous, so the real seller
    received no IQVIA data.  After the fix the discontinued sibling is dropped and
    02380196 matches unambiguously.
    """

    def _jamp_sheet1(self):
        return pd.DataFrame([
            # The real, currently-marketed JAMP METFORMIN 500MG.
            {"din": "02380196", "ingredient": "METFORMIN HYDROCHLORIDE",
             "brand_name": "JAMP METFORMIN", "company": "JAMP PHARMA CORPORATION",
             "strength": "500 MG", "status": "Marketed"},
            # Discontinued flavour variant — near-identical brand, same company.
            {"din": "02380722", "ingredient": "METFORMIN HYDROCHLORIDE",
             "brand_name": "JAMP-METFORMIN BLACKBERRY", "company": "JAMP PHARMA CORPORATION",
             "strength": "500 MG", "status": "Cancelled Post Market"},
            # Dormant sibling from a different company.
            {"din": "02167786", "ingredient": "METFORMIN HYDROCHLORIDE",
             "brand_name": "APO-METFORMIN", "company": "APOTEX INC",
             "strength": "500 MG", "status": "Dormant"},
        ])

    def _jamp_iqvia(self):
        return pd.DataFrame([{
            "Combined Molecule": "METFORMIN",
            "Product": "JAMP-METFORMIN",
            "Manufacturer": "JAMP PHARMA",
            "Strength": "0.5GM",
            "Units MAT 12/2025": 12345,
            "Dollars MAT 12/2025": 678910,
        }])

    def test_marketed_din_gets_data(self):
        enriched, _ = match_iqvia_to_sheet1(self._jamp_sheet1(), self._jamp_iqvia())
        row = enriched[enriched["din"] == "02380196"]
        assert int(row["Dollars MAT 12/2025"].iloc[0]) == 678910

    def test_discontinued_sibling_blank(self):
        enriched, _ = match_iqvia_to_sheet1(self._jamp_sheet1(), self._jamp_iqvia())
        for din in ("02380722", "02167786"):
            val = enriched[enriched["din"] == din]["Dollars MAT 12/2025"].iloc[0]
            assert val is None or pd.isna(val), f"discontinued DIN {din} must stay blank"

    def test_group_matched_not_ambiguous(self):
        _, recon = match_iqvia_to_sheet1(self._jamp_sheet1(), self._jamp_iqvia())
        jamp = recon[recon["iqvia_product"] == "JAMP-METFORMIN"]
        assert len(jamp) == 1
        assert jamp["status"].iloc[0] == "matched", (
            f"expected 'matched' got {jamp['status'].iloc[0]!r}"
        )

    def test_fallback_when_no_marketed_candidate(self):
        """When NO marketed DIN matches, a discontinued DIN still gets the data
        (historical-only products remain attributable)."""
        sheet1 = pd.DataFrame([
            {"din": "02380722", "ingredient": "METFORMIN HYDROCHLORIDE",
             "brand_name": "JAMP-METFORMIN", "company": "JAMP PHARMA CORPORATION",
             "strength": "500 MG", "status": "Cancelled Post Market"},
        ])
        enriched, recon = match_iqvia_to_sheet1(sheet1, self._jamp_iqvia())
        row = enriched[enriched["din"] == "02380722"]
        assert int(row["Dollars MAT 12/2025"].iloc[0]) == 678910

    def test_cancelled_pre_market_excluded_entirely(self):
        """'Cancelled Pre Market' = never sold; excluded like 'Approved' — it must
        not even be a fallback candidate, so the group finds no DIN."""
        sheet1 = pd.DataFrame([
            {"din": "02361264", "ingredient": "METFORMIN HYDROCHLORIDE",
             "brand_name": "JAMP-METFORMIN", "company": "JAMP PHARMA CORPORATION",
             "strength": "500 MG", "status": "Cancelled Pre Market"},
        ])
        enriched, recon = match_iqvia_to_sheet1(sheet1, self._jamp_iqvia())
        val = enriched[enriched["din"] == "02361264"]["Dollars MAT 12/2025"].iloc[0]
        assert val is None or pd.isna(val)
        jamp = recon[recon["iqvia_product"] == "JAMP-METFORMIN"]
        assert jamp["status"].iloc[0] == "no_din_match"


# ── Bare-number strength inference ───────────────────────────────────────────
# IQVIA encodes "160MG/12.5MG" as "160/12.5MG" — the unit is omitted on every
# component except the last.  _norm_strength must infer the unit from the
# last-component token and apply it to all bare-number tokens.

class TestNormStrengthBareNumber:
    def test_hctz_combo_two_components(self):
        # "160/12.5MG" → "DIOVAN HCT" style: valsartan 160mg + HCTZ 12.5mg
        assert _norm_strength("160/12.5MG") == frozenset({"160MG", "12.5MG"})

    def test_hctz_combo_three_components(self):
        # Hypothetical triple: "5/160/12.5MG" → 5MG + 160MG + 12.5MG
        assert _norm_strength("5/160/12.5MG") == frozenset({"5MG", "160MG", "12.5MG"})

    def test_high_dose_combo(self):
        # "320/25MG" (valsartan 320mg / HCTZ 25mg)
        assert _norm_strength("320/25MG") == frozenset({"320MG", "25MG"})

    def test_already_explicit_unchanged(self):
        # When every component already has its unit, no inference runs.
        assert _norm_strength("160MG/12.5MG") == frozenset({"160MG", "12.5MG"})

    def test_bare_number_matches_explicit(self):
        # "160/12.5MG" and "160MG/12.5MG" must produce identical frozensets.
        assert _norm_strength("160/12.5MG") == _norm_strength("160MG/12.5MG")

    def test_bare_number_with_mg_unit(self):
        # Bare-number inference works when the anchoring unit is MG (common IQVIA pattern).
        # "5/160MG" → both components get MG → {"5MG", "160MG"}
        assert _norm_strength("5/160MG") == frozenset({"5MG", "160MG"})

    def test_no_inference_without_unit(self):
        # If there is no unit anywhere, bare numbers stay as-is (can't infer).
        result = _norm_strength("160/12")
        # Both tokens have no unit — neither should gain a fabricated unit.
        # The frozenset should not contain tokens with "MG", "MCG", etc.
        for token in result:
            assert not any(u in token for u in ("MG", "MCG", "ML", "IU", "%")), (
                f"Fabricated unit in token {token!r} with no source unit"
            )


class TestNormStrengthDecimalCanon:
    """DPD stores some strengths with a trailing zero / extra decimal ('10.0 MG',
    '50.0 MG', '12.50 MG'); IQVIA writes them compact ('10MG'). The exact
    strength-set prefilter must treat them as equal, else any decimal-formatted
    DPD strength silently fails to match (real case: PMS-AMLODIPINE '10.0 MG')."""

    def test_trailing_zero_equals_integer(self):
        assert _norm_strength("10.0 MG") == frozenset({"10MG"})
        assert _norm_strength("10.0 MG") == _norm_strength("10MG")

    def test_extra_decimal_place(self):
        assert _norm_strength("12.50 MG") == frozenset({"12.5MG"})

    def test_decimal_combo(self):
        assert _norm_strength("50.0 MG; 12.5 MG") == _norm_strength("50MG/12.5MG")

    def test_real_decimal_preserved(self):
        # A genuine fractional strength must NOT be flattened to an integer.
        assert _norm_strength("0.5 MG") == frozenset({"0.5MG"})

    def test_percent_decimal(self):
        assert _norm_strength("5.0 %") == frozenset({"5%"})


class TestNormCompanyFrenchAndPrefix:
    """Company normalisation regressions surfaced by the 5-ingredient audit."""

    def test_strips_french_laboratoire(self):
        # 'LABORATOIRE RIVA INC' must reduce to 'riva' so it matches IQVIA's 'RIVA'.
        assert _norm_company("LABORATOIRE RIVA INC.") == "riva"

    def test_strips_french_laboratoires_plural(self):
        assert _norm_company("LABORATOIRES PALADIN") == "paladin"

    def test_pharma_prefix_not_eaten(self):
        # 'pharma' is a standalone corporate word, NOT a prefix to strip: company
        # names that merely start with 'PHARMA' must survive intact, else they
        # collapse to garbage roots that collide ('PHARMARIS' → 'ris' ≈ 'riva').
        assert _norm_company("PHARMARIS") == "pharmaris"
        assert _norm_company("PHARMASCIENCE INC") == "pharmascience"

    def test_standalone_pharma_still_stripped(self):
        assert _norm_company("JAMP PHARMA CORPORATION") == "jamp"
        assert _norm_company("AURO PHARMA INC") == "auro"


class TestCompanyFloorAndMoleculeGuard:
    """Integration guards against the two false-match classes the audit found:
    coincidental cross-company collisions and cross-molecule combo collisions."""

    def test_company_floor_rejects_cross_company_collision(self):
        """NORA PHARMA's 'NRA-METFORMIN' must NOT match LABORATOIRE RIVA's DIN.
        Brand similarity is inflated by the shared '-METFORMIN'; the manufacturers
        (NORA vs RIVA, sim ≈ 50) are different, so the company floor rejects it."""
        sheet1 = pd.DataFrame([
            {"din": "02239081", "ingredient": "METFORMIN HYDROCHLORIDE",
             "brand_name": "RIVA-METFORMIN", "company": "LABORATOIRE RIVA INC.",
             "strength": "500 MG", "status": "Marketed"},
        ])
        iqvia = pd.DataFrame([{
            "Combined Molecule": "METFORMIN", "Product": "NRA-METFORMIN",
            "Manufacturer": "NORA PHARMA INC", "Strength": "0.5GM",
            "Units MAT 12/2025": 999, "Dollars MAT 12/2025": 9999,
        }])
        enriched, _ = match_iqvia_to_sheet1(sheet1, iqvia)
        val = enriched[enriched["din"] == "02239081"]["Dollars MAT 12/2025"].iloc[0]
        assert val is None or pd.isna(val), "NORA→RIVA cross-company collision must be rejected"

    def test_riva_matches_its_own_din(self):
        """The genuine RIVA product (manufacturer 'RIVA') must match the RIVA DIN
        once 'LABORATOIRE' is stripped (company 100)."""
        sheet1 = pd.DataFrame([
            {"din": "02239081", "ingredient": "METFORMIN HYDROCHLORIDE",
             "brand_name": "RIVA-METFORMIN", "company": "LABORATOIRE RIVA INC.",
             "strength": "500 MG", "status": "Marketed"},
        ])
        iqvia = pd.DataFrame([{
            "Combined Molecule": "METFORMIN", "Product": "RIVA-METFORMIN",
            "Manufacturer": "RIVA", "Strength": "0.5GM",
            "Units MAT 12/2025": 100, "Dollars MAT 12/2025": 2000,
        }])
        enriched, _ = match_iqvia_to_sheet1(sheet1, iqvia)
        assert int(enriched[enriched["din"] == "02239081"]["Dollars MAT 12/2025"].iloc[0]) == 2000

    def test_cross_molecule_combo_rejected(self):
        """IQVIA TELMISARTAN/HCTZ must NOT match a DPD VALSARTAN/HCTZ DIN despite
        identical strength and the shared HYDROCHLOROTHIAZIDE component."""
        sheet1 = pd.DataFrame([
            {"din": "02384736", "ingredient": "VALSARTAN; HYDROCHLOROTHIAZIDE",
             "brand_name": "VALSARTAN HCT", "company": "SIVEM PHARMACEUTICALS ULC",
             "strength": "80 MG; 12.5 MG", "status": "Marketed"},
        ])
        iqvia = pd.DataFrame([{
            "Combined Molecule": "HYDROCHLOROTHIAZIDE:TELMISARTAN",
            "Product": "TELMISARTAN HCTZ", "Manufacturer": "SIVEM PHARMA ULC",
            "Strength": "80MG/12.5MG",
            "Units MAT 12/2025": 50, "Dollars MAT 12/2025": 5000,
        }])
        enriched, _ = match_iqvia_to_sheet1(sheet1, iqvia)
        val = enriched[enriched["din"] == "02384736"]["Dollars MAT 12/2025"].iloc[0]
        assert val is None or pd.isna(val), "telmisartan/HCTZ must not match valsartan/HCTZ DIN"

    def test_mono_does_not_match_combo(self):
        """A mono AMLODIPINE IQVIA group must NOT match an AMLODIPINE+ATORVASTATIN
        combo DIN even when both per-component strengths are 10MG (collapsing to
        the same {10MG} set)."""
        sheet1 = pd.DataFrame([
            {"din": "02362791",
             "ingredient": "AMLODIPINE (AMLODIPINE BESYLATE); ATORVASTATIN (ATORVASTATIN CALCIUM)",
             "brand_name": "MYLAN-AMLODIPINE/ATORVASTATIN", "company": "MYLAN PHARMACEUTICALS ULC",
             "strength": "10 MG; 10 MG", "status": "Marketed"},
        ])
        iqvia = pd.DataFrame([{
            "Combined Molecule": "AMLODIPINE", "Product": "AMLODIPINE",
            "Manufacturer": "MYLAN PHARMA", "Strength": "10MG",
            "Units MAT 12/2025": 7, "Dollars MAT 12/2025": 700,
        }])
        enriched, _ = match_iqvia_to_sheet1(sheet1, iqvia)
        val = enriched[enriched["din"] == "02362791"]["Dollars MAT 12/2025"].iloc[0]
        assert val is None or pd.isna(val), "mono amlodipine must not match the combo DIN"

    def test_real_combo_still_matches(self):
        """A genuine VALSARTAN/HCTZ IQVIA group MUST still match the VALSARTAN/HCTZ
        DIN — the molecule guard rejects only cross-molecule combos."""
        sheet1 = pd.DataFrame([
            {"din": "02384736", "ingredient": "VALSARTAN; HYDROCHLOROTHIAZIDE",
             "brand_name": "VALSARTAN HCT", "company": "SIVEM PHARMACEUTICALS ULC",
             "strength": "80 MG; 12.5 MG", "status": "Marketed"},
        ])
        iqvia = pd.DataFrame([{
            "Combined Molecule": "HYDROCHLOROTHIAZIDE:VALSARTAN",
            "Product": "VALSARTAN HCT", "Manufacturer": "SIVEM PHARMA ULC",
            "Strength": "80MG/12.5MG",
            "Units MAT 12/2025": 50, "Dollars MAT 12/2025": 5000,
        }])
        enriched, _ = match_iqvia_to_sheet1(sheet1, iqvia)
        assert int(enriched[enriched["din"] == "02384736"]["Dollars MAT 12/2025"].iloc[0]) == 5000


# ── Exact-brand priority + generic-label aggregation ──────────────────────────
# These reproduce, as self-contained fixtures, the amlodipine/metformin matcher
# errors found in the real IQVIA sweep. Root cause: scoring was 0.5*brand +
# 0.5*company with no priority for an exact brand-name match, so a generic group
# or a same-prefix sibling could out-claim a DIN that belongs, by exact brand, to
# another group. The fix reserves exact-brand matches before the fuzzy stage and
# aggregates same-company generic-label aliases onto the reserved DIN.

def _amlo(units):
    """One collapsed AMLODIPINE IQVIA group row (units, dollars=units*10)."""
    def _g(product, manufacturer, strength, u):
        return {"Combined Molecule": "AMLODIPINE", "Product": product,
                "Manufacturer": manufacturer, "Strength": strength,
                "Units MAT 12/2025": u, "Dollars MAT 12/2025": u * 10}
    return _g


def _u(enriched, din):
    row = enriched[enriched["din"] == din]
    if not len(row):
        return None
    v = row["Units MAT 12/2025"].iloc[0]
    return None if (v is None or pd.isna(v)) else int(v)


def _status(recon, product):
    r = recon[recon["iqvia_product"] == product]
    return r["status"].iloc[0] if len(r) else "MISSING"


class TestExactBrandPriority:
    """Real swap: PHARMASCIENCE sells PMS-AMLODIPINE (marketed) and PHARMA-
    AMLODIPINE (a distinct brand whose DIN is Dormant). PHARMARIS sells PRZ-
    AMLODIPINE. Because PHARMASCIENCE↔PHARMARIS scores 63.6 (above the 55 floor),
    the old fuzzy stage let the generic-ish PHARMA group claim the PMS DIN, which
    then pushed the real 33,489-unit PMS group cross-company onto the PRZ DIN."""

    def _g(self, product, manufacturer, strength, u):
        return {"Combined Molecule": "AMLODIPINE", "Product": product,
                "Manufacturer": manufacturer, "Strength": strength,
                "Units MAT 12/2025": u, "Dollars MAT 12/2025": u * 10}

    def _sheet1(self):
        return pd.DataFrame([
            {"din": "02284065", "ingredient": "AMLODIPINE (AMLODIPINE BESYLATE)",
             "brand_name": "PMS-AMLODIPINE", "company": "PHARMASCIENCE INC",
             "strength": "5 MG", "status": "Marketed"},
            {"din": "02522519", "ingredient": "AMLODIPINE (AMLODIPINE BESYLATE)",
             "brand_name": "PRZ-AMLODIPINE", "company": "PHARMARIS CANADA INC",
             "strength": "5 MG", "status": "Marketed"},
        ])

    def _iqvia(self):
        # Collapse sort order (alphabetical by Product) processes PHARMA before PMS,
        # which is exactly what triggered the old greedy claim.
        return pd.DataFrame([
            self._g("PHARMA-AMLODIPINE", "PHARMASCIENCE", "5MG", 13),
            self._g("PMS-AMLODIPINE", "PHARMASCIENCE", "5MG", 33489),
            self._g("PRZ-AMLODIPINE", "PHARMARIS", "5MG", 760),
        ])

    def test_pms_gets_its_own_exact_group(self):
        enriched, _ = match_iqvia_to_sheet1(self._sheet1(), self._iqvia())
        assert _u(enriched, "02284065") == 33489

    def test_pms_not_inflated_by_distinct_brand(self):
        """The distinct PHARMA-AMLODIPINE brand (13u) must NOT aggregate onto the
        PMS DIN — only generic molecule labels aggregate, never sibling brands."""
        enriched, _ = match_iqvia_to_sheet1(self._sheet1(), self._iqvia())
        assert _u(enriched, "02284065") != 33489 + 13

    def test_prz_gets_its_own_group_not_cross_company(self):
        """The 33,489-unit PMS group must never land on the PRZ DIN; PRZ gets PRZ."""
        enriched, _ = match_iqvia_to_sheet1(self._sheet1(), self._iqvia())
        assert _u(enriched, "02522519") == 760

    def test_distinct_brand_without_din_is_unmatched(self):
        """PHARMA-AMLODIPINE has no DIN here, and is a distinct brand (not a generic
        label), so it must stay unmatched rather than aggregate onto a sibling."""
        _, recon = match_iqvia_to_sheet1(self._sheet1(), self._iqvia())
        assert _status(recon, "PHARMA-AMLODIPINE") == "no_din_match"

    def test_pack_size_siblings_stay_ambiguous(self):
        """Exact-brand priority must NOT resolve genuine pack-size duplicates:
        PROVERA vs PROVERA PAK (same company, same strength) stays ambiguous."""
        sheet1 = pd.DataFrame([
            {"din": "00030937", "ingredient": "MEDROXYPROGESTERONE ACETATE",
             "brand_name": "PROVERA", "company": "PFIZER CANADA ULC",
             "strength": "5 MG", "status": "Marketed"},
            {"din": "02010739", "ingredient": "MEDROXYPROGESTERONE ACETATE",
             "brand_name": "PROVERA PAK", "company": "PFIZER CANADA ULC",
             "strength": "5 MG", "status": "Marketed"},
        ])
        iqvia = pd.DataFrame([{
            "Combined Molecule": "MEDROXYPROGESTERONE ACETATE", "Product": "PROVERA",
            "Manufacturer": "PFIZER", "Strength": "5MG",
            "Units MAT 12/2025": 1000, "Dollars MAT 12/2025": 10000,
        }])
        enriched, recon = match_iqvia_to_sheet1(sheet1, iqvia)
        assert _u(enriched, "00030937") is None
        assert _u(enriched, "02010739") is None
        assert _status(recon, "PROVERA") == "ambiguous"

    def test_two_marketed_siblings_stay_ambiguous(self):
        """JAMP at 5 mg has TWO marketed DINs — a generic-labelled AMLODIPINE DIN
        and a JAMP-AMLODIPINE DIN — so both the generic and exact groups are
        genuinely ambiguous and must stay blank (the same-company near-tie guard)."""
        sheet1 = pd.DataFrame([
            {"din": "02429217", "ingredient": "AMLODIPINE (AMLODIPINE BESYLATE)",
             "brand_name": "AMLODIPINE", "company": "JAMP PHARMA CORPORATION",
             "strength": "5 MG", "status": "Marketed"},
            {"din": "02357194", "ingredient": "AMLODIPINE (AMLODIPINE BESYLATE)",
             "brand_name": "JAMP-AMLODIPINE", "company": "JAMP PHARMA CORPORATION",
             "strength": "5 MG", "status": "Marketed"},
        ])
        iqvia = pd.DataFrame([
            self._g("AMLODIPINE", "JAMP PHARMA", "5MG", 8651),
            self._g("JAMP-AMLODIPINE", "JAMP PHARMA", "5MG", 28106),
        ])
        enriched, recon = match_iqvia_to_sheet1(sheet1, iqvia)
        assert _u(enriched, "02429217") is None
        assert _u(enriched, "02357194") is None
        assert _status(recon, "JAMP-AMLODIPINE") == "ambiguous"


class TestGenericLabelAggregation:
    """A generic molecule label (IQVIA "METFORMIN" / "AMLODIPINE") is the same
    physical product a firm also sells under its brand. When that firm's only
    marketed DIN is brand-named, the generic-labelled sales must aggregate onto it
    rather than orphan — but only for genuine generic labels and same company."""

    def _met(self, product, manufacturer, strength, u):
        return {"Combined Molecule": "METFORMIN", "Product": product,
                "Manufacturer": manufacturer, "Strength": strength,
                "Units MAT 12/2025": u, "Dollars MAT 12/2025": u * 10}

    def test_pro_metformin_stays_whole(self):
        """PRO DOC's only marketed metformin 500 DIN is PRO-METFORMIN; IQVIA splits
        its sales into a big generic "METFORMIN" group (37,942) and a small exact
        "PRO-METFORMIN" group (48). The DIN must reflect BOTH — not just the small
        exact-labelled one (the naive remove-both trap)."""
        sheet1 = pd.DataFrame([
            {"din": "02314908", "ingredient": "METFORMIN HYDROCHLORIDE",
             "brand_name": "PRO-METFORMIN", "company": "PRO DOC LIMITEE",
             "strength": "500 MG", "status": "Marketed"},
            # a different firm's marketed DIN — present (as in the real market) so the
            # lifecycle tier excludes any discontinued same-company sibling and the
            # generic PRO DOC label has no fuzzy target except the reserved DIN.
            {"din": "02246834", "ingredient": "METFORMIN HYDROCHLORIDE",
             "brand_name": "SANDOZ METFORMIN", "company": "SANDOZ CANADA INC",
             "strength": "500 MG", "status": "Marketed"},
        ])
        iqvia = pd.DataFrame([
            self._met("METFORMIN", "PRO DOC", "500MG", 37942),
            self._met("PRO-METFORMIN", "PRO DOC", "0.5GM", 48),
        ])
        enriched, recon = match_iqvia_to_sheet1(sheet1, iqvia)
        assert _u(enriched, "02314908") == 37942 + 48
        assert _u(enriched, "02246834") is None       # unrelated firm, no group
        assert _status(recon, "METFORMIN") == "matched"
        assert _status(recon, "PRO-METFORMIN") == "matched"

    def test_generic_alias_does_not_aggregate_cross_company(self):
        """MANTRA's "M-AMLODIPINE" (a one-letter manufacturer prefix, NOT a generic
        molecule label) must not be mistaken for generic and aggregated onto a
        coincidentally similar-named company (MINT, company ≈ 60)."""
        sheet1 = pd.DataFrame([
            {"din": "02362651", "ingredient": "AMLODIPINE (AMLODIPINE BESYLATE)",
             "brand_name": "MINT-AMLODIPINE", "company": "MINT PHARMACEUTICALS INC",
             "strength": "5 MG", "status": "Marketed"},
        ])
        iqvia = pd.DataFrame([
            _amlo(0)("M-AMLODIPINE", "MANTRA PHARMA", "5MG", 15851),
            _amlo(0)("MINT-AMLODIPINE", "MINT PHARMACEUTICALS", "5MG", 47937),
        ])
        enriched, recon = match_iqvia_to_sheet1(sheet1, iqvia)
        assert _u(enriched, "02362651") == 47937        # its own group, never summed
        assert _status(recon, "M-AMLODIPINE") == "no_din_match"

    def test_conservation_with_aggregation(self):
        """Every collapsed unit is still accounted for once after aggregation:
        matched + unmatched group rows == collapsed total."""
        sheet1 = pd.DataFrame([
            {"din": "02314908", "ingredient": "METFORMIN HYDROCHLORIDE",
             "brand_name": "PRO-METFORMIN", "company": "PRO DOC LIMITEE",
             "strength": "500 MG", "status": "Marketed"},
            {"din": "02246834", "ingredient": "METFORMIN HYDROCHLORIDE",
             "brand_name": "SANDOZ METFORMIN", "company": "SANDOZ CANADA INC",
             "strength": "500 MG", "status": "Marketed"},
        ])
        iqvia = pd.DataFrame([
            self._met("METFORMIN", "PRO DOC", "500MG", 37942),
            self._met("PRO-METFORMIN", "PRO DOC", "0.5GM", 48),
        ])
        enriched, recon = match_iqvia_to_sheet1(sheet1, iqvia)
        group_rows = recon[recon["status"] != "din_no_iqvia_match"]
        for c in ("Units MAT 12/2025", "Dollars MAT 12/2025"):
            assert int(pd.to_numeric(group_rows[c], errors="coerce").fillna(0).sum()) == \
                int(iqvia[c].sum())
            assert int(pd.to_numeric(enriched[c], errors="coerce").fillna(0).sum()) == \
                int(iqvia[c].sum())


# ── Parsing and collapsing ────────────────────────────────────────────────────

class TestParseIqvia:
    def test_shape(self, iqvia_raw):
        assert len(iqvia_raw) == 392  # known sample size

    def test_metric_cols_detected(self, iqvia_raw):
        mc = detect_metric_columns(iqvia_raw)
        assert len(mc) == 12  # 4 years × 3 metrics
        assert "Units MAT 12/2025" in mc
        assert "Dollars MAT 12/2025" in mc

    def test_dash_converted_to_zero(self, iqvia_raw):
        # The raw file has '-' in many cells; they must be numeric after parsing.
        # Actual negative values (returns/corrections) are preserved as-is.
        mc = detect_metric_columns(iqvia_raw)
        for col in mc:
            assert iqvia_raw[col].dtype in ("int64", "int32", "float64")
        # Spot-check: a known all-zero row should not have NaN
        assert iqvia_raw[mc[0]].isna().sum() == 0


class TestCollapseIqvia:
    def test_row_count(self, iqvia_collapsed):
        # 22 unique (molecule, product, manufacturer, strength) groups
        assert len(iqvia_collapsed) == 22

    def test_sanis_units(self, iqvia_collapsed):
        row = iqvia_collapsed[
            (iqvia_collapsed["Product"] == "PROGESTERONE") &
            (iqvia_collapsed["Manufacturer"] == "SANIS HEALTH INC")
        ]
        assert len(row) == 1
        assert int(row["Units MAT 12/2025"].iloc[0]) == 218591

    def test_sanis_dollars(self, iqvia_collapsed):
        row = iqvia_collapsed[
            (iqvia_collapsed["Product"] == "PROGESTERONE") &
            (iqvia_collapsed["Manufacturer"] == "SANIS HEALTH INC")
        ]
        assert int(row["Dollars MAT 12/2025"].iloc[0]) == 21215081

    def test_auro_units(self, iqvia_collapsed):
        row = iqvia_collapsed[iqvia_collapsed["Product"] == "AURO-PROGESTERONE"]
        assert len(row) == 1
        assert int(row["Units MAT 12/2025"].iloc[0]) == 233159

    def test_auro_dollars(self, iqvia_collapsed):
        row = iqvia_collapsed[iqvia_collapsed["Product"] == "AURO-PROGESTERONE"]
        assert int(row["Dollars MAT 12/2025"].iloc[0]) == 13005865

    def test_depo_provera_units(self, iqvia_collapsed):
        # Combines SYRINGE + VIAL × Drugstore + Hospital (36 raw rows → 1 collapsed)
        row = iqvia_collapsed[iqvia_collapsed["Product"] == "DEPO-PROVERA"]
        assert len(row) == 1
        assert int(row["Units MAT 12/2025"].iloc[0]) == 262834

    def test_depo_provera_dollars(self, iqvia_collapsed):
        row = iqvia_collapsed[iqvia_collapsed["Product"] == "DEPO-PROVERA"]
        assert int(row["Dollars MAT 12/2025"].iloc[0]) == 8853659


# ── Matching ──────────────────────────────────────────────────────────────────

def _make_sheet1(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal Sheet 1 DataFrame for matching tests."""
    return pd.DataFrame(rows)


class TestMatchIqvia:
    @pytest.fixture(scope="class")
    def sheet1_progesterone(self):
        """Minimal Sheet 1 representing a progesterone search result."""
        return _make_sheet1([
            # DIN 02516187 — SANIS HEALTH INC / PROGESTERONE / 100MG
            {
                "din": "02516187",
                "ingredient": "PROGESTERONE",
                "brand_name": "PROGESTERONE",
                "company": "SANIS HEALTH INC",
                "strength": "100 MG",
                "dosage_form": "Capsule",
            },
            # DIN 02493578 — AURO PHARMA INC / AURO-PROGESTERONE / 100MG
            {
                "din": "02493578",
                "ingredient": "PROGESTERONE",
                "brand_name": "AURO-PROGESTERONE",
                "company": "AURO PHARMA INC",
                "strength": "100 MG",
                "dosage_form": "Capsule",
            },
            # DIN 00585092 — PFIZER CANADA ULC / DEPO-PROVERA / 150 MG (injectable)
            {
                "din": "00585092",
                "ingredient": "MEDROXYPROGESTERONE ACETATE",
                "brand_name": "DEPO-PROVERA",
                "company": "PFIZER CANADA ULC",
                "strength": "150 MG",
                "dosage_form": "Injection",
            },
            # DIN 00030937 — PFIZER CANADA ULC / PROVERA 5MG TABLETS (ambiguous)
            {
                "din": "00030937",
                "ingredient": "MEDROXYPROGESTERONE ACETATE",
                "brand_name": "PROVERA 5MG TABLETS",
                "company": "PFIZER CANADA ULC",
                "strength": "5 MG",
                "dosage_form": "Tablet",
            },
            # DIN 02010739 — PFIZER CANADA ULC / PROVERA PAK 5MG (ambiguous)
            {
                "din": "02010739",
                "ingredient": "MEDROXYPROGESTERONE ACETATE",
                "brand_name": "PROVERA PAK 5MG",
                "company": "PFIZER CANADA ULC",
                "strength": "5 MG",
                "dosage_form": "Tablet",
            },
            # DIN 00262056 — veterinary; no IQVIA entry expected
            {
                "din": "00262056",
                "ingredient": "PROGESTERONE",
                "brand_name": "SYNOVEX S",
                "company": "ZOETIS CANADA INC",
                "strength": "200 MG",
                "dosage_form": "Implant",
            },
        ])

    def test_sanis_units_in_sheet1(self, sheet1_progesterone, iqvia_collapsed):
        enriched, _ = match_iqvia_to_sheet1(sheet1_progesterone, iqvia_collapsed)
        row = enriched[enriched["din"] == "02516187"]
        assert len(row) == 1
        assert int(row["Units MAT 12/2025"].iloc[0]) == 218591

    def test_sanis_dollars_in_sheet1(self, sheet1_progesterone, iqvia_collapsed):
        enriched, _ = match_iqvia_to_sheet1(sheet1_progesterone, iqvia_collapsed)
        row = enriched[enriched["din"] == "02516187"]
        assert int(row["Dollars MAT 12/2025"].iloc[0]) == 21215081

    def test_auro_units_in_sheet1(self, sheet1_progesterone, iqvia_collapsed):
        enriched, _ = match_iqvia_to_sheet1(sheet1_progesterone, iqvia_collapsed)
        row = enriched[enriched["din"] == "02493578"]
        assert int(row["Units MAT 12/2025"].iloc[0]) == 233159

    def test_auro_dollars_in_sheet1(self, sheet1_progesterone, iqvia_collapsed):
        enriched, _ = match_iqvia_to_sheet1(sheet1_progesterone, iqvia_collapsed)
        row = enriched[enriched["din"] == "02493578"]
        assert int(row["Dollars MAT 12/2025"].iloc[0]) == 13005865

    def test_depo_provera_units_in_sheet1(self, sheet1_progesterone, iqvia_collapsed):
        enriched, _ = match_iqvia_to_sheet1(sheet1_progesterone, iqvia_collapsed)
        row = enriched[enriched["din"] == "00585092"]
        assert int(row["Units MAT 12/2025"].iloc[0]) == 262834

    def test_depo_provera_dollars_in_sheet1(self, sheet1_progesterone, iqvia_collapsed):
        enriched, _ = match_iqvia_to_sheet1(sheet1_progesterone, iqvia_collapsed)
        row = enriched[enriched["din"] == "00585092"]
        assert int(row["Dollars MAT 12/2025"].iloc[0]) == 8853659

    def test_provera_5mg_ambiguous_din1(self, sheet1_progesterone, iqvia_collapsed):
        """DIN 00030937 (PROVERA 5MG) must remain blank — PROVERA 5MG is ambiguous."""
        enriched, _ = match_iqvia_to_sheet1(sheet1_progesterone, iqvia_collapsed)
        row = enriched[enriched["din"] == "00030937"]
        assert len(row) == 1
        val = row["Units MAT 12/2025"].iloc[0]
        assert val is None or (pd.isna(val))

    def test_provera_5mg_ambiguous_din2(self, sheet1_progesterone, iqvia_collapsed):
        """DIN 02010739 (PROVERA PAK 5MG) must remain blank — PROVERA 5MG is ambiguous."""
        enriched, _ = match_iqvia_to_sheet1(sheet1_progesterone, iqvia_collapsed)
        row = enriched[enriched["din"] == "02010739"]
        assert len(row) == 1
        val = row["Units MAT 12/2025"].iloc[0]
        assert val is None or (pd.isna(val))

    def test_synovex_no_iqvia_data(self, sheet1_progesterone, iqvia_collapsed):
        """Veterinary implant DIN 00262056 has no IQVIA match — cells must be None."""
        enriched, _ = match_iqvia_to_sheet1(sheet1_progesterone, iqvia_collapsed)
        row = enriched[enriched["din"] == "00262056"]
        assert len(row) == 1
        val = row["Units MAT 12/2025"].iloc[0]
        assert val is None or (pd.isna(val))

    def test_one_row_per_din(self, sheet1_progesterone, iqvia_collapsed):
        """Sheet 1 row count must not change after enrichment."""
        enriched, _ = match_iqvia_to_sheet1(sheet1_progesterone, iqvia_collapsed)
        assert len(enriched) == len(sheet1_progesterone)

    def test_reconciliation_contains_provera_ambiguous(self, sheet1_progesterone, iqvia_collapsed):
        _, recon = match_iqvia_to_sheet1(sheet1_progesterone, iqvia_collapsed)
        ambig = recon[
            (recon["status"] == "ambiguous") &
            (recon["iqvia_product"] == "PROVERA") &
            (recon["iqvia_strength"] == "5MG")
        ]
        assert len(ambig) >= 1

    def test_reconciliation_cytex_unmatched(self, sheet1_progesterone, iqvia_collapsed):
        """CYTEX PROGESTERONE 50MG/ML has no DIN in Sheet 1 → no_din_match."""
        _, recon = match_iqvia_to_sheet1(sheet1_progesterone, iqvia_collapsed)
        cytex = recon[
            (recon["status"] == "no_din_match") &
            (recon["iqvia_manufacturer"].str.upper() == "CYTEX")
        ]
        assert len(cytex) >= 1

    def test_reconciliation_hikma_unmatched(self, sheet1_progesterone, iqvia_collapsed):
        """HIKMA PROGESTERONE 50MG/ML has no DIN in Sheet 1 → no_din_match."""
        _, recon = match_iqvia_to_sheet1(sheet1_progesterone, iqvia_collapsed)
        hikma = recon[
            (recon["status"] == "no_din_match") &
            (recon["iqvia_manufacturer"].str.upper().str.contains("HIKMA"))
        ]
        assert len(hikma) >= 1

    def test_no_fabrication_zeros(self, sheet1_progesterone, iqvia_collapsed):
        """Unmatched DINs must have None/NaN, never 0, in metric columns."""
        enriched, _ = match_iqvia_to_sheet1(sheet1_progesterone, iqvia_collapsed)
        mc = detect_metric_columns(iqvia_collapsed)
        for col in mc:
            for din in ["00030937", "02010739", "00262056"]:
                row = enriched[enriched["din"] == din]
                val = row[col].iloc[0]
                # Must be None or NaN, NOT 0
                assert val is None or pd.isna(val), (
                    f"DIN {din} column {col!r} = {val!r} — should be None/NaN, not 0"
                )


# ── Company normalization: corporation / incorporated / limited ───────────────

class TestNormCompanyExtended:
    def test_strips_corporation(self):
        assert _norm_company("JAMP PHARMA CORPORATION") == "jamp"

    def test_strips_incorporated(self):
        assert _norm_company("SOME DRUG INCORPORATED") == "some drug"

    def test_strips_limited(self):
        assert _norm_company("TEVA CANADA LIMITED") == "teva"

    def test_strips_labs(self):
        assert _norm_company("PENDOPHARM LABS INC") == "pendopharm"

    def test_corporation_equals_corp(self):
        assert _norm_company("JAMP PHARMA CORP") == _norm_company("JAMP PHARMA CORPORATION")

    # ── French / Quebec legal suffixes ────────────────────────────────────────
    # "PRO DOC LIMITÉE / S.E.C." is the DPD-registered form; IQVIA writes "PRO DOC".
    # Without unicode normalisation + French suffix stripping the company_sim
    # drops from 100 to ~54, causing scores of ~77 instead of 100.

    def test_strips_limitee_accented(self):
        assert _norm_company("PRO DOC LIMITÉE") == "pro doc"

    def test_strips_limitee_plain(self):
        assert _norm_company("PRO DOC LIMITEE") == "pro doc"

    def test_strips_ltee(self):
        assert _norm_company("PRO DOC LTÉE") == "pro doc"

    def test_strips_sec_abbreviation(self):
        # "S.E.C." (société en commandite) — dots stripped first, then "sec" removed.
        assert _norm_company("PRO DOC LIMITÉE / S.E.C.") == "pro doc"

    def test_strips_ampersand(self):
        # "&" must be removed so "Smith & Nephew" and "Smith Nephew" normalise alike.
        assert _norm_company("SMITH & NEPHEW INC") == "smith nephew"

    def test_unicode_normalisation_general(self):
        # "é" must fold to "e" so accented and unaccented versions compare equal.
        # "PRO DOC LIMITÉE" (accented) vs "PRO DOC LIMITEE" (plain) must normalise identically.
        assert _norm_company("PRO DOC LIMITÉE") == _norm_company("PRO DOC LIMITEE")

    def test_limitee_equals_limited(self):
        # French "Limitée" and English "Limited" must reduce to the same root.
        assert _norm_company("PRO DOC LIMITÉE") == _norm_company("PRO DOC LIMITED")


# ── Claimed-DIN exclusion: DINs matched to an earlier IQVIA group must not ───
# reappear as candidates for later groups, preventing false near-ties.         ─

class TestClaimedDinExclusion:
    """Regression for DIN 02497654 (JAMP ABACAVIR / LAMIVUDINE, JAMP PHARMA
    CORPORATION).  Before the fix, the alphabetically-earlier APO group claimed
    DIN 02399539 first.  02399539 then still appeared in JAMP's candidate list
    (score 62.6), creating a near-tie gap of 5 against the correct JAMP DIN
    (score 68) — below TIE_MARGIN=15 → falsely flagged ambiguous.

    After the fix, claimed DINs are excluded from later groups' candidate lists,
    leaving 02497654 as the sole candidate → unambiguously matched.
    """

    @pytest.fixture(scope="class")
    def abacavir_sheet1(self):
        return pd.DataFrame([
            {"din": "02399539", "ingredient": "ABACAVIR SULFATE; LAMIVUDINE",
             "brand_name": "APO-ABACAVIR-LAMIVUDINE",    "company": "APOTEX INC",
             "strength": "600 MG; 300 MG", "status": "Marketed"},
            {"din": "02454513", "ingredient": "ABACAVIR SULFATE; LAMIVUDINE",
             "brand_name": "AURO-ABACAVIR/LAMIVUDINE",   "company": "AURO PHARMA INC",
             "strength": "600 MG; 300 MG", "status": "Marketed"},
            {"din": "02497654", "ingredient": "ABACAVIR SULFATE; LAMIVUDINE",
             "brand_name": "JAMP ABACAVIR / LAMIVUDINE", "company": "JAMP PHARMA CORPORATION",
             "strength": "600 MG; 300 MG", "status": "Marketed"},
            {"din": "02518287", "ingredient": "ABACAVIR SULFATE; LAMIVUDINE",
             "brand_name": "APO-ABACAVIR-LAMIVUDINE TABLETS", "company": "APOTEX INC",
             "strength": "600 MG; 300 MG", "status": "Approved"},
        ])

    @pytest.fixture(scope="class")
    def combinations_iqvia(self):
        if not COMBINATIONS_PATH:
            pytest.skip(_MISSING_SAMPLE_MSG)
        with open(COMBINATIONS_PATH, "rb") as f:
            return collapse_iqvia(parse_iqvia(f.read()))

    def test_jamp_din_gets_iqvia_data(self, abacavir_sheet1, combinations_iqvia):
        """DIN 02497654 (JAMP ABACAVIR / LAMIVUDINE) must be matched — not ambiguous."""
        enriched, _ = match_iqvia_to_sheet1(abacavir_sheet1, combinations_iqvia)
        row = enriched[enriched["din"] == "02497654"]
        assert len(row) == 1
        dollars = row["Dollars MAT 12/2022"].iloc[0]
        assert dollars is not None and not pd.isna(dollars), (
            "DIN 02497654 got no IQVIA data — claimed-DIN exclusion may not be working"
        )

    def test_jamp_not_ambiguous_in_recon(self, abacavir_sheet1, combinations_iqvia):
        """The JAMP IQVIA group must have status='matched', not 'ambiguous'."""
        _, recon = match_iqvia_to_sheet1(abacavir_sheet1, combinations_iqvia)
        jamp_rows = recon[recon["iqvia_product"] == "JAMP ABACAVIR/LAMIVUDINE"]
        assert len(jamp_rows) == 1
        assert jamp_rows["status"].iloc[0] == "matched", (
            f"JAMP group status = {jamp_rows['status'].iloc[0]!r}; "
            "expected 'matched' — claimed DIN 02399539 must not pollute JAMP's candidate list"
        )

    def test_apo_din_matched(self, abacavir_sheet1, combinations_iqvia):
        """DIN 02399539 (APO) must be matched to its own IQVIA group."""
        enriched, _ = match_iqvia_to_sheet1(abacavir_sheet1, combinations_iqvia)
        row = enriched[enriched["din"] == "02399539"]
        assert not pd.isna(row["Dollars MAT 12/2022"].iloc[0])

    def test_auro_din_matched(self, abacavir_sheet1, combinations_iqvia):
        """DIN 02454513 (AURO) must be matched to its own IQVIA group."""
        enriched, _ = match_iqvia_to_sheet1(abacavir_sheet1, combinations_iqvia)
        row = enriched[enriched["din"] == "02454513"]
        assert not pd.isna(row["Dollars MAT 12/2022"].iloc[0])

    def test_all_three_matched_in_recon(self, abacavir_sheet1, combinations_iqvia):
        """All three marketed abacavir/lamivudine DINs must each match a distinct group."""
        _, recon = match_iqvia_to_sheet1(abacavir_sheet1, combinations_iqvia)
        matched = recon[recon["status"] == "matched"]
        matched_dins = set(matched["din"].tolist())
        for expected_din in ("02399539", "02454513", "02497654"):
            assert expected_din in matched_dins, (
                f"DIN {expected_din} not in matched set {matched_dins}"
            )

    def test_no_cross_din_data_bleed(self, abacavir_sheet1, combinations_iqvia):
        """Each DIN must receive only its own IQVIA group's data — no zeros, no bleeds."""
        enriched, _ = match_iqvia_to_sheet1(abacavir_sheet1, combinations_iqvia)
        for din in ("02399539", "02454513", "02497654"):
            row = enriched[enriched["din"] == din]
            dollars = row["Dollars MAT 12/2022"].iloc[0]
            assert dollars is not None and not pd.isna(dollars) and dollars > 0, (
                f"DIN {din} has bad Dollars value: {dollars!r}"
            )


# ── Aggregation / conservation correctness (parse → collapse → match → stamp) ──
# These guard the OTHER half of correctness: that dollars are neither dropped,
# silently zeroed, double-counted, nor merged across distinct products.

def _make_raw_iqvia(rows: list[dict], metric_cols: list[str]) -> bytes:
    """Write a minimal raw IQVIA workbook (a single 'data' sheet) to bytes.

    ``rows`` are raw per-channel/province rows (pre-collapse); metric cell values
    may be strings (to exercise the parser) or numbers.
    """
    cols = ["Combined Molecule", "Product", "Manufacturer", "Strength"] + metric_cols
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    df = df[cols]
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        df.to_excel(xw, sheet_name="data", index=False)
    return buf.getvalue()


_MC = ["Dollars MAT 12/2025", "Units MAT 12/2025"]


class TestConservationOfDollars:
    """Backbone invariant: every dollar in the raw file is accounted for at each
    stage. raw-sum == collapsed-sum == reconciliation(group rows)-sum, and the
    value stamped onto Sheet 1 == the matched groups' sum (never doubled)."""

    def _raw(self):
        # Two products; PROGESTERONE is split across two channel rows that must
        # collapse (and sum) into one group.
        return _make_raw_iqvia([
            {"Combined Molecule": "PROGESTERONE", "Product": "PROGESTERONE",
             "Manufacturer": "SANIS HEALTH INC", "Strength": "100MG",
             "Dollars MAT 12/2025": "1000", "Units MAT 12/2025": "10"},
            {"Combined Molecule": "PROGESTERONE", "Product": "PROGESTERONE",
             "Manufacturer": "SANIS HEALTH INC", "Strength": "100MG",
             "Dollars MAT 12/2025": "2500", "Units MAT 12/2025": "25"},
            {"Combined Molecule": "PROGESTERONE", "Product": "AURO-PROGESTERONE",
             "Manufacturer": "AURO PHARMA INC", "Strength": "100MG",
             "Dollars MAT 12/2025": "777", "Units MAT 12/2025": "7"},
        ], _MC)

    def _sheet1(self):
        return pd.DataFrame([
            {"din": "02516187", "ingredient": "PROGESTERONE", "brand_name": "PROGESTERONE",
             "company": "SANIS HEALTH INC", "strength": "100 MG", "status": "Marketed"},
            {"din": "02493578", "ingredient": "PROGESTERONE", "brand_name": "AURO-PROGESTERONE",
             "company": "AURO PHARMA INC", "strength": "100 MG", "status": "Marketed"},
        ])

    def test_raw_equals_collapsed(self):
        raw = parse_iqvia(self._raw())
        collapsed = collapse_iqvia(raw)
        for c in _MC:
            assert int(raw[c].sum()) == int(collapsed[c].sum())

    def test_collapsed_equals_reconciliation_group_rows(self):
        collapsed = collapse_iqvia(parse_iqvia(self._raw()))
        _, recon = match_iqvia_to_sheet1(self._sheet1(), collapsed)
        group_rows = recon[recon["status"] != "din_no_iqvia_match"]
        for c in _MC:
            assert int(collapsed[c].sum()) == int(
                pd.to_numeric(group_rows[c], errors="coerce").fillna(0).sum()
            )

    def test_stamped_equals_matched_groups(self):
        collapsed = collapse_iqvia(parse_iqvia(self._raw()))
        enriched, recon = match_iqvia_to_sheet1(self._sheet1(), collapsed)
        matched = recon[recon["status"] == "matched"]
        for c in _MC:
            assert int(pd.to_numeric(matched[c], errors="coerce").fillna(0).sum()) == int(
                pd.to_numeric(enriched[c], errors="coerce").fillna(0).sum()
            )

    def test_full_chain_to_the_cent(self):
        raw = parse_iqvia(self._raw())
        collapsed = collapse_iqvia(raw)
        enriched, _ = match_iqvia_to_sheet1(self._sheet1(), collapsed)
        # Every raw dollar lands on exactly one DIN (both products are matched).
        assert int(raw["Dollars MAT 12/2025"].sum()) == 4277
        assert int(pd.to_numeric(enriched["Dollars MAT 12/2025"], errors="coerce").fillna(0).sum()) == 4277


class TestParseSilentZeroing:
    """parse_iqvia must not silently turn real, non-numeric sales cells into 0."""

    def test_thousands_separator_parsed(self):
        raw = parse_iqvia(_make_raw_iqvia([
            {"Combined Molecule": "X", "Product": "X", "Manufacturer": "ACME",
             "Strength": "10MG", "Dollars MAT 12/2025": "1,234,567", "Units MAT 12/2025": "1 200"},
        ], _MC))
        assert int(raw["Dollars MAT 12/2025"].iloc[0]) == 1234567
        assert int(raw["Units MAT 12/2025"].iloc[0]) == 1200

    def test_dash_and_blank_are_zero(self):
        raw = parse_iqvia(_make_raw_iqvia([
            {"Combined Molecule": "X", "Product": "X", "Manufacturer": "ACME",
             "Strength": "10MG", "Dollars MAT 12/2025": "-", "Units MAT 12/2025": ""},
        ], _MC))
        assert int(raw["Dollars MAT 12/2025"].iloc[0]) == 0
        assert int(raw["Units MAT 12/2025"].iloc[0]) == 0

    def test_unparseable_cell_raises_loudly(self):
        for bad in ["*", "<10", "N/A", "1.2K"]:
            raw_bytes = _make_raw_iqvia([
                {"Combined Molecule": "X", "Product": "X", "Manufacturer": "ACME",
                 "Strength": "10MG", "Dollars MAT 12/2025": bad, "Units MAT 12/2025": "5"},
            ], _MC)
            with pytest.raises(ValueError, match="non-numeric"):
                parse_iqvia(raw_bytes)


class TestCollapseMissingKeyLoud:
    """collapse_iqvia must fail loudly when a grouping key column is missing,
    rather than silently merging distinct products and overcounting."""

    def test_missing_manufacturer_raises(self):
        df = pd.DataFrame([
            {"Combined Molecule": "METFORMIN", "Product": "METFORMIN", "Strength": "500MG",
             "Dollars MAT 12/2025": 100},
            {"Combined Molecule": "METFORMIN", "Product": "METFORMIN", "Strength": "500MG",
             "Dollars MAT 12/2025": 200},
        ])  # no Manufacturer column
        with pytest.raises(ValueError, match="grouping column"):
            collapse_iqvia(df)

    def test_all_keys_present_collapses(self):
        df = pd.DataFrame([
            {"Combined Molecule": "METFORMIN", "Product": "METFORMIN",
             "Manufacturer": "ACME", "Strength": "500MG", "Dollars MAT 12/2025": 100},
            {"Combined Molecule": "METFORMIN", "Product": "METFORMIN",
             "Manufacturer": "ACME", "Strength": "500MG", "Dollars MAT 12/2025": 200},
        ])
        out = collapse_iqvia(df)
        assert len(out) == 1
        assert int(out["Dollars MAT 12/2025"].iloc[0]) == 300


class TestDuplicateDinAcrossBlocks:
    """A combination drug queried under multiple ingredients appears once per
    block (same DIN, multiple Sheet-1 rows). The IQVIA value must be stamped on
    exactly ONE row so a column-sum is not double-counted."""

    def _collapsed(self):
        return collapse_iqvia(parse_iqvia(_make_raw_iqvia([
            {"Combined Molecule": "HYDROCHLOROTHIAZIDE:VALSARTAN", "Product": "VALSARTAN HCT",
             "Manufacturer": "SANIS HEALTH INC", "Strength": "80MG/12.5MG",
             "Dollars MAT 12/2025": "5000", "Units MAT 12/2025": "50"},
        ], _MC)))

    def _sheet1_two_blocks(self):
        # Same DIN appears in the valsartan block AND the hydrochlorothiazide block.
        row = {"din": "02330482", "ingredient": "VALSARTAN; HYDROCHLOROTHIAZIDE",
               "brand_name": "VALSARTAN HCT", "company": "SANIS HEALTH INC",
               "strength": "80 MG; 12.5 MG", "status": "Marketed"}
        return pd.DataFrame([dict(row, block="valsartan"), dict(row, block="hydrochlorothiazide")])

    def test_value_stamped_once(self):
        enriched, _ = match_iqvia_to_sheet1(self._sheet1_two_blocks(), self._collapsed())
        rows = enriched[enriched["din"] == "02330482"]
        assert len(rows) == 2
        # Column-sum must equal the single group value, not double it.
        assert int(pd.to_numeric(rows["Dollars MAT 12/2025"], errors="coerce").fillna(0).sum()) == 5000
        # Exactly one of the two rows carries the value; the other is blank.
        carried = pd.to_numeric(rows["Dollars MAT 12/2025"], errors="coerce")
        assert carried.notna().sum() == 1

    def test_unique_din_unaffected(self):
        enriched, _ = match_iqvia_to_sheet1(self._sheet1_two_blocks().iloc[[0]], self._collapsed())
        assert int(enriched["Dollars MAT 12/2025"].iloc[0]) == 5000
