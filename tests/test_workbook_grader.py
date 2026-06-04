"""Self-tests for the pure helper functions in grade_workbook.py.

These tests exercise only workbook-reading logic (no network, no xlsx).
Run with:  python -m pytest tests/test_workbook_grader.py -v
"""
from __future__ import annotations

import io
import os
import sys
import datetime

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from grade_workbook import (
    # sentinels / predicates
    NOT_IN_PM, NO_PM_AVAILABLE, NEEDS_OCR, PH_SOLUBILITY_ONLY, NO_NOC_RECORD,
    NA_UNCOATED, _is_sentinel, _is_real, _is_no_pm,
    # fuzzy helper
    _fuzzy_contains,
    # stage detection
    detect_stages,
    # Family 1 helpers
    check_excipient_field,
    check_pack_style,
    check_pack_size,
    check_size_mm,
    check_ph,
    check_preservatives,
    check_colour,
    check_shape,
    check_noc_consistency,
    check_patent_count,
    check_column_names,
    # Family runners
    run_family1,
    run_family2,
    # Sample selector
    select_sample,
    # OCR probe helpers
    _make_synthetic_scanned_pdf,
    _run_synthetic_ocr_probe,
)


# ─── _is_sentinel ─────────────────────────────────────────────────────────────

class TestIsSentinel:
    def test_none_is_sentinel(self):
        assert _is_sentinel(None)

    def test_empty_string(self):
        assert _is_sentinel("")

    def test_not_in_pm(self):
        assert _is_sentinel(NOT_IN_PM)

    def test_no_pm_available(self):
        assert _is_sentinel(NO_PM_AVAILABLE)

    def test_no_noc_record(self):
        assert _is_sentinel(NO_NOC_RECORD)

    def test_ph_solubility_sentinel(self):
        # NOT a sentinel — it conveys real information
        assert not _is_sentinel(PH_SOLUBILITY_ONLY)

    def test_real_value_not_sentinel(self):
        assert not _is_sentinel("microcrystalline cellulose")

    def test_whitespace_is_sentinel(self):
        assert _is_sentinel("   ")

    def test_nan_string(self):
        assert _is_sentinel("nan")


# ─── _fuzzy_contains ──────────────────────────────────────────────────────────

class TestFuzzyContains:
    def test_exact_match(self):
        found, score = _fuzzy_contains("cellulose", "microcrystalline cellulose")
        assert found and score == 1.0

    def test_case_insensitive_exact(self):
        found, score = _fuzzy_contains("Cellulose", "microcrystalline cellulose")
        assert found and score == 1.0

    def test_ocr_noise_typo(self):
        # "rnagnesium" is a common tesseract misread of "magnesium"
        found, score = _fuzzy_contains("magnesium stearate", "rnagnesium stearate", threshold=0.9)
        assert found and score >= 0.9

    def test_completely_absent(self):
        found, score = _fuzzy_contains("titanium dioxide", "microcrystalline cellulose")
        assert not found

    def test_short_token_skipped(self):
        found, _ = _fuzzy_contains("ab", "ab cd ef")
        assert not found  # too short, returns False

    def test_empty_token(self):
        found, _ = _fuzzy_contains("", "some text")
        assert not found

    def test_threshold_respected(self):
        # "zzzzzz" is totally unlike "magnesium"
        found, _ = _fuzzy_contains("zzzzzzzzzz", "magnesium stearate", threshold=0.9)
        assert not found


# ─── check_excipient_field ────────────────────────────────────────────────────

class TestCheckExcipientField:
    def test_clean_excipient_no_findings(self):
        assert check_excipient_field("12345678", "excipients_core",
                                     "microcrystalline cellulose, stearic acid") == []

    def test_sentinel_no_findings(self):
        assert check_excipient_field("12345678", "excipients_core", NOT_IN_PM) == []

    def test_debossed_flagged(self):
        findings = check_excipient_field("12345678", "excipients_core",
                                         "White debossed tablets")
        assert any(f.check_id == "F1_EXCIPIENT_POISON" for f in findings)

    def test_administration_form_flagged(self):
        findings = check_excipient_field("12345678", "excipients_core",
                                         "Administration Form/Strength")
        assert any(f.check_id == "F1_EXCIPIENT_POISON" for f in findings)

    def test_heading_fragment_colon(self):
        findings = check_excipient_field("12345678", "excipients_core",
                                         "Non-Medicinal Ingredients:")
        assert any(f.check_id == "F1_EXCIPIENT_HEADING_FRAG" for f in findings)

    def test_severity_is_error(self):
        findings = check_excipient_field("12345678", "excipients_core",
                                         "Dosage Form and Strength")
        assert all(f.severity == "ERROR" for f in findings)


# ─── check_pack_style ─────────────────────────────────────────────────────────

class TestCheckPackStyle:
    def test_valid_bottle_no_findings(self):
        assert check_pack_style("12345678", "Bottle") == []

    def test_valid_blister_pack(self):
        assert check_pack_style("12345678", "Blister Pack") == []

    def test_sentinel_no_findings(self):
        assert check_pack_style("12345678", NOT_IN_PM) == []

    def test_ends_in_colon_flagged(self):
        findings = check_pack_style("12345678", "Packaging:")
        assert any(f.check_id == "F1_PACK_STYLE_COLON" for f in findings)

    def test_heading_phrase_flagged(self):
        findings = check_pack_style("12345678", "the following dosage strengths")
        assert any(f.check_id == "F1_PACK_STYLE_HDR_TEXT" for f in findings)

    def test_no_container_word_warn(self):
        findings = check_pack_style("12345678", "round white tablet")
        assert any(f.check_id == "F1_PACK_STYLE_NO_VOCAB" and f.severity == "WARN"
                   for f in findings)


# ─── check_pack_size ──────────────────────────────────────────────────────────

class TestCheckPackSize:
    def test_count_no_findings(self):
        assert check_pack_size("12345678", "100 count") == []

    def test_volume_no_findings(self):
        assert check_pack_size("12345678", "5 mL") == []

    def test_sentinel_no_findings(self):
        assert check_pack_size("12345678", NOT_IN_PM) == []

    def test_vial_in_pack_size_flagged(self):
        findings = check_pack_size("12345678", "5 mL Vial")
        assert any(f.check_id == "F1_PACK_SIZE_CONTAINER" for f in findings)

    def test_bottle_in_pack_size_flagged(self):
        findings = check_pack_size("12345678", "100 count Bottle")
        assert any(f.check_id == "F1_PACK_SIZE_CONTAINER" for f in findings)


# ─── check_size_mm ────────────────────────────────────────────────────────────

class TestCheckSizeMm:
    def test_valid_single_dim(self):
        assert check_size_mm("12345678", "9.5 mm") == []

    def test_valid_cross_dim(self):
        assert check_size_mm("12345678", "11 × 7 mm") == []

    def test_sentinel_no_findings(self):
        assert check_size_mm("12345678", NOT_IN_PM) == []

    def test_raw_section_text_flagged(self):
        findings = check_size_mm("12345678", "6. DOSAGE FORMS, COMPOSITION AND PACKAGING")
        assert any(f.check_id == "F1_SIZE_MM_FORMAT" for f in findings)

    def test_out_of_range_warns(self):
        findings = check_size_mm("12345678", "50 mm")
        assert any(f.check_id == "F1_SIZE_MM_RANGE" and f.severity == "WARN"
                   for f in findings)

    def test_too_small_warns(self):
        findings = check_size_mm("12345678", "1 mm")
        assert any(f.check_id == "F1_SIZE_MM_RANGE" for f in findings)

    def test_x_notation_valid(self):
        assert check_size_mm("12345678", "12x7 mm") == []


# ─── check_ph ─────────────────────────────────────────────────────────────────

class TestCheckPh:
    def test_valid_single(self):
        assert check_ph("12345678", "6.8") == []

    def test_valid_range(self):
        assert check_ph("12345678", "4.5–7.0") == []

    def test_sentinel_ok(self):
        assert check_ph("12345678", NOT_IN_PM) == []

    def test_ph_solubility_only_ok(self):
        assert check_ph("12345678", PH_SOLUBILITY_ONLY) == []

    def test_out_of_range_error(self):
        findings = check_ph("12345678", "15")
        assert any(f.check_id == "F1_PH_RANGE" and f.severity == "ERROR" for f in findings)

    def test_text_not_numeric_error(self):
        findings = check_ph("12345678", "pH approximately neutral")
        assert any(f.check_id == "F1_PH_FORMAT" for f in findings)

    def test_negative_ph_error(self):
        findings = check_ph("12345678", "-1")
        assert any(f.check_id in ("F1_PH_FORMAT", "F1_PH_RANGE") for f in findings)


# ─── check_preservatives ─────────────────────────────────────────────────────

class TestCheckPreservatives:
    def test_y_ok(self):
        assert check_preservatives("12345678", "Y") == []

    def test_n_ok(self):
        assert check_preservatives("12345678", "N") == []

    def test_sentinel_ok(self):
        assert check_preservatives("12345678", NOT_IN_PM) == []

    def test_free_text_flagged(self):
        findings = check_preservatives("12345678", "benzalkonium chloride")
        assert any(f.check_id == "F1_PRESERVATIVES_VALUE" for f in findings)

    def test_yes_flagged(self):
        findings = check_preservatives("12345678", "Yes")
        assert any(f.check_id == "F1_PRESERVATIVES_VALUE" for f in findings)


# ─── check_colour ─────────────────────────────────────────────────────────────

class TestCheckColour:
    def test_white_ok(self):
        assert check_colour("12345678", "white") == []

    def test_light_blue_ok(self):
        assert check_colour("12345678", "light blue") == []

    def test_pale_yellow_ok(self):
        assert check_colour("12345678", "pale yellow") == []

    def test_sentinel_ok(self):
        assert check_colour("12345678", NOT_IN_PM) == []

    def test_novel_colour_warns(self):
        findings = check_colour("12345678", "cerulean")
        assert any(f.check_id == "F1_COLOUR_NOVEL" and f.severity == "WARN"
                   for f in findings)

    def test_multi_colour_one_novel(self):
        findings = check_colour("12345678", "white, cerulean")
        novel = [f for f in findings if f.check_id == "F1_COLOUR_NOVEL"]
        assert len(novel) >= 1


# ─── check_shape ─────────────────────────────────────────────────────────────

class TestCheckShape:
    def test_round_ok(self):
        assert check_shape("12345678", "round") == []

    def test_oblong_ok(self):
        assert check_shape("12345678", "oblong") == []

    def test_biconvex_ok(self):
        assert check_shape("12345678", "biconvex") == []

    def test_sentinel_ok(self):
        assert check_shape("12345678", NOT_IN_PM) == []

    def test_novel_shape_warns(self):
        findings = check_shape("12345678", "heptagonal")
        assert any(f.check_id == "F1_SHAPE_NOVEL" and f.severity == "WARN"
                   for f in findings)

    def test_caplet_ok(self):
        assert check_shape("12345678", "caplet") == []


# ─── check_noc_consistency (revised) ─────────────────────────────────────────

class TestCheckNocConsistency:
    def _full_noc(self, sub_type: str = "NDS") -> dict:
        return {
            "noc_brand_name":        "PIQRAY",
            "noc_company":           "Novartis",
            "noc_date":              "2020-01-01",
            "noc_submission_type":   sub_type,
            "noc_therapeutic_class": "Antineoplastic",
        }

    def _no_noc(self) -> dict:
        return {c: NO_NOC_RECORD for c in (
            "noc_brand_name", "noc_company", "noc_date",
            "noc_submission_type", "noc_therapeutic_class",
        )}

    def test_all_real_nds_ok(self):
        assert check_noc_consistency("12345678", self._full_noc("NDS")) == []

    def test_all_real_ands_ok(self):
        assert check_noc_consistency("12345678", self._full_noc("ANDS")) == []

    def test_all_no_noc_record_ok(self):
        assert check_noc_consistency("12345678", self._no_noc()) == []

    def test_blank_company_not_inconsistent_error(self):
        # Blank string ≠ "No NOC record" sentinel, so NOT an ERROR
        row = self._full_noc()
        row["noc_company"] = ""
        findings = check_noc_consistency("12345678", row)
        assert not any(f.check_id == "F1_NOC_INCONSISTENT" for f in findings)

    def test_mixed_no_noc_and_real_is_error(self):
        # brand + company real, date + sub_type = "No NOC record" → ERROR
        row = {
            "noc_brand_name":        "PIQRAY",
            "noc_company":           "Novartis",
            "noc_date":              NO_NOC_RECORD,
            "noc_submission_type":   NO_NOC_RECORD,
            "noc_therapeutic_class": NOT_IN_PM,
        }
        findings = check_noc_consistency("12345678", row)
        assert any(f.check_id == "F1_NOC_INCONSISTENT" and f.severity == "ERROR"
                   for f in findings)

    def test_tclass_missing_is_info_not_error(self):
        # All 4 core fields populated, therapeutic_class blank → INFO
        row = self._full_noc()
        row["noc_therapeutic_class"] = ""
        findings = check_noc_consistency("12345678", row)
        info = [f for f in findings if f.check_id == "F1_NOC_TCLASS_MISSING"]
        assert info, "Expected INFO finding for missing tclass"
        assert info[0].severity == "INFO"
        errors = [f for f in findings if f.severity == "ERROR"]
        assert not errors, f"No ERROR should fire for blank tclass: {errors}"

    def test_tclass_missing_no_error(self):
        row = self._full_noc()
        row["noc_therapeutic_class"] = NOT_IN_PM
        errors = [f for f in check_noc_consistency("12345678", row) if f.severity == "ERROR"]
        assert not errors

    def test_snds_flagged(self):
        findings = check_noc_consistency("12345678", self._full_noc("SNDS"))
        assert any(f.check_id == "F1_NOC_SUB_TYPE" for f in findings)

    def test_sands_flagged(self):
        findings = check_noc_consistency("12345678", self._full_noc("SANDS"))
        assert any(f.check_id == "F1_NOC_SUB_TYPE" for f in findings)


# ─── check_patent_count ──────────────────────────────────────────────────────

class TestCheckPatentCount:
    def test_count_matches_ok(self):
        row  = {"patent_count": 2, "patent_1_number": "2709025", "patent_2_number": "2845123"}
        cols = ["patent_1_number", "patent_2_number"]
        assert check_patent_count("12345678", row, cols) == []

    def test_count_mismatch_error(self):
        row  = {"patent_count": 3, "patent_1_number": "2709025", "patent_2_number": "2845123"}
        cols = ["patent_1_number", "patent_2_number"]
        findings = check_patent_count("12345678", row, cols)
        assert any(f.check_id == "F1_PATENT_COUNT_MISMATCH" for f in findings)

    def test_no_patents_zero_ok(self):
        assert check_patent_count("12345678", {"patent_count": 0}, []) == []

    def test_long_patent_number_warns(self):
        row  = {"patent_count": 1, "patent_1_number": "CA 2645810 3022097"}
        cols = ["patent_1_number"]
        findings = check_patent_count("12345678", row, cols)
        assert any(f.check_id == "F1_PATENT_NUMBER_LEN" for f in findings)

    def test_normal_7digit_ok(self):
        row  = {"patent_count": 1, "patent_1_number": "2709025"}
        cols = ["patent_1_number"]
        assert check_patent_count("12345678", row, cols) == []


# ─── check_column_names ──────────────────────────────────────────────────────

class TestCheckColumnNames:
    def test_clean_columns_ok(self):
        assert check_column_names(["din", "brand_name", "patent_1_number", "colour"]) == []

    def test_url_column_flagged(self):
        findings = check_column_names(["din", "record_url"])
        assert any(f.check_id == "F1_COL_URL" for f in findings)

    def test_page_column_flagged(self):
        findings = check_column_names(["din", "active_ingredient_page"])
        assert any(f.check_id == "F1_COL_PAGE" for f in findings)

    def test_us_spelling_flagged(self):
        findings = check_column_names(["din", "color"])
        assert any(f.check_id == "F1_COL_SPELLING" for f in findings)

    def test_colour_ok(self):
        assert check_column_names(["colour"]) == []


# ─── Stage detection ─────────────────────────────────────────────────────────

def _stage_df(**col_vals) -> pd.DataFrame:
    """Build a minimal one-row DataFrame for stage detection tests."""
    defaults: dict = {
        "din":                 "12345678",
        "active_ingredient":   "",
        "excipients_core":     "",
        "noc_brand_name":      NO_NOC_RECORD,
        "noc_company":         NO_NOC_RECORD,
        "noc_date":            NO_NOC_RECORD,
        "noc_submission_type": NO_NOC_RECORD,
        "patent_count":        "0",
        "data_protection_ends": "",
    }
    defaults.update(col_vals)
    return pd.DataFrame([defaults])


class TestDetectStages:
    def test_patents_only_labeling_false(self):
        df = _stage_df(
            patent_1_number="2709025",
            noc_brand_name=NO_NOC_RECORD,
            active_ingredient="",
            excipients_core="",
        )
        stages = detect_stages(df)
        assert stages["PATENTS"] is True
        assert stages["LABELING"] is False
        assert stages["NOC"] is False

    def test_full_workbook_all_stages(self):
        df = _stage_df(
            active_ingredient="alpelisib",
            excipients_core="microcrystalline cellulose",
            noc_brand_name="PIQRAY",
            noc_company="Novartis",
            noc_date="2020-01-01",
            noc_submission_type="NDS",
            patent_1_number="2709025",
            data_protection_ends="2030-01-01",
        )
        stages = detect_stages(df)
        assert stages["LABELING"] is True
        assert stages["PATENTS"] is True
        assert stages["NOC"] is True
        assert stages["DP"] is True

    def test_all_no_noc_record_noc_false(self):
        df = _stage_df(
            noc_brand_name=NO_NOC_RECORD,
            noc_company=NO_NOC_RECORD,
            noc_date=NO_NOC_RECORD,
            noc_submission_type=NO_NOC_RECORD,
        )
        stages = detect_stages(df)
        assert stages["NOC"] is False

    def test_no_stages_empty_workbook(self):
        df = _stage_df()
        stages = detect_stages(df)
        assert stages["LABELING"] is False
        assert stages["PATENTS"] is False
        assert stages["NOC"] is False
        assert stages["DP"] is False


# ─── Family 1: stage-aware gating ────────────────────────────────────────────

class TestFamily1StageAware:
    def _df(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "din": "12345678",
            "excipients_core": "Debossed tablet heading:",  # would trigger F1_EXCIPIENT_POISON + HEADING_FRAG
            "excipients_coating": NOT_IN_PM,
            "pack_style": NOT_IN_PM,
            "pack_size": NOT_IN_PM,
            "size_mm": NOT_IN_PM,
            "ph": NOT_IN_PM,
            "preservatives": NOT_IN_PM,
            "colour": NOT_IN_PM,
            "shape": NOT_IN_PM,
            "noc_brand_name": NO_NOC_RECORD,
            "noc_company": NO_NOC_RECORD,
            "noc_date": NO_NOC_RECORD,
            "noc_submission_type": NO_NOC_RECORD,
            "noc_therapeutic_class": NO_NOC_RECORD,
            "patent_count": 0,
        }])

    def test_labeling_active_flags_excipient_poison(self):
        df = self._df()
        findings = run_family1(df, stages={"LABELING": True})
        assert any(f.check_id == "F1_EXCIPIENT_POISON" for f in findings)

    def test_labeling_absent_skips_excipient_checks(self):
        df = self._df()
        findings = run_family1(df, stages={"LABELING": False})
        labeling_checks = {
            "F1_EXCIPIENT_POISON", "F1_EXCIPIENT_HEADING_FRAG", "F1_PACK_STYLE_COLON",
            "F1_PACK_STYLE_HDR_TEXT", "F1_PACK_STYLE_NO_VOCAB", "F1_PACK_SIZE_CONTAINER",
            "F1_SIZE_MM_FORMAT", "F1_PH_FORMAT", "F1_PRESERVATIVES_VALUE",
            "F1_COLOUR_NOVEL", "F1_SHAPE_NOVEL",
        }
        assert not any(f.check_id in labeling_checks for f in findings)

    def test_noc_patent_checks_always_run(self):
        # Even with LABELING=False, patent count mismatch should still fire
        df = pd.DataFrame([{
            "din": "12345678",
            "noc_brand_name": NO_NOC_RECORD,
            "noc_company": NO_NOC_RECORD,
            "noc_date": NO_NOC_RECORD,
            "noc_submission_type": NO_NOC_RECORD,
            "noc_therapeutic_class": NO_NOC_RECORD,
            "patent_count": "3",
            "patent_1_number": "2709025",
            # patent_2 and _3 absent
        }])
        findings = run_family1(df, stages={"LABELING": False})
        assert any(f.check_id == "F1_PATENT_COUNT_MISMATCH" for f in findings)


# ─── Family 2 coherence helpers ──────────────────────────────────────────────

def _make_df(**kwargs) -> pd.DataFrame:
    defaults = {
        "din": "12345678",
        "_drug_code": "99001",
        "ingredient": "alpelisib",
        "dosage_form": "Tablet",
        "active_ingredient": "alpelisib",
        "excipients_core": NOT_IN_PM,
        "excipients_coating": NOT_IN_PM,
        "preservatives": NOT_IN_PM,
        "ph": NOT_IN_PM,
        "colour": NOT_IN_PM,
        "shape": NOT_IN_PM,
        "size_mm": NOT_IN_PM,
        "weight": NOT_IN_PM,
        "noc_brand_name": NO_NOC_RECORD,
        "noc_company": NO_NOC_RECORD,
        "noc_date": NO_NOC_RECORD,
        "noc_submission_type": NO_NOC_RECORD,
        "noc_therapeutic_class": NO_NOC_RECORD,
        "patent_count": 0,
        "data_protection_ends": NOT_IN_PM,
    }
    defaults.update(kwargs)
    return pd.DataFrame([defaults])


class TestFamily2Coherence:
    def test_clean_row_no_findings(self):
        df = _make_df()
        assert run_family2(df) == []

    def test_blank_active_ingredient_error_when_labeling(self):
        df = _make_df(active_ingredient=NOT_IN_PM)
        findings = run_family2(df, stages={"LABELING": True})
        assert any(f.check_id == "F2_AI_BLANK" for f in findings)

    def test_blank_active_ingredient_no_labeling_not_flagged(self):
        df = _make_df(active_ingredient=NOT_IN_PM)
        findings = run_family2(df, stages={"LABELING": False})
        assert not any(f.check_id == "F2_AI_BLANK" for f in findings)

    def test_blank_active_ingredient_no_drug_code_not_flagged(self):
        # F2_AI_BLANK requires _drug_code to be present
        df = _make_df(active_ingredient=NOT_IN_PM, _drug_code="")
        findings = run_family2(df, stages={"LABELING": True})
        assert not any(f.check_id == "F2_AI_BLANK" for f in findings)

    def test_pm_mixed_sentinel_error(self):
        df = _make_df(
            excipients_core="microcrystalline cellulose",
            excipients_coating=NO_PM_AVAILABLE,
            preservatives=NO_PM_AVAILABLE,
        )
        findings = run_family2(df, stages={"LABELING": True})
        assert any(f.check_id == "F2_PM_MIXED_SENTINEL" for f in findings)

    def test_pm_mixed_sentinel_skipped_without_labeling(self):
        df = _make_df(
            excipients_core="microcrystalline cellulose",
            excipients_coating=NO_PM_AVAILABLE,
        )
        findings = run_family2(df, stages={"LABELING": False})
        assert not any(f.check_id == "F2_PM_MIXED_SENTINEL" for f in findings)

    def test_excipient_no_appearance_warn(self):
        df = _make_df(
            excipients_core="microcrystalline cellulose, stearic acid",
            colour=NOT_IN_PM,
            shape=NOT_IN_PM,
        )
        findings = run_family2(df, stages={"LABELING": True})
        assert any(f.check_id == "F2_EXCIPIENT_NO_APPEARANCE" for f in findings)

    def test_weight_no_size_warn(self):
        df = _make_df(weight="250 mg", size_mm=NOT_IN_PM)
        findings = run_family2(df, stages={"LABELING": True})
        assert any(f.check_id == "F2_WEIGHT_NO_SIZE" for f in findings)

    def test_size_no_weight_warn(self):
        df = _make_df(size_mm="9 mm", weight=NOT_IN_PM)
        findings = run_family2(df, stages={"LABELING": True})
        assert any(f.check_id == "F2_SIZE_NO_WEIGHT" for f in findings)

    def test_coating_no_core_error(self):
        df = _make_df(
            excipients_core=NOT_IN_PM,
            excipients_coating="hydroxypropyl methylcellulose, titanium dioxide",
        )
        findings = run_family2(df, stages={"LABELING": True})
        assert any(f.check_id == "F2_COATING_NO_CORE" for f in findings)

    def test_dp_past_date_error(self):
        df = _make_df(data_protection_ends="2010-01-01")
        findings = run_family2(df)
        assert any(f.check_id == "F2_DP_PAST_DATE" for f in findings)

    def test_dp_future_date_ok(self):
        future = (datetime.date.today() + datetime.timedelta(days=365)).isoformat()
        df = _make_df(data_protection_ends=future)
        findings = run_family2(df)
        assert not any(f.check_id == "F2_DP_PAST_DATE" for f in findings)

    def test_liquid_no_pres_no_ph_warn(self):
        df = _make_df(
            dosage_form="Solution for injection",
            preservatives="N",
            ph=NOT_IN_PM,
        )
        findings = run_family2(df, stages={"LABELING": True})
        assert any(f.check_id == "F2_LIQUID_NO_PH" for f in findings)

    def test_dp_past_date_fires_even_without_labeling(self):
        # DP check is not gated on LABELING stage
        df = _make_df(data_protection_ends="2010-01-01")
        findings = run_family2(df, stages={"LABELING": False})
        assert any(f.check_id == "F2_DP_PAST_DATE" for f in findings)


# ─── Synthetic OCR probe ─────────────────────────────────────────────────────

class TestSyntheticOcrProbe:
    def test_synthetic_pdf_has_no_text_layer(self):
        """PIL creates a true image-only PDF with zero selectable characters."""
        import pdfplumber
        pdf_bytes = _make_synthetic_scanned_pdf()
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdoc:
            chars = sum(len(pg.extract_text() or "") for pg in pdoc.pages)
        assert chars < 50, f"Expected image-only PDF but got {chars} text-layer chars"

    def test_synthetic_pdf_is_nonempty(self):
        """Synthetic PDF is a valid non-trivial file."""
        assert len(_make_synthetic_scanned_pdf()) > 1000

    def test_synthetic_probe_never_returns_error(self):
        """
        Probe returns INFO on pass, WARN/SKIPPED if OCR unavailable — never ERROR.
        (ERROR would mean the probe itself is broken, not just unverified.)
        """
        findings = _run_synthetic_ocr_probe()
        assert findings, "Must produce at least one finding"
        errors = [f for f in findings if f.severity == "ERROR"]
        assert not errors, f"Synthetic probe returned unexpected ERRORs: {errors}"

    def test_synthetic_probe_all_family4(self):
        """All probe findings belong to Family 4."""
        findings = _run_synthetic_ocr_probe()
        assert all(f.family == 4 for f in findings)

    def test_synthetic_probe_pass_or_skipped_when_ocr_available(self):
        """
        When OCR runs successfully, probe produces F4_OCR_SYNTH_PASS (INFO).
        If OCR unavailable in this environment, findings are SKIPPED — that is OK.
        """
        findings = _run_synthetic_ocr_probe()
        skipped  = all(f.severity == "SKIPPED" for f in findings)
        if skipped:
            pytest.skip("OCR not available in this environment (tesseract/pdf2image)")
        passes = [f for f in findings if f.check_id == "F4_OCR_SYNTH_PASS"]
        assert passes, (
            f"Expected F4_OCR_SYNTH_PASS when OCR is available, got: "
            f"{[(f.check_id, f.severity, f.workbook_value) for f in findings]}"
        )
        assert passes[0].severity == "INFO"


# ─── OCR fuzzy matching in anti-hallucination ────────────────────────────────

class TestOcrFuzzyInExcipientCheck:
    """Unit tests for _fuzzy_contains behaviour that backs the OCR-aware check."""

    def test_exact_match_returns_1_0(self):
        found, score = _fuzzy_contains("magnesium stearate",
                                       "excipients: magnesium stearate, cellulose")
        assert found and score == 1.0

    def test_ocr_noise_above_threshold(self):
        # "rnagnesium stearate" — common tesseract mis-read
        found, score = _fuzzy_contains("magnesium stearate",
                                       "rnagnesium stearate, cellulose",
                                       threshold=0.9)
        assert found, f"Expected fuzzy match, score={score:.3f}"
        assert score >= 0.9

    def test_completely_different_below_threshold(self):
        found, score = _fuzzy_contains("magnesium stearate",
                                       "lactose monohydrate, povidone K30",
                                       threshold=0.9)
        assert not found

    def test_short_tokens_skipped(self):
        # < 3 chars → always returns False (avoid false positives)
        found, _ = _fuzzy_contains("mg", "magnesium stearate")
        assert not found

    def test_exact_beats_fuzzy_threshold(self):
        # Even with threshold=0.99, exact match returns True
        found, score = _fuzzy_contains("cellulose", "microcrystalline cellulose",
                                       threshold=0.99)
        assert found and score == 1.0


# ─── select_sample ────────────────────────────────────────────────────────────

class TestSelectSample:
    def _df(self, n: int) -> pd.DataFrame:
        rows = []
        for i in range(n):
            rows.append({
                "din": f"{10000000 + i}",
                "ingredient": "alpelisib",
                "dosage_form": "Tablet" if i % 5 != 0 else "Solution",
                "noc_submission_type": ["NDS", "ANDS", NO_NOC_RECORD][i % 3],
                "data_protection_ends": NOT_IN_PM,
                "patent_count": 0,
            })
        return pd.DataFrame(rows)

    def test_returns_all_when_smaller(self):
        df     = self._df(10)
        sample = select_sample(df, 40)
        assert len(sample) == 10

    def test_caps_at_n(self):
        df     = self._df(200)
        sample = select_sample(df, 40)
        assert len(sample) == 40

    def test_unique_dins(self):
        df     = self._df(100)
        sample = select_sample(df, 40)
        assert len(sample) == len(set(sample))
