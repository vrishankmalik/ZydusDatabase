"""Tests for Change 1: wide patent columns, defensive split, zip-by-DIN parsing."""
from __future__ import annotations

import io
import zipfile
import csv

import pytest


# ── _split_merged_patent_number ───────────────────────────────────────────────

def test_split_merged_14_digit_token():
    from app.enrichment.patents import _split_merged_patent_number
    result = _split_merged_patent_number("26458103022097")
    assert result == ["2645810", "3022097"], f"Expected split, got: {result}"


def test_split_clean_7_digit_patent():
    from app.enrichment.patents import _split_merged_patent_number
    result = _split_merged_patent_number("2709025")
    assert result == ["2709025"]


def test_split_handles_ca_prefix():
    from app.enrichment.patents import _split_merged_patent_number
    # "CA 2709025" → cleaned to "2709025" → single token
    result = _split_merged_patent_number("CA 2709025")
    assert result == ["2709025"]


def test_split_empty_string():
    from app.enrichment.patents import _split_merged_patent_number
    result = _split_merged_patent_number("")
    assert result == []


# ── _parse_patent_zip_by_din ──────────────────────────────────────────────────

def _make_zip_csv(rows: list[dict], filename: str = "Patent.csv") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        csv_buf = io.StringIO()
        if rows:
            writer = csv.DictWriter(csv_buf, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        zf.writestr(filename, csv_buf.getvalue())
    return buf.getvalue()


def test_parse_patent_zip_by_din_basic():
    from app.enrichment.patents import _parse_patent_zip_by_din

    zip_bytes = _make_zip_csv([
        {"DIN": "02322285", "PATENT_NO": "2645810", "FILING_DATE": "2008-12-10"},
        {"DIN": "02322285", "PATENT_NO": "3022097", "FILING_DATE": "2015-06-01"},
        {"DIN": "02498014", "PATENT_NO": "2709025", "FILING_DATE": "2008-12-10"},
    ])
    result = _parse_patent_zip_by_din(zip_bytes)

    assert "02322285" in result
    assert set(result["02322285"]) == {"2645810", "3022097"}
    assert "02498014" in result
    assert result["02498014"] == ["2709025"]


def test_parse_patent_zip_by_din_pads_din_to_8_digits():
    from app.enrichment.patents import _parse_patent_zip_by_din

    zip_bytes = _make_zip_csv([
        {"DIN": "2322285", "PATENT_NO": "9999999"},  # 7-digit — should pad to "02322285"
    ])
    result = _parse_patent_zip_by_din(zip_bytes)
    assert "02322285" in result


def test_parse_patent_zip_by_din_defensive_split_on_merged():
    from app.enrichment.patents import _parse_patent_zip_by_din

    zip_bytes = _make_zip_csv([
        {"DIN": "02322285", "PATENT_NO": "26458103022097"},  # merged 14-digit
    ])
    result = _parse_patent_zip_by_din(zip_bytes)
    assert "02322285" in result
    assert set(result["02322285"]) == {"2645810", "3022097"}


def test_parse_patent_zip_by_din_empty_zip():
    from app.enrichment.patents import _parse_patent_zip_by_din

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("README.txt", "no csv here")
    result = _parse_patent_zip_by_din(buf.getvalue())
    assert result == {}


def test_parse_patent_zip_by_din_empty_bytes():
    from app.enrichment.patents import _parse_patent_zip_by_din
    assert _parse_patent_zip_by_din(b"") == {}


# ── _aggregate_patents_wide ───────────────────────────────────────────────────

def test_aggregate_wide_two_patents(tmp_path):
    import app.enrichment.store as store_mod
    store_mod.reset_for_testing(str(tmp_path / "enrich.db"))
    store_mod.upsert_patent("02322285", "2645810", "2000-01-01", "2005-01-01", "2020-01-01")
    store_mod.upsert_patent("02322285", "3022097", "2015-06-01", "2020-01-01", "2035-06-01")

    from app.enrichment.workbook import _aggregate_patents_wide
    wide = _aggregate_patents_wide("02322285", 2)

    assert wide["patent_count"] == 2
    numbers = {wide["patent_1_number"], wide["patent_2_number"]}
    assert numbers == {"2645810", "3022097"}, f"Unexpected patent numbers: {numbers}"
    # Each patent group must have its own dates
    for i in (1, 2):
        pn = wide[f"patent_{i}_number"]
        assert wide[f"patent_{i}_filing_date"] is not None, f"patent_{i} has no filing_date"
        assert wide[f"patent_{i}_expiry_date"] is not None, f"patent_{i} has no expiry_date"


def test_aggregate_wide_trailing_groups_blank(tmp_path):
    """DIN with 1 patent and M=3 → groups 2 and 3 all None."""
    import app.enrichment.store as store_mod
    store_mod.reset_for_testing(str(tmp_path / "enrich.db"))
    store_mod.upsert_patent("00000001", "1111111", "2000-01-01", "2005-01-01", "2020-01-01")

    from app.enrichment.workbook import _aggregate_patents_wide
    wide = _aggregate_patents_wide("00000001", 3)

    assert wide["patent_count"] == 1
    assert wide["patent_1_number"] == "1111111"
    assert wide["patent_2_number"] is None
    assert wide["patent_2_filing_date"] is None
    assert wide["patent_2_grant_date"] is None
    assert wide["patent_2_expiry_date"] is None
    assert wide["patent_3_number"] is None


def test_no_cross_group_bleed(tmp_path):
    """Patent columns for DIN-A must not appear in DIN-B's row."""
    import app.enrichment.store as store_mod
    store_mod.reset_for_testing(str(tmp_path / "enrich.db"))
    store_mod.upsert_patent("00000001", "1111111", "2000-01-01", "2005-01-01", "2020-01-01")
    store_mod.upsert_patent("00000002", "9999999", "2010-01-01", "2015-01-01", "2030-01-01")

    from app.enrichment.workbook import _aggregate_patents_wide
    wide_a = _aggregate_patents_wide("00000001", 1)
    wide_b = _aggregate_patents_wide("00000002", 1)

    assert wide_a["patent_1_number"] == "1111111"
    assert wide_b["patent_1_number"] == "9999999"
    assert wide_a["patent_1_number"] != wide_b["patent_1_number"]


def test_zero_patent_din_gives_patent_count_0(tmp_path):
    import app.enrichment.store as store_mod
    store_mod.reset_for_testing(str(tmp_path / "enrich.db"))

    from app.enrichment.workbook import _aggregate_patents_wide
    wide = _aggregate_patents_wide("99999999", 1)

    assert wide["patent_count"] == 0
    assert wide["patent_1_number"] is None


# ── build_sheet1 uses wide columns ────────────────────────────────────────────

def test_wide_columns_in_sheet1(tmp_path):
    """build_sheet1 uses patent_N_* columns; old merged columns absent."""
    import app.enrichment.store as store_mod
    store_mod.reset_for_testing(str(tmp_path / "enrich.db"))
    store_mod.upsert_patent("02498014", "2709025", "2008-12-10", "2014-08-26", "2028-12-10")
    store_mod.upsert_patent("02498014", "3022097", "2015-01-01", "2020-03-01", "2035-01-01")

    from tests.test_build_workbook import _dpd, _make_response
    from app.enrichment.workbook import build_sheet1

    response = _make_response(dpd_records=[_dpd("02498014")])
    df = build_sheet1(response)

    assert "patent_1_number" in df.columns
    assert "patent_2_number" in df.columns
    assert "patent_numbers" not in df.columns
    assert "all_patents_detail" not in df.columns
    # Count should match
    assert df.loc[0, "patent_count"] == 2


def test_m_computed_as_max_across_dins(tmp_path):
    """M is the global max: DIN with fewer patents gets trailing blank groups."""
    import app.enrichment.store as store_mod
    store_mod.reset_for_testing(str(tmp_path / "enrich.db"))
    # DIN-A: 2 patents
    store_mod.upsert_patent("00000001", "1111111", "2000-01-01", "2005-01-01", "2020-01-01")
    store_mod.upsert_patent("00000001", "2222222", "2010-01-01", "2015-01-01", "2030-01-01")
    # DIN-B: 1 patent
    store_mod.upsert_patent("00000002", "3333333", "2012-01-01", "2016-01-01", "2032-01-01")

    from tests.test_build_workbook import _dpd, _make_response
    from app.enrichment.workbook import build_sheet1

    response = _make_response(dpd_records=[_dpd("00000001"), _dpd("00000002")])
    df = build_sheet1(response)

    # M = 2 → both DINs have patent_2_* columns
    assert "patent_2_number" in df.columns

    row_a = df[df["din"] == "00000001"].iloc[0]
    row_b = df[df["din"] == "00000002"].iloc[0]

    assert row_a["patent_count"] == 2
    assert row_b["patent_count"] == 1
    # DIN-B's second group is blank
    assert row_b["patent_2_number"] is None or str(row_b["patent_2_number"]) in ("None", "nan", "")


# ── Change 2: no *_url or *_page columns ─────────────────────────────────────

def test_columns_no_url_or_page(tmp_path):
    """Sheet 1 must not contain any column whose name ends in _url or _page."""
    import app.enrichment.store as store_mod
    store_mod.reset_for_testing(str(tmp_path / "enrich.db"))

    from tests.test_build_workbook import _dpd, _noc, _make_response
    from app.enrichment.workbook import build_sheet1

    response = _make_response(dpd_records=[_dpd("02498014")], noc_records=[_noc("02498014")])
    df = build_sheet1(response)

    for col in df.columns:
        assert not col.endswith("_url"), f"URL column should not appear in output: {col!r}"
        assert not col.endswith("_page"), f"Page citation column should not appear in output: {col!r}"


def test_drug_code_and_needs_ocr_present(tmp_path):
    """_drug_code and needs_ocr must always be present in Sheet 1."""
    import app.enrichment.store as store_mod
    store_mod.reset_for_testing(str(tmp_path / "enrich.db"))

    from tests.test_build_workbook import _dpd, _make_response
    from app.enrichment.workbook import build_sheet1

    response = _make_response(dpd_records=[_dpd("02498014")])
    df = build_sheet1(response)

    assert "_drug_code" in df.columns, "_drug_code must be kept in output"
    assert "needs_ocr" in df.columns, "needs_ocr must be kept in output"


def test_no_patent_numbers_cell_exceeds_8_chars(tmp_path):
    """Every patent_N_number cell must be ≤ 8 characters (no merged 14-digit tokens)."""
    import app.enrichment.store as store_mod
    store_mod.reset_for_testing(str(tmp_path / "enrich.db"))
    store_mod.upsert_patent("02322285", "2645810", "2000-01-01", "2005-01-01", "2020-01-01")
    store_mod.upsert_patent("02322285", "3022097", "2015-01-01", "2020-01-01", "2035-01-01")

    from tests.test_build_workbook import _dpd, _make_response
    from app.enrichment.workbook import build_sheet1

    response = _make_response(dpd_records=[_dpd("02322285")])
    df = build_sheet1(response)

    number_cols = [c for c in df.columns if c.endswith("_number") and c.startswith("patent_")]
    for col in number_cols:
        for val in df[col].dropna():
            assert len(str(val)) <= 8, f"Patent number {val!r} exceeds 8 chars in column {col}"
