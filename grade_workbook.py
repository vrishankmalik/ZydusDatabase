#!/usr/bin/env python3
"""Non-gating workbook accuracy harness for the Canadian drug database scraper.

Reads Sheet 1 ("DPD + NOC + Patents") from an enriched .xlsx, grades values
across four check families, writes checks_report.md + checks_report.csv
alongside the workbook, and prints a summary to stdout. Exit code is always 0.

Families:
  1 – Field invariants       (all rows, workbook-only reads)
  2 – Cross-field coherence  (all rows, workbook-only reads)
  3 – Anti-hallucination     (live source re-fetch on stratified sample)
  4 – Determinism + OCR      (re-run extraction on sample; OCR liveness probe)

Severity levels:  ERROR | WARN | INFO | SKIPPED

Usage:
  python grade_workbook.py path/to/workbook.xlsx [--sample N] [--cache-dir DIR]

Live endpoint overrides (env vars):
  GRADER_DPD_BASE       GRADER_DPD_INFO_URL
  GRADER_CPD_BASE       GRADER_REGISTER_URL
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import datetime
import difflib
import hashlib
import io
import json
import logging
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import httpx
import pandas as pd

# ─── Sentinel catalogue (from app/enrichment/labeling.py + workbook.py) ───────

NOT_IN_PM          = "Not in PM"
NO_PM_AVAILABLE    = "No PM available"
NEEDS_OCR          = "needs OCR / manual check"
PH_SOLUBILITY_ONLY = "Not stated (pH-dependent solubility only)"
NO_NOC_RECORD      = "No NOC record"
NA_UNCOATED        = "N/A (uncoated)"

_SENTINEL_LOWER = frozenset({
    "", "not in pm", "no pm available", "needs ocr / manual check",
    "no noc record", "n/a", "na", "none", "not applicable", "nan",
})


def _is_sentinel(v: Any) -> bool:
    if v is None:
        return True
    return str(v).strip().lower() in _SENTINEL_LOWER


def _is_real(v: Any) -> bool:
    return not _is_sentinel(v)


def _is_no_pm(v: Any) -> bool:
    return str(v).strip() == NO_PM_AVAILABLE if v is not None else False


# ─── Stage detection ──────────────────────────────────────────────────────────

def detect_stages(df: pd.DataFrame) -> dict[str, bool]:
    """
    Detect which enrichment stages ran by inspecting column fill-rates.
    Does NOT rely on column existence alone — checks actual populated rows.

    LABELING present  := active_ingredient OR any PM field non-empty in >0 rows
    PATENTS present   := any patent_N_number non-empty in >0 rows
    NOC present       := any noc_* (other than 'No NOC record') non-empty
    DP present        := data_protection_ends non-empty in >0 rows
    """
    cols = set(df.columns)

    def _any_real_in(col_names: list[str]) -> bool:
        for col in col_names:
            if col in cols:
                for v in df[col]:
                    sv = str(v).strip()
                    if sv and sv.lower() not in {"nan", "no noc record"} and _is_real(sv):
                        return True
        return False

    labeling_cols = [
        "active_ingredient", "excipients_core", "excipients_coating",
        "preservatives", "ph", "colour", "shape", "size_mm", "weight",
    ]
    patent_cols = [c for c in cols if re.match(r"patent_\d+_number$", c)]
    noc_data_cols = ["noc_brand_name", "noc_company", "noc_date", "noc_submission_type"]

    noc_present = False
    for col in noc_data_cols:
        if col in cols:
            for v in df[col]:
                sv = str(v).strip()
                if _is_real(sv) and sv != NO_NOC_RECORD:
                    noc_present = True
                    break
        if noc_present:
            break

    return {
        "LABELING": _any_real_in(labeling_cols),
        "PATENTS":  _any_real_in(list(patent_cols)),
        "NOC":      noc_present,
        "DP":       _any_real_in(["data_protection_ends"]),
    }


# ─── Coverage Ledger ──────────────────────────────────────────────────────────

@dataclass
class _LedgerEntry:
    check_id: str
    family: int
    n_eligible: int = 0
    n_executed: int = 0
    n_skipped: int = 0

    @property
    def coverage_pct(self) -> float:
        if self.n_eligible == 0:
            return 0.0
        return 100.0 * self.n_executed / self.n_eligible


class CoverageLedger:
    """Tracks per-check-id eligibility/execution/skip counts for the report."""

    def __init__(self) -> None:
        self._d: dict[str, _LedgerEntry] = {}

    def add(self, check_id: str, family: int, *,
            eligible: int = 0, executed: int = 0, skipped: int = 0) -> None:
        if check_id not in self._d:
            self._d[check_id] = _LedgerEntry(check_id, family)
        e = self._d[check_id]
        e.n_eligible += eligible
        e.n_executed += executed
        e.n_skipped  += skipped

    def rows(self) -> list[_LedgerEntry]:
        return sorted(self._d.values(), key=lambda r: (r.family, r.check_id))


# ─── Fuzzy matching helper ────────────────────────────────────────────────────

def _fuzzy_contains(token: str, text: str, threshold: float = 0.9) -> tuple[bool, float]:
    """
    Return (found, best_ratio).
    Exact substring first; then sliding-window SequenceMatcher.

    Strategy: always probe positions 0..stride-1 individually (so single-char
    offsets are never skipped), then stride through the remainder.  This catches
    OCR prefix insertions (e.g. 'rnagnesium' for 'magnesium') without scanning
    the full text at stride=1.
    """
    tl = token.lower().strip()
    textl = text.lower()
    if not tl or len(tl) < 3:
        return False, 0.0
    if tl in textl:
        return True, 1.0
    n = len(tl)
    max_start = max(0, len(textl) - n)
    if max_start == 0:
        return False, 0.0
    stride = max(1, n // 4)
    best  = 0.0

    def _check(i: int) -> float:
        return difflib.SequenceMatcher(None, tl, textl[i : i + n]).ratio()

    # Probe first stride positions one-by-one (catches near-start best windows)
    for i in range(min(stride, max_start + 1)):
        s = _check(i)
        if s > best:
            best = s
        if best >= threshold:
            return True, best

    # Stride through remainder
    for i in range(stride, max_start + 1, stride):
        s = _check(i)
        if s > best:
            best = s
        if best >= threshold:
            return True, best

    return best >= threshold, best


# ─── Vocabulary ───────────────────────────────────────────────────────────────

_CONTAINER_KEYWORDS: frozenset[str] = frozenset({
    "prefilled syringe", "pre-filled syringe", "auto-injector", "auto injector",
    "autoinjector", "stick pack", "blister pack", "ampoule", "ampule", "ampul",
    "vial", "syringe", "cartridge", "blister", "bottle", "jar", "tube",
    "sachet", "pouch", "carton", "bag", "canister", "inhaler", "dropper",
    "suppository", "pen", "kit",
})

_KNOWN_COLOURS: frozenset[str] = frozenset({
    "white", "red", "pink", "orange", "yellow", "green", "blue", "purple",
    "violet", "brown", "beige", "grey", "gray", "black", "cream", "tan",
    "teal", "maroon", "ivory",
})
_COLOUR_MODIFIERS: frozenset[str] = frozenset({"light", "pale", "dark", "bright", "deep", "off"})

_KNOWN_SHAPES: frozenset[str] = frozenset({
    "round", "oval", "ovaloid", "oblong", "capsule-shaped", "capsule shaped",
    "caplet", "biconvex", "pentagonal", "hexagonal", "octagonal", "triangular",
    "diamond", "shield", "kidney", "bean",
})

_NOC_DATA_COLS = ["noc_brand_name", "noc_company", "noc_date", "noc_submission_type"]
_NOC_COLS = _NOC_DATA_COLS + ["noc_therapeutic_class"]

# Patterns that must NOT appear in excipient fields
_EXCIPIENT_POISON_RE = re.compile(
    r"debossed|[Aa]dministration\s+[Ff]orm|[Ff]orm\s*/\s*[Ss]trength|"
    r"[Aa]dministration\s+[Ss]trength|tablets?\s+with\s+['\"]|"
    r"[Ss]trength\s+and\s+[Dd]osage|[Dd]osage\s+[Ff]orm",
    re.IGNORECASE,
)
_HEADING_FRAG_RE   = re.compile(r":\s*$")
_PACK_STYLE_HDR_RE = re.compile(
    r"the following dosage strengths|dosage strengths|^\s*dosage\s+form",
    re.IGNORECASE,
)
_SIZE_MM_RE    = re.compile(
    r"^\d+(?:\.\d+)?(?:\s*[×xX×]\s*\d+(?:\.\d+)?)?\s*mm$", re.IGNORECASE
)
_SIZE_RANGE_RE = re.compile(r"^\d+(?:\.\d+)?\s*[-–]\s*\d+(?:\.\d+)?\s*mm$", re.IGNORECASE)
_PH_NUM_RE     = re.compile(r"^\d+(?:\.\d+)?(?:\s*[-–]\s*\d+(?:\.\d+)?)?$")

# ─── Finding ──────────────────────────────────────────────────────────────────

@dataclass
class Finding:
    check_id: str
    din: str
    field: str
    severity: str        # ERROR | WARN | INFO | SKIPPED
    workbook_value: str
    reason: str
    family: int = 0


# ─── Family 1: Invariants ─────────────────────────────────────────────────────

def check_excipient_field(din: str, field: str, val: Any) -> list[Finding]:
    if _is_sentinel(val):
        return []
    s = str(val).strip()
    out: list[Finding] = []
    if _EXCIPIENT_POISON_RE.search(s):
        out.append(Finding("F1_EXCIPIENT_POISON", din, field, "ERROR", s[:120],
                           "Excipient field contains dosage/appearance/heading wording", 1))
    if _HEADING_FRAG_RE.search(s):
        out.append(Finding("F1_EXCIPIENT_HEADING_FRAG", din, field, "ERROR", s[:120],
                           "Excipient field ends in ':' — heading fragment leaked in", 1))
    return out


def check_pack_style(din: str, val: Any) -> list[Finding]:
    if _is_sentinel(val):
        return []
    s = str(val).strip()
    out: list[Finding] = []
    if _HEADING_FRAG_RE.search(s):
        out.append(Finding("F1_PACK_STYLE_COLON", din, "pack_style", "ERROR", s[:120],
                           "pack_style ends in ':'", 1))
    if _PACK_STYLE_HDR_RE.search(s):
        out.append(Finding("F1_PACK_STYLE_HDR_TEXT", din, "pack_style", "ERROR", s[:120],
                           "pack_style equals a section-heading phrase", 1))
    sl = s.lower()
    if not any(kw in sl for kw in _CONTAINER_KEYWORDS):
        out.append(Finding("F1_PACK_STYLE_NO_VOCAB", din, "pack_style", "WARN", s[:120],
                           "pack_style has no container-vocab keyword", 1))
    return out


def check_pack_size(din: str, val: Any) -> list[Finding]:
    if _is_sentinel(val):
        return []
    s  = str(val).strip()
    sl = s.lower()
    for kw in _CONTAINER_KEYWORDS:
        if re.search(r"\b" + re.escape(kw) + r"\b", sl):
            return [Finding("F1_PACK_SIZE_CONTAINER", din, "pack_size", "ERROR", s[:120],
                            f"pack_size contains container word '{kw}' — should be number+unit only", 1)]
    return []


def check_size_mm(din: str, val: Any) -> list[Finding]:
    if _is_sentinel(val):
        return []
    s = str(val).strip()
    if not (_SIZE_MM_RE.match(s) or _SIZE_RANGE_RE.match(s)):
        return [Finding("F1_SIZE_MM_FORMAT", din, "size_mm", "ERROR", s[:80],
                        "size_mm does not parse to N mm or N×N mm", 1)]
    nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", s)]
    out: list[Finding] = []
    for v in nums:
        if v < 2.0 or v > 30.0:
            out.append(Finding("F1_SIZE_MM_RANGE", din, "size_mm", "WARN", s,
                               f"size_mm value {v} mm outside expected 2–30 mm for solid dosage", 1))
    return out


def check_ph(din: str, val: Any) -> list[Finding]:
    if _is_sentinel(val) or str(val).strip() == PH_SOLUBILITY_ONLY:
        return []
    s = str(val).strip()
    if not _PH_NUM_RE.match(s):
        return [Finding("F1_PH_FORMAT", din, "ph", "ERROR", s[:80],
                        "ph is not a numeric 0–14 value, range, or documented-absence sentinel", 1)]
    out: list[Finding] = []
    for n in re.findall(r"\d+(?:\.\d+)?", s):
        v = float(n)
        if v < 0 or v > 14:
            out.append(Finding("F1_PH_RANGE", din, "ph", "ERROR", s,
                               f"pH value {v} is outside 0–14", 1))
    return out


def check_preservatives(din: str, val: Any) -> list[Finding]:
    if _is_sentinel(val):
        return []
    s = str(val).strip()
    if s not in ("Y", "N"):
        return [Finding("F1_PRESERVATIVES_VALUE", din, "preservatives", "ERROR", s[:80],
                        "preservatives must be 'Y', 'N', or a sentinel — got something else", 1)]
    return []


def check_colour(din: str, val: Any) -> list[Finding]:
    """Each comma/slash-separated part must contain at least one known colour word."""
    if _is_sentinel(val):
        return []
    s     = str(val).strip().lower()
    parts = [p.strip() for p in re.split(r"[,/]+", s) if p.strip()]
    out: list[Finding] = []
    for part in parts:
        if not any(re.search(r"\b" + re.escape(c) + r"\b", part) for c in _KNOWN_COLOURS):
            out.append(Finding("F1_COLOUR_NOVEL", din, "colour", "WARN", str(val)[:80],
                               f"Colour part '{part}' contains no known colour vocab word", 1))
    return out


def check_shape(din: str, val: Any) -> list[Finding]:
    if _is_sentinel(val):
        return []
    s = str(val).strip().lower()
    if not any(shape in s for shape in _KNOWN_SHAPES):
        return [Finding("F1_SHAPE_NOVEL", din, "shape", "WARN", str(val)[:80],
                        "Shape value not in known shape synonym vocab", 1)]
    return []


def check_noc_consistency(din: str, row: dict) -> list[Finding]:
    """
    ERROR: some noc_* data fields have real values while others say 'No NOC record'
           (mixed sentinel + real value in the same row).
    INFO:  noc_therapeutic_class absent while all four core NOC fields are populated
           (expected — NOC source often omits therapeutic class).
    """
    out: list[Finding] = []

    real_cols   = [c for c in _NOC_DATA_COLS
                   if _is_real(row.get(c)) and str(row.get(c, "")).strip() != NO_NOC_RECORD]
    no_noc_cols = [c for c in _NOC_DATA_COLS
                   if str(row.get(c, "")).strip() == NO_NOC_RECORD]

    if real_cols and no_noc_cols:
        out.append(Finding("F1_NOC_INCONSISTENT", din, "noc_*", "ERROR",
                           f"real={real_cols}",
                           f"Mixed 'No NOC record' + real values in same row: "
                           f"real={real_cols}, no_noc={no_noc_cols}", 1))

    if len(real_cols) == len(_NOC_DATA_COLS) and not no_noc_cols:
        tclass = row.get("noc_therapeutic_class")
        if _is_sentinel(tclass):
            out.append(Finding("F1_NOC_TCLASS_MISSING", din, "noc_therapeutic_class", "INFO",
                               str(tclass),
                               "noc_therapeutic_class absent while all four core NOC fields are "
                               "populated (common — NOC source often omits therapeutic class)", 1))

    sub = str(row.get("noc_submission_type") or "").strip()
    _NOC_VALID_SUBS = {
        "NDS", "ANDS", NO_NOC_RECORD,
        "New Drug Submission (NDS)",
        "Abbreviated New Drug Submission (ANDS)",
    }
    if sub and sub not in _NOC_VALID_SUBS and not _is_sentinel(sub):
        out.append(Finding("F1_NOC_SUB_TYPE", din, "noc_submission_type", "ERROR", sub[:80],
                           "noc_submission_type should be NDS/ANDS (or full text equivalent) "
                           "or 'No NOC record' (SNDS/SANDS supplemental submissions are filtered)", 1))

    return out


def check_patent_count(din: str, row: dict, patent_cols: list[str]) -> list[Finding]:
    raw = row.get("patent_count")
    if not _is_real(raw):
        return []
    try:
        declared = int(float(str(raw)))
    except ValueError:
        return []
    actual = sum(
        1 for c in patent_cols
        if re.match(r"patent_\d+_number$", c) and _is_real(row.get(c))
    )
    out: list[Finding] = []
    if declared != actual:
        out.append(Finding("F1_PATENT_COUNT_MISMATCH", din, "patent_count", "ERROR",
                           str(declared),
                           f"patent_count={declared} but {actual} patent_N_number columns populated", 1))
    for col in patent_cols:
        if not re.match(r"patent_\d+_number$", col):
            continue
        pn = row.get(col)
        if not _is_real(pn):
            continue
        clean = re.sub(r"(?i)^\s*ca\s*", "", str(pn).strip())
        clean = re.sub(r"[,\s]+", "", clean)
        if len(clean) > 8:
            out.append(Finding("F1_PATENT_NUMBER_LEN", din, col, "WARN", str(pn)[:20],
                               f"Patent number '{clean}' exceeds 8 chars (merged or malformed?)", 1))
    return out


def check_column_names(columns: list[str]) -> list[Finding]:
    out: list[Finding] = []
    for col in columns:
        if col.endswith("_url"):
            out.append(Finding("F1_COL_URL", "(header)", col, "ERROR", col,
                               "_url columns removed in Change 2; must not appear in Sheet 1", 1))
        if col.endswith("_page"):
            out.append(Finding("F1_COL_PAGE", "(header)", col, "ERROR", col,
                               "_page citation columns removed in Change 2; must not appear", 1))
        if "color" in col.lower() and "colour" not in col.lower():
            out.append(Finding("F1_COL_SPELLING", "(header)", col, "ERROR", col,
                               "Column uses 'color' (US); must be 'colour' (CA)", 1))
    return out


def run_family1(
    df: pd.DataFrame,
    stages: dict[str, bool] | None = None,
) -> list[Finding]:
    """
    Run all Field-Invariant checks.

    When stages["LABELING"] is False, PM-field and active-ingredient checks are
    skipped entirely (not emitted as ERROR). NOC and patent checks always run.
    """
    if stages is None:
        stages = {}
    labeling_active = stages.get("LABELING", True)

    out: list[Finding] = []
    cols         = list(df.columns)
    patent_cols  = [c for c in cols if c.startswith("patent_")]

    out.extend(check_column_names(cols))

    for _, row in df.iterrows():
        din   = str(row.get("din", "")).strip()
        rdict = dict(row)

        if labeling_active:
            out.extend(check_excipient_field(din, "excipients_core",    rdict.get("excipients_core")))
            out.extend(check_excipient_field(din, "excipients_coating", rdict.get("excipients_coating")))
            out.extend(check_pack_style(din, rdict.get("pack_style")))
            out.extend(check_pack_size(din,  rdict.get("pack_size")))
            out.extend(check_size_mm(din,    rdict.get("size_mm")))
            out.extend(check_ph(din,         rdict.get("ph")))
            out.extend(check_preservatives(din, rdict.get("preservatives")))
            out.extend(check_colour(din,     rdict.get("colour")))
            out.extend(check_shape(din,      rdict.get("shape")))

        out.extend(check_noc_consistency(din, rdict))
        out.extend(check_patent_count(din, rdict, patent_cols))

    return out


# ─── Family 2: Cross-field coherence ─────────────────────────────────────────

_PM_FIELDS = frozenset({
    "excipients_core", "excipients_coating", "preservatives",
    "ph", "colour", "shape", "size_mm", "weight",
})
_LIQUID_FORMS = frozenset({
    "solution", "liquid", "injection", "infusion",
    "suspension", "syrup", "elixir", "oral drops",
})


def run_family2(
    df: pd.DataFrame,
    stages: dict[str, bool] | None = None,
) -> list[Finding]:
    """
    Cross-field coherence checks.

    PM-field checks (F2_AI_BLANK, F2_PM_MIXED_SENTINEL, etc.) are only run when
    stages["LABELING"] is True (default).  F2_DP_PAST_DATE always runs.
    """
    if stages is None:
        stages = {}
    labeling_active = stages.get("LABELING", True)

    out: list[Finding] = []
    today = datetime.date.today()

    for _, row in df.iterrows():
        din   = str(row.get("din", "")).strip()
        rdict = dict(row)

        if labeling_active:
            ai = rdict.get("active_ingredient")
            dc = rdict.get("_drug_code")
            if din and _is_real(str(dc) if dc is not None else "") and _is_sentinel(ai):
                out.append(Finding("F2_AI_BLANK", din, "active_ingredient", "ERROR", str(ai),
                                   "active_ingredient is blank/sentinel for a DIN row with a drug_code "
                                   "(Tier-A field)", 2))

            pm_real  = {f for f in _PM_FIELDS if _is_real(rdict.get(f)) and not _is_no_pm(rdict.get(f))}
            pm_no_pm = {f for f in _PM_FIELDS if _is_no_pm(rdict.get(f))}
            if pm_real and pm_no_pm:
                out.append(Finding("F2_PM_MIXED_SENTINEL", din,
                                   ", ".join(sorted(pm_no_pm)), "ERROR",
                                   f"real={sorted(pm_real)}",
                                   "PM demonstrably exists: some fields populated, others say "
                                   "'No PM available'", 2))

            exc_real   = (_is_real(rdict.get("excipients_core")) and
                          not _is_no_pm(rdict.get("excipients_core")))
            clr_absent = _is_sentinel(rdict.get("colour")) or _is_no_pm(rdict.get("colour"))
            shp_absent = _is_sentinel(rdict.get("shape"))  or _is_no_pm(rdict.get("shape"))
            if exc_real and clr_absent and shp_absent:
                out.append(Finding("F2_EXCIPIENT_NO_APPEARANCE", din, "colour,shape", "WARN",
                                   str(rdict.get("excipients_core"))[:60],
                                   "Excipients populated but both colour and shape absent — "
                                   "same §6 section", 2))

            wt_real   = _is_real(rdict.get("weight"))  and not _is_no_pm(rdict.get("weight"))
            sz_real   = _is_real(rdict.get("size_mm")) and not _is_no_pm(rdict.get("size_mm"))
            wt_absent = _is_sentinel(rdict.get("weight"))  or _is_no_pm(rdict.get("weight"))
            sz_absent = _is_sentinel(rdict.get("size_mm")) or _is_no_pm(rdict.get("size_mm"))
            if wt_real and sz_absent:
                out.append(Finding("F2_WEIGHT_NO_SIZE", din, "size_mm", "WARN",
                                   f"weight={rdict.get('weight')}",
                                   "weight present but size_mm absent — same physical descriptor block", 2))
            if sz_real and wt_absent:
                out.append(Finding("F2_SIZE_NO_WEIGHT", din, "weight", "WARN",
                                   f"size_mm={rdict.get('size_mm')}",
                                   "size_mm present but weight absent — same physical descriptor block", 2))

            dosage   = str(rdict.get("dosage_form") or "").lower()
            is_liq   = any(w in dosage for w in _LIQUID_FORMS)
            pres_val = str(rdict.get("preservatives") or "").strip()
            ph_val   = rdict.get("ph")
            if is_liq and pres_val == "N" and _is_sentinel(ph_val):
                out.append(Finding("F2_LIQUID_NO_PH", din, "ph", "WARN",
                                   f"dosage_form={rdict.get('dosage_form')}, preservatives=N",
                                   "Liquid dosage form with preservatives=N and no pH — both expected", 2))

            core_val  = rdict.get("excipients_core")
            coat_val  = rdict.get("excipients_coating")
            coat_real = (_is_real(coat_val) and not _is_no_pm(coat_val)
                         and str(coat_val).strip() != NA_UNCOATED)
            core_absent = _is_sentinel(core_val) or _is_no_pm(core_val)
            if coat_real and core_absent:
                out.append(Finding("F2_COATING_NO_CORE", din, "excipients_core", "ERROR",
                                   str(coat_val)[:60],
                                   "excipients_coating populated but excipients_core absent — "
                                   "must appear as a pair", 2))

        dp_ends = rdict.get("data_protection_ends")
        if dp_ends and _is_real(dp_ends):
            dp_s = str(dp_ends).strip()
            for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%B %d, %Y", "%d %B %Y"):
                try:
                    dt = datetime.datetime.strptime(dp_s, fmt).date()
                    if dt < today:
                        out.append(Finding("F2_DP_PAST_DATE", din, "data_protection_ends",
                                           "ERROR", dp_s,
                                           f"data_protection_ends={dp_s} is past today — "
                                           "active table must have future dates", 2))
                    break
                except ValueError:
                    continue

    return out


# ─── Disk cache ───────────────────────────────────────────────────────────────

class _DiskCache:
    def __init__(self, cache_dir: str) -> None:
        self.dir = Path(cache_dir) / "grade_workbook_cache"
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, ns: str, key: str) -> Path:
        h = hashlib.sha256(f"{ns}:{key}".encode()).hexdigest()[:16]
        return self.dir / f"{ns}_{h}.json"

    def get(self, ns: str, key: str) -> Any:
        p = self._path(ns, key)
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                return None
        return None

    def set(self, ns: str, key: str, value: Any) -> None:
        self._path(ns, key).write_text(json.dumps(value, default=str))


# ─── Live endpoint config (overridable via env for proxy/network testing) ─────

_DPD_BASE     = "https://health-products.canada.ca/api/drug"
_DPD_INFO_URL = "https://health-products.canada.ca/dpd-bdpp/info"
_CPD_BASE     = "https://brevets-patents.ic.gc.ca/opic-cipo/cpd/eng/patent"
_REGISTER_URL = (
    "https://www.canada.ca/en/health-canada/services/drugs-health-products"
    "/drug-products/applications-submissions/register-innovative-drugs.html"
)

def _live(key: str) -> str:
    env_key = f"GRADER_{key}"
    defaults = {
        "DPD_BASE":     _DPD_BASE,
        "DPD_INFO_URL": _DPD_INFO_URL,
        "CPD_BASE":     _CPD_BASE,
        "REGISTER_URL": _REGISTER_URL,
    }
    return os.environ.get(env_key, defaults[key])

_UA = "Mozilla/5.0 (compatible; WorkbookGrader/1.0)"


# ─── HTTP helpers ─────────────────────────────────────────────────────────────

async def _http_get_retry(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: dict | None = None,
    headers: dict | None = None,
    timeout: float = 20.0,
    max_retries: int = 3,
) -> Optional[httpx.Response]:
    """GET with exponential-backoff retry on transport errors. Does NOT retry 4xx."""
    for attempt in range(max_retries):
        try:
            r = await client.get(url, params=params or {}, headers=headers or {},
                                 timeout=timeout)
            if r.status_code < 500:
                return r
        except (httpx.TransportError, httpx.TimeoutException):
            pass
        except Exception:
            pass
        if attempt < max_retries - 1:
            await asyncio.sleep(2 ** attempt)
    return None


async def _get_json(client: httpx.AsyncClient, url: str, params: dict) -> Any:
    r = await _http_get_retry(client, url, params=params,
                              headers={"User-Agent": _UA, "Accept": "application/json"})
    if r is None or r.status_code != 200:
        return None
    try:
        return r.json()
    except Exception:
        return None


# ─── Family 3 helpers ─────────────────────────────────────────────────────────

async def _check_active_ingredient(
    client: httpx.AsyncClient, cache: _DiskCache,
    din: str, drug_code: Any, wb_val: Any,
) -> list[Finding]:
    if not _is_real(str(drug_code)):
        return []
    dc     = str(drug_code).strip()
    cached = cache.get("dpd_ai", dc)
    if cached is None:
        data = await _get_json(client, f"{_live('DPD_BASE')}/activeingredient/",
                               {"id": dc, "lang": "en", "type": "json"})
        if data is None:
            return [Finding("F3_AI_SKIPPED", din, "active_ingredient", "SKIPPED",
                            str(wb_val), "DPD API unreachable", 3)]
        entries = data if isinstance(data, list) else ([data] if isinstance(data, dict) and data else [])
        live = set()
        for e in entries:
            if isinstance(e, dict):
                n = (e.get("ingredient_name") or e.get("active_ingredient_name")
                     or e.get("ingredientName") or "")
                if n:
                    live.add(n.strip().upper())
        cache.set("dpd_ai", dc, list(live))
        cached = list(live)
    live_set = {x.upper() for x in cached}
    if not live_set or _is_sentinel(wb_val):
        return []
    wb_set   = {x.strip().upper() for x in re.split(r"[;,]+", str(wb_val)) if x.strip()}
    invented = wb_set - live_set
    if invented:
        return [Finding("F3_AI_MISMATCH", din, "active_ingredient", "ERROR",
                        str(wb_val)[:120],
                        f"Workbook ingredient(s) {invented} not in DPD live API (live={live_set})", 3)]
    return []


async def _check_pack_live(
    client: httpx.AsyncClient, cache: _DiskCache,
    din: str, drug_code: Any, wb_size: Any, wb_style: Any,
) -> list[Finding]:
    if not _is_real(str(drug_code)):
        return []
    dc     = str(drug_code).strip()
    cached = cache.get("dpd_pkg", dc)
    if cached is None:
        data = await _get_json(client, f"{_live('DPD_BASE')}/packaging/",
                               {"id": dc, "type": "json"})
        if data is None:
            return [Finding("F3_PKG_SKIPPED", din, "pack_size", "SKIPPED",
                            str(wb_size), "DPD packaging API unreachable", 3)]
        entries    = data if isinstance(data, list) else ([data] if isinstance(data, dict) and data else [])
        prod_infos = [str(e.get("product_information") or "") for e in entries if isinstance(e, dict)]
        cache.set("dpd_pkg", dc, prod_infos)
        cached = prod_infos
    combined = " ".join(cached).upper()
    out: list[Finding] = []

    if _is_real(wb_size):
        for n in re.findall(r"\d+(?:\.\d+)?", str(wb_size)):
            int_part = n.split(".")[0]
            if n not in combined and int_part not in combined:
                out.append(Finding("F3_PACK_SIZE_INVENTED", din, "pack_size", "ERROR",
                                   str(wb_size)[:80],
                                   f"Number '{n}' from workbook pack_size absent in DPD product_information", 3))

    if _is_real(wb_style):
        sl = str(wb_style).lower()
        if (any(kw in sl for kw in _CONTAINER_KEYWORDS) and
                not any(kw.upper() in combined for kw in _CONTAINER_KEYWORDS)):
            out.append(Finding("F3_PACK_STYLE_INVENTED", din, "pack_style", "WARN",
                               str(wb_style)[:80],
                               "Workbook pack_style container word absent in DPD product_information", 3))
    return out


async def _get_pdf_text(
    client: httpx.AsyncClient, cache: _DiskCache, dc: str
) -> Optional[str]:
    """Fetch DPD info page → PDF URL → PDF text (pdfplumber, no OCR)."""
    cached = cache.get("pdf_text", dc)
    if cached is not None:
        return cached or None

    html = cache.get("dpd_info_html", dc)
    if html is None:
        r = await _http_get_retry(client, _live("DPD_INFO_URL"),
                                  params={"lang": "eng", "code": dc},
                                  headers={"User-Agent": _UA, "Accept": "text/html"})
        html = r.text if (r and r.status_code == 200) else ""
        cache.set("dpd_info_html", dc, html)
    if not html:
        return None

    from bs4 import BeautifulSoup
    soup    = BeautifulSoup(html, "html.parser")
    pdf_url: Optional[str] = None
    for link in soup.find_all("a", href=True):
        href: str = link["href"]
        if href.lower().endswith(".pdf") or "pdf.hres.ca" in href.lower():
            if not href.startswith("http"):
                href = f"https://health-products.canada.ca{href}"
            pdf_url = href
            break
    if not pdf_url:
        cache.set("pdf_text", dc, "")
        return None

    pdf_cached = cache.get("pdf_text_by_url", pdf_url)
    if pdf_cached is not None:
        cache.set("pdf_text", dc, pdf_cached)
        return pdf_cached or None

    r = await _http_get_retry(client, pdf_url, headers={"User-Agent": _UA}, timeout=60.0)
    if r is None or r.status_code != 200:
        cache.set("pdf_text", dc, "")
        return None

    try:
        import pdfplumber
        parts: list[str] = []
        with pdfplumber.open(io.BytesIO(r.content)) as pdf:
            for page in pdf.pages:
                parts.append(page.extract_text() or "")
        text = "\n".join(parts)
    except Exception:
        text = ""

    cache.set("pdf_text_by_url", pdf_url, text)
    cache.set("pdf_text", dc, text)
    return text if text else None


async def _check_excipients_pdf(
    client: httpx.AsyncClient, cache: _DiskCache,
    din: str, drug_code: Any, wb_core: Any, wb_coat: Any,
    needs_ocr: bool = False,
) -> list[Finding]:
    """
    Anti-hallucination check for excipient fields.

    When needs_ocr=True (workbook row has needs_ocr=1), uses fuzzy matching
    (≥ 0.9) to tolerate OCR noise.  Fuzzy-but-not-exact matches are INFO, not ERROR.
    On text-layer pages uses exact substring match.
    """
    has_real = ((_is_real(wb_core) and not _is_no_pm(wb_core)) or
                (_is_real(wb_coat) and not _is_no_pm(wb_coat)))
    if not has_real or not _is_real(str(drug_code)):
        return []
    dc       = str(drug_code).strip()
    pdf_text = await _get_pdf_text(client, cache, dc)
    if pdf_text is None:
        return [Finding("F3_PDF_SKIPPED", din, "excipients", "SKIPPED", "",
                        "Could not fetch/parse PM PDF for anti-hallucination check", 3)]
    pdf_lower = pdf_text.lower()
    out: list[Finding] = []
    for field_name, val in (("excipients_core", wb_core), ("excipients_coating", wb_coat)):
        if not _is_real(val) or _is_no_pm(val) or str(val).strip() in (NA_UNCOATED, ""):
            continue
        for token in re.split(r"[,;]\s*", str(val).strip()):
            token = token.strip()
            if len(token) < 3:
                continue
            if needs_ocr:
                found, score = _fuzzy_contains(token, pdf_lower, threshold=0.9)
                if not found:
                    out.append(Finding("F3_EXCIPIENT_FABRICATED", din, field_name, "ERROR",
                                       token[:60],
                                       f"Excipient '{token}' not in PM PDF (fuzzy<0.9, "
                                       f"score={score:.2f}, OCR page)", 3))
                elif score < 1.0:
                    out.append(Finding("F3_EXCIPIENT_OCR_FUZZY", din, field_name, "INFO",
                                       token[:60],
                                       f"Excipient '{token}' matched via fuzzy (score={score:.2f}) — "
                                       "OCR noise tolerance applied", 3))
            else:
                if token.lower() not in pdf_lower:
                    out.append(Finding("F3_EXCIPIENT_FABRICATED", din, field_name, "ERROR",
                                       token[:60],
                                       f"Excipient token '{token}' not found verbatim "
                                       "(case-insensitive) in PM PDF", 3))
    return out


async def _check_patents_cpd(
    client: httpx.AsyncClient, cache: _DiskCache,
    din: str, patent_numbers: list[str],
) -> list[Finding]:
    out: list[Finding] = []
    for pn_raw in patent_numbers:
        pn = re.sub(r"(?i)^\s*ca\s*", "", str(pn_raw).strip())
        pn = re.sub(r"[,\s]+", "", pn)
        if not pn:
            continue
        cached = cache.get("cpd", pn)
        if cached is None:
            url = f"{_live('CPD_BASE')}/{pn}/summary.html"
            r   = await _http_get_retry(client, url, headers={"User-Agent": _UA})
            if r is None:
                out.append(Finding("F3_PATENT_SKIPPED", din, "patent_number", "SKIPPED",
                                   pn_raw, "CPD domain unreachable after retries", 3))
                continue
            status    = r.status_code
            text_html = r.text if status == 200 else ""
            has_filed = bool(re.search(r"\(22\)\s*Filed|Filing Date", text_html, re.IGNORECASE))
            cached    = {"status": status, "has_filed": has_filed}
            cache.set("cpd", pn, cached)
        if cached["status"] != 200 or not cached["has_filed"]:
            out.append(Finding("F3_GHOST_PATENT", din, "patent_number", "ERROR",
                               pn_raw,
                               f"Patent {pn} → HTTP {cached['status']} or no filing date on CPD (ghost)", 3))
    return out


async def _check_dp_live(
    client: httpx.AsyncClient, cache: _DiskCache,
    din: str, ingredient: Any, company: Any, dp_ends: Any,
) -> list[Finding]:
    if _is_sentinel(dp_ends) or not _is_real(dp_ends):
        return []
    if not ingredient or not company:
        return []

    cached = cache.get("dp_table_rows", "active")
    if cached is None:
        r = await _http_get_retry(client, _live("REGISTER_URL"),
                                  headers={"User-Agent": _UA, "Accept": "text/html"},
                                  timeout=30.0)
        html = r.text if (r and r.status_code == 200) else ""
        if not html:
            return [Finding("F3_DP_SKIPPED", din, "data_protection_ends", "SKIPPED",
                            str(dp_ends), "Register of Innovative Drugs unreachable", 3)]
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        rows: list[list[str]] = []
        for tbl in soup.find_all("table"):
            hdrs = [th.get_text(" ", strip=True).lower() for th in tbl.find_all("th")]
            if any("medicinal" in h for h in hdrs):
                for tr in tbl.find_all("tr"):
                    cells = tr.find_all("td")
                    if cells:
                        rows.append([c.get_text(" ", strip=True) for c in cells])
                break
        cache.set("dp_table_rows", "active", rows)
        cached = rows

    def _n(s: str) -> str:
        s = s.lower().strip()
        s = re.sub(r"\b(inc|ltd|llc|corp|corporation|gmbh|limited|canada|pharmaceuticals|"
                   r"pharmaceutical|pharma|laboratories|laboratory|labs)\b\.?", "", s)
        return re.sub(r"\s+", " ", s).strip()

    ing_n = _n(str(ingredient))
    cmp_n = _n(str(company))
    found = any(
        (ing_n in _n(row[0]) or _n(row[0]) in ing_n) and
        (len(cmp_n) < 4 or cmp_n in _n(row[3]) or _n(row[3]) in cmp_n)
        for row in cached if len(row) > 3
    )
    if not found:
        return [Finding("F3_DP_NO_MATCH", din, "data_protection_ends", "ERROR",
                        str(dp_ends)[:60],
                        f"dp fields populated but no live active Register match for "
                        f"ingredient={ingredient!r} company={company!r}", 3)]
    return []


async def run_family3(
    df: pd.DataFrame,
    sample_dins: list[str],
    cache: _DiskCache,
    stages: dict[str, bool] | None = None,
) -> list[Finding]:
    if stages is None:
        stages = {}
    sample_set  = set(sample_dins)
    patent_cols = [c for c in df.columns if re.match(r"patent_\d+_number$", c)]

    tasks: list = []
    async with httpx.AsyncClient(follow_redirects=True, verify=False) as client:
        for _, row in df.iterrows():
            din = str(row.get("din", "")).strip()
            if din not in sample_set:
                continue
            rdict  = dict(row)
            dc     = rdict.get("_drug_code")
            nocr   = str(rdict.get("needs_ocr", "0")).strip() in ("1", "True", "true", "yes")

            if stages.get("LABELING", True):
                tasks.append(_check_active_ingredient(
                    client, cache, din, dc, rdict.get("active_ingredient")))
                tasks.append(_check_pack_live(
                    client, cache, din, dc,
                    rdict.get("pack_size"), rdict.get("pack_style")))
                tasks.append(_check_excipients_pdf(
                    client, cache, din, dc,
                    rdict.get("excipients_core"), rdict.get("excipients_coating"),
                    needs_ocr=nocr))

            if stages.get("PATENTS", True):
                pnums = [str(rdict.get(c)).strip() for c in patent_cols if _is_real(rdict.get(c))]
                if pnums:
                    tasks.append(_check_patents_cpd(client, cache, din, pnums))

            if stages.get("DP", True):
                tasks.append(_check_dp_live(
                    client, cache, din,
                    rdict.get("ingredient"), rdict.get("company"),
                    rdict.get("data_protection_ends")))

        if not tasks:
            return []
        results = await asyncio.gather(*tasks, return_exceptions=True)

    out: list[Finding] = []
    for r in results:
        if isinstance(r, list):
            out.extend(r)
        elif isinstance(r, Exception):
            logging.warning("F3 task exception: %s", r)
    return out


# ─── Family 4: Determinism + OCR ─────────────────────────────────────────────

_EXTRACT_FIELDS = (
    "excipients_core", "excipients_coating", "preservatives",
    "colour", "shape", "size_mm", "weight", "ph",
)

_SYNTH_FONT_PATHS = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
]
_SYNTH_TOKENS = ["microcrystalline cellulose", "magnesium stearate"]


def _make_synthetic_scanned_pdf() -> bytes:
    """
    Create a single-page image-only PDF (no text layer) for OCR testing.
    Uses Helvetica (or a system fallback) at 28pt on a 1200×300 white canvas.
    """
    from PIL import Image, ImageDraw, ImageFont

    img  = Image.new("RGB", (1200, 300), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font: Any = None
    for fp in _SYNTH_FONT_PATHS:
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, 28)
                break
            except Exception:
                pass
    if font is None:
        try:
            font = ImageFont.load_default(size=28)
        except TypeError:
            font = ImageFont.load_default()

    lines = [
        "6. DOSAGE FORMS, COMPOSITION AND PACKAGING",
        "",
        "Non-Medicinal Ingredients: microcrystalline cellulose, magnesium stearate",
        "titanium dioxide, lactose monohydrate",
    ]
    y = 30
    for line in lines:
        draw.text((40, y), line, fill=(0, 0, 0), font=font)
        y += 60

    buf = io.BytesIO()
    img.save(buf, format="PDF", resolution=150.0)
    return buf.getvalue()


def _run_synthetic_ocr_probe() -> list[Finding]:
    """
    Deterministic, self-contained OCR liveness probe.

    Creates a synthetic image-only PDF (no text layer), feeds it through the
    real _extract_text_with_ocr / section-locate pipeline, and asserts:
      1. Text-layer extraction returns < 50 chars.
      2. OCR is triggered (ocr_used=True).
      3. OCR text length > 10 chars.
      4. Known tokens are recovered with fuzzy >= 0.9 (tolerates OCR noise).
      5. §6 section locator finds the section.

    Returns INFO on full pass, WARN/SKIPPED on partial/unavailable.
    Never returns ERROR — this probe's job is to characterise capability,
    not to gate builds.
    """
    try:
        from app.enrichment.labeling import (
            _extract_text_with_ocr, _find_section, _S6_MARKERS, _S6_END,
        )
        import pdfplumber
        from PIL import Image  # noqa: F401 – verify PIL importable
    except ImportError as e:
        return [Finding("F4_OCR_SYNTH_IMPORT", "(probe)", "ocr_liveness", "SKIPPED", "",
                        f"Synthetic OCR probe skipped — missing dependency: {e}", 4)]

    pdf_bytes = _make_synthetic_scanned_pdf()

    # Step 1 — verify no text layer
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdoc:
            text_layer_chars = sum(len(pg.extract_text() or "") for pg in pdoc.pages)
    except Exception:
        text_layer_chars = 0

    if text_layer_chars > 50:
        return [Finding("F4_OCR_SYNTH_HAS_TEXT", "(probe)", "ocr_liveness", "WARN",
                        str(text_layer_chars),
                        f"Synthetic PDF has unexpected text layer ({text_layer_chars} chars) — "
                        "PIL behaviour may have changed", 4)]

    # Step 2 — run OCR pipeline
    cache_key = "_grade_synth_" + hashlib.md5(pdf_bytes).hexdigest()[:8]
    try:
        pages, ocr_used = _extract_text_with_ocr(
            pdf_bytes, cache_key=cache_key, enable_ocr=True,
        )
    except Exception as exc:
        return [Finding("F4_OCR_SYNTH_EXTRACT_ERR", "(probe)", "ocr_liveness", "WARN", "",
                        f"_extract_text_with_ocr raised on synthetic PDF: {exc}", 4)]

    ocr_text = " ".join(t for _, t in pages)

    if not ocr_used:
        return [Finding(
            "F4_OCR_SYNTH_NOT_TRIGGERED", "(probe)", "ocr_liveness", "SKIPPED",
            f"text_layer={text_layer_chars} chars, ocr_used=False",
            f"OCR was NOT triggered on image-only PDF (text_layer={text_layer_chars} chars). "
            "Possible causes: ENABLE_OCR=0, pytesseract/pdf2image unavailable, "
            "or tesseract binary missing. OCR path is UNVERIFIED.",
            4,
        )]

    if len(ocr_text) < 10:
        return [Finding("F4_OCR_SYNTH_ZERO", "(probe)", "ocr_liveness", "WARN",
                        str(len(ocr_text)),
                        f"OCR was triggered but produced only {len(ocr_text)} chars — "
                        "tesseract may be misconfigured", 4)]

    # Step 3 — token recovery (fuzzy >= 0.9)
    token_misses: list[Finding] = []
    for token in _SYNTH_TOKENS:
        found, score = _fuzzy_contains(token, ocr_text, threshold=0.9)
        if not found:
            token_misses.append(Finding(
                "F4_OCR_SYNTH_TOKEN_MISS", "(probe)", "ocr_liveness", "WARN",
                token,
                f"Synthetic token '{token}' not recovered from OCR output "
                f"(best_score={score:.2f}, ocr_text={len(ocr_text)} chars)",
                4,
            ))
    if token_misses:
        return token_misses

    # Step 4 — section locator
    s6       = _find_section(pages, _S6_MARKERS, _S6_END)
    sect_ok  = s6 is not None

    return [Finding(
        "F4_OCR_SYNTH_PASS", "(probe)", "ocr_liveness", "INFO",
        f"text_layer={text_layer_chars} chars → OCR → {len(ocr_text)} chars; "
        f"§6={'found' if sect_ok else 'not found'}",
        f"Synthetic OCR probe PASSED: image-only PDF ({text_layer_chars} text-layer chars) → "
        f"OCR triggered → {len(ocr_text)} chars recovered; tokens found at ≥0.9 similarity; "
        f"§6 section {'located' if sect_ok else 'NOT located (section regex may need tuning)'}",
        4,
    )]


async def _reextract_din(
    client: httpx.AsyncClient, cache: _DiskCache,
    din: str, dc: str, strength: str,
) -> Optional[dict]:
    """Re-run extraction on PM PDF (no Ollama) for determinism check."""
    try:
        from app.enrichment.labeling import parse_labeling_fields_async, _extract_text_with_ocr
    except ImportError:
        return None

    html = cache.get("dpd_info_html", dc)
    if html is None:
        r = await _http_get_retry(client, _live("DPD_INFO_URL"),
                                  params={"lang": "eng", "code": dc},
                                  headers={"User-Agent": _UA})
        html = r.text if (r and r.status_code == 200) else ""
        cache.set("dpd_info_html", dc, html)
    if not html:
        return None

    from bs4 import BeautifulSoup
    soup    = BeautifulSoup(html, "html.parser")
    pdf_url: Optional[str] = None
    for link in soup.find_all("a", href=True):
        href: str = link["href"]
        if href.lower().endswith(".pdf") or "pdf.hres.ca" in href.lower():
            if not href.startswith("http"):
                href = f"https://health-products.canada.ca{href}"
            pdf_url = href
            break
    if not pdf_url:
        return None

    b64 = cache.get("pdf_b64", pdf_url)
    if b64 is None:
        r = await _http_get_retry(client, pdf_url, headers={"User-Agent": _UA}, timeout=60.0)
        if r is None or r.status_code != 200:
            return None
        import base64
        b64 = base64.b64encode(r.content).decode()
        cache.set("pdf_b64", pdf_url, b64)

    import base64
    pdf_bytes = base64.b64decode(b64)
    try:
        pages, _ = _extract_text_with_ocr(pdf_bytes, cache_key="", enable_ocr=False)
        result   = await parse_labeling_fields_async(pages, strength or None)
    except Exception as exc:
        logging.warning("F4 re-extraction failed din=%s: %s", din, exc)
        return None
    return result


async def _check_determinism(
    df: pd.DataFrame, sample_dins: list[str], cache: _DiskCache,
    labeling_active: bool = True,
) -> list[Finding]:
    out: list[Finding] = []
    if not labeling_active:
        return out
    try:
        import app.enrichment.labeling  # noqa: F401
    except ImportError as e:
        out.append(Finding("F4_IMPORT_FAILED", "(all)", "determinism", "SKIPPED", "",
                           f"Cannot import app.enrichment.labeling: {e}", 4))
        return out

    async with httpx.AsyncClient(follow_redirects=True, verify=False) as client:
        for din in sample_dins[:12]:
            wb_rows = df[df["din"].astype(str) == din]
            if wb_rows.empty:
                continue
            wb = dict(wb_rows.iloc[0])
            dc = str(wb.get("_drug_code") or "").strip()
            if not dc:
                continue
            extracted = await _reextract_din(client, cache, din, dc,
                                             str(wb.get("strength") or ""))
            if extracted is None:
                out.append(Finding("F4_DET_NO_PDF", din, "determinism", "SKIPPED", "",
                                   f"No PDF available for re-extraction (drug_code={dc})", 4))
                continue
            for f in _EXTRACT_FIELDS:
                wb_val = str(wb.get(f) or "").strip()
                re_val = str(extracted.get(f) or "").strip()
                if wb_val != re_val and not (_is_sentinel(wb_val) and _is_sentinel(re_val)):
                    out.append(Finding("F4_NONDETERMINISTIC", din, f, "WARN",
                                       wb_val[:80],
                                       f"Re-extraction differs: workbook={wb_val!r} "
                                       f"rerun={re_val!r}", 4))
    return out


async def _check_ocr_liveness_live(cache: _DiskCache) -> list[Finding]:
    """
    Live informational probe: downloads real PMs and logs text-vs-OCR usage.
    Results are WARN/INFO/SKIPPED only — for information, not assertion.
    The synthetic probe (_run_synthetic_ocr_probe) is the authoritative test.
    """
    out: list[Finding] = []
    try:
        from app.enrichment.labeling import (
            _extract_text_with_ocr, _find_section,
            _S6_MARKERS, _S6_END, _S13_MARKERS, _S13_END,
        )
    except ImportError as e:
        out.append(Finding("F4_OCR_IMPORT", "(probe)", "ocr_liveness", "SKIPPED", "",
                           f"Cannot import labeling module: {e}", 4))
        return out

    _LIVE_PROBE_DCS = [83088, 48982]

    async with httpx.AsyncClient(follow_redirects=True, verify=False) as client:
        for dc in _LIVE_PROBE_DCS:
            dc_s = str(dc)
            html = cache.get("dpd_info_html", dc_s)
            if html is None:
                r    = await _http_get_retry(client, _live("DPD_INFO_URL"),
                                             params={"lang": "eng", "code": dc_s},
                                             headers={"User-Agent": _UA})
                html = r.text if (r and r.status_code == 200) else ""
                cache.set("dpd_info_html", dc_s, html)
            if not html:
                out.append(Finding("F4_OCR_LIVE_NO_INFO", dc_s, "ocr_liveness", "SKIPPED", "",
                                   f"drug_code={dc}: DPD info page unreachable (live probe, informational)", 4))
                continue

            from bs4 import BeautifulSoup
            soup    = BeautifulSoup(html, "html.parser")
            pdf_url: Optional[str] = None
            for link in soup.find_all("a", href=True):
                href: str = link["href"]
                if href.lower().endswith(".pdf") or "pdf.hres.ca" in href.lower():
                    if not href.startswith("http"):
                        href = f"https://health-products.canada.ca{href}"
                    pdf_url = href
                    break
            if not pdf_url:
                out.append(Finding("F4_OCR_LIVE_NO_PDF", dc_s, "ocr_liveness", "WARN", "",
                                   f"drug_code={dc}: no PDF link (informational live probe)", 4))
                continue

            b64 = cache.get("pdf_b64", pdf_url)
            if b64 is None:
                r = await _http_get_retry(client, pdf_url, headers={"User-Agent": _UA},
                                          timeout=60.0)
                if r is None or r.status_code != 200:
                    out.append(Finding("F4_OCR_LIVE_PDF_FAIL", dc_s, "ocr_liveness", "SKIPPED", "",
                                       f"drug_code={dc}: PDF download failed (live probe, informational)", 4))
                    continue
                import base64
                b64 = base64.b64encode(r.content).decode()
                cache.set("pdf_b64", pdf_url, b64)

            import base64
            pdf_bytes = base64.b64decode(b64)
            try:
                pages, ocr_used = _extract_text_with_ocr(
                    pdf_bytes, cache_key=f"ocr_live:{dc}", enable_ocr=True,
                )
            except Exception as exc:
                out.append(Finding("F4_OCR_LIVE_ERR", dc_s, "ocr_liveness", "WARN", "",
                                   f"drug_code={dc}: extraction raised: {exc} (informational)", 4))
                continue

            total_chars  = sum(len(t) for _, t in pages)
            s6           = _find_section(pages, _S6_MARKERS, _S6_END)
            s13          = _find_section(pages, _S13_MARKERS, _S13_END)
            section_used = "§6" if s6 else ("§13" if s13 else "none")
            out.append(Finding(
                "F4_OCR_LIVE_INFO", dc_s, "ocr_liveness", "INFO",
                f"{total_chars} chars, ocr_used={ocr_used}, section={section_used}",
                f"Live OCR probe drug_code={dc}: {total_chars} chars; "
                f"ocr_used={ocr_used}; section={section_used} (informational — not assertion)",
                4,
            ))
    return out


async def run_family4(
    df: pd.DataFrame, sample_dins: list[str], cache: _DiskCache,
    stages: dict[str, bool] | None = None,
) -> list[Finding]:
    if stages is None:
        stages = {}
    labeling_active = stages.get("LABELING", True)

    synth = _run_synthetic_ocr_probe()                              # sync, deterministic
    det   = await _check_determinism(df, sample_dins, cache,
                                     labeling_active=labeling_active)
    live  = await _check_ocr_liveness_live(cache)                   # informational only
    return synth + det + live


# ─── Stratified sample ────────────────────────────────────────────────────────

def select_sample(df: pd.DataFrame, n: int) -> list[str]:
    all_dins = df["din"].astype(str).tolist()
    if len(all_dins) <= n:
        return all_dins

    strats: dict[str, list[str]] = defaultdict(list)
    patent_cols = [c for c in df.columns if re.match(r"patent_\d+_number$", c)]

    for _, row in df.iterrows():
        din    = str(row.get("din", "")).strip()
        rdict  = dict(row)
        dosage = str(rdict.get("dosage_form") or "").lower()
        sub    = str(rdict.get("noc_submission_type") or "")

        if ";" in str(rdict.get("ingredient") or ""):
            strats["combo"].append(din)
        if any(w in dosage for w in ("solution", "injection", "infusion")):
            strats["solution"].append(din)
        if re.search(r"\b(er|extended|modified|controlled|sustained)\b", dosage, re.IGNORECASE):
            strats["er"].append(din)
        if sub == "NDS":
            strats["nds"].append(din)
        elif sub == "ANDS":
            strats["ands"].append(din)
        elif sub == NO_NOC_RECORD:
            strats["no_noc"].append(din)
        if _is_real(rdict.get("data_protection_ends")):
            strats["dp"].append(din)
        if any(_is_real(rdict.get(c)) for c in patent_cols):
            strats["patent"].append(din)

    seen: set[str] = set()
    selected: list[str] = []
    quota = max(1, n // max(len(strats), 1))

    for key in ("combo", "solution", "dp", "patent", "nds", "ands", "er", "no_noc"):
        for din in strats.get(key, [])[:quota]:
            if din not in seen:
                selected.append(din)
                seen.add(din)
            if len(selected) >= n:
                return selected

    remainder = [d for d in all_dins if d not in seen]
    random.shuffle(remainder)
    for din in remainder:
        if len(selected) >= n:
            break
        selected.append(din)
        seen.add(din)
    return selected[:n]


# ─── Report ───────────────────────────────────────────────────────────────────

def _md_table(findings: list[Finding], limit: int = 60) -> str:
    if not findings:
        return "_None._\n"
    rows = ["| check_id | din | field | sev | value | reason |",
            "|---|---|---|---|---|---|"]
    for f in findings[:limit]:
        val    = str(f.workbook_value).replace("|", "\\|")[:60]
        reason = f.reason.replace("|", "\\|")[:90]
        rows.append(f"| {f.check_id} | {f.din} | {f.field} | {f.severity} | {val} | {reason} |")
    if len(findings) > limit:
        rows.append(f"\n*… and {len(findings) - limit} more — see CSV for full list.*")
    return "\n".join(rows) + "\n"


def format_report(
    findings: list[Finding],
    xlsx_path: str,
    sample_n: int,
    total_rows: int,
    elapsed: float,
    stages: dict[str, bool] | None = None,
) -> str:
    if stages is None:
        stages = {}

    lines: list[str] = [
        "# Workbook Accuracy Check Report",
        "",
        f"**Workbook:** `{xlsx_path}`  ",
        f"**Sheet 1 rows:** {total_rows}  ",
        f"**Family-3/4 sample:** {sample_n} DINs  ",
        f"**Run date:** {datetime.date.today()}  ",
        f"**Elapsed:** {elapsed:.1f} s  ",
        "",
        "> These are consistency/faithfulness checks, not absolute accuracy."
        " Absolute accuracy requires a hand-labeled gold set.",
        "",
    ]

    # ── Stage Coverage Map ───────────────────────────────────────────────────
    lines += ["## Stage Coverage Map", ""]
    stage_order = [("LABELING", "Labeling (PDF extraction)"),
                   ("PATENTS",  "Patents (patent register)"),
                   ("NOC",      "NOC (Notice of Compliance)"),
                   ("DP",       "Data Protection register")]
    for key, label in stage_order:
        present = stages.get(key, False)
        icon    = "✓" if present else "✗"
        note    = "" if present else "  ← checks for this stage skipped"
        lines.append(f"- **{key}** ({label}): {icon}{note}")
    lines.append("")

    labeling_active = stages.get("LABELING", True)
    if not labeling_active:
        lines += [
            "> ⚠ **LABELING stage not present in this workbook — all PM-field and "
            "active_ingredient checks skipped.**",
            "",
        ]

    # ── Coverage Ledger ──────────────────────────────────────────────────────
    f3_all       = [f for f in findings if f.family == 3]
    f3_skip_dins = {f.din for f in f3_all if f.severity == "SKIPPED"}
    f3_exec_dins = {f.din for f in f3_all if f.severity != "SKIPPED"} - f3_skip_dins
    f3_errors    = [f for f in f3_all if f.severity == "ERROR"]

    synth_pass = any(f.check_id == "F4_OCR_SYNTH_PASS"   for f in findings)
    synth_skip = any("F4_OCR_SYNTH" in f.check_id and f.severity in ("SKIPPED", "WARN")
                     for f in findings if f.check_id != "F4_OCR_SYNTH_PASS")
    synth_label = ("PASS ✓" if synth_pass else
                   ("SKIPPED/WARN — unverified" if synth_skip else "FAIL ✗"))

    det_ran  = len({f.din for f in findings if f.family == 4 and f.check_id == "F4_NONDETERMINISTIC"})
    det_skip = len({f.din for f in findings if f.family == 4 and f.check_id == "F4_DET_NO_PDF"})

    lines += [
        "## Coverage Ledger",
        "",
        "| Family | Eligible | Executed | Skipped | Coverage | Notes |",
        "|--------|----------|----------|---------|----------|-------|",
        f"| F1 Field Invariants | {total_rows} rows | {total_rows} rows | 0 | 100% | "
        + ("PM checks active" if labeling_active else "PM checks N/A (no labeling)") + " |",
        f"| F2 Coherence | {total_rows} rows | {total_rows} rows | 0 | 100% | "
        + ("PM checks active" if labeling_active else "PM checks N/A") + " |",
        f"| F3 Anti-hallucination | {sample_n} DINs | {len(f3_exec_dins)} DINs | "
        f"{len(f3_skip_dins)} DINs | "
        f"{int(100*len(f3_exec_dins)/max(sample_n,1))}% | "
        + ("source(s) unreachable for skipped DINs" if f3_skip_dins else "all checked") + " |",
        f"| F4 Determinism | {min(12, sample_n)} DINs | {det_ran} | {det_skip} | "
        f"{int(100*det_ran/max(min(12,sample_n),1))}% | labeling stage required |",
        f"| F4 Synthetic OCR probe | 1 | 1 | 0 | 100% | result: {synth_label} |",
        "",
    ]

    # ── Verified vs Unverifiable headline ────────────────────────────────────
    lines += ["## Verified vs Unverifiable", ""]
    if f3_exec_dins:
        lines.append(
            f"- **Anti-hallucination:** {len(f3_errors)} error(s) across "
            f"**{len(f3_exec_dins)} DIN(s) ACTUALLY CHECKED**."
        )
    else:
        lines.append("- **Anti-hallucination:** 0 DINs checked — ALL UNVERIFIED.")

    if f3_skip_dins:
        lines.append(
            f"- **{len(f3_skip_dins)} DIN(s) in sample could NOT be verified** "
            "(DPD/CPD unreachable) — those fields are NOT verified."
        )

    if not labeling_active:
        lines.append(
            "- **Labeling fields (active_ingredient, excipients, pH, colour, shape, etc.):** "
            "UNVERIFIED — labeling stage was not run for this workbook."
        )

    if synth_pass:
        lines.append("- **OCR pipeline:** VERIFIED via synthetic image-only PDF probe.")
    elif synth_skip:
        lines.append(
            "- **OCR pipeline:** UNVERIFIED — synthetic probe could not run "
            "(check tesseract binary and ENABLE_OCR setting)."
        )

    lines.append("")

    # ── Fabrication / Ghost top section ──────────────────────────────────────
    fab = [f for f in findings if f.family == 3 and f.severity == "ERROR"]
    if fab:
        lines += [
            "## ⚠ FABRICATION / GHOST FINDINGS  (Family-3 ERRORs — highest-priority fixes)",
            "",
            "Values present in the workbook that do not exist in their authoritative source:",
            "",
        ]
        for f in fab:
            lines.append(f"- **{f.check_id}** DIN={f.din} field=`{f.field}`: "
                         f"`{str(f.workbook_value)[:80]}` — {f.reason}")
        lines.append("")
    else:
        lines += ["## Fabrication / Ghost Section", "",
                  "No Family-3 ERRORs found"
                  + (" (among checked DINs)" if f3_exec_dins else " (no DINs checked — cannot confirm)")
                  + ".", ""]

    # ── Per-family tables ─────────────────────────────────────────────────────
    for fam_num, fam_name in (
        (1, "Field Invariants"),
        (2, "Cross-Field Coherence"),
        (3, "Anti-Hallucination (live fetch)"),
        (4, "Determinism + OCR Liveness"),
    ):
        ff  = [f for f in findings if f.family == fam_num]
        ctr = Counter(f.severity for f in ff)
        ran_any = bool(ff) or fam_num <= 2  # F1/F2 always run

        status = ""
        if fam_num == 3 and not f3_exec_dins:
            status = "  **⚠ UNVERIFIED — no DINs successfully checked**"
        elif fam_num == 4 and not labeling_active and not synth_pass:
            status = "  *(determinism not run; synthetic OCR probe skipped)*"

        lines += [
            f"## Family {fam_num}: {fam_name}{status}",
            "",
            f"ERROR: **{ctr['ERROR']}**  |  WARN: **{ctr['WARN']}**  |  "
            f"INFO: **{ctr['INFO']}**  |  SKIPPED: **{ctr['SKIPPED']}**",
            "",
            _md_table(ff),
        ]

    # ── Totals ────────────────────────────────────────────────────────────────
    total_e = sum(1 for f in findings if f.severity == "ERROR")
    total_w = sum(1 for f in findings if f.severity == "WARN")
    total_i = sum(1 for f in findings if f.severity == "INFO")
    total_s = sum(1 for f in findings if f.severity == "SKIPPED")
    lines += [
        "## Totals",
        "",
        "| Severity | Count |",
        "|---|---|",
        f"| ERROR    | {total_e} |",
        f"| WARN     | {total_w} |",
        f"| INFO     | {total_i} |",
        f"| SKIPPED  | {total_s} |",
        f"| **Total**| **{len(findings)}** |",
        "",
        "---",
        "*Consistency/faithfulness checks only — absolute accuracy requires a gold set.*",
    ]
    return "\n".join(lines)


def write_csv(findings: list[Finding], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "family", "check_id", "din", "field", "severity", "workbook_value", "reason",
        ])
        w.writeheader()
        for f in findings:
            w.writerow({
                "family": f.family, "check_id": f.check_id, "din": f.din,
                "field": f.field, "severity": f.severity,
                "workbook_value": str(f.workbook_value)[:200],
                "reason": f.reason,
            })


# ─── Main ─────────────────────────────────────────────────────────────────────

async def _run(xlsx_path: str, sample_n: int, cache_dir: str) -> None:
    t0 = time.time()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    if not os.path.exists(xlsx_path):
        print(f"ERROR: workbook not found at {xlsx_path}")
        print("Generate one first:  python -m app.enrichment.workbook --q <term> "
              "--field ingredient --out workbook.xlsx")
        return

    print(f"Loading: {xlsx_path}")
    try:
        df = pd.read_excel(xlsx_path, sheet_name="DPD + NOC + Patents", dtype=str)
    except Exception as e:
        print(f"ERROR reading Sheet 1: {e}")
        return
    df = df.fillna("").replace("nan", "")
    total_rows = len(df)
    print(f"  {total_rows} rows × {len(df.columns)} columns")

    # Stage detection
    stages = detect_stages(df)
    print("\nSTAGE COVERAGE MAP:")
    for k, v in stages.items():
        print(f"  {k}: {'present' if v else 'NOT present'}")
    if not stages["LABELING"]:
        print("\n  ⚠  LABELING stage not present in this workbook — labeling checks skipped.")

    cache = _DiskCache(cache_dir)

    print("\nFamily 1 (invariants, all rows)...")
    f1 = run_family1(df, stages=stages)
    print(f"  ERR={sum(1 for f in f1 if f.severity=='ERROR')}  "
          f"WARN={sum(1 for f in f1 if f.severity=='WARN')}  "
          f"INFO={sum(1 for f in f1 if f.severity=='INFO')}")

    print("Family 2 (coherence, all rows)...")
    f2 = run_family2(df, stages=stages)
    print(f"  ERR={sum(1 for f in f2 if f.severity=='ERROR')}  "
          f"WARN={sum(1 for f in f2 if f.severity=='WARN')}")

    sample = select_sample(df, sample_n)
    print(f"\nSample: {len(sample)} DINs for live checks")

    print("Family 3 (anti-hallucination, live)...")
    f3 = await run_family3(df, sample, cache, stages=stages)
    print(f"  ERR={sum(1 for f in f3 if f.severity=='ERROR')}  "
          f"WARN={sum(1 for f in f3 if f.severity=='WARN')}  "
          f"SKIP={sum(1 for f in f3 if f.severity=='SKIPPED')}")

    print("Family 4 (determinism + OCR liveness)...")
    f4 = await run_family4(df, sample, cache, stages=stages)
    print(f"  ERR={sum(1 for f in f4 if f.severity=='ERROR')}  "
          f"WARN={sum(1 for f in f4 if f.severity=='WARN')}  "
          f"INFO={sum(1 for f in f4 if f.severity=='INFO')}  "
          f"SKIP={sum(1 for f in f4 if f.severity=='SKIPPED')}")

    all_findings = f1 + f2 + f3 + f4
    elapsed      = time.time() - t0

    out_dir  = os.path.dirname(os.path.abspath(xlsx_path))
    md_path  = os.path.join(out_dir, "checks_report.md")
    csv_path = os.path.join(out_dir, "checks_report.csv")

    report = format_report(all_findings, xlsx_path, len(sample), total_rows, elapsed,
                           stages=stages)
    Path(md_path).write_text(report, encoding="utf-8")
    write_csv(all_findings, csv_path)

    print("\n" + "=" * 70)
    print(report)
    print("=" * 70)
    print(f"\n→ {md_path}")
    print(f"→ {csv_path}")
    e = sum(1 for f in all_findings if f.severity == "ERROR")
    w = sum(1 for f in all_findings if f.severity == "WARN")
    i = sum(1 for f in all_findings if f.severity == "INFO")
    s = sum(1 for f in all_findings if f.severity == "SKIPPED")
    print(f"\nTotal: {len(all_findings)} findings  "
          f"(ERR={e}  WARN={w}  INFO={i}  SKIP={s})")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Non-gating workbook accuracy harness. Always exits 0.",
    )
    ap.add_argument("xlsx", help="Path to the enriched workbook .xlsx")
    ap.add_argument("--sample", type=int, default=40,
                    help="DINs to sample for live checks (default 40)")
    ap.add_argument("--cache-dir", default="/tmp/canadian_drug_db_cache",
                    help="Disk cache directory (default /tmp/canadian_drug_db_cache)")
    args = ap.parse_args()
    asyncio.run(_run(args.xlsx, args.sample, args.cache_dir))
    sys.exit(0)


if __name__ == "__main__":
    main()
