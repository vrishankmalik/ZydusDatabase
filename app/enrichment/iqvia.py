"""IQVIA Canada metrics: parse, collapse, and match to DINs.

Parsing
-------
Reads the 'data' sheet from an IQVIA Excel export.  Metric columns are
detected by pattern (``Dollars|Units|Ext Units MAT MM/YYYY``) so the file
can roll forward every refresh without code changes.

Collapsing
----------
Each product appears multiple times — once per channel (Drugstore/Hospital),
once per province (up to 9), and sometimes once per container type (SYRINGE
vs VIAL).  All of those rows represent the same DIN, so they are summed.
Group key: (Combined Molecule, Product, Manufacturer, Strength).

Matching (cite-or-blank)
------------------------
For each collapsed IQVIA group the algorithm finds candidate DINs in Sheet 1:

  1. Prefilter by strength: normalize both sides to a frozenset of
     ``NUMBERunit`` tokens; require exact set equality.  ``150MG/ML`` drops
     the ``/ML`` denominator before comparison; ``;``-separated DPD strengths
     and ``/``-separated IQVIA combos are split the same way.

  2. Score by brand + company similarity (0–100 each), weighted 50/50 after
     stripping corporate suffixes and trailing strength / form words.

  3. Accept only when:
        • exactly ONE candidate exceeds CONFIDENT_THRESHOLD (65), OR
        • the top candidate exceeds CONFIDENT_THRESHOLD AND the gap to the
          second candidate exceeds TIE_MARGIN (8).
     Any other outcome → blank for all involved DINs + reconciliation entry.

  4. Each IQVIA group is assigned to at most one DIN.  If two DINs both score
     above MIN_CANDIDATE (55) the group is marked ambiguous and no DIN
     receives data.

Reconciliation
--------------
Returns a DataFrame listing every IQVIA group with:
  - ``status``: matched / ambiguous / low_score / no_din_match
  - the matched DIN (or blank)
  - the top-2 candidate scores (for human review)
"""
from __future__ import annotations

import io
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Optional

import pandas as pd

# ── Matching thresholds ───────────────────────────────────────────────────────

# Combined score (0–100) required to accept a match.
CONFIDENT_THRESHOLD = 65

# Minimum brand+company score for a DIN to be considered a candidate at all.
MIN_CANDIDATE = 55

# Minimum manufacturer-name similarity (0–100) for a DIN to be eligible at all.
# Generic brand names are dominated by the shared molecule, so brand similarity
# can run high for the wrong company; the manufacturer is the true discriminator.
# Set below the lowest observed genuine same-firm match (CHEPLAPHARM vs
# CHEPLAPHARM ARZNEIMITTEL GMBH ≈ 63) and above coincidental cross-company
# collisions (NORA vs RIVA ≈ 50).
MIN_COMPANY_SIM = 55

# If the gap between top and second candidate is less than this, flag as a tie.
# Set at 15 so that cases like PROVERA 5MG (gap ≈ 11) are surfaced for review
# rather than silently assigned to the slightly-higher-scoring DIN.
TIE_MARGIN = 15

# Minimum company similarity for a generic-label alias to AGGREGATE onto an
# exact-brand-reserved DIN (Pass 2).  Aggregation is a high-confidence "same firm,
# same product, different IQVIA label" merge, so it demands a much stronger company
# match than the fuzzy floor: a genuine alias (IQVIA "METFORMIN / PRO DOC" onto the
# reserved PRO-METFORMIN / PRO DOC DIN) scores ~100, while a coincidental
# cross-company collision (MANTRA's "M-METFORMIN" vs ANGITA, company ≈ 67) must be
# rejected.  This is independent of MIN_COMPANY_SIM (which stays at 55 so genuine
# CHEPLAPHARM-style fuzzy matches survive); raising the fuzzy floor instead would
# kill those, so the aggregation path carries its own stricter bar.
AGG_MIN_COMPANY_SIM = 85

# ── Lifecycle-status tiers ─────────────────────────────────────────────────────
#
# A DPD DIN carries a lifecycle status.  IQVIA reports recent sales (a Moving
# Annual Total), so only a *currently marketed* product can legitimately own that
# revenue.  Discontinued siblings — same molecule, same strength, near-identical
# brand/company — would otherwise pollute the candidate list and trigger false
# near-ties that blank the one real seller (e.g. JAMP METFORMIN 500MG DIN
# 02380196 (Marketed) vs JAMP-METFORMIN BLACKBERRY DIN 02380722 (Cancelled Post
# Market): gap 11 < TIE_MARGIN → both wrongly blanked).
#
# Two tiers:
#   • _NEVER_MARKETED_STATUSES — the product was never sold at all, so it can
#     never own IQVIA sales.  Dropped from candidacy entirely (the original code
#     dropped only "approved"; "cancelled pre market" is the same case).
#   • _MARKETED_STATUS — current market presence.  When ANY candidate for an
#     IQVIA group is marketed, only marketed candidates are considered; once-sold
#     but discontinued statuses (dormant, cancelled post market, cancelled
#     (unreturned annual), …) are used only as a fallback when no marketed DIN
#     matches, so historical-only products are still attributable.
_MARKETED_STATUS = "marketed"
_NEVER_MARKETED_STATUSES = {"approved", "cancelled pre market"}

# ── Regex helpers ─────────────────────────────────────────────────────────────

# Detects IQVIA metric column names: "Dollars MAT 12/2025", "Units MAT 12/2024", etc.
_METRIC_COL_RE = re.compile(
    r"^(Dollars|Units|Ext\s+Units)\s+MAT\s+\d{2}/\d{4}$",
    re.IGNORECASE,
)

# Concentration denominator to strip: /ML, /G, /L (but NOT /MG which is a combo separator)
_CONC_DENOM_RE = re.compile(r"\s*/\s*(ml|g|l)\s*$", re.IGNORECASE)

# Internal spaces between a number and its unit: "100 MG" → "100MG"
_NUM_SPACE_UNIT_RE = re.compile(r"(\d)\s+([A-Za-z%])")

# Tokens that carry no company identity — stripped before fuzzy comparison.
# Longer alternatives precede shorter ones so the regex engine matches the
# longest word first (e.g. "corporation" before "corp", "incorporated" before "inc").
# After unicode normalisation + punctuation stripping, French abbreviations
# resolve to plain ASCII: "S.E.C." → "sec", "Ltée." → "ltee".
_CORP_STRIP_RE = re.compile(
    r"\b("
    r"incorporated|inc|limited|limitee|ltee|ltd\b|llc|llp|ulc|corporation|corp|co\b|"
    r"sa\b|ag\b|gmbh|plc|sencrl|senc|sec\b|"
    r"pharmaceuticals|pharmaceutical|pharma\b|therapeutics|"
    r"laboratoires|laboratoire|laboratories|laboratory|labs\b|lab\b|healthcare|health|canada|"
    r"a division of|division|serono|consumer"
    r")[.,]*",
    re.IGNORECASE,
)

# Strip trailing strength/dose/form tokens from a brand name before comparison.
# Requires a leading digit so "PROVERA 5MG TABLETS" → "PROVERA".
_BRAND_TRAILING_RE = re.compile(
    r"\s+\d[\d.\s]*(%|mg|mcg|ug|g\b|ml|iu|miu|units?|cap|capsule|tablet|tab|pak\b|pack).*$",
    re.IGNORECASE,
)
# Strip bare dosage-form words at the end of a brand name when no digit precedes them.
# Handles cases like "APO-ABACAVIR-LAMIVUDINE TABLETS" where the DPD brand name
# includes the form but IQVIA omits it — without this, the trailing word inflates
# the dissimilarity and can cause a false near-tie.
_BRAND_TRAILING_FORM_RE = re.compile(
    r"\s+(?:tablets?|capsules?|caps?|injections?|solution|suspension|cream|ointment|gel|patch|spray|drops?|syrup|elixir|lotion)\s*$",
    re.IGNORECASE,
)

# IQVIA sometimes omits the unit on all but the last component of a combination
# (e.g. "160/12.5MG" meaning "160MG/12.5MG"). These regexes detect that case.
_BARE_NUM_RE = re.compile(r'^\d+(?:\.\d+)?$')      # token with no unit at all
_UNIT_TAIL_RE = re.compile(r'(MG|MCG|UG|ML|IU|MIU|%)$', re.IGNORECASE)  # unit suffix

# ── Normalization helpers ─────────────────────────────────────────────────────

def _norm_strength(s: object) -> frozenset[str]:
    """Return a frozenset of normalised 'NUMBER+UNIT' tokens.

    Examples
    --------
    '100 MG'          -> frozenset({'100MG'})
    '1 MG; 100 MG'    -> frozenset({'1MG', '100MG'})   # DPD semicolon format
    '100MG/1MG'       -> frozenset({'100MG', '1MG'})   # IQVIA combo slash
    '150MG/ML'        -> frozenset({'150MG'})           # concentration → drop /ML
    '0.6GM'           -> frozenset({'600MG'})           # GM → MG unit conversion
    '0.6GM/300MG'     -> frozenset({'600MG', '300MG'}) # IQVIA combo with GM unit
    '160/12.5MG'      -> frozenset({'160MG', '12.5MG'}) # bare-number: unit inferred
    '10MG/G'          -> frozenset({'1%'})              # MG/G → % (10 MG/G = 1%)
    '50M'             -> frozenset({'50MG'})            # IQVIA field-width truncation
    '8 %'             -> frozenset({'8%'})
    ''                -> frozenset()
    """
    if s is None:
        return frozenset()
    raw = str(s).strip()
    if not raw or raw.lower() in ("none", "nan", "not applicable", "n/a"):
        return frozenset()
    # MG/G → % BEFORE splitting (/G would otherwise be stripped as a denominator).
    # 10 MG/G = 1% by definition (milligrams per gram).
    raw = re.sub(
        r'(\d+(?:\.\d+)?)\s*MG/G',
        lambda m: f"{float(m.group(1)) / 10:g}%",
        raw,
        flags=re.IGNORECASE,
    )
    # Drop per-volume denominator (/ML, /G, /L) — signals concentration, not combo.
    raw = _CONC_DENOM_RE.sub("", raw)
    # Split on / or ; to get individual components.
    parts = re.split(r"[/;]", raw)
    result: set[str] = set()
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # Collapse "100 MG" → "100MG"
        norm = _NUM_SPACE_UNIT_RE.sub(r"\1\2", p)
        norm = norm.upper().strip()
        if not norm:
            continue
        # Unit conversions — longest suffixes first to avoid partial matches.
        converted = False
        for suffix, factor in [("KG", 1_000_000.0), ("GM", 1_000.0), ("MCG", 0.001), ("UG", 0.001)]:
            if norm.endswith(suffix):
                val_str = norm[: -len(suffix)]
                try:
                    norm = f"{float(val_str) * factor:g}MG"
                    converted = True
                    break
                except ValueError:
                    pass
        if not converted:
            # "G" alone (e.g. "0.5G" → "500MG") — checked last so "MG"/"MCG" don't match.
            m_g = re.match(r'^(\d+(?:\.\d+)?)G$', norm)
            if m_g:
                try:
                    norm = f"{float(m_g.group(1)) * 1000:g}MG"
                    converted = True
                except ValueError:
                    pass
        if not converted:
            # IQVIA field-width truncation: "50M" → "50MG".
            m_trunc = re.match(r'^(\d+(?:\.\d+)?)M$', norm)
            if m_trunc:
                try:
                    norm = f"{float(m_trunc.group(1)):g}MG"
                except ValueError:
                    pass
        if norm:
            # Canonicalize the numeric prefix so DPD's decimal/trailing-zero
            # formatting matches IQVIA's compact form: '10.0MG' → '10MG',
            # '12.50MG' → '12.5MG', '50.0MG' → '50MG'. Without this, the exact
            # strength-set prefilter silently drops any DPD strength stored with a
            # trailing zero (e.g. PMS-AMLODIPINE '10.0 MG' vs IQVIA '10MG').
            m_num = re.match(r'^(\d+(?:\.\d+)?)(.*)$', norm)
            if m_num:
                norm = f"{float(m_num.group(1)):g}{m_num.group(2)}"
            result.add(norm)

    # Bare-number inference: IQVIA omits the unit on all but the last component
    # when all components share the same unit, e.g. "160/12.5MG" means "160MG/12.5MG".
    # Find any bare-number tokens (digits only, no unit) and apply the unit from
    # the last non-bare token in the set.  If every token is a bare number (no
    # unit context exists) we leave them unchanged rather than guessing.
    bare = {t for t in result if _BARE_NUM_RE.match(t)}
    if bare:
        inferred_unit: Optional[str] = None
        for t in (result - bare):
            m = _UNIT_TAIL_RE.search(t)
            if m:
                inferred_unit = m.group(1).upper()
        if inferred_unit:
            result -= bare
            result |= {t + inferred_unit for t in bare}

    return frozenset(result)


def _norm_company(s: object) -> str:
    """Strip corporate/legal suffixes and collapse whitespace.

    Processing order:
      1. Unicode NFKD → ASCII so accented variants match plain equivalents
         ("Limitée" → "Limitee", "ltée" → "ltee").
      2. Strip punctuation that appears in French legal abbreviations
         ("S.E.C." → "sec", "Smith & Nephew" → "Smith Nephew").
      3. Apply _CORP_STRIP_RE to remove generic company-type words.
      4. Collapse whitespace.
    """
    if s is None:
        return ""
    t = str(s)
    # Step 1: flatten accented chars to ASCII equivalents
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii")
    t = t.lower()
    # Step 2: strip punctuation used in abbreviations and separators
    # "." and "," are components of abbreviations (s.e.c.); "/" and "&" are
    # separators ("Smith & Nephew", "Limitée / S.E.C.").
    t = re.sub(r"[.,/&]", "", t)
    # Step 3: remove generic corporate-type tokens
    t = _CORP_STRIP_RE.sub(" ", t)
    # Step 4: normalise whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _norm_brand(s: object) -> str:
    """Strip trailing strength / form tokens and lowercase."""
    if s is None:
        return ""
    t = str(s).strip()
    t = _BRAND_TRAILING_RE.sub("", t)       # "PROVERA 5MG TABLETS" → "PROVERA"
    t = _BRAND_TRAILING_FORM_RE.sub("", t)  # "APO-DRUG TABLETS" → "APO-DRUG"
    return t.lower().strip()


def _sim(a: str, b: str) -> float:
    """SequenceMatcher ratio scaled to 0–100."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio() * 100.0


def _molecule_overlap(iq_mol: str, din_ing: str) -> bool:
    """True when an IQVIA molecule genuinely corresponds to a DIN's ingredient.

    The number of active components must match (mono ≠ combo), and every IQVIA
    molecule component must appear in the DIN's ingredient string.  DPD lists salt
    forms IQVIA omits, so containment is one-directional (IQVIA ⊆ DPD); the
    component-count equality supplies the other direction.  Rejects cross-molecule
    combos that merely share one component (TELMISARTAN/HCTZ vs VALSARTAN/HCTZ) and
    mono-vs-combo collisions whose per-component strengths coincide.
    """
    iq_parts = [p for p in re.split(r"[:/;+]", str(iq_mol)) if p.strip()]
    din_parts = [p for p in re.split(r";", str(din_ing)) if p.strip()]
    if not iq_parts or not din_parts:
        return False
    if len(iq_parts) != len(din_parts):
        return False
    din_words = set(re.findall(r"[A-Za-z]{4,}", str(din_ing).upper()))
    for p in iq_parts:
        p_words = set(re.findall(r"[A-Za-z]{4,}", p.upper()))
        if p_words and not (p_words & din_words):
            return False
    return True


# Salt forms, pharmacopoeia standards and dosage-form words that carry no brand
# identity — a label made only of molecule + these is still a "generic" label.
# Used by the verifier's downrank tier (passed as ``extra``) so "AMLODIPINE
# BESYLATE USP" reads as generic; the matcher's aggregation path passes nothing,
# keeping aggregation strict (molecule words only).
_GENERIC_FILLER_WORDS = frozenset({
    "besylate", "mesylate", "maleate", "sulfate", "sulphate", "hydrochloride",
    "hcl", "hydrobromide", "fumarate", "succinate", "tartrate", "acetate",
    "sodium", "potassium", "calcium", "magnesium", "dihydrate",
    "usp", "bp", "ep", "ph", "eur",
    "tablet", "tablets", "tab", "tabs", "capsule", "capsules", "cap", "caps",
    "fc", "ec", "xr", "sr", "er", "la", "cr", "od", "mr",
})


def _is_generic_brand(brand_norm: str, molecule: str, extra: frozenset = frozenset()) -> bool:
    """True when an IQVIA Product is a *generic molecule label*, not a real brand.

    A generic label (e.g. IQVIA "AMLODIPINE" / "METFORMIN") carries only the
    molecule name(s) — no manufacturer prefix.  Such a label is one of the ways
    IQVIA spells the *same physical product* a company also sells under its brand;
    when that company's only marketed DIN is brand-named, the generic-labelled
    sales belong to (and must aggregate onto) that DIN.

    A distinct brand (e.g. "PHARMA-AMLODIPINE", "PRO-METFORMIN") carries a
    manufacturer token absent from the molecule, so it is its OWN product and must
    never be aggregated onto a sibling brand's DIN.

    Test: every alphabetic run in the normalised brand appears in the molecule's
    word set.  "amlodipine" ⊆ {amlodipine} → generic; "pms-amlodipine" has "pms"
    ∉ molecule → distinct.  Runs of ANY length are checked (not just ≥3 letters)
    so a one- or two-letter manufacturer prefix is not silently ignored:
    "m-amlodipine" (Mantra) keeps its "m" token and is correctly distinct — vital,
    because treating it as generic would let it aggregate onto a coincidentally
    similar-named company's DIN (M-AMLODIPINE/MANTRA → MINT-AMLODIPINE).
    """
    bwords = set(re.findall(r"[a-z]+", brand_norm.lower()))
    if not bwords:
        return False
    mwords = set(re.findall(r"[a-z]+", str(molecule).lower())) | extra
    return bwords <= mwords


# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_iqvia(file_bytes: bytes) -> pd.DataFrame:
    """Read the 'data' sheet from an IQVIA Excel export.

    Returns the raw DataFrame (one row per channel × province × pack).
    Header row is assumed to be row 1 (0-indexed row 0); data from row 2.

    Metric cells containing '-' or blank are converted to 0.

    A hidden ``_excel_row`` column is added recording the 1-based Excel row
    number for each data row (header=row 1, first data row=row 2).  This is
    used by collapse_iqvia() to produce provenance strings for the debug column.
    """
    # keep_default_na=False is critical: pandas otherwise converts "N/A", "NA",
    # "NULL", "#N/A" etc. to NaN at read time, which would then be indistinguishable
    # from a truly empty cell and silently zeroed. Keeping them as literal strings
    # lets the metric parser below decide (true blanks → 0; everything else → loud
    # error) instead of losing real sales to pandas' NA inference.
    df = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="data",
        header=0,
        dtype=str,
        keep_default_na=False,
    )
    df.columns = [str(c).strip() for c in df.columns]

    # Excel row 1 = header; data rows start at Excel row 2 (pandas index 0).
    df["_excel_row"] = range(2, len(df) + 2)

    metric_cols = [c for c in df.columns if _METRIC_COL_RE.match(c)]
    for col in metric_cols:
        raw = df[col].astype(str).str.strip()
        # Cells that legitimately mean "no data" → 0. Detect real nulls via
        # .isna() (an empty Excel cell read with dtype=str comes back as NaN, not
        # the string "nan") plus the textual "no data" sentinels.
        is_blank = df[col].isna() | raw.isin(["-", "", "nan", "None", "NaN", "<NA>"])
        # Strip thousands separators ("1,234" / "1 234" → "1234") before parsing.
        cleaned = (
            raw.str.replace(",", "", regex=False)
               .str.replace(r"\s+", "", regex=True)
               .where(~is_blank, "0")
        )
        numeric = pd.to_numeric(cleaned, errors="coerce")
        # A non-blank cell that fails to parse would be silently zeroed, vanishing
        # real sales with no error. Fail loud instead so the offending formats are
        # surfaced and mapped rather than lost (e.g. "*", "<10", "N/A", "1.2K").
        bad = numeric.isna() & ~is_blank
        if bad.any():
            offending = sorted(set(raw[bad]))
            raise ValueError(
                f"IQVIA metric column {col!r} has {int(bad.sum())} non-numeric "
                f"cell(s) that cannot be parsed and would be silently zeroed: "
                f"{offending[:10]}{' …' if len(offending) > 10 else ''}. "
                "Clean or map these values before import — they represent real sales."
            )
        df[col] = numeric.fillna(0).astype("int64")

    return df


def detect_metric_columns(df: pd.DataFrame) -> list[str]:
    """Return metric column names in the order they appear."""
    return [c for c in df.columns if _METRIC_COL_RE.match(c)]


# ── Collapsing ────────────────────────────────────────────────────────────────

_GROUP_KEY = ["Combined Molecule", "Product", "Manufacturer", "Strength"]


def collapse_iqvia(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse channel × province × pack rows to one row per product group.

    Groups by (Combined Molecule, Product, Manufacturer, Strength).
    All metric columns are summed; non-metric columns (Channel, Province,
    Pack, Product Form, Form 3, Corporation) are dropped.

    When ``_excel_row`` is present (added by parse_iqvia), an additional
    ``_source_excel_rows`` column is emitted containing the sorted, comma-
    separated Excel row numbers for every raw row summed into each group.
    This column is the provenance source for the debug audit column.
    """
    metric_cols = detect_metric_columns(df)
    # A missing grouping key (renamed/dropped column) must fail loudly: silently
    # grouping on the remaining keys would merge distinct products — e.g. dropping
    # Manufacturer collapses every company's "METFORMIN 500MG" into one row and
    # overcounts — with no error and no way to notice downstream.
    missing_keys = [k for k in _GROUP_KEY if k not in df.columns]
    if missing_keys:
        raise ValueError(
            f"IQVIA data is missing required grouping column(s): {missing_keys}. "
            f"Present columns: {[c for c in df.columns if not str(c).startswith('_')]}. "
            "Grouping without them would silently merge distinct products and overcount."
        )
    present_keys = list(_GROUP_KEY)
    grouped = (
        df[present_keys + metric_cols]
        .groupby(present_keys, as_index=False)[metric_cols]
        .sum()
    )
    if "_excel_row" in df.columns:
        prov = (
            df.groupby(present_keys)["_excel_row"]
            .apply(lambda s: ", ".join(str(r) for r in sorted(s)))
            .reset_index()
            .rename(columns={"_excel_row": "_source_excel_rows"})
        )
        grouped = grouped.merge(prov, on=present_keys, how="left")
    return grouped


# ── Matching ──────────────────────────────────────────────────────────────────

def match_iqvia_to_sheet1(
    sheet1_df: pd.DataFrame,
    iqvia_collapsed: pd.DataFrame,
    debug_iqvia_rows: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach IQVIA metric columns to Sheet 1 by matching IQVIA groups to DINs.

    Returns
    -------
    enriched_df
        Sheet 1 DataFrame with metric columns appended.  DINs that could not
        be confidently and unambiguously matched have None in those columns.
    reconciliation_df
        One row per IQVIA group, plus one row per DIN that had no IQVIA match,
        documenting the outcome of every matching decision.
    """
    if sheet1_df.empty or iqvia_collapsed.empty:
        return sheet1_df.copy(), _empty_reconciliation()

    metric_cols = detect_metric_columns(iqvia_collapsed)
    if not metric_cols:
        return sheet1_df.copy(), _empty_reconciliation()

    # ── Build normalised lookup for each Sheet 1 row ──────────────────────────
    # Each row: din, ingredient, brand_name, company, strength
    s1 = sheet1_df.copy()

    _PLACEHOLDER = {"not in dpd", "not applicable", "n/a", "none", ""}

    def _s1_rows() -> list[dict]:
        result = []
        seen_dins: set[str] = set()
        for _, row in s1.iterrows():
            din = str(row.get("din", "") or "").strip()
            if not din:
                continue
            # A combination drug queried under several ingredients appears in
            # multiple Sheet-1 rows with the SAME DIN. Each DIN is one product, so
            # it must be a single candidate — otherwise the duplicates compete
            # against each other (identical score, gap 0) and the DIN is wrongly
            # flagged ambiguous and blanked. Keep the first occurrence only.
            if din in seen_dins:
                continue
            seen_dins.add(din)
            ing = str(row.get("ingredient", "") or "").strip()
            brand = str(row.get("brand_name", "") or "").strip()
            company = str(row.get("company", "") or "").strip()
            strength = str(row.get("strength", "") or "").strip()
            status = str(row.get("status", "") or "").strip().lower()
            # Skip rows that are DPD sentinels ("Not in DPD" etc.)
            if brand.lower() in _PLACEHOLDER or ing.lower() in _PLACEHOLDER:
                continue
            # Never-marketed DINs ("Approved" = approved but never launched;
            # "Cancelled Pre Market" = cancelled before launch) have no sales
            # history and can never own IQVIA revenue. Including them as candidates
            # creates false near-ties against the correctly marketed DIN from the
            # same manufacturer (e.g. APO-ABACAVIR-LAMIVUDINE TABLETS DIN 02518287
            # vs the marketed DIN 02399539), causing the real match to be flagged
            # ambiguous and receive no IQVIA data.
            if status in _NEVER_MARKETED_STATUSES:
                continue
            result.append({
                "din": din,
                "ingredient": ing,
                "brand_norm": _norm_brand(brand),
                "company_norm": _norm_company(company),
                "strength_set": _norm_strength(strength),
                "is_marketed": status == _MARKETED_STATUS,
            })
        return result

    s1_rows = _s1_rows()

    # ── Precompute static per-group candidate sets (the expensive filters) ────
    # Strength + molecule prefilters do not depend on what has been claimed, so
    # they run ONCE per group here; both passes below only do cheap set-membership
    # and scoring on the cached lists, keeping the two-pass design as fast as the
    # old single pass.
    groups: list[dict] = []
    for iq_idx, iq_row in iqvia_collapsed.iterrows():
        molecule = str(iq_row.get("Combined Molecule", "") or "").strip()
        product = str(iq_row.get("Product", "") or "").strip()
        manufacturer = str(iq_row.get("Manufacturer", "") or "").strip()
        strength_raw = str(iq_row.get("Strength", "") or "").strip()
        iq_strength_set = _norm_strength(strength_raw)
        iq_molecule_norm = molecule.upper()
        # Step 1: strength prefilter; Step 1b: molecule prefilter (see
        # _molecule_overlap — rejects cross-molecule and mono-vs-combo collisions).
        base_cands = [
            r for r in s1_rows
            if r["strength_set"] and r["strength_set"] == iq_strength_set
            and _molecule_overlap(iq_molecule_norm, r["ingredient"])
        ]
        groups.append({
            "idx": iq_idx,
            "row": iq_row,
            "grp": (molecule, product, manufacturer, strength_raw),
            "molecule_norm": iq_molecule_norm,
            "brand_norm": _norm_brand(product),
            "company_norm": _norm_company(manufacturer),
            "strength_raw": strength_raw,
            "base_cands": base_cands,
        })

    def _live_candidates(g: dict, claimed: set[str]) -> list[dict]:
        """Candidates after excluding claimed DINs and applying the lifecycle tier.

        Step 1c: when any remaining candidate is currently marketed, restrict to
        marketed ones so discontinued siblings cannot create false near-ties;
        fall back to discontinued candidates only when no marketed DIN matches.
        """
        avail = [r for r in g["base_cands"] if r["din"] not in claimed]
        marketed = [r for r in avail if r["is_marketed"]]
        return marketed if marketed else avail

    # din_to_groups: DIN → list of IQVIA group indices whose metrics it carries.
    # A DIN normally carries exactly one group; same-company generic-label aliases
    # (Pass 2 aggregation) can add more, so the value is a list and metrics are
    # summed at stamp time.
    din_to_groups: dict[str, list] = {}
    exact_reserved: dict[str, object] = {}  # DIN → group idx (Pass-1 reservations)
    recon_by_idx: dict[object, dict] = {}   # group idx → its single recon row

    # ── Pass 1: exact-brand reservation (highest-confidence signal) ───────────
    # When a same-strength/molecule IQVIA group's normalised brand exactly equals
    # one candidate DIN's — and no same-company competitor is within TIE_MARGIN of
    # it — reserve that DIN for that group BEFORE the fuzzy stage. This stops a
    # generic or same-prefix sibling group (processed earlier in sort order) from
    # greedily claiming a DIN that belongs, by exact brand, to a later group — the
    # PMS↔PMS / PRZ↔PRZ / JAMP↔JAMP / PRO-METFORMIN swaps. The same-company
    # TIE_MARGIN guard keeps genuine pack-size / two-marketed-brand ambiguities
    # (PROVERA vs PROVERA PAK; JAMP's generic vs JAMP-AMLODIPINE DINs) unreserved
    # so the fuzzy pass can flag them ambiguous, and avoids needing to raise
    # MIN_COMPANY_SIM (which would kill the CHEPLAPHARM-style genuine matches).
    for g in groups:
        if not g["brand_norm"]:
            continue
        avail = _live_candidates(g, set(exact_reserved))
        exact = [
            r for r in avail
            if r["brand_norm"] == g["brand_norm"]
            and _sim(g["company_norm"], r["company_norm"]) >= MIN_COMPANY_SIM
        ]
        if len(exact) != 1:
            # 0 → no exact match, defer to fuzzy; >1 → genuinely ambiguous
            # (pack-size duplicates / two DINs share the brand), also deferred so
            # the fuzzy near-tie gate blanks it rather than arbitrarily assigning.
            continue
        D = exact[0]
        d_score = 50.0 + 0.5 * _sim(g["company_norm"], D["company_norm"])  # brand leg = 100
        tie = False
        for r in avail:
            if r["din"] == D["din"]:
                continue
            cs = _sim(g["company_norm"], r["company_norm"])
            if cs < MIN_COMPANY_SIM:
                continue
            if d_score - (0.5 * _sim(g["brand_norm"], r["brand_norm"]) + 0.5 * cs) < TIE_MARGIN:
                tie = True
                break
        if tie:
            continue
        exact_reserved[D["din"]] = g["idx"]
        din_to_groups[D["din"]] = [g["idx"]]
        recon_by_idx[g["idx"]] = _recon_row(
            iqvia_group=g["grp"], metric_cols=metric_cols, iq_row=g["row"],
            status="matched", notes=f"exact-brand match; score={d_score:.0f}",
            din=D["din"], top_score=d_score, second_score=0.0,
        )

    # ── Pass 2: fuzzy gate (unchanged) + generic-label alias aggregation ──────
    for g in groups:
        if g["idx"] in recon_by_idx:
            continue  # already reserved in Pass 1
        molecule_candidates = _live_candidates(g, set(din_to_groups))

        # Step 2: score each candidate. Brand+company weighted 50/50 after a
        # company floor (MIN_COMPANY_SIM) — the manufacturer is the real
        # discriminator for generic molecule brands, rejecting coincidental
        # cross-company collisions (NORA's NRA-METFORMIN vs RIVA's DIN) while
        # keeping legal-suffix-only company differences (CHEPLAPHARM ≈ 63).
        scored: list[tuple[float, dict]] = []
        for r in molecule_candidates:
            company_sim = _sim(g["company_norm"], r["company_norm"])
            if company_sim < MIN_COMPANY_SIM:
                continue
            scored.append((_sim(g["brand_norm"], r["brand_norm"]) * 0.5 + company_sim * 0.5, r))
        scored.sort(key=lambda t: t[0], reverse=True)
        top_candidates = [(sc, r) for sc, r in scored if sc >= MIN_CANDIDATE]

        # Step 3: decide
        if not top_candidates:
            # Generic-label alias aggregation. A generic molecule label
            # (IQVIA "METFORMIN" / "AMLODIPINE") is the same physical product a
            # firm also sells under its brand. When that firm's only marketed DIN
            # is the brand-named one already reserved in Pass 1, no fuzzy candidate
            # survives (it was reserved out) — but the sales still belong to that
            # DIN, so aggregate (sum) rather than orphan. Distinct brands
            # (PHARMA-AMLODIPINE) are not generic, so they never aggregate onto a
            # sibling's DIN; their sales stay unmatched.
            agg = None
            if _is_generic_brand(g["brand_norm"], g["molecule_norm"]):
                best_cs = -1.0
                for r in g["base_cands"]:
                    if r["din"] not in exact_reserved:
                        continue
                    cs = _sim(g["company_norm"], r["company_norm"])
                    if cs >= AGG_MIN_COMPANY_SIM and cs > best_cs:
                        best_cs, agg = cs, r
            if agg is not None:
                din_to_groups[agg["din"]].append(g["idx"])
                recon_by_idx[g["idx"]] = _recon_row(
                    iqvia_group=g["grp"], metric_cols=metric_cols, iq_row=g["row"],
                    status="matched",
                    notes=f"generic-label alias aggregated onto exact-brand DIN {agg['din']}",
                    din=agg["din"], top_score=50.0 + 0.5 * best_cs, second_score=0.0,
                )
                continue
            recon_by_idx[g["idx"]] = _recon_row(
                iqvia_group=g["grp"], metric_cols=metric_cols, iq_row=g["row"],
                status="no_din_match",
                notes=f"No DIN had strength={g['strength_raw']!r} + score≥{MIN_CANDIDATE} (searched {len(molecule_candidates)} candidates)",
                din="", top_score=scored[0][0] if scored else 0.0,
                second_score=scored[1][0] if len(scored) > 1 else 0.0,
            )
            continue

        if len(top_candidates) >= 2:
            top_score, top_r = top_candidates[0]
            sec_score, _ = top_candidates[1]
            gap = top_score - sec_score
            if gap < TIE_MARGIN:
                dins_involved = ", ".join(r["din"] for _, r in top_candidates[:4])
                recon_by_idx[g["idx"]] = _recon_row(
                    iqvia_group=g["grp"], metric_cols=metric_cols, iq_row=g["row"],
                    status="ambiguous",
                    notes=f"Near-tie: top={top_score:.0f} gap={gap:.0f}<{TIE_MARGIN}; candidates: {dins_involved}",
                    din="", top_score=top_score, second_score=sec_score,
                )
                continue
            if top_score < CONFIDENT_THRESHOLD:
                recon_by_idx[g["idx"]] = _recon_row(
                    iqvia_group=g["grp"], metric_cols=metric_cols, iq_row=g["row"],
                    status="low_score", notes=f"Top score {top_score:.0f} < {CONFIDENT_THRESHOLD}",
                    din=top_r["din"], top_score=top_score, second_score=sec_score,
                )
                continue
            assigned_din = top_r["din"]
        else:
            top_score, top_r = top_candidates[0]
            if top_score < CONFIDENT_THRESHOLD:
                recon_by_idx[g["idx"]] = _recon_row(
                    iqvia_group=g["grp"], metric_cols=metric_cols, iq_row=g["row"],
                    status="low_score", notes=f"Top score {top_score:.0f} < {CONFIDENT_THRESHOLD}",
                    din=top_r["din"], top_score=top_score, second_score=0.0,
                )
                continue
            assigned_din = top_r["din"]
            sec_score = 0.0

        din_to_groups[assigned_din] = [g["idx"]]
        recon_by_idx[g["idx"]] = _recon_row(
            iqvia_group=g["grp"], metric_cols=metric_cols, iq_row=g["row"],
            status="matched", notes=f"score={top_score:.0f}",
            din=assigned_din, top_score=top_score,
            second_score=sec_score if len(top_candidates) >= 2 else 0.0,
        )

    # Emit recon rows in stable group order (matches collapse_iqvia's sort).
    recon_rows: list[dict] = [recon_by_idx[g["idx"]] for g in groups]

    # ── Append DINs that got no IQVIA group assigned ──────────────────────────
    matched_dins = set(din_to_groups.keys())
    for r in s1_rows:
        if r["din"] not in matched_dins:
            recon_rows.append({
                "iqvia_molecule": "",
                "iqvia_product": "",
                "iqvia_manufacturer": "",
                "iqvia_strength": "",
                "din": r["din"],
                "status": "din_no_iqvia_match",
                "top_score": None,
                "second_score": None,
                "notes": "No IQVIA group matched this DIN",
                **{c: None for c in metric_cols},
            })

    # ── Merge metric cols into sheet1 ─────────────────────────────────────────
    # Build a mapping: DIN → metric values + debug info, summing every IQVIA group
    # the DIN carries (≥1; >1 only for same-company generic-label aggregation).
    din_metric_map: dict[str, dict] = {}
    din_debug_rows: dict[str, str] = {}    # DIN → source excel rows string
    din_debug_product: dict[str, str] = {} # DIN → "Product (Manufacturer)" label
    has_provenance = "_source_excel_rows" in iqvia_collapsed.columns
    for din, iq_idxs in din_to_groups.items():
        iq_rows = [iqvia_collapsed.loc[ix] for ix in iq_idxs]
        din_metric_map[din] = {
            col: sum(int(ir[col]) for ir in iq_rows) for col in metric_cols
        }
        if debug_iqvia_rows:
            if has_provenance:
                din_debug_rows[din] = "; ".join(
                    str(ir.get("_source_excel_rows") or "") for ir in iq_rows
                )
            labels = []
            for ir in iq_rows:
                product = str(ir.get("Product") or "").strip()
                mfr = str(ir.get("Manufacturer") or "").strip()
                labels.append(f"{product} ({mfr})" if mfr else product)
            din_debug_product[din] = " + ".join(labels)

    # Stamp metric values onto Sheet 1, but only on the FIRST row per DIN.
    # A combination drug queried under several of its ingredients appears once in
    # each ingredient's vertical block — multiple Sheet-1 rows sharing one DIN.
    # Mapping the IQVIA value onto every such row would double-count it in any
    # column-sum (dashboard KPIs, totals). Assigning it to the first occurrence
    # only keeps each IQVIA group's sales present exactly once across the sheet.
    din_order = [str(d).strip() for d in s1["din"].tolist()]
    carried: set[str] = set()
    row_carries: list[bool] = []
    for d in din_order:
        if d in din_metric_map and d not in carried:
            carried.add(d)
            row_carries.append(True)
        else:
            row_carries.append(False)

    for col in metric_cols:
        s1[col] = [
            din_metric_map[din_order[i]][col] if row_carries[i] else None
            for i in range(len(din_order))
        ]

    if debug_iqvia_rows:
        s1["IQVIA Source Rows (debug)"] = [
            din_debug_rows.get(din_order[i]) if row_carries[i] else None
            for i in range(len(din_order))
        ]
        s1["IQVIA Matched Product (debug)"] = [
            din_debug_product.get(din_order[i]) if row_carries[i] else None
            for i in range(len(din_order))
        ]

    recon_df = pd.DataFrame(recon_rows) if recon_rows else _empty_reconciliation()
    # Ensure consistent column order
    recon_col_order = [
        "status", "iqvia_molecule", "iqvia_product", "iqvia_manufacturer",
        "iqvia_strength", "din", "top_score", "second_score", "notes",
    ] + metric_cols
    existing = [c for c in recon_col_order if c in recon_df.columns]
    extra = [c for c in recon_df.columns if c not in set(recon_col_order)]
    recon_df = recon_df[existing + extra]

    return s1, recon_df


def _recon_row(
    iqvia_group: tuple[str, str, str, str],
    metric_cols: list[str],
    iq_row: "pd.Series",
    status: str,
    notes: str,
    din: str,
    top_score: float,
    second_score: float,
) -> dict:
    molecule, product, manufacturer, strength = iqvia_group
    row: dict = {
        "iqvia_molecule": molecule,
        "iqvia_product": product,
        "iqvia_manufacturer": manufacturer,
        "iqvia_strength": strength,
        "din": din,
        "status": status,
        "top_score": round(top_score, 1),
        "second_score": round(second_score, 1),
        "notes": notes,
    }
    for c in metric_cols:
        row[c] = int(iq_row[c]) if c in iq_row.index else None
    return row


def _empty_reconciliation() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "status", "iqvia_molecule", "iqvia_product", "iqvia_manufacturer",
        "iqvia_strength", "din", "top_score", "second_score", "notes",
    ])
