"""Go/no-go product screening over already-built workbook data.

A *product* is a unique (ingredient(s) + dosage form) pair.  Dosage form is taken
VERBATIM — "Tablet" and "Tablet (extended-release)" are different products and are
never collapsed.  Strength is NOT part of the product key.

Six criteria are computed per product from the built Sheet 1 ("DPD + NOC +
Patents", one row per DIN, with IQVIA metric columns already appended when an
IQVIA file was loaded) and Sheet 2 ("Generic Submissions"):

  1. competitors    distinct companies among the product's MARKETED DINs (DPD)
  2. filings        GSUR submission rows matching the product's ingredient(s)
  3. approvals      distinct companies among ALL the product's DINs (every
                    Sheet-1 DIN has an NOC by construction, so this is the count
                    of distinct companies with an approval)
  4. value          sum of the latest 'Dollars MAT' column over the product DINs
  5. quantity       sum of the latest 'Units MAT' column over the product DINs
  6. quantity_ext   sum of the latest 'Ext Units MAT' column over the product DINs

DINs with no IQVIA match contribute 0 to sums 4-6 (their metric cells are None).
Criteria 4-6 are meaningful only when an IQVIA file is loaded; without it the
sums are 0 and a value/quantity criterion raises.

This module performs NO scraping — it operates only on the supplied DataFrames,
which are the exact contents of the full export workbook.

Output is a two-tab workbook:
  - Summary: one row per QUALIFYING product, showing the EXACT value of all six
    criteria (including those not filtered on).
  - Detail: the full DIN-level Sheet-1 rows for every DIN in a qualifying product.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from app.grouping import COMBINATION_SEPARATOR, _normalize_name

# ── IQVIA latest-period metric detection ──────────────────────────────────────

# "Dollars MAT 12/2025", "Units MAT 12/2024", "Ext Units MAT 12/2025"
_MAT_RE = re.compile(
    r"^(Dollars|Units|Ext\s+Units)\s+MAT\s+(\d{2})/(\d{4})$",
    re.IGNORECASE,
)

# Family display key → the criterion metric name it backs.
_FAMILY_TO_METRIC = {
    "dollars": "value",
    "units": "quantity",
    "ext units": "quantity_ext",
}

# Criterion metric → the computed product column it filters on.
_METRIC_TO_COLUMN = {
    "competitors": "competitors",
    "filings": "filings",
    "approvals": "approvals",
    "value": "value_sizeable",
    "quantity": "quantity_sizeable",
    "quantity_ext": "quantity_ext_sizeable",
}

# Metrics that require IQVIA data to be present.
_IQVIA_METRICS = frozenset({"value", "quantity", "quantity_ext"})

_VALID_OPERATORS = frozenset({"above", "below", "exactly"})

# Summary tab column order + display headers.
_SUMMARY_COLS = (
    "ingredient", "dosage_form",
    "competitors", "filings", "approvals",
    "value_sizeable", "quantity_sizeable", "quantity_ext_sizeable",
)
_SUMMARY_HEADERS = {
    "ingredient": "Ingredient",
    "dosage_form": "Dosage Form",
    "competitors": "Number of Competitors",
    "filings": "Number of Filings",
    "approvals": "Number of Approvals",
    "value_sizeable": "Value Sizeable ($)",
    "quantity_sizeable": "Quantity Sizeable (Units)",
    "quantity_ext_sizeable": "Quantity Ext Sizeable",
}

# Salt-form / pharmacopoeia filler words stripped before molecule comparison so a
# GSUR "metformin hydrochloride" filing matches a DPD "metformin" product.
_SALT_FILLER_WORDS = frozenset({
    "hydrochloride", "hcl", "hydrobromide", "sulfate", "sulphate", "mesylate",
    "besylate", "maleate", "fumarate", "succinate", "tartrate", "acetate",
    "sodium", "potassium", "calcium", "magnesium", "dihydrate", "monohydrate",
    "usp", "bp", "ep",
})


@dataclass(frozen=True)
class Criterion:
    """A single go/no-go test: ``metric operator value`` (e.g. competitors above 3)."""
    metric: str
    operator: str
    value: float

    def __post_init__(self) -> None:
        if self.metric not in _METRIC_TO_COLUMN:
            raise ValueError(
                f"Unknown criterion metric {self.metric!r}; "
                f"expected one of {sorted(_METRIC_TO_COLUMN)}"
            )
        if self.operator not in _VALID_OPERATORS:
            raise ValueError(
                f"Unknown operator {self.operator!r}; expected one of {sorted(_VALID_OPERATORS)}"
            )


def parse_criteria(raw: Optional[list[dict]]) -> list[Criterion]:
    """Build a list of Criterion from request dicts, skipping blank/unset entries.

    Each dict is ``{"metric": ..., "operator": ..., "value": ...}``.  Entries with
    a missing metric/operator or a blank value are silently dropped so the form can
    send all six rows and only the filled-in ones take effect.
    """
    out: list[Criterion] = []
    for entry in raw or []:
        metric = str(entry.get("metric") or "").strip()
        operator = str(entry.get("operator") or "").strip().lower()
        value = entry.get("value")
        if not metric or not operator:
            continue
        if value is None or str(value).strip() == "":
            continue
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"Criterion value for {metric!r} is not numeric: {value!r}")
        out.append(Criterion(metric=metric, operator=operator, value=value_f))
    return out


def latest_metric_columns(columns: list[str]) -> dict[str, Optional[str]]:
    """Return the latest-period column name for each IQVIA metric family.

    Picks the column with the maximum (year, month) for each of Dollars / Units /
    Ext Units.  Families absent from ``columns`` map to None.
    """
    best: dict[str, tuple[tuple[int, int], str]] = {}
    for col in columns:
        m = _MAT_RE.match(str(col).strip())
        if not m:
            continue
        family = re.sub(r"\s+", " ", m.group(1)).strip().lower()
        metric = _FAMILY_TO_METRIC.get(family)
        if metric is None:
            continue
        period = (int(m.group(3)), int(m.group(2)))  # (year, month)
        prev = best.get(metric)
        if prev is None or period > prev[0]:
            best[metric] = (period, str(col))
    return {
        "value": best["value"][1] if "value" in best else None,
        "quantity": best["quantity"][1] if "quantity" in best else None,
        "quantity_ext": best["quantity_ext"][1] if "quantity_ext" in best else None,
    }


# ── Grouping helpers ──────────────────────────────────────────────────────────

# A trailing dose/strength token on an ingredient component, e.g. the " 150 MG"
# in DPD's "ALPELISIB 150 MG", or " 10 MG/ML" / " 10 MG/5 ML" concentrations.
# Strength is NEVER part of the product key — a manufacturer's "metformin 200 MG
# Tablet" and "metformin 400 MG Tablet" are the SAME product (same molecule, same
# form).  Each component is split off first, so only one trailing dose can appear.
_STRENGTH_TAIL_RE = re.compile(
    r"\s+\d[\d.,]*\s*"
    r"(?:mcg|mg|miu|iu|meq|ml|kg|ug|units?|g|l|%)"
    r"(?:\s*/\s*[\d.,]*\s*(?:mcg|mg|ug|ml|g|l)?)?"
    r"\s*$",
    re.IGNORECASE,
)


def _strip_strength(component: str) -> str:
    """Remove a trailing strength/dose token from one ingredient component."""
    return _STRENGTH_TAIL_RE.sub("", str(component)).strip()


def _ingredient_key(ingredient: Any) -> tuple[str, ...]:
    """Sorted, deduplicated tuple of normalized ingredient names (combo-safe).

    Strength tokens are stripped per component so the key is strength-agnostic.
    """
    parts = re.split(r"[;+]", str(ingredient or ""))
    names = {_normalize_name(_strip_strength(p)) for p in parts if p.strip()}
    return tuple(sorted(n for n in names if n))


def _ingredient_label(key: tuple[str, ...]) -> str:
    return COMBINATION_SEPARATOR.join(key)


def _dosage_form_key(dosage_form: Any) -> str:
    """Verbatim dosage form (whitespace-trimmed only) — modifiers preserved."""
    if dosage_form is None:
        return ""
    try:
        if pd.isna(dosage_form):
            return ""
    except (TypeError, ValueError):
        pass
    return re.sub(r"\s+", " ", str(dosage_form).strip())


def _molecule_tokens(ingredient: Any) -> frozenset[str]:
    """Significant molecule word tokens, salt-form fillers removed."""
    toks = set(re.findall(r"[a-z]{3,}", str(ingredient or "").lower()))
    return frozenset(toks - _SALT_FILLER_WORDS)


def _num(v: Any) -> float:
    """Coerce a metric cell to a float; None/NaN/blank → 0.0."""
    if v is None:
        return 0.0
    try:
        if pd.isna(v):
            return 0.0
    except (TypeError, ValueError):
        pass
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


# ── Product computation ───────────────────────────────────────────────────────

def compute_products(
    sheet1_df: pd.DataFrame,
    sheet2_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Compute the six criteria for every (ingredient + dosage form) product.

    Returns ``(products_df, warnings)``.  ``products_df`` has one row per product
    with all six criteria values plus an internal ``_dins`` (list of DINs) and
    ``_ing_key`` (tuple) column.  ``warnings`` lists products whose dosage form is
    blank (which should not normally happen).
    """
    metric_cols = latest_metric_columns(list(sheet1_df.columns))
    col_value = metric_cols["value"]
    col_qty = metric_cols["quantity"]
    col_qty_ext = metric_cols["quantity_ext"]

    # Pre-index GSUR filings by molecule-token set for O(1) per-product lookup.
    filings_by_tokens: dict[frozenset[str], int] = {}
    if not sheet2_df.empty and "medicinal_ingredient" in sheet2_df.columns:
        for ing in sheet2_df["medicinal_ingredient"]:
            toks = _molecule_tokens(ing)
            if toks:
                filings_by_tokens[toks] = filings_by_tokens.get(toks, 0) + 1

    # Accumulate per product key.
    products: dict[tuple[tuple[str, ...], str], dict[str, Any]] = {}
    if not sheet1_df.empty:
        for _, row in sheet1_df.iterrows():
            din = str(row.get("din", "") or "").strip()
            if not din:
                continue
            ing_key = _ingredient_key(row.get("ingredient"))
            if not ing_key:
                continue
            form = _dosage_form_key(row.get("dosage_form"))
            pkey = (ing_key, form)
            p = products.get(pkey)
            if p is None:
                p = {
                    "ing_key": ing_key,
                    "dosage_form": form,
                    "dins": [],
                    "din_set": set(),
                    "all_companies": set(),
                    "marketed_companies": set(),
                    "value": 0.0,
                    "quantity": 0.0,
                    "quantity_ext": 0.0,
                }
                products[pkey] = p

            company = str(row.get("company", "") or "").strip()
            status = str(row.get("status", "") or "").strip().lower()
            if din not in p["din_set"]:
                p["din_set"].add(din)
                p["dins"].append(din)
            if company:
                p["all_companies"].add(company)
                if status == "marketed":
                    p["marketed_companies"].add(company)
            # Metric cells are populated on the FIRST Sheet-1 row of each DIN only
            # (the matcher stamps once per DIN), so summing every row never
            # double-counts; unmatched DINs contribute 0 via _num().
            if col_value:
                p["value"] += _num(row.get(col_value))
            if col_qty:
                p["quantity"] += _num(row.get(col_qty))
            if col_qty_ext:
                p["quantity_ext"] += _num(row.get(col_qty_ext))

    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for (ing_key, form), p in products.items():
        label = _ingredient_label(ing_key)
        if not form:
            warnings.append(label)
        filings = filings_by_tokens.get(_tokens_of_key(ing_key), 0)
        rows.append({
            "ingredient": label,
            "dosage_form": form,
            "competitors": len(p["marketed_companies"]),
            "filings": filings,
            "approvals": len(p["all_companies"]),
            "value_sizeable": _as_int(p["value"]),
            "quantity_sizeable": _as_int(p["quantity"]),
            "quantity_ext_sizeable": _as_int(p["quantity_ext"]),
            "_dins": list(p["dins"]),
            "_ing_key": ing_key,
        })

    # Deterministic order: ingredient label, then dosage form.
    rows.sort(key=lambda r: (r["ingredient"], r["dosage_form"]))
    products_df = pd.DataFrame(
        rows,
        columns=list(_SUMMARY_COLS) + ["_dins", "_ing_key"],
    )
    return products_df, sorted(set(warnings))


def _tokens_of_key(ing_key: tuple[str, ...]) -> frozenset[str]:
    """Molecule-token set for a product's ingredient key (matches GSUR indexing)."""
    return _molecule_tokens(" ".join(ing_key))


def _as_int(v: float) -> int:
    """IQVIA metrics are whole numbers (int64 cells); present sums as ints."""
    return int(round(v))


# ── Filtering ─────────────────────────────────────────────────────────────────

def _passes(value: float, operator: str, threshold: float) -> bool:
    if operator == "above":
        return value > threshold
    if operator == "below":
        return value < threshold
    # exactly
    return value == threshold


def apply_criteria(
    products_df: pd.DataFrame,
    criteria: list[Criterion],
) -> pd.DataFrame:
    """Return the subset of products passing ALL criteria (logical AND).

    Raises if any criterion targets an IQVIA metric while no IQVIA data was loaded
    (detected by every product's value/quantity sums being absent) — guarded at the
    caller, but re-checked here so a stray criterion can't silently pass on zeros.
    """
    if products_df.empty or not criteria:
        return products_df.copy()

    mask = pd.Series(True, index=products_df.index)
    for c in criteria:
        col = _METRIC_TO_COLUMN[c.metric]
        col_vals = products_df[col].apply(_num)
        mask &= col_vals.apply(lambda v, c=c: _passes(v, c.operator, c.value))
    return products_df[mask].reset_index(drop=True)


def requires_iqvia(criteria: list[Criterion]) -> bool:
    return any(c.metric in _IQVIA_METRICS for c in criteria)


# ── Workbook assembly ─────────────────────────────────────────────────────────

def build_summary_sheet(qualifying: pd.DataFrame) -> pd.DataFrame:
    """Public summary frame (display headers, no internal columns)."""
    cols = [c for c in _SUMMARY_COLS if c in qualifying.columns]
    df = qualifying[cols].copy() if not qualifying.empty else pd.DataFrame(columns=cols)
    return df.rename(columns=_SUMMARY_HEADERS)


def build_detail_sheet(
    sheet1_df: pd.DataFrame,
    qualifying: pd.DataFrame,
) -> pd.DataFrame:
    """DIN-level Sheet-1 rows for every DIN in a qualifying product (deduped, sorted)."""
    from app.enrichment.workbook import _apply_display_names

    if qualifying.empty or sheet1_df.empty:
        return _apply_display_names(sheet1_df.iloc[0:0].copy())

    keep: list[str] = []
    seen: set[str] = set()
    for dins in qualifying["_dins"]:
        for d in dins:
            if d not in seen:
                seen.add(d)
                keep.append(d)

    din_str = sheet1_df["din"].astype(str).str.strip()
    detail = sheet1_df[din_str.isin(seen)].copy()
    detail = detail.sort_values("din", kind="stable").reset_index(drop=True)
    return _apply_display_names(detail)


def build_filtered_workbook(
    sheet1_df: pd.DataFrame,
    sheet2_df: pd.DataFrame,
    criteria: list[Criterion],
) -> tuple[bytes, pd.DataFrame, pd.DataFrame, list[str]]:
    """Screen the built workbook data and return a two-tab filtered XLSX.

    Returns ``(xlsx_bytes, summary_df, detail_df, warnings)`` where summary_df and
    detail_df carry display-name headers (the same frames written to the file).
    """
    from app.enrichment.workbook import _style_sheet

    if requires_iqvia(criteria):
        metric_cols = latest_metric_columns(list(sheet1_df.columns))
        if not any(metric_cols.values()):
            raise ValueError(
                "Value / Quantity criteria require an IQVIA file, but none of the "
                "Dollars/Units/Ext Units MAT columns are present in the data."
            )

    products_df, warnings = compute_products(sheet1_df, sheet2_df)
    qualifying = apply_criteria(products_df, criteria)

    summary_out = build_summary_sheet(qualifying)
    detail_out = build_detail_sheet(sheet1_df, qualifying)

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        summary_out.to_excel(writer, sheet_name="Summary", index=False)
        _style_sheet(writer.sheets["Summary"], summary_out)
        detail_out.to_excel(writer, sheet_name="Detail", index=False)
        _style_sheet(writer.sheets["Detail"], detail_out)

    return buf.getvalue(), summary_out, detail_out, warnings
