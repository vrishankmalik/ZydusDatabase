"""Quarter-over-quarter comparison of two IQVIA Canada extracts.

The user pulls one IQVIA extract per quarter and wants to see ONLY what moved
since the previous pull — not a row-level dump.  A naive "any metric differs"
diff is useless: a MAT (Moving Annual Total) is a rolling 12-month sum that ticks
on nearly every row every period, so it flags ~80% of shared rows as "changed".

This module reuses the platform's canonical IQVIA path end-to-end:

  parse_iqvia → collapse_iqvia        (one row per product, summed across
                                       channel × province × pack — the same grain
                                       the DIN matcher consumes)
  _norm_brand / _norm_company /       (the matcher's identity normalisation, so
  _norm_strength                       quarter-to-quarter formatting jitter does
                                       not masquerade as add/remove churn)
  latest_mat_metrics                   (resolve the newest MAT period in EACH file
                                       independently — the two extracts do not
                                       share column names or even date order)

It then compares the latest-MAT value of each product across the two files and
emits three signals:

  • NEW entrants  — present in new, absent (or zero) in old
  • EXITS         — present in old, absent (or zero) in new
  • MATERIAL MOVES— present in both, and the move clears the materiality gate
                    (config.IQVIA_DIFF_* — absolute AND percent floor on Dollars
                    or Units; Ext Units is shown for context but, being collinear
                    with Units, is not an independent trigger)

Below-threshold moves are dropped.  Entrants and exits are never thresholded —
appearing or disappearing from the market is always material — and are sorted by
size so the largest land first.  Nothing is ever invented: a missing metric for a
period is read as 0, and Δ% is left blank when the old base is 0.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from app.config import IQVIA_DIFF_DOLLARS_ABS, IQVIA_DIFF_UNITS_ABS, IQVIA_DIFF_PCT
from app.enrichment.iqvia import (
    parse_iqvia,
    collapse_iqvia,
    detect_metric_columns,
    latest_mat_metrics,
    _norm_brand,
    _norm_company,
    _norm_strength,
)

# Identity / display column names (match the IQVIA schema verbatim).
_ID_COLS = ["Combined Molecule", "Product", "Manufacturer", "Strength"]
# Canonical metric keys, in display order.
_METRICS = ["dollars", "units", "ext_units"]
_METRIC_LABEL = {"dollars": "Dollars", "units": "Units", "ext_units": "Ext Units"}


def _period_label(period: Optional[tuple[int, int]]) -> str:
    """Format a (year, month) MAT period as 'YYYY/MM', or '—' when unknown."""
    if not period:
        return "—"
    return f"{period[0]}/{period[1]:02d}"


def _identity(molecule: object, product: object, manufacturer: object, strength: object) -> tuple:
    """Normalised cross-file identity key for one product group.

    Reuses the matcher's normalisation so trivial formatting differences between
    quarters (legal-suffix changes, whitespace, trailing form words, strength
    punctuation) do not split one product into a phantom exit + entrant pair.
    """
    mol = " ".join(str(molecule or "").split()).upper()
    return (
        mol,
        _norm_brand(product),
        _norm_company(manufacturer),
        tuple(sorted(_norm_strength(strength))),
    )


def _aggregate_latest(file_bytes: bytes) -> tuple[Optional[tuple[int, int]], dict[tuple, dict]]:
    """Collapse a file and fold it to one latest-MAT triple per normalised identity.

    Returns ``(latest_period, {identity: {dollars, units, ext_units, + raw display
    fields}})``.  Collapsed groups that normalise to the same identity are summed
    (accuracy-conservative: never drops sales).  Display fields keep the raw values
    of the first collapsed group seen for that identity.
    """
    collapsed = collapse_iqvia(parse_iqvia(file_bytes))
    metric_cols = detect_metric_columns(collapsed)
    period, latest = latest_mat_metrics(metric_cols)

    agg: dict[tuple, dict] = {}
    for _, row in collapsed.iterrows():
        ident = _identity(
            row.get("Combined Molecule"), row.get("Product"),
            row.get("Manufacturer"), row.get("Strength"),
        )
        rec = agg.get(ident)
        if rec is None:
            rec = {
                "dollars": 0, "units": 0, "ext_units": 0,
                "Combined Molecule": str(row.get("Combined Molecule") or "").strip(),
                "Product": str(row.get("Product") or "").strip(),
                "Manufacturer": str(row.get("Manufacturer") or "").strip(),
                "Strength": str(row.get("Strength") or "").strip(),
            }
            agg[ident] = rec
        for key in _METRICS:
            col = latest.get(key)
            if col is not None:
                rec[key] += int(row.get(col, 0) or 0)
    return period, agg


def _present(rec: Optional[dict]) -> bool:
    """True when a product has any non-zero latest-MAT metric (i.e. is on market)."""
    return rec is not None and (rec["dollars"] > 0 or rec["units"] > 0 or rec["ext_units"] > 0)


def _is_material(old: dict, new: dict) -> bool:
    """True when the move clears the absolute AND percent floor on Dollars or Units.

    Ext Units is intentionally excluded as a trigger: it is units × pack size and
    moves in lock-step with Units, so triggering on it would add no independent
    signal while widening the noise.  It is still reported in the output columns.
    """
    for key, abs_floor in (("dollars", IQVIA_DIFF_DOLLARS_ABS), ("units", IQVIA_DIFF_UNITS_ABS)):
        delta = new[key] - old[key]
        if abs(delta) < abs_floor:
            continue
        base = old[key]
        pct = abs(delta) / base if base else float("inf")  # base 0 → infinite move
        if pct >= IQVIA_DIFF_PCT:
            return True
    return False


def _pct(delta: int, base: int) -> Optional[float]:
    """Signed percent change, or None when the base is 0 (Δ% undefined — never faked)."""
    if not base:
        return None
    return round(delta / base * 100.0, 1)


@dataclass
class IqviaDiff:
    """Result of comparing two IQVIA extracts at the canonical product grain."""
    entrants: pd.DataFrame
    exits: pd.DataFrame
    moves: pd.DataFrame
    old_period: Optional[tuple[int, int]]
    new_period: Optional[tuple[int, int]]
    warnings: list[str] = field(default_factory=list)
    reordered: bool = False


def compare_iqvia(old_bytes: bytes, new_bytes: bytes) -> IqviaDiff:
    """Compare an older and a newer IQVIA extract; return only what changed.

    ``old_bytes`` / ``new_bytes`` are the two uploaded files (xlsx or CSV) in the
    caller's slot order (slot 1 = old, slot 2 = new).  Old vs new is decided by the
    latest MAT period resolved per file, NOT by slot order: the file with the
    earlier latest period is old, the later is new.  If that reverses the slots, the
    files are auto-ordered older → newer and ``reordered`` is set (the workbook
    surfaces a prominent notice).  On a tie (same latest period in both) the order
    given is respected, with a plain informational note — never an error.
    """
    slot1_period, slot1_agg = _aggregate_latest(old_bytes)   # caller's slot 1 ("old")
    slot2_period, slot2_agg = _aggregate_latest(new_bytes)   # caller's slot 2 ("new")

    if slot1_period is None or slot2_period is None:
        raise ValueError(
            "No dated IQVIA metric columns (Dollars/Units/Ext Units MAT …) found in "
            f"{'slot 1 (the OLD file)' if slot1_period is None else 'slot 2 (the NEW file)'}. "
            "Make sure you uploaded the data extract, not a pivot/summary."
        )

    warnings: list[str] = []
    reordered = False
    if slot2_period < slot1_period:
        # Slot 1 is newer than slot 2 → the files were uploaded in the wrong order.
        # Auto-order older → newer so the comparison below always runs old → new;
        # the reversal is surfaced as a prominent banner in build_diff_workbook.
        old_period, old_agg = slot2_period, slot2_agg
        new_period, new_agg = slot1_period, slot1_agg
        reordered = True
    else:
        # slot 2 newer (correct order) OR equal (tie → respect the order given).
        old_period, old_agg = slot1_period, slot1_agg
        new_period, new_agg = slot2_period, slot2_agg
        if slot2_period == slot1_period:
            warnings.append(
                f"Both files share the same latest period ({_period_label(slot1_period)}); "
                "compared in the order given (slot 1 = old, slot 2 = new). Moves reflect "
                "revisions/additions between two pulls of the same period."
            )

    entrant_rows: list[dict] = []
    exit_rows: list[dict] = []
    move_rows: list[dict] = []

    for ident in set(old_agg) | set(new_agg):
        o = old_agg.get(ident)
        n = new_agg.get(ident)
        o_present, n_present = _present(o), _present(n)

        if n_present and not o_present:
            entrant_rows.append({
                **{c: n[c] for c in _ID_COLS},
                **{_METRIC_LABEL[k]: n[k] for k in _METRICS},
            })
        elif o_present and not n_present:
            exit_rows.append({
                **{c: o[c] for c in _ID_COLS},
                **{_METRIC_LABEL[k]: o[k] for k in _METRICS},
            })
        elif o_present and n_present and _is_material(o, n):
            row = {c: n[c] for c in _ID_COLS}
            for k in _METRICS:
                label = _METRIC_LABEL[k]
                delta = n[k] - o[k]
                row[f"{label} Old"] = o[k]
                row[f"{label} New"] = n[k]
                row[f"{label} Δ"] = delta
                row[f"{label} Δ%"] = _pct(delta, o[k])
            move_rows.append(row)

    entrant_cols = _ID_COLS + [_METRIC_LABEL[k] for k in _METRICS]
    move_cols = _ID_COLS + [
        f"{_METRIC_LABEL[k]} {suffix}"
        for k in _METRICS for suffix in ("Old", "New", "Δ", "Δ%")
    ]

    entrants = pd.DataFrame(entrant_rows, columns=entrant_cols)
    exits = pd.DataFrame(exit_rows, columns=entrant_cols)
    moves = pd.DataFrame(move_rows, columns=move_cols)

    if not entrants.empty:
        entrants = entrants.sort_values("Dollars", ascending=False, kind="mergesort").reset_index(drop=True)
    if not exits.empty:
        exits = exits.sort_values("Dollars", ascending=False, kind="mergesort").reset_index(drop=True)
    if not moves.empty:
        moves = (
            moves.assign(_absd=moves["Dollars Δ"].abs())
            .sort_values("_absd", ascending=False, kind="mergesort")
            .drop(columns="_absd")
            .reset_index(drop=True)
        )

    return IqviaDiff(entrants, exits, moves, old_period, new_period, warnings, reordered)


# ── Workbook ──────────────────────────────────────────────────────────────────

def build_diff_workbook(diff: IqviaDiff) -> bytes:
    """Render an IqviaDiff to a changes-only XLSX (Summary + 3 signal sheets)."""
    from app.enrichment.workbook import _style_sheet  # reuse the existing styling

    # A prominent reorder notice leads the sheet (above the counts) when the upload
    # slots were auto-corrected, so it cannot be missed as a buried warning row.
    # Periods are rendered via _period_label → always YYYY/MM regardless of source
    # file date format, so the two dates can never look mismatched.
    banner_rows: list[dict] = []
    if diff.reordered:
        banner_rows.append({
            "Metric": "⚠ FILES REORDERED",
            "Value": (
                f"Upload slots were reversed: slot 1's latest period "
                f"({_period_label(diff.new_period)}) was NEWER than slot 2's "
                f"({_period_label(diff.old_period)}). Compared as older → newer "
                "automatically — 'Old' = the file uploaded in slot 2, 'New' = slot 1."
            ),
        })

    summary = pd.DataFrame(
        banner_rows
        + [
            {"Metric": "New entrants", "Value": len(diff.entrants)},
            {"Metric": "Exits", "Value": len(diff.exits)},
            {"Metric": "Material moves", "Value": len(diff.moves)},
            {"Metric": "Materiality gate — Dollars", "Value": f"|Δ| ≥ {int(IQVIA_DIFF_DOLLARS_ABS):,} and ≥ {IQVIA_DIFF_PCT:.0%}"},
            {"Metric": "Materiality gate — Units", "Value": f"|Δ| ≥ {int(IQVIA_DIFF_UNITS_ABS):,} and ≥ {IQVIA_DIFF_PCT:.0%}"},
        ]
        + [{"Metric": "⚠ Warning", "Value": w} for w in diff.warnings],
        columns=["Metric", "Value"],
    )

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, df in (
            ("Summary", summary),
            ("New Entrants", diff.entrants),
            ("Exits", diff.exits),
            ("Material Moves", diff.moves),
        ):
            df.to_excel(writer, sheet_name=name, index=False)
            _style_sheet(writer.sheets[name], df)
    return buf.getvalue()


def build_iqvia_diff_workbook(old_bytes: bytes, new_bytes: bytes) -> bytes:
    """End-to-end convenience: compare two extracts and return the changes XLSX."""
    return build_diff_workbook(compare_iqvia(old_bytes, new_bytes))
