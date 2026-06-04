"""Precision / recall test for the Patent Register ingredient fuzzy matcher.

Why this exists
---------------
The `_find_matching_options` function in `app/sources/patent_register.py` maps
a free-text user query to one of ~469 dropdown values.  A false positive merge
(wrong option matched) silently corrupts patent-linkage decisions and is harder
to detect than a missed match.  This test guards against regressions by keeping
precision ≥ 0.95 on a hand-labeled benchmark.

Benchmark
---------
`tests/fixtures/fuzzy_pairs.csv`  — 25 (query, option, expected_match) triples.
Labels reflect semantic correctness, not current matcher behaviour, so the test
fails when the matcher is tuned too loosely.

Metrics
-------
Precision = TP / (TP + FP)  — asserted ≥ 0.95
Recall    = TP / (TP + FN)  — printed but not hard-asserted (a missed link is
                               safer than a wrong one per spec)

Fixture options
---------------
Pulled from `tests/fixtures/patent_register/index.html` (the mock dropdown used
by other tests).  The same dropdown is used here so the benchmark is comparable
to production behaviour.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent.parent))

FIXTURES_DIR = Path(__file__).parent / "fixtures"
FUZZY_PAIRS_CSV = FIXTURES_DIR / "fuzzy_pairs.csv"


def _load_fixture_dropdown() -> list[str]:
    """Return the ingredient options from the Patent Register fixture HTML."""
    html = (FIXTURES_DIR / "patent_register" / "index.html").read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")
    sel = soup.find("select", {"id": "medicinalIngredient"})
    if sel is None:
        pytest.skip("Patent Register index fixture missing — run make refresh-fixtures")
    return [o["value"] for o in sel.find_all("option") if o.get("value")]


def _load_pairs() -> list[tuple[str, str, bool, str]]:
    """Return list of (query, option, expected_match, notes) from the CSV."""
    rows = []
    with open(FUZZY_PAIRS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                (
                    row["query"].strip(),
                    row["option"].strip(),
                    row["expected_match"].strip().lower() == "true",
                    row.get("notes", "").strip(),
                )
            )
    return rows


class TestFuzzyMatcherPrecisionRecall:
    """Run the matcher against the labeled benchmark and assert precision ≥ 0.95."""

    PRECISION_THRESHOLD = 0.95

    def test_precision_recall(self, capsys) -> None:
        from app.sources.patent_register import _find_matching_options

        dropdown = _load_fixture_dropdown()
        pairs = _load_pairs()

        tp = fp = fn = tn = 0
        false_positives: list[str] = []
        false_negatives: list[str] = []

        for query, option, expected, notes in pairs:
            matched_opts = _find_matching_options(query, dropdown)
            got_match = option in matched_opts

            if expected and got_match:
                tp += 1
            elif not expected and got_match:
                fp += 1
                false_positives.append(
                    f"  FP: query={query!r} matched {option!r}"
                    + (f" — {notes}" if notes else "")
                )
            elif expected and not got_match:
                fn += 1
                false_negatives.append(
                    f"  FN: query={query!r} did not match {option!r}"
                    + (f" — {notes}" if notes else "")
                )
            else:
                tn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0

        print(
            f"\nFuzzy matcher metrics on {len(pairs)}-pair benchmark:\n"
            f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}\n"
            f"  Precision={precision:.3f}  Recall={recall:.3f}"
        )
        if false_positives:
            print("False positives (regressions):\n" + "\n".join(false_positives))
        if false_negatives:
            print("False negatives (acceptable recall misses):\n" + "\n".join(false_negatives))

        assert precision >= self.PRECISION_THRESHOLD, (
            f"Fuzzy matcher precision {precision:.3f} is below the required "
            f"{self.PRECISION_THRESHOLD} threshold.\n"
            f"False positives:\n" + "\n".join(false_positives or ["(none)"])
        )

    def test_exact_substring_match_always_takes_priority(self) -> None:
        """Substring match must be returned even when a fuzzy match also exists,
        and must not be displaced by the fuzzy fallback path."""
        from app.sources.patent_register import _find_matching_options

        dropdown = _load_fixture_dropdown()
        result = _find_matching_options("metformin hydrochloride", dropdown)

        assert "METFORMIN HYDROCHLORIDE" in result, (
            "Exact-match option 'METFORMIN HYDROCHLORIDE' missing from results for "
            "'metformin hydrochloride'"
        )

    def test_unrelated_query_returns_empty(self) -> None:
        """A query with no substring or fuzzy match must return an empty list."""
        from app.sources.patent_register import _find_matching_options

        dropdown = _load_fixture_dropdown()
        result = _find_matching_options("xyz_no_such_drug_123", dropdown)
        assert result == [], (
            f"Expected empty result for nonsense query, got: {result}"
        )

    def test_combo_ingredient_is_reachable_by_component(self) -> None:
        """Searching 'canagliflozin' must return the mono AND the combo option."""
        from app.sources.patent_register import _find_matching_options

        dropdown = _load_fixture_dropdown()
        result = _find_matching_options("canagliflozin", dropdown)

        assert "CANAGLIFLOZIN" in result
        combo = "METFORMIN HYDROCHLORIDE / CANAGLIFLOZIN"
        assert combo in result, (
            f"Combo option '{combo}' not found for 'canagliflozin' search"
        )
