"""The Bulletin 17C reference check (#519): published numbers in, honest differences out."""

from __future__ import annotations

from pathlib import Path

from aquascope.hydrology import b17c_check as b

DOC = Path(__file__).parents[2] / "docs" / "engineering_exports.md"


def test_the_reference_cases_match_the_publication_counts():
    moose, ores = b.CASES
    assert len(moose["peaks"]) == 68 and moose["peaks"][0] == 2080 and moose["peaks"][-1] == 4250
    assert len(ores["peaks"]) == 82 and sum(1 for v in ores["peaks"] if v == 0) == 12
    assert ores["published"]["n_censored"] == 30 and ores["published"]["low_outlier_threshold"] == 782
    # Station skew is negative in the Bulletin ("-0.929 (station skew)", figure 10-5).
    assert ores["published"]["skew"] == -0.929 and moose["published"]["skew"] == 0.421


def test_moose_river_agrees_within_about_one_percent():
    case = b.run_case(b.CASES[0])
    assert case["moments"]["aquascope"]["mean"] == case["moments"]["published"]["mean"]
    assert case["moments"]["aquascope"]["std"] == case["moments"]["published"]["std"]
    assert abs(case["moments"]["aquascope"]["skew"] - 0.421) < 0.005
    assert case["max_abs_diff_pct"] <= 1.5


def test_the_report_names_what_was_not_run():
    rep = b.run()
    assert [c["id"] for c in rep["cases"]] == ["moose", "orestimba"]
    assert len(rep["not_run"]) == 5 and rep["source"] == "https://doi.org/10.3133/tm4B5"


def test_the_docs_page_carries_exactly_the_computed_table():
    assert b.markdown_table() in DOC.read_text(encoding="utf-8")
