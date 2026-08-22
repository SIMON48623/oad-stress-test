from __future__ import annotations

from fractions import Fraction

from analytic_counterexample import center_construction, grid_cross_check, interval_proof


def test_exact_center_counterexample() -> None:
    center = center_construction()
    assert center["raw_accuracy"] == center["smoothed_accuracy"] == Fraction(4, 5)
    assert center["raw_delay"] == 0
    assert center["smoothed_delay"] == 1
    assert center["smoothed_ece_15"] < center["raw_ece_15"]


def test_interval_box_proves_strict_ranking_inversion() -> None:
    proof = interval_proof()
    assert proof["overall"]
    assert proof["ece_gap_lower_bound"] > 0


def test_bounded_grid_cross_check() -> None:
    rows, summary = grid_cross_check()
    assert len(rows) == 4
    assert summary["grid_points"] == 625
    assert summary["all_accuracy_delay_pass"]
    assert summary["all_bins_pass"]

