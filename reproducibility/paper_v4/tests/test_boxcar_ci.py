from __future__ import annotations

import csv
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = PACKAGE_ROOT / "audits" / "five_instance_boxcar_ci.csv"
EXPECTED_INSTANCES = {
    "thumos14_causal_gru",
    "thumos14_linear_probe",
    "ek100_causal_gru",
    "testra_ek100",
    "cmert_thumos14",
}


def _rows() -> list[dict[str, str]]:
    assert AUDIT_PATH.exists(), "five-instance boxcar audit is missing"
    with AUDIT_PATH.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_boxcar_audit_covers_exactly_five_instances_and_four_metrics_each():
    rows = _rows()

    assert {row["instance"] for row in rows} == EXPECTED_INSTANCES
    assert len(rows) == 5 * 4
    for instance in EXPECTED_INSTANCES:
        assert {row["metric"] for row in rows if row["instance"] == instance} == {
            "accuracy",
            "global_ece",
            "mean_transition_delay",
            "missed_transition_rate",
        }


def test_all_five_boxcar_intervals_support_the_stated_ece_delay_and_miss_directions():
    rows = _rows()

    for row in rows:
        if row["metric"] == "global_ece":
            assert float(row["ci_high"]) < 0.0
        elif row["metric"] in {"mean_transition_delay", "missed_transition_rate"}:
            assert float(row["ci_low"]) > 0.0
