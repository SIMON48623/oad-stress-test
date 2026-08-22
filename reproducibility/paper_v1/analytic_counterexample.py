from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, getcontext
from fractions import Fraction
from pathlib import Path

from confirmatory_cluster_bootstrap import sha256_file, write_csv


getcontext().prec = 40


Y = (0, 0, 1, 1, 1)
L = Fraction(1, 25)
G = Fraction(71, 100)
H = Fraction(113, 200)
A = Fraction(1, 2)
PRIMARY_BINS = 15
BIN_SENSITIVITY = (10, 15, 20, 30)


@dataclass(frozen=True)
class Interval:
    low: Decimal
    high: Decimal

    def __add__(self, other: "Interval") -> "Interval":
        return Interval(self.low + other.low, self.high + other.high)

    def __sub__(self, other: "Interval") -> "Interval":
        return Interval(self.low - other.high, self.high - other.low)

    def __mul__(self, other: "Interval") -> "Interval":
        products = (
            self.low * other.low,
            self.low * other.high,
            self.high * other.low,
            self.high * other.high,
        )
        return Interval(min(products), max(products))


ONE = Interval(Decimal(1), Decimal(1))


def d(value: str) -> Decimal:
    return Decimal(value)


def ema_fraction(probabilities: tuple[Fraction, ...], alpha: Fraction) -> tuple[Fraction, ...]:
    values = [probabilities[0]]
    for probability in probabilities[1:]:
        values.append(alpha * values[-1] + (1 - alpha) * probability)
    return tuple(values)


def ema_float(probabilities: tuple[float, ...], alpha: float) -> tuple[float, ...]:
    values = [probabilities[0]]
    for probability in probabilities[1:]:
        values.append(alpha * values[-1] + (1.0 - alpha) * probability)
    return tuple(values)


def predicted_label(probability_class_1: Fraction | float) -> int:
    return int(probability_class_1 > Fraction(1, 2) if isinstance(probability_class_1, Fraction) else probability_class_1 > 0.5)


def accuracy(labels: tuple[int, ...], probabilities: tuple[Fraction | float, ...]) -> Fraction:
    correct = sum(predicted_label(probability) == label for label, probability in zip(labels, probabilities))
    return Fraction(correct, len(labels))


def transition_delay(probabilities: tuple[Fraction | float, ...], transition_index: int = 2) -> int:
    for delay, probability in enumerate(probabilities[transition_index:]):
        if predicted_label(probability) == 1:
            return delay
    return len(probabilities) - transition_index


def fixed_bin_ece(labels: tuple[int, ...], probabilities: tuple[Fraction, ...], bins: int) -> Fraction:
    counts = [0 for _ in range(bins)]
    correct = [0 for _ in range(bins)]
    confidence_sums = [Fraction(0, 1) for _ in range(bins)]
    for label, probability in zip(labels, probabilities):
        prediction = predicted_label(probability)
        confidence = probability if prediction == 1 else 1 - probability
        bin_index = min(bins - 1, int(confidence * bins))
        counts[bin_index] += 1
        correct[bin_index] += int(prediction == label)
        confidence_sums[bin_index] += confidence
    value = Fraction(0, 1)
    n = len(labels)
    for bin_index in range(bins):
        if counts[bin_index] == 0:
            continue
        bin_accuracy = Fraction(correct[bin_index], counts[bin_index])
        bin_confidence = confidence_sums[bin_index] / counts[bin_index]
        value += Fraction(counts[bin_index], n) * abs(bin_accuracy - bin_confidence)
    return value


def center_construction() -> dict[str, object]:
    raw = (L, G, H, H, H)
    smoothed = ema_fraction(raw, A)
    return {
        "raw": raw,
        "smoothed": smoothed,
        "raw_accuracy": accuracy(Y, raw),
        "smoothed_accuracy": accuracy(Y, smoothed),
        "raw_delay": transition_delay(raw),
        "smoothed_delay": transition_delay(smoothed),
        "raw_ece_15": fixed_bin_ece(Y, raw, PRIMARY_BINS),
        "smoothed_ece_15": fixed_bin_ece(Y, smoothed, PRIMARY_BINS),
    }


def interval_proof() -> dict[str, object]:
    l = Interval(d("0.039"), d("0.041"))
    g = Interval(d("0.709"), d("0.711"))
    h = Interval(d("0.564"), d("0.566"))
    a = Interval(d("0.499"), d("0.501"))
    one_minus_a = ONE - a
    s0 = l
    s1 = a * s0 + one_minus_a * g
    s2 = a * s1 + one_minus_a * h
    s3 = a * s2 + one_minus_a * h
    s4 = a * s3 + one_minus_a * h
    raw_ece = Interval(
        (l.low + g.low + d("3") * (d("1") - h.high)) / d("5"),
        (l.high + g.high + d("3") * (d("1") - h.low)) / d("5"),
    )
    smooth_ece = Interval(
        (l.low + s1.low + (s3.low - s2.high) + (d("1") - s4.high)) / d("5"),
        (l.high + s1.high + (s3.high - s2.low) + (d("1") - s4.low)) / d("5"),
    )

    def inside(interval: Interval, lower: Decimal, upper: Decimal) -> bool:
        return interval.low > lower and interval.high < upper

    bin_width = d("1") / d("15")
    memberships = {
        "raw_conf_1_minus_l_bin14": inside(ONE - l, d("14") * bin_width, d("1.0000000001")),
        "raw_conf_g_bin10": inside(g, d("10") * bin_width, d("11") * bin_width),
        "raw_conf_h_bin8": inside(h, d("8") * bin_width, d("9") * bin_width),
        "smooth_conf_1_minus_s1_bin9": inside(ONE - s1, d("9") * bin_width, d("10") * bin_width),
        "smooth_conf_1_minus_s2_bin7": inside(ONE - s2, d("7") * bin_width, d("8") * bin_width),
        "smooth_conf_s3_bin7": inside(s3, d("7") * bin_width, d("8") * bin_width),
        "smooth_conf_s4_bin8": inside(s4, d("8") * bin_width, d("9") * bin_width),
    }
    checks = {
        "s1_below_half": s1.high < d("0.5"),
        "s2_below_half": s2.high < d("0.5"),
        "s3_above_half": s3.low > d("0.5"),
        "all_primary_bin_memberships_fixed": all(memberships.values()),
        "raw_ece_lower_exceeds_smoothed_ece_upper": raw_ece.low > smooth_ece.high,
    }
    return {
        "states": {"s0": s0, "s1": s1, "s2": s2, "s3": s3, "s4": s4},
        "raw_ece": raw_ece,
        "smooth_ece": smooth_ece,
        "ece_gap_lower_bound": raw_ece.low - smooth_ece.high,
        "memberships": memberships,
        "checks": checks,
        "overall": all(checks.values()),
    }


def grid_cross_check() -> tuple[list[dict[str, object]], dict[str, object]]:
    axes = {
        "l": [0.039, 0.0395, 0.0400, 0.0405, 0.041],
        "g": [0.709, 0.7095, 0.7100, 0.7105, 0.711],
        "h": [0.564, 0.5645, 0.5650, 0.5655, 0.566],
        "a": [0.499, 0.4995, 0.5000, 0.5005, 0.501],
    }
    totals = 0
    accuracy_delay_pass = 0
    bin_pass = {bins: 0 for bins in BIN_SENSITIVITY}
    minimum_improvement = {bins: float("inf") for bins in BIN_SENSITIVITY}
    for l, g, h, a in itertools.product(axes["l"], axes["g"], axes["h"], axes["a"]):
        totals += 1
        raw_float = (l, g, h, h, h)
        smooth_float = ema_float(raw_float, a)
        raw_fraction = tuple(Fraction(str(value)) for value in raw_float)
        smooth_fraction = tuple(Fraction(str(value)) for value in smooth_float)
        structural = (
            accuracy(Y, raw_float) == accuracy(Y, smooth_float) == Fraction(4, 5)
            and transition_delay(raw_float) == 0
            and transition_delay(smooth_float) == 1
        )
        accuracy_delay_pass += int(structural)
        for bins in BIN_SENSITIVITY:
            raw_ece = float(fixed_bin_ece(Y, raw_fraction, bins))
            smooth_ece = float(fixed_bin_ece(Y, smooth_fraction, bins))
            improvement = raw_ece - smooth_ece
            minimum_improvement[bins] = min(minimum_improvement[bins], improvement)
            bin_pass[bins] += int(improvement > 0.0)
    rows = [
        {
            "ece_bins": bins,
            "grid_points": totals,
            "points_with_lower_smoothed_ece": bin_pass[bins],
            "minimum_raw_minus_smoothed_ece": minimum_improvement[bins],
        }
        for bins in BIN_SENSITIVITY
    ]
    summary = {
        "grid_points": totals,
        "accuracy_and_delay_pass_points": accuracy_delay_pass,
        "all_accuracy_delay_pass": accuracy_delay_pass == totals,
        "all_bins_pass": all(bin_pass[bins] == totals for bins in BIN_SENSITIVITY),
    }
    return rows, summary


def fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator} = {float(value):.6f}"


def interval_text(value: Interval) -> str:
    return f"[{value.low}, {value.high}]"


def write_analytic_note(path: Path, center: dict[str, object], proof: dict[str, object]) -> None:
    raw = center["raw"]
    smooth = center["smoothed"]
    states = proof["states"]
    lines = [
        "# Minimal analytical ranking-inversion counterexample",
        "",
        "## Construction",
        "",
        "Let the binary ground truth be `y=(0,0,1,1,1)` and let `p_t` denote the raw probability of class 1. Choose",
        "",
        "`p=(0.04, 0.71, 0.565, 0.565, 0.565)` and causal EMA `s_t = 0.5 s_{t-1} + 0.5 p_t`, initialized with `s_0=p_0`.",
        "",
        "The exact smoothed sequence is",
        "",
        f"`s=({', '.join(str(value) for value in smooth)}) = ({', '.join(f'{float(value):.6f}' for value in smooth)})`.",
        "",
        "The raw predictor makes one isolated pre-transition error and then reacts at the transition immediately. EMA removes that isolated error but transfers the error to the first post-transition frame.",
        "",
        "## Exact outcome",
        "",
        f"- Raw accuracy: {fraction_text(center['raw_accuracy'])}",
        f"- EMA accuracy: {fraction_text(center['smoothed_accuracy'])}",
        f"- Raw transition delay: {center['raw_delay']}",
        f"- EMA transition delay: {center['smoothed_delay']}",
        f"- Raw 15-bin ECE: {fraction_text(center['raw_ece_15'])}",
        f"- EMA 15-bin ECE: {fraction_text(center['smoothed_ece_15'])}",
        "",
        "Thus accuracy is unchanged, global ECE improves, and transition delay worsens.",
        "",
        "## Non-singleton parameter region",
        "",
        "Let `l∈[0.039,0.041]`, `g∈[0.709,0.711]`, `h∈[0.564,0.566]`, and `a∈[0.499,0.501]`, with raw sequence `(l,g,h,h,h)` and EMA coefficient `a`. Interval propagation gives:",
        "",
        f"- s1: {interval_text(states['s1'])}",
        f"- s2: {interval_text(states['s2'])}",
        f"- s3: {interval_text(states['s3'])}",
        f"- s4: {interval_text(states['s4'])}",
        "",
        "Throughout the box, `s1<0.5`, `s2<0.5`, and `s3>0.5`. Therefore EMA always corrects the isolated old-segment spike and always responds one frame later than the raw predictor.",
        "",
        "Under the fixed 15-bin assignments,",
        "",
        "`ECE_raw = [l + g + 3(1-h)]/5`,",
        "",
        "`ECE_EMA = [l + s1 + (s3-s2) + (1-s4)]/5`.",
        "",
        f"Interval bounds give `ECE_raw ∈ {interval_text(proof['raw_ece'])}` and `ECE_EMA ∈ {interval_text(proof['smooth_ece'])}`. The guaranteed gap is at least {proof['ece_gap_lower_bound']}.",
        "",
        "## Scope boundary",
        "",
        "This construction proves existence, not inevitability. It formalizes how aggregate calibration can reward suppression of an isolated fluctuation while ignoring that the same causal memory delays a true change.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_gate_report(
    path: Path,
    center: dict[str, object],
    proof: dict[str, object],
    bin_rows: list[dict[str, object]],
    grid_summary: dict[str, object],
) -> None:
    exact_pass = (
        center["smoothed_accuracy"] >= center["raw_accuracy"]
        and center["smoothed_delay"] > center["raw_delay"]
        and center["smoothed_ece_15"] < center["raw_ece_15"]
    )
    bin_pass = all(float(row["ema_ece"]) < float(row["raw_ece"]) for row in bin_rows)
    overall = exact_pass and bool(proof["overall"]) and bin_pass and bool(grid_summary["all_accuracy_delay_pass"]) and bool(grid_summary["all_bins_pass"])
    lines = [
        "# G3 analytical counterexample gate",
        "",
        f"- Decision: **{'PASS' if overall else 'FAIL_TOY_ONLY'}**",
        f"- Exact center construction: **{'PASS' if exact_pass else 'FAIL'}**",
        f"- Interval-box proof: **{'PASS' if proof['overall'] else 'FAIL'}**",
        f"- ECE-bin numerical sensitivity: **{'PASS' if bin_pass else 'FAIL'}**",
        f"- 625-point neighborhood cross-check: **{'PASS' if grid_summary['all_accuracy_delay_pass'] and grid_summary['all_bins_pass'] else 'FAIL'}**",
        "",
        "| Bins | Raw ECE | EMA ECE | Raw minus EMA |",
        "|---:|---:|---:|---:|",
    ]
    for row in bin_rows:
        lines.append(
            f"| {row['ece_bins']} | {float(row['raw_ece']):.6f} | {float(row['ema_ece']):.6f} | {float(row['raw_minus_ema']):.6f} |"
        )
    lines.extend(
        [
            "",
            f"- Accuracy: raw {float(center['raw_accuracy']):.3f}, EMA {float(center['smoothed_accuracy']):.3f}",
            f"- Delay: raw {center['raw_delay']}, EMA {center['smoothed_delay']}",
            f"- Guaranteed 15-bin ECE improvement throughout the parameter box: at least {proof['ece_gap_lower_bound']}",
            "",
            "Interpretation: the analytical gate establishes possibility and mechanism only. Empirical prevalence remains supported by G1, not by this five-frame construction.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exact and interval verification of the frozen G3 counterexample.")
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.protocol.exists():
        raise FileNotFoundError(args.protocol)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_names = (
        "analytic_counterexample.md",
        "counterexample_values.csv",
        "bin_sensitivity.csv",
        "neighborhood_grid_summary.csv",
        "metadata.json",
        "gate_G3.md",
    )
    if any((args.output_dir / name).exists() for name in output_names):
        raise FileExistsError("G3 outputs already exist; use a new directory to preserve immutability")
    json.loads(args.protocol.read_text(encoding="utf-8"))
    center = center_construction()
    proof = interval_proof()
    raw = center["raw"]
    smooth = center["smoothed"]
    value_rows = []
    for index, (label, raw_p, smooth_p) in enumerate(zip(Y, raw, smooth)):
        value_rows.append(
            {
                "time": index,
                "ground_truth": label,
                "raw_p_class1_fraction": str(raw_p),
                "raw_p_class1": float(raw_p),
                "ema_p_class1_fraction": str(smooth_p),
                "ema_p_class1": float(smooth_p),
                "raw_prediction": predicted_label(raw_p),
                "ema_prediction": predicted_label(smooth_p),
            }
        )
    bin_rows = []
    for bins in BIN_SENSITIVITY:
        raw_ece = fixed_bin_ece(Y, raw, bins)
        smooth_ece = fixed_bin_ece(Y, smooth, bins)
        bin_rows.append(
            {
                "ece_bins": bins,
                "raw_ece_fraction": str(raw_ece),
                "raw_ece": float(raw_ece),
                "ema_ece_fraction": str(smooth_ece),
                "ema_ece": float(smooth_ece),
                "raw_minus_ema": float(raw_ece - smooth_ece),
            }
        )
    grid_rows, grid_summary = grid_cross_check()
    write_csv(args.output_dir / "counterexample_values.csv", value_rows)
    write_csv(args.output_dir / "bin_sensitivity.csv", bin_rows)
    write_csv(args.output_dir / "neighborhood_grid_summary.csv", grid_rows)
    write_analytic_note(args.output_dir / "analytic_counterexample.md", center, proof)
    write_gate_report(args.output_dir / "gate_G3.md", center, proof, bin_rows, grid_summary)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "analytical_confirmation",
        "protocol": str(args.protocol),
        "protocol_sha256": sha256_file(args.protocol),
        "script_sha256": sha256_file(Path(__file__)),
        "exact_center": {
            "raw_accuracy": float(center["raw_accuracy"]),
            "ema_accuracy": float(center["smoothed_accuracy"]),
            "raw_delay": center["raw_delay"],
            "ema_delay": center["smoothed_delay"],
            "raw_ece_15": float(center["raw_ece_15"]),
            "ema_ece_15": float(center["smoothed_ece_15"]),
        },
        "interval_proof_pass": proof["overall"],
        "grid_summary": grid_summary,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(args.output_dir / "gate_G3.md")


if __name__ == "__main__":
    main()

