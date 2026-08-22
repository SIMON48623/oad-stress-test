import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("results/thumos14_debug")


def read_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def pick_col(df, names):
    for n in names:
        if n in df.columns:
            return n
    raise KeyError(f"None of these columns exist: {names}. Available: {list(df.columns)}")


def transition_points(gt):
    return [(i, gt[i]) for i in range(1, len(gt)) if gt[i] != gt[i - 1]]


def observed_transition_delay(df):
    t_col = pick_col(df, ["t", "timestep", "time_idx"])
    gt_col = pick_col(df, ["gt_label", "label", "ground_truth"])
    pred_col = pick_col(df, ["pred_label", "prediction", "pred"])
    observed_col = pick_col(df, ["observed", "is_observed"])

    df = df.sort_values(t_col).reset_index(drop=True)
    gt = df[gt_col].tolist()
    pred = df[pred_col].tolist()
    obs = df[observed_col].astype(bool).tolist()

    delays = []
    for t0, new_label in transition_points(gt):
        for t in range(t0, len(df)):
            if obs[t] and pred[t] == new_label:
                delays.append(t - t0)
                break
    return delays


def carried_transition_delay(df):
    t_col = pick_col(df, ["t", "timestep", "time_idx"])
    gt_col = pick_col(df, ["gt_label", "label", "ground_truth"])
    pred_col = pick_col(df, ["pred_label", "prediction", "pred"])
    observed_col = pick_col(df, ["observed", "is_observed"])

    df = df.sort_values(t_col).reset_index(drop=True)
    gt = df[gt_col].tolist()
    raw_pred = df[pred_col].tolist()
    obs = df[observed_col].astype(bool).tolist()

    carried = []
    last_pred = None
    for p, o in zip(raw_pred, obs):
        if o:
            last_pred = p
        carried.append(last_pred)

    delays = []
    for t0, new_label in transition_points(gt):
        for t in range(t0, len(df)):
            if carried[t] == new_label:
                delays.append(t - t0)
                break
    return delays


def summarize(delays):
    if not delays:
        return {"count": 0, "median": np.nan, "p90": np.nan, "mean": np.nan}
    arr = np.asarray(delays, dtype=float)
    return {
        "count": int(len(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "mean": float(np.mean(arr)),
    }


def main():
    if not ROOT.exists():
        raise FileNotFoundError(
            f"{ROOT} does not exist. First run debug baselines into results/thumos14_debug."
        )

    files = sorted(ROOT.glob("*_budget*.jsonl"))
    if not files:
        raise FileNotFoundError(
            f"No *_budget*.jsonl files found in {ROOT}. Run debug baselines without --summary-only first."
        )

    rows = []
    for path in files:
        name = path.stem
        policy = name.split("_budget")[0]
        budget = float(name.split("_budget")[1])

        df = read_jsonl(path)
        video_col = pick_col(df, ["video_id", "video"])

        observed_all = []
        carried_all = []

        for _, vdf in df.groupby(video_col):
            observed_all.extend(observed_transition_delay(vdf))
            carried_all.extend(carried_transition_delay(vdf))

        obs_s = summarize(observed_all)
        car_s = summarize(carried_all)

        rows.append({
            "policy": policy,
            "budget": budget,
            "observed_count": obs_s["count"],
            "observed_median": obs_s["median"],
            "observed_p90": obs_s["p90"],
            "observed_mean": obs_s["mean"],
            "carried_count": car_s["count"],
            "carried_median": car_s["median"],
            "carried_p90": car_s["p90"],
            "carried_mean": car_s["mean"],
            "log_path": str(path),
        })

    out = pd.DataFrame(rows).sort_values(["policy", "budget"])
    print(out.to_string(index=False))
    out.to_csv(ROOT / "transition_sparse_diagnostic.csv", index=False)
    print(f"\nSaved: {ROOT / 'transition_sparse_diagnostic.csv'}")


if __name__ == "__main__":
    main()
