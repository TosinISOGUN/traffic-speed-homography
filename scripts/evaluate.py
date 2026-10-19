"""Compute accuracy, precision, recall and F1 (Sec. 4.VI, Table 1, Fig. 5)."""
import argparse
import json
from pathlib import Path

import pandas as pd

import _path  # noqa: F401
from _path import ROOT
from speed_estimation.evaluation import evaluate
from speed_estimation.plots import plot_metrics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--estimates", default=str(ROOT / "results" / "video" / "per_vehicle_speeds.csv"))
    p.add_argument("--ground-truth", required=True,
                   help="CSV with vehicle_id, tracker_id, true_speed_kmh")
    p.add_argument("--tol-kmh", type=float, default=5.0, help="absolute tolerance (km/h)")
    p.add_argument("--tol-rel", type=float, default=None, help="relative tolerance, e.g. 0.1 for 10%%")
    p.add_argument("--out-dir", default=str(ROOT / "results" / "evaluation"))
    a = p.parse_args()

    metrics, detail = evaluate(pd.read_csv(a.estimates), pd.read_csv(a.ground_truth),
                               tol_kmh=a.tol_kmh, tol_rel=a.tol_rel)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    detail.to_csv(out / "per_vehicle_outcomes.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(metrics.to_dict(), indent=2))
    pd.DataFrame([{"Metrics": "YOLO Model", "Accuracy": metrics.accuracy, "Precision": metrics.precision,
                   "Recall": metrics.recall, "F1 Score": metrics.f1}]).round(2).to_csv(
        out / "table1_metrics.csv", index=False)
    plot_metrics(metrics.to_dict(), out / "fig5_metrics.png",
                 "Performance metrics for vehicle speed estimation")
    print(json.dumps(metrics.to_dict(), indent=2))
    if metrics.n_unlabelled_tracks:
        print(f"note: {metrics.n_unlabelled_tracks} tracks are not in the ground truth and were ignored")


if __name__ == "__main__":
    main()
