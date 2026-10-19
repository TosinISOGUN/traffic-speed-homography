"""Recompute Table 1 from the labels in the manuscript's evaluation script."""
import csv

import _path  # noqa: F401
from _path import ROOT
from speed_estimation.published import PUBLISHED, reproduce_table1


def main() -> None:
    m = reproduce_table1()
    print(f"{m['n']} vehicles: TP {m['tp']}, FP {m['fp']}, FN {m['fn']}, TN {m['tn']}\n")
    print(f"{'metric':<10}{'recomputed':>12}{'published':>11}")
    for key in PUBLISHED:
        print(f"{key:<10}{m[key]:>12.4f}{PUBLISHED[key]:>11.2f}")

    out = ROOT / "results" / "published"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "table1_reproduced.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "recomputed", "rounded", "published"])
        for key, published in PUBLISHED.items():
            writer.writerow([key, m[key], round(m[key], 2), published])
    print(f"\nsaved {out / 'table1_reproduced.csv'}")


if __name__ == "__main__":
    main()
