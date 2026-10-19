"""Turn hand-timed post crossings into a ground-truth CSV for evaluate.py."""
import argparse
from pathlib import Path

import pandas as pd

import _path  # noqa: F401
from _path import ROOT
from speed_estimation.evaluation import ground_truth_from_timings


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--timings", required=True,
                   help="CSV with tracker_id, frame_start, frame_end and optionally posts_crossed")
    p.add_argument("--fps", type=float, default=25.0, help="video frame rate (vehicles.mp4 is 25)")
    p.add_argument("--post-spacing", type=float, default=50.0, help="metres between posts")
    p.add_argument("--out", default=str(ROOT / "results" / "video" / "ground_truth.csv"))
    a = p.parse_args()

    gt = ground_truth_from_timings(pd.read_csv(a.timings), a.fps, a.post_spacing)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    gt.to_csv(a.out, index=False)
    print(gt.round(1).to_string(index=False))
    print(f"\n{len(gt)} vehicles. Saved {a.out}")


if __name__ == "__main__":
    main()
