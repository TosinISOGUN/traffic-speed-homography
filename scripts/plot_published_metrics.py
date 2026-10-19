"""Redraw Fig. 5 from the values published in Table 1 of the manuscript."""
from pathlib import Path

import _path  # noqa: F401
from _path import ROOT
from speed_estimation.plots import plot_metrics
from speed_estimation.published import PUBLISHED

if __name__ == "__main__":
    out = ROOT / "results" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    plot_metrics(PUBLISHED, Path(out) / "fig5_published_metrics.png",
                 "Performance metrics for vehicle speed estimation (published)")
    print(f"saved {out / 'fig5_published_metrics.png'}")
