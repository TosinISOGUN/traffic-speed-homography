"""Synthetic validation: known speeds, same pipeline, with vs without perspective correction."""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

import _path  # noqa: F401
from _path import ROOT
from speed_estimation.plots import plot_simulation
from speed_estimation.simulation import SimConfig, run_experiment, summarize_experiment


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenarios", type=int, default=20)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--jitter", type=float, default=0.01, help="box-edge noise, fraction of box height")
    p.add_argument("--tol-kmh", type=float, default=5.0)
    p.add_argument("--out-dir", default=str(ROOT / "results" / "simulation"))
    a = p.parse_args()

    sim = SimConfig(edge_jitter_frac=a.jitter)
    per, details = run_experiment(n_scenarios=a.scenarios, seed=a.seed, sim=sim, tol_kmh=a.tol_kmh)
    summary = summarize_experiment(per, details)

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    per.to_csv(out / "per_scenario_metrics.csv", index=False)
    details.to_csv(out / "per_vehicle_outcomes.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    (out / "summary.json").write_text(json.dumps({
        "settings": {**asdict(sim), "scenarios": a.scenarios, "seed": a.seed, "tol_kmh": a.tol_kmh},
        "results": summary.to_dict(orient="records"),
    }, indent=2, default=str))
    plot_simulation(summary, details, out / "simulation_comparison.png")
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
