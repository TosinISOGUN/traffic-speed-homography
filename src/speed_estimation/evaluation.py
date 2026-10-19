"""Evaluation (Sec. 4.VI): accuracy, precision, recall and F1 score.

The published Table 1 was computed differently: binary-classification metrics
over 15 hand-entered correct/incorrect labels, with no true speeds recorded.
`published.py` reproduces it exactly. This module is the evaluation for new
ground truth with measured speeds, using the following vehicle-level
definitions (see README, "How the metrics are defined"):

* A *real vehicle* is a ground-truth row with a true speed.
* An estimate is *correct* when |estimated - true| <= tolerance.
* TP  = real vehicles whose estimate is correct.
* FP  = estimates that are wrong, plus estimates on spurious tracks
        (ground-truth rows with a tracker ID but no true speed).
* FN  = real vehicles with no estimate (missed) or a wrong estimate.

    accuracy  = TP / (real vehicles + spurious tracks)
    precision = TP / (all estimates reported)      = TP / (TP + FP)
    recall    = TP / real vehicles                 = TP / (TP + FN)
    F1        = 2PR / (P + R)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    n_real_vehicles: int
    n_correct: int
    n_wrong: int
    n_missed: int
    n_spurious: int
    n_unlabelled_tracks: int
    mae_kmh: float
    rmse_kmh: float
    mape_pct: float
    tolerance: str

    def to_dict(self) -> dict:
        return asdict(self)


def _parse_ids(value) -> list[int]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [int(float(v)) for v in str(value).replace(",", ";").split(";") if v.strip()]


def _is_correct(est: float, true: float, tol_kmh: float | None, tol_rel: float | None) -> bool:
    err = abs(est - true)
    ok = True
    if tol_kmh is not None:
        ok &= err <= tol_kmh
    if tol_rel is not None:
        ok &= err <= tol_rel * abs(true)
    return bool(ok)


def evaluate(
    estimates: pd.DataFrame,
    ground_truth: pd.DataFrame,
    tol_kmh: float | None = 5.0,
    tol_rel: float | None = None,
) -> tuple[Metrics, pd.DataFrame]:
    """Compare per-track estimates with ground truth.

    estimates    : columns tracker_id, estimated_speed_kmh (and optionally n_estimates)
    ground_truth : columns vehicle_id, tracker_id, true_speed_kmh
                   - tracker_id may list several IDs ("3;7") after an ID switch;
                     the track with the most estimates is used.
                   - blank tracker_id  -> the vehicle was never tracked (missed).
                   - blank true speed  -> the track is not a real vehicle (spurious).
    """
    if tol_kmh is None and tol_rel is None:
        raise ValueError("give tol_kmh and/or tol_rel")
    est = estimates.set_index("tracker_id")
    n_est = est["n_estimates"] if "n_estimates" in est else pd.Series(1, index=est.index)

    rows = []
    labelled_ids: set[int] = set()
    for _, g in ground_truth.iterrows():
        ids = _parse_ids(g.get("tracker_id"))
        labelled_ids.update(ids)
        true = g.get("true_speed_kmh")
        has_true = true is not None and not (isinstance(true, float) and np.isnan(true))
        present = [i for i in ids if i in est.index]
        if not has_true:
            if present:
                rows.append({"vehicle_id": g.get("vehicle_id"), "tracker_id": present[0],
                             "true_speed_kmh": np.nan,
                             "estimated_speed_kmh": float(est.loc[present[0], "estimated_speed_kmh"]),
                             "outcome": "spurious"})
            continue
        true = float(true)
        if not present:
            rows.append({"vehicle_id": g.get("vehicle_id"), "tracker_id": ids[0] if ids else np.nan,
                         "true_speed_kmh": true, "estimated_speed_kmh": np.nan, "outcome": "missed"})
            continue
        best = max(present, key=lambda i: n_est.loc[i])
        value = float(est.loc[best, "estimated_speed_kmh"])
        outcome = "correct" if _is_correct(value, true, tol_kmh, tol_rel) else "wrong"
        rows.append({"vehicle_id": g.get("vehicle_id"), "tracker_id": best,
                     "true_speed_kmh": true, "estimated_speed_kmh": value, "outcome": outcome})

    detail = pd.DataFrame(rows, columns=["vehicle_id", "tracker_id", "true_speed_kmh",
                                         "estimated_speed_kmh", "outcome"])
    count = detail["outcome"].value_counts()
    n_correct, n_wrong = int(count.get("correct", 0)), int(count.get("wrong", 0))
    n_missed, n_spurious = int(count.get("missed", 0)), int(count.get("spurious", 0))
    n_real = n_correct + n_wrong + n_missed
    tp, fp, fn = n_correct, n_wrong + n_spurious, n_wrong + n_missed

    def ratio(a: float, b: float) -> float:
        return float(a / b) if b else float("nan")

    precision, recall = ratio(tp, tp + fp), ratio(tp, tp + fn)
    f1 = ratio(2 * precision * recall, precision + recall) if tp else 0.0
    matched = detail[detail["outcome"].isin(["correct", "wrong"])]
    err = (matched["estimated_speed_kmh"] - matched["true_speed_kmh"]).to_numpy(dtype=float)
    tol_text = " and ".join(t for t in [
        f"±{tol_kmh:g} km/h" if tol_kmh is not None else "",
        f"±{tol_rel:.0%}" if tol_rel is not None else "",
    ] if t)

    metrics = Metrics(
        accuracy=ratio(tp, n_real + n_spurious), precision=precision, recall=recall, f1=f1,
        tp=tp, fp=fp, fn=fn, n_real_vehicles=n_real, n_correct=n_correct, n_wrong=n_wrong,
        n_missed=n_missed, n_spurious=n_spurious,
        n_unlabelled_tracks=int(len(set(est.index) - labelled_ids)),
        mae_kmh=float(np.mean(np.abs(err))) if len(err) else float("nan"),
        rmse_kmh=float(np.sqrt(np.mean(err ** 2))) if len(err) else float("nan"),
        mape_pct=float(np.mean(np.abs(err) / matched["true_speed_kmh"].to_numpy(float)) * 100)
        if len(err) else float("nan"),
        tolerance=tol_text,
    )
    return metrics, detail


def ground_truth_from_timings(timings: pd.DataFrame, fps: float,
                              post_spacing_m: float = 50.0) -> pd.DataFrame:
    """True speeds from timing vehicles between roadside posts.

    timings : columns tracker_id, frame_start, frame_end and, optionally,
              posts_crossed (number of post gaps between the two frames, default 1).
    Speed = posts_crossed * post_spacing_m / ((frame_end - frame_start) / fps).
    """
    missing = {"tracker_id", "frame_start", "frame_end"} - set(timings.columns)
    if missing:
        raise ValueError(f"timings is missing columns: {sorted(missing)}")
    gaps = timings["posts_crossed"] if "posts_crossed" in timings else pd.Series(1, index=timings.index)
    frames = timings["frame_end"] - timings["frame_start"]
    if (frames <= 0).any():
        bad = timings.loc[frames <= 0, "tracker_id"].tolist()
        raise ValueError(f"frame_end must be after frame_start (tracker_id {bad})")
    if (gaps <= 0).any():
        raise ValueError("posts_crossed must be positive")
    speed = gaps * post_spacing_m / (frames / fps) * 3.6
    out = pd.DataFrame({"vehicle_id": range(1, len(timings) + 1),
                        "tracker_id": timings["tracker_id"].astype(str),
                        "true_speed_kmh": speed.to_numpy()})
    return out


def ground_truth_template(estimates: pd.DataFrame) -> pd.DataFrame:
    """A CSV to fill in by hand: one row per tracked vehicle, true speed blank."""
    t = estimates[["tracker_id"]].copy()
    t.insert(0, "vehicle_id", range(1, len(t) + 1))
    t["true_speed_kmh"] = np.nan
    t["notes"] = ""
    return t
