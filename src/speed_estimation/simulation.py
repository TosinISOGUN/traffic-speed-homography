"""Synthetic validation with known ground-truth speeds.

The manuscript's evaluation used correct/incorrect labels, not measured
speeds, so it cannot show how far off the estimates are. This module checks
the method itself: vehicles with known speeds are driven along a
25 m x 250 m road, projected into a 3840 x 2160 camera view through the inverse
of the paper's homography, turned into noisy YOLO-like boxes, and passed through
the *same* pipeline (zone filter, NMS, ByteTrack, transform, speed averaging).

Comparing the perspective-corrected pipeline with a constant pixel-to-metre
scale isolates the effect the manuscript is about: perspective distortion.
All noise parameters are assumptions, listed in SimConfig.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import supervision as sv

from .config import LEGACY_TARGET, PipelineConfig
from .evaluation import Metrics, evaluate
from .pipeline import SpeedPipeline, summarize_tracks
from .transform import NaiveScaleTransformer, ViewTransformer

COCO_CAR, COCO_TRUCK = 2, 7


@dataclass
class SimConfig:
    """Scenario and detector-noise assumptions."""

    fps: float = 25.0                 # Supervision vehicles.mp4 is 25 fps
    duration_s: float = 21.0          # PAPER: 21 s clip
    resolution_wh: tuple[int, int] = (3840, 2160)
    n_vehicles: int = 24
    speed_range_kmh: tuple[float, float] = (60.0, 140.0)
    truck_fraction: float = 0.25
    # Lane centres across the 25 m road; first three move toward the camera.
    lanes_toward: tuple[float, ...] = (2.5, 6.0, 9.5)
    lanes_away: tuple[float, ...] = (15.5, 19.0, 22.5)
    # Detector noise
    edge_jitter_frac: float = 0.01    # per-edge Gaussian sd, fraction of box height
    edge_jitter_min_px: float = 1.0
    p_detect_near: float = 0.98
    p_detect_far: float = 0.70        # boxes under small_box_px tall
    small_box_px: float = 40.0
    false_positive_rate: float = 0.05  # spurious boxes per frame
    speed_jitter_frac: float = 0.0    # optional per-vehicle speed drift (sd, fraction)


@dataclass
class SimVehicle:
    vid: int
    lane_x: float
    direction: int      # +1 toward camera (y increasing), -1 away
    speed_kmh: float
    start_frame: int
    start_y: float
    width_m: float
    length_m: float
    height_m: float
    class_id: int

    def y_at(self, frame: int, fps: float) -> float:
        t = (frame - self.start_frame) / fps
        return self.start_y + self.direction * self.speed_kmh / 3.6 * t


def make_vehicles(cfg: SimConfig, rng: np.random.Generator) -> list[SimVehicle]:
    n_frames = int(cfg.duration_s * cfg.fps)
    vehicles, lanes = [], [(x, +1) for x in cfg.lanes_toward] + [(x, -1) for x in cfg.lanes_away]
    for vid in range(1, cfg.n_vehicles + 1):
        lane_x, direction = lanes[rng.integers(len(lanes))]
        truck = rng.random() < cfg.truck_fraction
        speed = rng.uniform(*cfg.speed_range_kmh) * (0.75 if truck else 1.0)
        # Start just outside the zone so each vehicle crosses it.
        start_y = -15.0 if direction == +1 else 265.0
        # Spread entries so the vehicle is in view for a while.
        start = int(rng.uniform(-0.3, 0.7) * n_frames)
        vehicles.append(SimVehicle(
            vid=vid, lane_x=lane_x + rng.normal(0, 0.3), direction=direction, speed_kmh=float(speed),
            start_frame=start, start_y=start_y,
            width_m=rng.uniform(2.3, 2.6) if truck else rng.uniform(1.7, 1.95),
            length_m=rng.uniform(8, 12) if truck else rng.uniform(4.0, 5.0),
            height_m=rng.uniform(3.2, 3.9) if truck else rng.uniform(1.4, 1.6),
            class_id=COCO_TRUCK if truck else COCO_CAR,
        ))
    # Avoid two vehicles occupying the same spot in a lane.
    vehicles.sort(key=lambda v: (v.lane_x, v.start_frame))
    return vehicles


def _box_for(v: SimVehicle, y: float, vt: ViewTransformer) -> np.ndarray:
    """Image box of a vehicle whose near-end ground point is (lane_x, y)."""
    far_y = y - v.length_m  # the far end is always further up the road (smaller y)
    pts = np.array([
        [v.lane_x - v.width_m / 2, y], [v.lane_x + v.width_m / 2, y],
        [v.lane_x, far_y],
    ], dtype=np.float32)
    left, right, far = vt.inverse_transform_points(pts)
    near_bottom = (left[1] + right[1]) / 2
    width_px = abs(right[0] - left[0])
    px_per_m = width_px / v.width_m
    top = min(far[1], near_bottom) - v.height_m * px_per_m
    cx = (left[0] + right[0]) / 2
    return np.array([cx - width_px / 2, top, cx + width_px / 2, near_bottom], dtype=np.float32)


def simulate_detections(cfg: SimConfig, vehicles: list[SimVehicle], rng: np.random.Generator,
                        source: np.ndarray, target: np.ndarray) -> list[sv.Detections]:
    vt = ViewTransformer(source, target)
    w, h = cfg.resolution_wh
    frames = []
    for f in range(int(cfg.duration_s * cfg.fps)):
        boxes, conf, cls, ids = [], [], [], []
        for v in vehicles:
            if f < v.start_frame:
                continue
            y = v.y_at(f, cfg.fps)
            if not (-30 <= y <= 280):
                continue
            box = _box_for(v, y, vt)
            bh = box[3] - box[1]
            if box[3] < 0 or box[1] > h - 1 or box[2] < 0 or box[0] > w - 1 or bh <= 2:
                continue
            p = cfg.p_detect_far if bh < cfg.small_box_px else cfg.p_detect_near
            if rng.random() > p:
                continue
            sd = max(cfg.edge_jitter_min_px, cfg.edge_jitter_frac * bh)
            box = box + rng.normal(0, sd, 4)
            box = np.clip(box, [0, 0, 0, 0], [w - 1, h - 1, w - 1, h - 1])
            if box[2] - box[0] < 2 or box[3] - box[1] < 2:
                continue
            boxes.append(box)
            conf.append(float(np.clip(0.55 + 0.4 * min(bh / 300, 1) + rng.normal(0, 0.03), 0.31, 0.99)))
            cls.append(v.class_id)
            ids.append(v.vid)
        for _ in range(rng.poisson(cfg.false_positive_rate)):
            cx, cy = rng.uniform(1300, 2600), rng.uniform(820, 1500)
            s = rng.uniform(20, 120)
            boxes.append(np.array([cx - s / 2, cy - s, cx + s / 2, cy], dtype=np.float32))
            conf.append(float(rng.uniform(0.31, 0.5)))
            cls.append(COCO_CAR)
            ids.append(-1)
        if boxes:
            frames.append(sv.Detections(
                xyxy=np.array(boxes, dtype=np.float32), confidence=np.array(conf),
                class_id=np.array(cls), data={"sim_id": np.array(ids)},
            ))
        else:
            frames.append(sv.Detections.empty())
    return frames


def run_pipeline_on(frames: list[sv.Detections], fps: float, config: PipelineConfig,
                    transformer=None) -> pd.DataFrame:
    pipe = SpeedPipeline(fps=fps, config=config, transformer=transformer)
    rows = []
    for f, det in enumerate(frames):
        res = pipe.step(det)
        d = res.detections
        sim_ids = d.data.get("sim_id", np.full(len(d), -1)) if len(d) else []
        for i in range(len(d)):
            tid = int(d.tracker_id[i])
            rows.append({"frame": f, "tracker_id": tid, "sim_id": int(sim_ids[i]),
                         "class_name": "", "speed_kmh": res.speeds.get(tid)})
    return pd.DataFrame(rows, columns=["frame", "tracker_id", "sim_id", "class_name", "speed_kmh"])


def ground_truth_from(log: pd.DataFrame, vehicles: list[SimVehicle]) -> pd.DataFrame:
    """Map each track to the simulated vehicle it followed most often."""
    owner = log.groupby("tracker_id")["sim_id"].agg(lambda s: s.mode().iat[0])
    seen = set(log.loc[log["sim_id"] > 0, "sim_id"])
    rows = []
    for v in vehicles:
        if v.vid not in seen:
            continue  # never detected inside the zone: not part of the test set
        tids = [int(t) for t, s in owner.items() if s == v.vid]
        rows.append({"vehicle_id": v.vid, "tracker_id": ";".join(map(str, tids)) or None,
                     "true_speed_kmh": v.speed_kmh})
    for t, s in owner.items():
        if s == -1:
            rows.append({"vehicle_id": None, "tracker_id": str(int(t)), "true_speed_kmh": np.nan})
    return pd.DataFrame(rows)


# name -> (speed method, perspective corrected, use the original 24 x 249 target)
VARIANTS = {
    "Perspective corrected (net displacement)": ("displacement", True, False),
    "Perspective corrected (path length, literal)": ("path", True, False),
    "Original 2024 script (legacy)": ("legacy", True, True),
    "No correction (constant scale)": ("displacement", False, False),
}


def run_experiment(n_scenarios: int = 20, seed: int = 7, sim: SimConfig | None = None,
                   config: PipelineConfig | None = None, tol_kmh: float = 5.0
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (per-scenario metrics, per-vehicle details) for every variant."""
    sim = sim or SimConfig()
    config = config or PipelineConfig()
    metrics_rows, details = [], []
    for k in range(n_scenarios):
        rng = np.random.default_rng(seed + k)
        vehicles = make_vehicles(sim, rng)
        frames = simulate_detections(sim, vehicles, rng, config.source, config.target)
        for name, (method, corrected, legacy_target) in VARIANTS.items():
            cfg = replace(config, speed_method=method)
            if legacy_target:
                # The scene is still the true 25 m x 250 m road; only the
                # pipeline's metric frame is the original's 24 m x 249 m.
                cfg = replace(cfg, target=LEGACY_TARGET.copy())
            transformer = None if corrected else NaiveScaleTransformer(cfg.source, cfg.target)
            log = run_pipeline_on(frames, sim.fps, cfg, transformer)
            gt = ground_truth_from(log, vehicles)
            est = summarize_tracks(log)
            m, detail = evaluate(est, gt, tol_kmh=tol_kmh)
            metrics_rows.append({"scenario": k, "variant": name, **m.to_dict()})
            detail.insert(0, "variant", name)
            detail.insert(0, "scenario", k)
            details.append(detail)
    return pd.DataFrame(metrics_rows), pd.concat(details, ignore_index=True)


def summarize_experiment(per_scenario: pd.DataFrame, details: pd.DataFrame) -> pd.DataFrame:
    """Pooled metrics per variant (counts summed over scenarios)."""
    out = []
    for name, g in per_scenario.groupby("variant", sort=False):
        tp, fp, fn = g["tp"].sum(), g["fp"].sum(), g["fn"].sum()
        real, spur = g["n_real_vehicles"].sum(), g["n_spurious"].sum()
        p = tp / (tp + fp) if tp + fp else float("nan")
        r = tp / (tp + fn) if tp + fn else float("nan")
        f1 = 2 * p * r / (p + r) if tp else 0.0
        d = details[(details["variant"] == name) & details["outcome"].isin(["correct", "wrong"])]
        err = d["estimated_speed_kmh"] - d["true_speed_kmh"]
        out.append({
            "variant": name, "accuracy": tp / (real + spur), "precision": p, "recall": r,
            "f1": f1, "mae_kmh": err.abs().mean(), "bias_kmh": err.mean(),
            "rmse_kmh": float(np.sqrt((err ** 2).mean())), "n_real_vehicles": int(real),
            "tp": int(tp), "fp": int(fp), "fn": int(fn),
        })
    return pd.DataFrame(out)
