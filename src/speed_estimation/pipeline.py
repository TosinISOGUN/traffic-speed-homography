"""End-to-end pipeline (Fig. 1, Sec. 4).

detect -> confidence filter -> polygon-zone filter -> NMS -> ByteTrack ->
bottom-centre anchor -> perspective transform -> one-second speed averaging.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import supervision as sv

from .config import PipelineConfig
from .detection import Detector
from .speed import SpeedEstimator
from .transform import ViewTransformer

LOG_COLUMNS = [
    "frame", "time_s", "tracker_id", "class_id", "class_name", "confidence",
    "x_px", "y_px", "x_m", "y_m", "speed_kmh",
]


@dataclass
class FrameResult:
    detections: sv.Detections          # tracked detections inside the zone
    points_m: np.ndarray               # (n, 2) road-plane positions
    speeds: dict[int, float | None]    # tracker_id -> km/h (None while warming up)


class SpeedPipeline:
    """Stateful per-frame processor. Works on any detector output, so the
    synthetic validation can drive it with simulated detections."""

    def __init__(self, fps: float, config: PipelineConfig | None = None,
                 transformer=None) -> None:
        self.config = config or PipelineConfig()
        self.fps = fps
        c = self.config
        self.zone = sv.PolygonZone(
            polygon=c.source.astype(np.int64),
            triggering_anchors=(sv.Position.BOTTOM_CENTER,),
        )
        self.tracker = sv.ByteTrack(
            frame_rate=int(round(fps)),
            track_activation_threshold=c.confidence_threshold,
        )
        self.transformer = transformer or ViewTransformer(c.source, c.target)
        self.estimator = SpeedEstimator(
            fps=fps,
            window_seconds=c.window_seconds,
            min_window_fraction=c.min_window_fraction,
            method=c.speed_method,
        )

    def step(self, detections: sv.Detections) -> FrameResult:
        c = self.config
        if len(detections) and detections.confidence is not None:
            detections = detections[detections.confidence > c.confidence_threshold]
        if len(detections):
            detections = detections[self.zone.trigger(detections)]
        if len(detections):
            detections = detections.with_nms(threshold=c.iou_threshold)
        detections = self.tracker.update_with_detections(detections)

        anchors = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        points_m = self.transformer.transform_points(anchors)
        ids = detections.tracker_id if detections.tracker_id is not None else []
        speeds = self.estimator.update(ids, points_m)
        return FrameResult(detections=detections, points_m=points_m, speeds=speeds)


class Annotator:
    """Draws boxes, traces, the zone and '#id  NN km/h' labels (Fig. 4)."""

    def __init__(self, resolution_wh: tuple[int, int], fps: float, source: np.ndarray) -> None:
        thickness = sv.calculate_optimal_line_thickness(resolution_wh=resolution_wh)
        text_scale = sv.calculate_optimal_text_scale(resolution_wh=resolution_wh)
        self.source = source.astype(np.int32)
        self.thickness = thickness
        self.box = sv.BoxAnnotator(thickness=thickness, color_lookup=sv.ColorLookup.TRACK)
        self.label = sv.LabelAnnotator(
            text_scale=text_scale, text_thickness=thickness,
            text_position=sv.Position.BOTTOM_CENTER, color_lookup=sv.ColorLookup.TRACK,
        )
        self.trace = sv.TraceAnnotator(
            thickness=thickness, trace_length=int(fps * 2),
            position=sv.Position.BOTTOM_CENTER, color_lookup=sv.ColorLookup.TRACK,
        )

    def __call__(self, frame: np.ndarray, result: FrameResult) -> np.ndarray:
        d = result.detections
        labels = []
        for tid in (d.tracker_id if d.tracker_id is not None else []):
            speed = result.speeds.get(int(tid))
            labels.append(f"#{tid}" if speed is None else f"#{tid} {speed:.0f} km/h")
        out = frame.copy()
        out = sv.draw_polygon(out, polygon=self.source, color=sv.Color.RED, thickness=self.thickness)
        out = self.trace.annotate(out, d)
        out = self.box.annotate(out, d)
        out = self.label.annotate(out, d, labels=labels)
        return out


def process_video(
    source_path: str | Path,
    detector: Detector,
    target_path: str | Path | None = None,
    log_path: str | Path | None = None,
    config: PipelineConfig | None = None,
    max_frames: int | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Run the pipeline on a video. Returns (and optionally saves) the per-frame log."""
    config = config or PipelineConfig()
    info = sv.VideoInfo.from_video_path(str(source_path))
    pipeline = SpeedPipeline(fps=info.fps, config=config)
    annotator = Annotator(info.resolution_wh, info.fps, config.source)

    total = info.total_frames if max_frames is None else min(max_frames, info.total_frames or max_frames)
    frames = sv.get_video_frames_generator(str(source_path), end=total)
    rows: list[list] = []

    sink = sv.VideoSink(str(target_path), video_info=info) if target_path else None
    iterator = frames
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(frames, total=total, desc="frames")
        except ImportError:
            pass

    try:
        if sink:
            sink.__enter__()
        for index, frame in enumerate(iterator):
            result = pipeline.step(detector(frame))
            d = result.detections
            anchors = d.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            names = d.data.get("class_name", [""] * len(d)) if d.data else [""] * len(d)
            for i in range(len(d)):
                tid = int(d.tracker_id[i])
                rows.append([
                    index, index / info.fps, tid,
                    int(d.class_id[i]) if d.class_id is not None else -1,
                    str(names[i]),
                    float(d.confidence[i]) if d.confidence is not None else np.nan,
                    float(anchors[i, 0]), float(anchors[i, 1]),
                    float(result.points_m[i, 0]), float(result.points_m[i, 1]),
                    result.speeds.get(tid),
                ])
            if sink:
                sink.write_frame(annotator(frame, result))
    finally:
        if sink:
            sink.__exit__(None, None, None)

    log = pd.DataFrame(rows, columns=LOG_COLUMNS)
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        log.to_csv(log_path, index=False)
    return log


def summarize_tracks(log: pd.DataFrame, min_estimates: int = 1) -> pd.DataFrame:
    """One speed per vehicle: the median of its one-second estimates."""
    valid = log.dropna(subset=["speed_kmh"])
    if valid.empty:
        return pd.DataFrame(columns=["tracker_id", "class_name", "estimated_speed_kmh", "n_estimates",
                                     "first_frame", "last_frame"])
    grouped = valid.groupby("tracker_id")
    summary = pd.DataFrame({
        "class_name": grouped["class_name"].agg(lambda s: s.mode().iat[0] if not s.mode().empty else ""),
        "estimated_speed_kmh": grouped["speed_kmh"].median(),
        "n_estimates": grouped["speed_kmh"].size(),
        "first_frame": grouped["frame"].min(),
        "last_frame": grouped["frame"].max(),
    }).reset_index()
    return summary[summary["n_estimates"] >= min_estimates].reset_index(drop=True)


def read_first_frame(path: str | Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise IOError(f"cannot read {path}")
    return frame
