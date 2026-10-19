"""Experiment constants.

Every value is either stated in the manuscript (marked PAPER) or a documented
assumption (marked ASSUMPTION). See README, "Assumptions".
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# --------------------------------------------------------------------------- #
# Region of interest (Sec. 4.IV)
# --------------------------------------------------------------------------- #
# PAPER: the source quadrilateral A-B-C-D covers the full road width and the
# stretch between six roadside posts spaced 50 m apart, on the Supervision
# "vehicles.mp4" asset (3840 x 2160).
# ASSUMPTION: the manuscript gives the shape but not the pixel coordinates.
# These are the vertices used for this exact clip in the Roboflow tutorial the
# manuscript cites (Skalski, 2024). C and D fall outside the frame on purpose:
# the road edges are extended to the bottom image border.
SOURCE = np.array(
    [
        [1252, 787],   # A  far-left
        [2298, 803],   # B  far-right
        [5039, 2159],  # C  near-right
        [-550, 2159],  # D  near-left
    ],
    dtype=np.float32,
)

# PAPER: road width 25 m, stretch length 250 m, A' anchored at the origin.
TARGET_WIDTH_M = 25.0
TARGET_HEIGHT_M = 250.0
TARGET = np.array(
    [
        [0, 0],
        [TARGET_WIDTH_M, 0],
        [TARGET_WIDTH_M, TARGET_HEIGHT_M],
        [0, TARGET_HEIGHT_M],
    ],
    dtype=np.float32,
)

# The original 2024 script (Roboflow's example) used TARGET_WIDTH - 1 and
# TARGET_HEIGHT - 1, i.e. a 24 m x 249 m rectangle. Used only by the "legacy"
# variant that reproduces it.
LEGACY_TARGET = np.array(
    [
        [0, 0],
        [TARGET_WIDTH_M - 1, 0],
        [TARGET_WIDTH_M - 1, TARGET_HEIGHT_M - 1],
        [0, TARGET_HEIGHT_M - 1],
    ],
    dtype=np.float32,
)


@dataclass
class PipelineConfig:
    """Run-time settings for the video pipeline."""

    # PAPER: YOLOv8x at 640 x 640 ("yolov8x-640"). The Ultralytics weights are
    # the same architecture and COCO training as Roboflow's hosted model.
    model: str = "yolov8x.pt"
    imgsz: int = 640
    # ASSUMPTION: not stated in the manuscript; tutorial defaults.
    confidence_threshold: float = 0.3
    iou_threshold: float = 0.7
    # COCO ids kept as "vehicles". PAPER: the clip contains cars and trucks.
    # ASSUMPTION: bus and motorcycle are kept too so the code generalizes.
    vehicle_class_ids: tuple[int, ...] = (2, 3, 5, 7)  # car, motorcycle, bus, truck
    # PAPER: average over one second of frames (N = FPS).
    window_seconds: float = 1.0
    # ASSUMPTION: show a speed once at least half a second has been observed.
    min_window_fraction: float = 0.5
    # PAPER (Eq. for ΔD): ΔD = Σ Δd_i over the one-second window.
    # "displacement": Δd_i taken as signed steps, so the sum telescopes to the
    #                 net distance moved in the window (default; jitter cancels).
    # "path":         Δd_i taken as unsigned step lengths (literal reading;
    #                 detection jitter accumulates, so speeds are biased upward).
    # "legacy":       the original 2024 script's calculation (see speed.py).
    # The synthetic validation reports all three. See README, "Assumptions".
    speed_method: str = "displacement"
    source: np.ndarray = field(default_factory=lambda: SOURCE.copy())
    target: np.ndarray = field(default_factory=lambda: TARGET.copy())
