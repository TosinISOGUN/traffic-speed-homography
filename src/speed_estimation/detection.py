"""Object detection (Sec. 3.D.i, 4.II): YOLOv8x at 640 x 640."""
from __future__ import annotations

from typing import Protocol

import numpy as np
import supervision as sv


class Detector(Protocol):
    def __call__(self, frame: np.ndarray) -> sv.Detections: ...


class UltralyticsDetector:
    """YOLOv8 through the Ultralytics package (weights download on first use)."""

    def __init__(self, model: str = "yolov8x.pt", imgsz: int = 640, conf: float = 0.3,
                 class_ids: tuple[int, ...] | None = None) -> None:
        from ultralytics import YOLO

        self.model = YOLO(model)
        self.imgsz = imgsz
        self.conf = conf
        self.class_ids = class_ids

    def __call__(self, frame: np.ndarray) -> sv.Detections:
        result = self.model(frame, imgsz=self.imgsz, conf=self.conf, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        if self.class_ids is not None:
            detections = detections[np.isin(detections.class_id, self.class_ids)]
        return detections


class YoloNasDetector:
    """YOLO-NAS-L with COCO weights, the detector in the 2024 script behind the
    manuscript's results (Roboflow's `yolo_nas_example.py`).

    Needs `pip install super-gradients`. As in that script, no confidence
    threshold is passed to the model; `SpeedPipeline` applies its own.
    """

    def __init__(self, model_name: str = "yolo_nas_l",
                 class_ids: tuple[int, ...] | None = None) -> None:
        from super_gradients.training import models

        self.model = models.get(model_name, pretrained_weights="coco")
        self.class_ids = class_ids

    def __call__(self, frame: np.ndarray) -> sv.Detections:
        result = self.model.predict(frame)[0]
        detections = sv.Detections.from_yolo_nas(result)
        if self.class_ids is not None:
            detections = detections[np.isin(detections.class_id, self.class_ids)]
        return detections


class RoboflowDetector:
    """The exact hosted model named in the manuscript ("yolov8x-640").

    Needs `pip install inference` and a Roboflow API key in ROBOFLOW_API_KEY.
    """

    def __init__(self, model_id: str = "yolov8x-640", conf: float = 0.3,
                 class_ids: tuple[int, ...] | None = None) -> None:
        from inference import get_model

        self.model = get_model(model_id)
        self.conf = conf
        self.class_ids = class_ids

    def __call__(self, frame: np.ndarray) -> sv.Detections:
        result = self.model.infer(frame, confidence=self.conf)[0]
        detections = sv.Detections.from_inference(result)
        if self.class_ids is not None:
            detections = detections[np.isin(detections.class_id, self.class_ids)]
        return detections
