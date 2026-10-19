"""Estimate vehicle speeds in a video (Sec. 4, Fig. 4).

Writes an annotated video, a per-frame log, a per-vehicle summary and a
ground-truth template to fill in for evaluation.
"""
import argparse
import json
from pathlib import Path

import _path  # noqa: F401
from _path import ROOT
from speed_estimation.config import PipelineConfig
from speed_estimation.evaluation import ground_truth_template
from speed_estimation.pipeline import process_video, summarize_tracks


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default=str(ROOT / "data" / "vehicles.mp4"), help="input video")
    p.add_argument("--out-dir", default=str(ROOT / "results" / "video"))
    p.add_argument("--backend", choices=["ultralytics", "roboflow", "yolo-nas"], default="ultralytics",
                   help="yolo-nas is the detector of the 2024 script behind the manuscript's results")
    p.add_argument("--model", default=None,
                   help="Ultralytics weights (default yolov8x.pt), Roboflow model id (default "
                        "yolov8x-640) or super-gradients model name (default yolo_nas_l)")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.3, help="confidence threshold")
    p.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    p.add_argument("--method", choices=["displacement", "path", "legacy"], default="displacement",
                   help="legacy reproduces the original 2024 script, including its "
                        "(width - 1) x (length - 1) target rectangle")
    p.add_argument("--source-polygon", type=str, default=None,
                   help='JSON list of 4 [x, y] image points A,B,C,D for another camera')
    p.add_argument("--road-width", type=float, default=25.0)
    p.add_argument("--road-length", type=float, default=250.0)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--no-video", action="store_true", help="skip writing the annotated video")
    a = p.parse_args()

    import numpy as np
    from speed_estimation.config import SOURCE

    source = np.array(json.loads(a.source_polygon), dtype=np.float32) if a.source_polygon else SOURCE
    width, length = a.road_width, a.road_length
    if a.method == "legacy":
        width, length = width - 1, length - 1  # as the original script did
    target = np.array([[0, 0], [width, 0], [width, length], [0, length]], dtype=np.float32)
    model = a.model or {"ultralytics": "yolov8x.pt", "roboflow": "yolov8x-640",
                        "yolo-nas": "yolo_nas_l"}[a.backend]
    cfg = PipelineConfig(model=model, imgsz=a.imgsz, confidence_threshold=a.conf,
                         iou_threshold=a.iou, speed_method=a.method, source=source, target=target)

    if a.backend == "ultralytics":
        from speed_estimation.detection import UltralyticsDetector
        detector = UltralyticsDetector(model, a.imgsz, a.conf, cfg.vehicle_class_ids)
    elif a.backend == "yolo-nas":
        from speed_estimation.detection import YoloNasDetector
        detector = YoloNasDetector(model, cfg.vehicle_class_ids)
    else:
        from speed_estimation.detection import RoboflowDetector
        detector = RoboflowDetector(model, a.conf, cfg.vehicle_class_ids)

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = process_video(a.source, detector,
                        target_path=None if a.no_video else out / "annotated.mp4",
                        log_path=out / "frame_log.csv", config=cfg, max_frames=a.max_frames)
    summary = summarize_tracks(log)
    summary.to_csv(out / "per_vehicle_speeds.csv", index=False)
    template = out / "ground_truth_template.csv"
    ground_truth_template(summary).to_csv(template, index=False)
    if len(summary):
        print(summary.to_string(index=False))
    print(f"\n{len(summary)} vehicles with a speed estimate. Outputs in {out}")
    print(f"To evaluate, fill in true speeds in {template} and run scripts/evaluate.py")


if __name__ == "__main__":
    main()
