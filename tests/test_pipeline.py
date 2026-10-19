import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import supervision as sv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from speed_estimation.config import SOURCE, TARGET, PipelineConfig  # noqa: E402
from speed_estimation.evaluation import evaluate, ground_truth_from_timings  # noqa: E402
from speed_estimation.pipeline import SpeedPipeline, process_video, summarize_tracks  # noqa: E402
from speed_estimation.published import PUBLISHED, reproduce_table1  # noqa: E402
from speed_estimation.simulation import SimConfig, run_experiment, summarize_experiment  # noqa: E402
from speed_estimation.speed import SpeedEstimator, window_speed_kmh  # noqa: E402
from speed_estimation.transform import ViewTransformer  # noqa: E402


def test_homography_maps_source_to_paper_target():
    vt = ViewTransformer(SOURCE, TARGET)
    np.testing.assert_allclose(vt.transform_points(SOURCE), TARGET, atol=1e-3)
    np.testing.assert_allclose(TARGET, [[0, 0], [25, 0], [25, 250], [0, 250]])
    pts = np.array([[1800, 1000], [2500, 2000]], dtype=np.float32)
    np.testing.assert_allclose(vt.inverse_transform_points(vt.transform_points(pts)), pts, atol=1e-2)


@pytest.mark.parametrize("method", ["displacement", "path"])
def test_constant_speed_is_recovered(method):
    fps, v = 25, 100 / 3.6
    pts = np.column_stack([np.full(26, 5.0), np.arange(26) * v / fps])
    assert window_speed_kmh(pts, fps, method) == pytest.approx(100, abs=1e-6)


def test_one_second_average_is_distance_in_one_second():
    # Paper: with ΔT = 1 s, the averaged speed equals the distance covered in that second.
    fps = 25
    pts = np.column_stack([np.zeros(fps + 1), np.linspace(0, 30, fps + 1)])  # 30 m in 1 s
    assert window_speed_kmh(pts, fps) == pytest.approx(30 * 3.6)


def test_path_length_accumulates_jitter_but_displacement_does_not():
    fps, rng = 25, np.random.default_rng(0)
    truth = np.column_stack([np.zeros(26), np.arange(26) * (80 / 3.6) / fps])
    path, disp = [], []
    for _ in range(200):
        noisy = truth + rng.normal(0, 0.3, truth.shape)
        path.append(window_speed_kmh(noisy, fps, "path"))
        disp.append(window_speed_kmh(noisy, fps, "displacement"))
    assert np.mean(disp) == pytest.approx(80, abs=1.5)   # unbiased
    assert np.mean(path) > 85                              # biased upward by jitter


def test_estimator_warm_up_and_window():
    est = SpeedEstimator(fps=25)
    assert est.window == 26 and est.min_points == 13
    out = None
    for f in range(40):
        out = est.update([7], [[0.0, f * 1.0]])  # 1 m per frame = 90 km/h
        if f < 12:
            assert out[7] is None
    assert out[7] == pytest.approx(90.0)


def test_missed_detections_do_not_inflate_speed():
    est = SpeedEstimator(fps=25)
    out = {}
    for f in range(60):
        ids = [3] if f % 3 else []          # detector misses every third frame
        out = est.update(ids, [[0.0, f * 1.0]] if ids else np.empty((0, 2))) or out
    assert out[3] == pytest.approx(90.0)


def test_metric_definitions():
    est = pd.DataFrame({"tracker_id": [1, 2, 3, 4], "estimated_speed_kmh": [100, 80, 150, 60],
                        "n_estimates": [10, 10, 10, 10]})
    gt = pd.DataFrame({
        "vehicle_id": [1, 2, 3, 4, None],
        "tracker_id": ["1", "2", "3", None, "4"],
        "true_speed_kmh": [102, 79, 120, 90, np.nan],
    })
    m, detail = evaluate(est, gt, tol_kmh=5)
    assert (m.n_correct, m.n_wrong, m.n_missed, m.n_spurious) == (2, 1, 1, 1)
    assert (m.tp, m.fp, m.fn) == (2, 2, 2)
    assert m.precision == pytest.approx(0.5) and m.recall == pytest.approx(0.5)
    assert m.accuracy == pytest.approx(2 / 5)
    assert m.f1 == pytest.approx(0.5)


def test_labels_reproduce_table1():
    m = reproduce_table1()
    assert (m["n"], m["tp"], m["fp"], m["fn"], m["tn"]) == (15, 9, 1, 2, 3)
    for key, value in PUBLISHED.items():
        assert round(m[key], 2) == value


def test_legacy_method_reproduces_original_calculation():
    # 1 m per frame at 25 fps = 90 km/h. The original counts 25 points as 25
    # intervals (24 m in "1 s"), so it reports 86.4 km/h: 4% low.
    est = SpeedEstimator(fps=25, method="legacy")
    out = {}
    for f in range(40):
        out = est.update([7], [[3.7, f * 1.0 + 0.5]])
    assert out[7] == pytest.approx(86.4)
    # Distances are truncated to whole metres and only y counts.
    pts = np.array([[0.0, 0.9], [9.0, 10.2]])
    assert window_speed_kmh(pts, fps=1, method="legacy") == pytest.approx(10 / 2 * 3.6)


def test_timings_give_true_speeds_that_evaluate_accepts():
    # 50 m in 2 s = 90 km/h; 100 m (two gaps) in 3.6 s = 100 km/h.
    timings = pd.DataFrame({"tracker_id": [3, 5], "frame_start": [100, 40],
                            "frame_end": [150, 130], "posts_crossed": [1, 2]})
    gt = ground_truth_from_timings(timings, fps=25)
    assert gt["true_speed_kmh"].tolist() == pytest.approx([90.0, 100.0])
    est = pd.DataFrame({"tracker_id": [3, 5], "estimated_speed_kmh": [91.0, 98.0], "n_estimates": [20, 20]})
    m, _ = evaluate(est, gt, tol_kmh=5)
    assert (m.tp, m.fp, m.fn) == (2, 0, 0)
    with pytest.raises(ValueError):
        ground_truth_from_timings(timings.assign(frame_end=[100, 130]), fps=25)


def test_yolo_nas_detector_adapts_prediction_and_filters_classes(monkeypatch):
    """Uses a stand-in for super-gradients (not installed in the test environment)."""
    import sys
    import types

    from speed_estimation.detection import YoloNasDetector

    class FakeModel:
        def predict(self, frame):
            prediction = types.SimpleNamespace(
                bboxes_xyxy=np.array([[10, 20, 50, 80], [100, 100, 150, 160]], np.float32),
                confidence=np.array([0.9, 0.8], np.float32),
                labels=np.array([2.0, 0.0]),   # a car and a person
            )
            return [types.SimpleNamespace(prediction=prediction)]

    asked = {}
    models = types.SimpleNamespace(get=lambda name, pretrained_weights: asked.update(
        name=name, weights=pretrained_weights) or FakeModel())
    for name, module in {"super_gradients": types.ModuleType("super_gradients"),
                         "super_gradients.training": types.SimpleNamespace(models=models)}.items():
        monkeypatch.setitem(sys.modules, name, module)

    detections = YoloNasDetector(class_ids=(2, 3, 5, 7))(np.zeros((200, 200, 3), np.uint8))
    assert asked == {"name": "yolo_nas_l", "weights": "coco"}
    assert len(detections) == 1 and detections.class_id.tolist() == [2]


def test_estimator_forgets_finished_tracks():
    est = SpeedEstimator(fps=25, forget_after_s=2.0)
    for f in range(10):
        est.update([1], [[0.0, f * 1.0]])
    for _ in range(60):
        est.update([], np.empty((0, 2)))
    assert 1 not in est._history


def _moving_box_frames(speed_kmh, fps=25, n=60, x_m=6.0, y0=40.0):
    vt = ViewTransformer(SOURCE, TARGET)
    frames = []
    for f in range(n):
        y = y0 + speed_kmh / 3.6 * f / fps
        (cx, by), = vt.inverse_transform_points([[x_m, y]])
        w = 2.0 * abs(vt.inverse_transform_points([[x_m + 1, y]])[0, 0] - cx)
        frames.append(sv.Detections(xyxy=np.array([[cx - w / 2, by - w, cx + w / 2, by]], np.float32),
                                    confidence=np.array([0.9]), class_id=np.array([2])))
    return frames


def test_pipeline_recovers_known_speed_through_tracker():
    pipe = SpeedPipeline(fps=25, config=PipelineConfig())
    speeds = []
    for det in _moving_box_frames(110):
        res = pipe.step(det)
        speeds += [s for s in res.speeds.values() if s is not None]
    assert speeds and np.median(speeds) == pytest.approx(110, abs=0.5)


def test_simulation_perspective_correction_beats_constant_scale():
    per, det = run_experiment(n_scenarios=2, seed=1, sim=SimConfig(n_vehicles=12))
    s = summarize_experiment(per, det).set_index("variant")
    corrected = s.loc["Perspective corrected (net displacement)"]
    naive = s.loc["No correction (constant scale)"]
    assert corrected["f1"] > 0.85
    assert corrected["mae_kmh"] < naive["mae_kmh"] / 3


def test_process_video_end_to_end(tmp_path):
    """Runs video I/O, tracking, annotation and logging with a stand-in detector."""
    fps, w, h = 25, 3840, 2160
    frames = _moving_box_frames(90, fps=fps, n=40)
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for _ in frames:
        writer.write(np.zeros((h, w, 3), np.uint8))
    writer.release()
    it = iter(frames)
    log = process_video(path, lambda frame: next(it), target_path=tmp_path / "out.mp4",
                        log_path=tmp_path / "log.csv", show_progress=False)
    assert (tmp_path / "out.mp4").stat().st_size > 0
    summary = summarize_tracks(log)
    assert len(summary) == 1
    assert summary["estimated_speed_kmh"].iat[0] == pytest.approx(90, abs=0.5)
