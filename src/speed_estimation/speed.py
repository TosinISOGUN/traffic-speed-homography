"""Speed calculation (Sec. 4.V).

Per-frame speed v_t = Δd_t * FPS is noisy, so the manuscript averages over
N = FPS frames (one second):

    v̄ = ΔD / ΔT = (Σ_{i=t}^{t+N-1} Δd_i) * FPS / N

and, with ΔT = 1 s, v̄ is simply the distance covered in the last second.

method="legacy" reproduces the original 2024 script (Roboflow's speed
estimation example): it keeps the last FPS road y-coordinates, truncated to
whole metres, and divides |y_last - y_first| by (number of points / FPS). That
uses only the along-road axis, quantises distance to 1 m, and counts N points
as N frame intervals instead of N - 1, so speeds come out about 4% low.
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

MS_TO_KMH = 3.6


def window_speed_kmh(points_m: np.ndarray, fps: float, method: str = "displacement",
                     frames: np.ndarray | None = None) -> float:
    """Speed over a window of metric positions.

    points_m : (k, 2) array of road-plane positions, oldest first.
    frames   : frame index of each position. Needed when the detector missed
               frames: elapsed time is (last - first) / fps, not (k - 1) / fps.
               If omitted, positions are assumed to be on consecutive frames.
    method   : "displacement" -> net distance from first to last point
               (= Σ signed Δd, the manuscript's ΔD);
               "path" -> sum of unsigned frame-to-frame step lengths;
               "legacy" -> the original script's calculation (see module
               docstring); `frames` is ignored, as it was originally.
    """
    points_m = np.asarray(points_m, dtype=float)
    k = len(points_m)
    if k < 2:
        raise ValueError("need at least two positions")
    if frames is None:
        elapsed_s = (k - 1) / fps
    else:
        elapsed_s = (float(frames[-1]) - float(frames[0])) / fps
        if elapsed_s <= 0:
            raise ValueError("frames must be increasing")
    if method == "path":
        distance_m = float(np.linalg.norm(np.diff(points_m, axis=0), axis=1).sum())
    elif method == "displacement":
        distance_m = float(np.linalg.norm(points_m[-1] - points_m[0]))
    elif method == "legacy":
        y = points_m[:, 1].astype(int)  # the original cast road positions to int
        return abs(int(y[-1]) - int(y[0])) / (k / fps) * MS_TO_KMH
    else:
        raise ValueError(f"unknown method {method!r}")
    return distance_m / elapsed_s * MS_TO_KMH


class SpeedEstimator:
    """Keeps the last `window_seconds` of positions per tracker ID.

    Positions are stored with their frame index, so frames where the detector
    missed a vehicle do not shrink the measured time. Tracks not seen for
    `forget_after_s` are dropped, so memory stays bounded on long streams.

    method="legacy" instead keeps the last FPS positions regardless of missed
    frames and reports a speed once it holds FPS / 2 of them, as the original
    script did.
    """

    def __init__(
        self,
        fps: float,
        window_seconds: float = 1.0,
        min_window_fraction: float = 0.5,
        method: str = "displacement",
        forget_after_s: float = 5.0,
    ) -> None:
        if method not in ("displacement", "path", "legacy"):
            raise ValueError(f"unknown method {method!r}")
        self.fps = float(fps)
        # A one-second window spans N = FPS frame intervals (N + 1 positions).
        self.window_frames = max(1, int(round(self.fps * window_seconds)))
        self.window = self.window_frames + 1
        self.min_span = max(1, int(round(self.window_frames * min_window_fraction)))
        self.min_points = self.min_span + 1
        self.method = method
        # Must exceed the tracker's lost-track buffer (ByteTrack: about 1 s), so a
        # track is only forgotten once its ID can no longer come back.
        self.forget_after = max(self.window_frames, int(round(self.fps * forget_after_s)))
        self._frame = -1
        self._history: dict[int, deque] = defaultdict(deque)

    def update(self, tracker_ids, points_m, frame: int | None = None) -> dict[int, float | None]:
        """Add one frame of positions; return the current speed per tracker ID.

        frame : index of this frame. Defaults to one more than the last call.
        A speed is None until the track spans at least half the window.
        """
        self._frame = self._frame + 1 if frame is None else int(frame)
        speeds: dict[int, float | None] = {}
        for tid, point in zip(tracker_ids, np.asarray(points_m).reshape(-1, 2)):
            tid = int(tid)
            history = self._history[tid]
            history.append((self._frame, point))
            if self.method == "legacy":
                speeds[tid] = self._legacy_speed(history)
                continue
            while history and history[0][0] < self._frame - self.window_frames:
                history.popleft()
            span = history[-1][0] - history[0][0]
            if len(history) >= 2 and span >= self.min_span:
                frames = np.array([f for f, _ in history])
                pts = np.array([p for _, p in history])
                speeds[tid] = window_speed_kmh(pts, self.fps, self.method, frames)
            else:
                speeds[tid] = None
        self._forget_stale()
        return speeds

    def _legacy_speed(self, history: deque) -> float | None:
        # Original: deque(maxlen=fps); no speed while len < fps / 2.
        while len(history) > self.window_frames:
            history.popleft()
        if len(history) < self.fps / 2:
            return None
        pts = np.array([p for _, p in history])
        return window_speed_kmh(pts, self.fps, "legacy")

    def _forget_stale(self) -> None:
        cutoff = self._frame - self.forget_after
        for tid in [t for t, h in self._history.items() if h[-1][0] < cutoff]:
            del self._history[tid]
