"""Perspective correction (Sec. 4.IV).

H = getPerspectiveTransform(SOURCE, TARGET) maps image pixels on the road plane
to metric road coordinates: [x' y' w]^T = H [x y 1]^T, (x', y') / w.
"""
from __future__ import annotations

import cv2
import numpy as np


class ViewTransformer:
    """Maps points between the image and the metric road plane."""

    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = np.asarray(source, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)
        if source.shape != (4, 2) or target.shape != (4, 2):
            raise ValueError("SOURCE and TARGET must each hold four (x, y) points")
        self.m = cv2.getPerspectiveTransform(source, target)
        self.m_inv = cv2.getPerspectiveTransform(target, source)

    @staticmethod
    def _apply(m: np.ndarray, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float32)
        if points.size == 0:
            return points.reshape(-1, 2)
        reshaped = points.reshape(-1, 1, 2)
        return cv2.perspectiveTransform(reshaped, m).reshape(-1, 2)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Image pixels -> road metres."""
        return self._apply(self.m, points)

    def inverse_transform_points(self, points: np.ndarray) -> np.ndarray:
        """Road metres -> image pixels."""
        return self._apply(self.m_inv, points)


class NaiveScaleTransformer:
    """Baseline *without* perspective correction.

    Converts pixels to metres with one constant scale: the road stretch length
    divided by the vertical pixel extent of the source region. This is what a
    system that ignores perspective distortion would do, and it is used only
    to measure how much the homography helps.
    """

    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = np.asarray(source, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)
        px_height = float(source[:, 1].max() - source[:, 1].min())
        m_height = float(target[:, 1].max() - target[:, 1].min())
        self.metres_per_pixel = m_height / px_height

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(points, dtype=np.float32).reshape(-1, 2) * self.metres_per_pixel
