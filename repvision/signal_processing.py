from __future__ import annotations

import dataclasses

import numpy as np
from scipy import signal as sp_signal

from .pose import PoseResult

BODY_JOINTS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


@dataclasses.dataclass
class RepEvent:
    index: int
    t_start: float
    t_peak: float
    t_end: float
    prominence: float


@dataclasses.dataclass
class RepSignal:
    t: np.ndarray
    raw: np.ndarray
    filtered: np.ndarray
    reps: list[RepEvent]
    period_est: float

    @property
    def count(self) -> int:
        return len(self.reps)


def _normalize_skeleton(pose: PoseResult) -> np.ndarray:
    pts = pose.world_landmarks[:, BODY_JOINTS, :]
    mid_hip = (pose.world_landmarks[:, 23] + pose.world_landmarks[:, 24]) / 2
    mid_sho = (pose.world_landmarks[:, 11] + pose.world_landmarks[:, 12]) / 2
    torso = np.linalg.norm(mid_sho - mid_hip, axis=1)
    scale = np.median(torso[torso > 1e-6]) or 1.0
    return (pts - mid_hip[:, None, :]) / scale


def _pca_signal(traj: np.ndarray) -> np.ndarray:
    X = traj.reshape(len(traj), -1)
    X = X - X.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    return X @ vt[0]


def _dominant_period(x: np.ndarray, fps: float,
                     lo: float = 0.4, hi: float = 15.0) -> float:
    x = x - x.mean()
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    if ac[0] <= 0:
        return 1.5
    ac /= ac[0]
    i0, i1 = int(lo * fps), min(int(hi * fps), len(ac) - 1)
    if i1 <= i0:
        return 1.5
    peaks, _ = sp_signal.find_peaks(ac[i0:i1])
    if len(peaks) == 0:
        return 1.5
    return (i0 + peaks[np.argmax(ac[i0:i1][peaks])]) / fps


class RepDetector:

    def __init__(self, min_prominence: float = 0.35,
                 smooth_window_s: float = 0.35,
                 detrend_window_s: float = 6.0):
        self.min_prominence = min_prominence
        self.smooth_window_s = smooth_window_s
        self.detrend_window_s = detrend_window_s

    def detect(self, pose: PoseResult) -> RepSignal:
        fps = pose.fps
        traj = _normalize_skeleton(pose)
        raw = _pca_signal(traj)

        win = max(3, int(self.detrend_window_s * fps) | 1)
        trend = sp_signal.medfilt(raw, kernel_size=min(win, len(raw) // 2 * 2 - 1))
        x = raw - trend

        sw = max(5, int(self.smooth_window_s * fps) | 1)
        if len(x) > sw:
            x = sp_signal.savgol_filter(x, sw, polyorder=2)

        mad = np.median(np.abs(x - np.median(x))) or 1e-6
        xn = x / (1.4826 * mad)

        period = _dominant_period(xn, fps)
        min_dist = max(3, int(0.55 * period * fps))

        best = None
        for sgn in (+1, -1):
            amp = np.max(sgn * xn)
            peaks, props = sp_signal.find_peaks(
                sgn * xn, prominence=self.min_prominence * np.max(np.abs(xn)),
                height=0.3 * amp, distance=min_dist)
            score = props["prominences"].sum() if len(peaks) else 0.0
            if best is None or score > best[0]:
                best = (score, sgn, peaks, props)
        _, sgn, peaks, props = best
        y = sgn * xn

        max_half = max(min_dist, int(period * fps))
        reps = []
        for k, p in enumerate(peaks):
            left = max(peaks[k - 1] if k > 0 else 0, p - max_half)
            right = min(peaks[k + 1] if k + 1 < len(peaks) else len(y) - 1,
                        p + max_half)
            v0 = left + int(np.argmin(y[left:p + 1])) if p > left else left
            v1 = p + int(np.argmin(y[p:right + 1])) if right > p else right
            reps.append(RepEvent(
                index=k + 1,
                t_start=pose.timestamps[v0],
                t_peak=pose.timestamps[p],
                t_end=pose.timestamps[v1],
                prominence=float(props["prominences"][k]),
            ))

        return RepSignal(t=pose.timestamps, raw=raw, filtered=y,
                         reps=reps, period_est=period)
