"""Do esqueleto ao evento de repetição.

Etapas:
1. normalização do esqueleto (invariância a escala e posição na cena);
2. projeção das trajetórias articulares no 1º componente principal (PCA),
   obtendo um sinal 1D que oscila no ritmo do exercício;
3. filtragem passa-banda implícita (detrend por mediana móvel + suavização
   Savitzky-Golay);
4. detecção de picos com proeminência e distância adaptativas, estimadas
   a partir da autocorrelação do sinal.

Cada pico = uma repetição; os vales adjacentes delimitam o ciclo.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy import signal as sp_signal

from .pose import PoseResult

# Articulações usadas no sinal: ombros, cotovelos, punhos, quadris,
# joelhos e tornozelos — o rosto e os dedos só adicionam ruído.
BODY_JOINTS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


@dataclasses.dataclass
class RepEvent:
    """Uma repetição detectada."""

    index: int        # nº da repetição (1..N)
    t_start: float    # início do ciclo (s)
    t_peak: float     # ápice do movimento (s)
    t_end: float      # fim do ciclo (s)
    prominence: float # proeminência do pico no sinal normalizado


@dataclasses.dataclass
class RepSignal:
    """Sinal 1D de movimento e as repetições detectadas nele."""

    t: np.ndarray          # (T,) segundos
    raw: np.ndarray        # sinal PC1 bruto
    filtered: np.ndarray   # sinal filtrado usado na detecção
    reps: list[RepEvent]
    period_est: float      # período dominante estimado (s)

    @property
    def count(self) -> int:
        return len(self.reps)


def _normalize_skeleton(pose: PoseResult) -> np.ndarray:
    """Centraliza no ponto médio dos quadris e normaliza pelo tronco."""
    pts = pose.world_landmarks[:, BODY_JOINTS, :]          # (T, J, 3)
    mid_hip = (pose.world_landmarks[:, 23] + pose.world_landmarks[:, 24]) / 2
    mid_sho = (pose.world_landmarks[:, 11] + pose.world_landmarks[:, 12]) / 2
    torso = np.linalg.norm(mid_sho - mid_hip, axis=1)      # (T,)
    scale = np.median(torso[torso > 1e-6]) or 1.0
    return (pts - mid_hip[:, None, :]) / scale


def _pca_signal(traj: np.ndarray) -> np.ndarray:
    """Projeta as trajetórias (T, J, 3) no 1º componente principal."""
    X = traj.reshape(len(traj), -1)
    X = X - X.mean(axis=0, keepdims=True)
    # SVD é a forma estável de obter o PC1 sem depender de sklearn
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    return X @ vt[0]


def _dominant_period(x: np.ndarray, fps: float,
                     lo: float = 0.4, hi: float = 15.0) -> float:
    """Período dominante via autocorrelação, restrito a [lo, hi] segundos."""
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
    """Detecta repetições a partir de um :class:`PoseResult`."""

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

        # detrend: remove deriva lenta (deslocamentos do atleta na cena)
        win = max(3, int(self.detrend_window_s * fps) | 1)
        trend = sp_signal.medfilt(raw, kernel_size=min(win, len(raw) // 2 * 2 - 1))
        x = raw - trend

        # suavização Savitzky-Golay preserva a forma dos picos
        sw = max(5, int(self.smooth_window_s * fps) | 1)
        if len(x) > sw:
            x = sp_signal.savgol_filter(x, sw, polyorder=2)

        # amplitude robusta (MAD) para normalizar a proeminência
        mad = np.median(np.abs(x - np.median(x))) or 1e-6
        xn = x / (1.4826 * mad)

        period = _dominant_period(xn, fps)
        min_dist = max(3, int(0.55 * period * fps))

        # o sinal PCA tem sinal arbitrário: escolhe a orientação com
        # picos mais proeminentes; além da proeminência, exige altura
        # mínima (30% do pico máximo) para descartar oscilações fora do
        # exercício, como o atleta entrando em posição
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

        # o vale que delimita o ciclo é buscado no máximo a 1 período do
        # pico — sem isso, uma pausa longa entre séries seria absorvida
        # pela repetição vizinha
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
