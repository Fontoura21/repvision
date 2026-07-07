"""Saídas visuais: vídeo anotado com esqueleto + contador e gráfico do sinal."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .pose import PoseResult, SKELETON
from .set_grouping import SetInfo
from .signal_processing import RepSignal

# Paleta Material Design v2 (Indigo 500 / Amber 500 / Teal 400)
C_PRIMARY = (243, 81, 63)     # BGR de #3F51B5
C_ACCENT = (7, 193, 255)      # BGR de #FFC107
C_SKELETON = (136, 189, 38)   # BGR de #26BD88


def _hud(frame: np.ndarray, rep_n: int, set_n: int, set_rep: int,
         t: float) -> None:
    h, w = frame.shape[:2]
    bar_h = max(56, h // 12)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), C_PRIMARY, -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    scale = bar_h / 70.0
    cv2.putText(frame, f"Serie {set_n}  |  Rep {set_rep}  (total {rep_n})",
                (int(12 * scale), int(bar_h * 0.65)),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255),
                max(1, int(2 * scale)), cv2.LINE_AA)
    label = f"{t:6.1f}s"
    (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    cv2.putText(frame, label, (w - tw - int(12 * scale), int(bar_h * 0.65)),
                cv2.FONT_HERSHEY_SIMPLEX, scale, C_ACCENT,
                max(1, int(2 * scale)), cv2.LINE_AA)


def render_video(video_path: str | Path, out_path: str | Path,
                 pose: PoseResult, rep_signal: RepSignal,
                 sets: list[SetInfo]) -> None:
    """Gera o vídeo anotado (esqueleto + contador de séries/repetições)."""
    cap = cv2.VideoCapture(str(video_path))
    w, h = pose.frame_size
    writer = cv2.VideoWriter(str(out_path),
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             pose.fps, (w, h))

    # para cada instante, quantas repetições/séries já se completaram
    rep_peaks = [r.t_peak for r in rep_signal.reps]
    set_of_rep = {}
    for s in sets:
        for r in s.reps:
            set_of_rep[r.index] = s.index

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or idx >= pose.n_frames:
            break
        t = pose.timestamps[idx]

        # esqueleto
        pts = pose.landmarks[idx]
        vis = pose.visibility[idx]
        for a, b in SKELETON:
            if vis[a] > 0.5 and vis[b] > 0.5:
                pa = (int(pts[a][0] * w), int(pts[a][1] * h))
                pb = (int(pts[b][0] * w), int(pts[b][1] * h))
                cv2.line(frame, pa, pb, C_SKELETON, 2, cv2.LINE_AA)
        for j, (x, y) in enumerate(pts):
            if vis[j] > 0.5:
                cv2.circle(frame, (int(x * w), int(y * h)), 3,
                           C_ACCENT, -1, cv2.LINE_AA)

        done = sum(1 for tp in rep_peaks if tp <= t)
        cur_set = 1
        set_rep = done
        if done:
            cur_set = set_of_rep.get(done, 1)
            first_of_set = min(r.index for s in sets if s.index == cur_set
                               for r in s.reps)
            set_rep = done - first_of_set + 1
        _hud(frame, done, cur_set, set_rep, t)

        writer.write(frame)
        idx += 1
    cap.release()
    writer.release()


def plot_signal(out_path: str | Path, rep_signal: RepSignal,
                sets: list[SetInfo], title: str = "") -> None:
    """Gráfico do sinal de movimento com picos e faixas das séries."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 4), dpi=150)
    ax.plot(rep_signal.t, rep_signal.filtered, color="#3F51B5", lw=1.2,
            label="sinal de movimento (PC1 filtrado)")
    tp = [r.t_peak for r in rep_signal.reps]
    yp = np.interp(tp, rep_signal.t, rep_signal.filtered)
    ax.plot(tp, yp, "o", color="#FFC107", ms=6, mec="#3F51B5",
            label=f"repeticoes ({rep_signal.count})")
    for s in sets:
        ax.axvspan(s.t_start, s.t_end, color="#26BD88", alpha=0.12)
        ax.text((s.t_start + s.t_end) / 2, ax.get_ylim()[1] * 0.9,
                f"série {s.index}: {s.count} reps", ha="center",
                fontsize=9, color="#00695C")
    ax.set_xlabel("tempo (s)")
    ax.set_ylabel("amplitude (norm.)")
    if title:
        ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
