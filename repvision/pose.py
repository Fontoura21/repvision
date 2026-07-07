"""Extração de pose com a rede neural BlazePose (MediaPipe Pose Landmarker).

A rede neural utilizada é o BlazePose GHUM (checkpoint oficial do Google,
arquivos ``models/pose_landmarker_{lite,full}.task``). Ela devolve, por
quadro, 33 landmarks 2D normalizados + 33 landmarks 3D em metros (world),
com escore de visibilidade por articulação.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import cv2
import numpy as np

N_LANDMARKS = 33

# Conexões do esqueleto BlazePose (subconjunto estável, sem rosto detalhado)
SKELETON = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
    (24, 26), (26, 28), (27, 31), (28, 32), (15, 17),
    (16, 18), (15, 19), (16, 20), (15, 21), (16, 22),
]


@dataclasses.dataclass
class PoseResult:
    """Séries temporais de pose de um vídeo inteiro."""

    landmarks: np.ndarray        # (T, 33, 2) coords normalizadas da imagem
    world_landmarks: np.ndarray  # (T, 33, 3) coords 3D em metros (origem no quadril)
    visibility: np.ndarray       # (T, 33) escore de visibilidade por articulação
    timestamps: np.ndarray       # (T,) segundos
    fps: float
    frame_size: tuple[int, int]  # (largura, altura)

    @property
    def n_frames(self) -> int:
        return len(self.timestamps)


class PoseExtractor:
    """Envolve o MediaPipe Pose Landmarker em modo vídeo."""

    def __init__(self, model_path: str | Path, delegate_gpu: bool = False):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        delegate = (mp_python.BaseOptions.Delegate.GPU if delegate_gpu
                    else mp_python.BaseOptions.Delegate.CPU)
        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(model_path), delegate=delegate),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)

    def extract(self, video_path: str | Path,
                progress: bool = True) -> PoseResult:
        """Roda a rede de pose em todos os quadros do vídeo."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"não consegui abrir o vídeo: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        lm, wlm, vis, ts = [], [], [], []
        frame_idx = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            t_ms = int(round(frame_idx / fps * 1000))
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                                   data=rgb)
            result = self._landmarker.detect_for_video(image, t_ms)

            if result.pose_landmarks:
                pts = result.pose_landmarks[0]
                wpts = result.pose_world_landmarks[0]
                lm.append([(p.x, p.y) for p in pts])
                wlm.append([(p.x, p.y, p.z) for p in wpts])
                vis.append([p.visibility for p in pts])
            else:
                # pessoa não detectada: repete o último quadro válido
                lm.append(lm[-1] if lm else [(np.nan, np.nan)] * N_LANDMARKS)
                wlm.append(wlm[-1] if wlm else [(0.0, 0.0, 0.0)] * N_LANDMARKS)
                vis.append([0.0] * N_LANDMARKS)
            ts.append(frame_idx / fps)

            frame_idx += 1
            if progress and total and frame_idx % 100 == 0:
                print(f"  pose: {frame_idx}/{total} quadros", flush=True)
        cap.release()

        return PoseResult(
            landmarks=np.asarray(lm, dtype=np.float32),
            world_landmarks=np.asarray(wlm, dtype=np.float32),
            visibility=np.asarray(vis, dtype=np.float32),
            timestamps=np.asarray(ts, dtype=np.float64),
            fps=fps,
            frame_size=(width, height),
        )

    def close(self):
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
