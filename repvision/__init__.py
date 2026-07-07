"""RepVision — contagem de séries e repetições em exercícios de musculação.

Pipeline: vídeo → pose (BlazePose/MediaPipe) → sinal 1D (PCA) →
detecção de repetições (picos) → agrupamento em séries.
"""

__version__ = "1.0.0"

from .pose import PoseExtractor, PoseResult
from .signal_processing import RepDetector, RepEvent
from .set_grouping import group_sets, SetInfo

__all__ = [
    "PoseExtractor",
    "PoseResult",
    "RepDetector",
    "RepEvent",
    "group_sets",
    "SetInfo",
]
