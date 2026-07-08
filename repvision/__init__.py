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
