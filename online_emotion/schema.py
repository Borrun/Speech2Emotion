from dataclasses import dataclass
from typing import List, Optional


@dataclass
class DetectorConfig:
    fps: int = 30
    w_audio: float = 0.65
    w_text: float = 0.35
    thr_on: float = 0.58
    thr_off: float = 0.45
    confirm_win: int = 4
    min_gap: int = 6
    adapt_lr_offset: float = 0.15
    adapt_lr_scale: float = 0.05


@dataclass
class BoundaryEvent:
    frame_idx: int
    t_sec: float
    p_audio: float
    p_text: float
    p_fused: float
    reason: str = "fused_peak_confirmed"


@dataclass
class ChunkResult:
    frame_start: int
    frame_end: int
    fused_probs: List[float]
    text_probs: List[float]
    events: List[BoundaryEvent]
    pending_peak: Optional[int]

