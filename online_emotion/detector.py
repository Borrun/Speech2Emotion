from typing import List, Optional

from .schema import BoundaryEvent, ChunkResult, DetectorConfig
from .text_prior import TextPrior


class OnlineBoundaryDetector:
    """
    Stream detector API:
      1) init(text_prior, config)
      2) call process_chunk(frame_start, p_audio_chunk) repeatedly
      3) read returned events
    """
    def __init__(self, text_prior: TextPrior, cfg: Optional[DetectorConfig] = None):
        self.text_prior = text_prior
        self.cfg = cfg or DetectorConfig()

        self.last_confirmed = -10**9
        self._candidate_peak_idx: Optional[int] = None
        self._candidate_peak_fused = -1.0
        self._candidate_peak_audio = 0.0
        self._candidate_peak_text = 0.0

    def _mix(self, p_audio: float, p_text: float) -> float:
        p = self.cfg.w_audio * float(p_audio) + self.cfg.w_text * float(p_text)
        return float(max(0.0, min(1.0, p)))

    def _ready_to_confirm(self, cur_idx: int) -> bool:
        if self._candidate_peak_idx is None:
            return False
        return cur_idx >= self._candidate_peak_idx + int(self.cfg.confirm_win)

    def _can_emit(self, idx: int) -> bool:
        return (idx - self.last_confirmed) >= int(self.cfg.min_gap)

    def _emit_event(self) -> Optional[BoundaryEvent]:
        if self._candidate_peak_idx is None:
            return None
        idx = int(self._candidate_peak_idx)
        if not self._can_emit(idx):
            self._reset_candidate()
            return None

        ev = BoundaryEvent(
            frame_idx=idx,
            t_sec=idx / float(self.cfg.fps),
            p_audio=float(self._candidate_peak_audio),
            p_text=float(self._candidate_peak_text),
            p_fused=float(self._candidate_peak_fused),
        )
        self.last_confirmed = idx
        self.text_prior.register_boundary(
            frame_idx=idx,
            lr_offset=float(self.cfg.adapt_lr_offset),
            lr_scale=float(self.cfg.adapt_lr_scale),
        )
        self._reset_candidate()
        return ev

    def set_weights(self, w_audio: float, w_text: float) -> None:
        """Layer 1: update fusion weights dynamically between chunks."""
        self.cfg.w_audio = float(w_audio)
        self.cfg.w_text  = float(w_text)

    def _reset_candidate(self):
        self._candidate_peak_idx = None
        self._candidate_peak_fused = -1.0
        self._candidate_peak_audio = 0.0
        self._candidate_peak_text = 0.0

    def process_chunk(self, frame_start: int, p_audio_chunk: List[float]) -> ChunkResult:
        n = len(p_audio_chunk)
        text_probs = self.text_prior.probs_for_range(frame_start, n)
        fused_probs: List[float] = []
        events: List[BoundaryEvent] = []

        for i in range(n):
            idx = int(frame_start + i)
            p_audio = float(p_audio_chunk[i])
            p_text = float(text_probs[i])
            p = self._mix(p_audio, p_text)
            fused_probs.append(p)

            if self._candidate_peak_idx is None:
                if p >= float(self.cfg.thr_on):
                    self._candidate_peak_idx = idx
                    self._candidate_peak_fused = p
                    self._candidate_peak_audio = p_audio
                    self._candidate_peak_text = p_text
            else:
                if p > self._candidate_peak_fused:
                    self._candidate_peak_idx = idx
                    self._candidate_peak_fused = p
                    self._candidate_peak_audio = p_audio
                    self._candidate_peak_text = p_text

                if self._ready_to_confirm(idx):
                    ev = self._emit_event()
                    if ev is not None:
                        events.append(ev)
                    continue

                # hysteresis: when the curve falls under thr_off, commit peak early.
                if p < float(self.cfg.thr_off):
                    ev = self._emit_event()
                    if ev is not None:
                        events.append(ev)
                    else:
                        self._reset_candidate()

        return ChunkResult(
            frame_start=int(frame_start),
            frame_end=int(frame_start + n - 1 if n > 0 else frame_start),
            fused_probs=fused_probs,
            text_probs=text_probs,
            events=events,
            pending_peak=self._candidate_peak_idx,
        )
