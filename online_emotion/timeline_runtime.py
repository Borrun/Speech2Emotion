from collections import Counter
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Sequence, Tuple

from .detector import OnlineBoundaryDetector
from .runtime_utils import (
    Label,
    StreamEmotionState,
    align_text_to_frames,
    chunk_text_and_features,
)
from .schema import DetectorConfig
from .text_emotion import TextEmotionConstraint
from .text_prior import TextPriorBuilder


class FrameStatus(IntEnum):
    UNSEEN = 0
    UNREADY = 1
    PROVISIONAL = 2
    COMMITTED = 3


class EventStatus(IntEnum):
    PROVISIONAL = 0
    COMMITTED = 1


def samples_to_ready_frame_count(total_samples: int, n_fft: int, hop_length: int) -> int:
    if int(total_samples) < int(n_fft):
        return 0
    return 1 + (int(total_samples) - int(n_fft)) // int(hop_length)


def samples_to_ready_last_frame(total_samples: int, n_fft: int, hop_length: int) -> int:
    return samples_to_ready_frame_count(total_samples, n_fft, hop_length) - 1


@dataclass
class FrameRecord:
    frame_idx: int = -1
    t_sec: float = 0.0
    status: FrameStatus = FrameStatus.UNSEEN
    audio_arrived: bool = False
    inferred: bool = False
    emotion_logits: Tuple[float, ...] = ()
    level_logits: Tuple[float, ...] = ()
    boundary_prob: float = 0.0
    emotion_id: int = -1
    level_id: int = -1
    boundary_flag: bool = False
    emotion_conf: float = 0.0
    level_conf: float = 0.0
    final_conf: float = 0.0
    segment_id: int = -1
    revision_id: int = 0
    infer_pass_id: int = 0
    source_begin_frame: int = -1
    source_end_frame: int = -1


@dataclass
class TimelineBoundaryEvent:
    boundary_id: int = -1
    frame_idx: int = -1
    t_sec: float = 0.0
    confidence: float = 0.0
    status: EventStatus = EventStatus.PROVISIONAL
    left_segment_id: int = -1
    right_segment_id: int = -1
    revision_id: int = 0
    p_audio: float = 0.0
    p_text: float = 0.0
    p_fused: float = 0.0


@dataclass
class SegmentState:
    segment_id: int = -1
    start_frame: int = -1
    end_frame: int = -1
    end_closed: bool = False
    status: EventStatus = EventStatus.PROVISIONAL
    major_emotion_id: int = -1
    major_level_id: int = -1
    emotion_hist: Dict[int, int] = field(default_factory=dict)
    level_hist: Dict[int, int] = field(default_factory=dict)
    frame_count: int = 0
    confidence: float = 0.0


@dataclass
class PlaybackEmotionView:
    frame_idx: int = -1
    status: FrameStatus = FrameStatus.UNSEEN
    emotion_id: int = -1
    level_id: int = -1
    confidence: float = 0.0
    segment_id: int = -1
    is_boundary: bool = False
    source: str = "default"


@dataclass
class RhythmFrameView:
    frame_idx: int = -1
    status: FrameStatus = FrameStatus.UNSEEN
    emotion_id: int = -1
    level_id: int = -1
    boundary_flag: bool = False
    boundary_strength: float = 0.0
    source: str = "default"


@dataclass
class ControlStateView:
    frame_idx: int = -1
    status: FrameStatus = FrameStatus.UNSEEN
    emotion_id: int = -1
    level_id: int = -1
    confidence: float = 0.0
    segment_id: int = -1
    segment_major_emotion: int = -1
    segment_major_level: int = -1
    segment_progress: float = 0.0
    is_boundary_on_this_frame: bool = False
    source: str = "default"


@dataclass
class AcousticFrame:
    frame_idx: int
    emotion_id: int
    level_id: int
    boundary_prob: float
    emotion_logits: Tuple[float, ...] = ()
    level_logits: Tuple[float, ...] = ()
    emotion_conf: float = 0.0
    level_conf: float = 0.0


@dataclass
class AcousticSlice:
    frame_ctx_begin: int
    frame_begin: int
    frame_end: int
    total_frames: int
    type_map: Tuple[str, ...] = ()
    frames: List[AcousticFrame] = field(default_factory=list)


@dataclass
class FusedFrame:
    frame_idx: int
    emotion_id: int
    level_id: int
    boundary_prob: float
    boundary_flag: bool
    segment_id: int
    emotion_logits: Tuple[float, ...] = ()
    level_logits: Tuple[float, ...] = ()
    emotion_conf: float = 0.0
    level_conf: float = 0.0
    final_conf: float = 0.0


@dataclass
class FusedSlice:
    frame_begin: int
    frame_end: int
    frames: List[FusedFrame] = field(default_factory=list)
    boundary_events: List[TimelineBoundaryEvent] = field(default_factory=list)
    segments: List[SegmentState] = field(default_factory=list)


@dataclass
class TimelineBuffer:
    frames: List[FrameRecord] = field(default_factory=list)
    base_frame_idx: int = 0
    ingest_end: int = -1
    inferred_end: int = -1
    committed_end: int = -1

    def has_frame(self, frame_idx: int) -> bool:
        if not self.frames:
            return False
        return self.base_frame_idx <= int(frame_idx) < self.base_frame_idx + len(self.frames)

    def ensure_until(self, frame_idx: int, fps: int) -> None:
        frame_idx = int(frame_idx)
        if frame_idx < 0:
            return
        if not self.frames:
            self.base_frame_idx = 0
        needed = frame_idx - (self.base_frame_idx + len(self.frames) - 1)
        for _ in range(max(0, needed)):
            cur_idx = self.base_frame_idx + len(self.frames)
            self.frames.append(FrameRecord(frame_idx=cur_idx, t_sec=cur_idx / float(max(1, fps))))

    def at(self, frame_idx: int) -> FrameRecord:
        if not self.has_frame(frame_idx):
            raise IndexError("frame out of range: %s" % frame_idx)
        return self.frames[int(frame_idx) - self.base_frame_idx]


@dataclass
class AudioIngestTracker:
    sample_rate: int = 16000
    hop_length: int = 533
    n_fft: int = 800
    total_samples_received: int = 0

    def update(self, n_samples: int) -> Tuple[int, List[int]]:
        old_last = samples_to_ready_last_frame(self.total_samples_received, self.n_fft, self.hop_length)
        self.total_samples_received += int(max(0, n_samples))
        new_last = samples_to_ready_last_frame(self.total_samples_received, self.n_fft, self.hop_length)
        if new_last <= old_last:
            return new_last, []
        return new_last, list(range(old_last + 1, new_last + 1))


class PredJsonAcousticAdapter:
    def __init__(self, pred_obj: Dict):
        self.pred_obj = pred_obj
        self.frames = list(pred_obj.get("frames", []))
        self.total_frames = len(self.frames)
        self.type_map = tuple(pred_obj.get("type_map", []) or [])
        self.sample_rate = int(pred_obj.get("sample_rate", 16000))
        self.fps = int(pred_obj.get("fps", 30))

    @classmethod
    def from_path(cls, path: str):
        import json

        with open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f))

    def infer_window(self, frame_ctx_begin: int, frame_write_begin: int, frame_write_end: int) -> AcousticSlice:
        write_begin = int(max(0, frame_write_begin))
        write_end = int(min(frame_write_end, self.total_frames - 1))
        if write_begin > write_end:
            return AcousticSlice(
                frame_ctx_begin=int(frame_ctx_begin),
                frame_begin=write_begin,
                frame_end=write_begin - 1,
                total_frames=self.total_frames,
                type_map=self.type_map,
                frames=[],
            )

        frames = []
        for idx in range(write_begin, write_end + 1):
            fr = self.frames[idx]
            frames.append(
                AcousticFrame(
                    frame_idx=idx,
                    emotion_id=int(fr.get("type_id", -1)),
                    level_id=int(fr.get("level_id", -1)),
                    boundary_prob=float(fr.get("boundary_p", 0.0)),
                    emotion_logits=tuple(float(x) for x in fr.get("type_logits", []) or []),
                    level_logits=tuple(float(x) for x in fr.get("level_logits", []) or []),
                    emotion_conf=float(fr.get("emotion_conf", 1.0 if "type_id" in fr else 0.0)),
                    level_conf=float(fr.get("level_conf", 1.0 if "level_id" in fr else 0.0)),
                )
            )

        return AcousticSlice(
            frame_ctx_begin=int(frame_ctx_begin),
            frame_begin=write_begin,
            frame_end=write_end,
            total_frames=self.total_frames,
            type_map=self.type_map,
            frames=frames,
        )


@dataclass
class TimelineRuntimeConfig:
    sample_rate: int = 16000
    fps: int = 30
    n_fft: int = 800
    hop_length: int = 533
    infer_tick_ms: int = 300
    recompute_left_frames: int = 120
    stable_right_frames: int = 42
    history_keep_frames: int = 900
    revise_keep_frames: int = 300
    smooth_win: int = 5
    emo_hysteresis: int = 3
    dynamic_w_text: bool = False
    w_text_max: float = 0.5
    text_emotion: bool = False
    sentiment_thr: float = 0.3
    text_conf_thr: float = 0.50
    text_blend_w: float = 0.35
    detector_cfg: DetectorConfig = field(default_factory=DetectorConfig)


class SimpleFusionRefineAdapter:
    def __init__(self, text: str, cfg: TimelineRuntimeConfig, n_types: int = 6):
        self.text = str(text or "")
        self.cfg = cfg
        self.n_types = int(n_types)
        self._builder = TextPriorBuilder(fps=int(cfg.fps))
        self._text_constraint = (
            TextEmotionConstraint(
                sentiment_thr=float(cfg.sentiment_thr),
                text_conf_thr=float(cfg.text_conf_thr),
                text_blend_w=float(cfg.text_blend_w),
                n_types=int(self.n_types),
            )
            if bool(cfg.text_emotion)
            else None
        )

    @staticmethod
    def _copy_detector_cfg(cfg: DetectorConfig) -> DetectorConfig:
        return DetectorConfig(
            fps=int(cfg.fps),
            w_audio=float(cfg.w_audio),
            w_text=float(cfg.w_text),
            thr_on=float(cfg.thr_on),
            thr_off=float(cfg.thr_off),
            confirm_win=int(cfg.confirm_win),
            min_gap=int(cfg.min_gap),
            adapt_lr_offset=float(cfg.adapt_lr_offset),
            adapt_lr_scale=float(cfg.adapt_lr_scale),
        )

    @staticmethod
    def _major_id(counter: Counter) -> int:
        if not counter:
            return -1
        return int(counter.most_common(1)[0][0])

    @staticmethod
    def _segment_conf(counter: Counter, n: int) -> float:
        if not counter or n <= 0:
            return 0.0
        return float(counter.most_common(1)[0][1]) / float(max(1, n))

    def _seed_label(self, committed_segments: Sequence[SegmentState]) -> Optional[Label]:
        if not committed_segments:
            return None
        last = committed_segments[-1]
        if last.major_emotion_id < 0 or last.major_level_id < 0:
            return None
        return int(last.major_emotion_id), int(last.major_level_id)

    def _starting_segment_id(self, committed_segments: Sequence[SegmentState]) -> int:
        if not committed_segments:
            return 0
        last = committed_segments[-1]
        if not last.end_closed:
            return int(last.segment_id)
        return int(last.segment_id) + 1

    def _build_detector(self, total_frames: int, committed_boundary_events: Sequence[TimelineBoundaryEvent]) -> OnlineBoundaryDetector:
        total_sec_hint = float(max(1, total_frames)) / float(max(1, self.cfg.fps))
        prior = self._builder.build(
            text=self.text,
            token_timing=None,
            total_sec_hint=total_sec_hint,
        )
        for ev in committed_boundary_events:
            prior.register_boundary(
                frame_idx=int(ev.frame_idx),
                lr_offset=float(self.cfg.detector_cfg.adapt_lr_offset),
                lr_scale=float(self.cfg.detector_cfg.adapt_lr_scale),
            )
        det = OnlineBoundaryDetector(text_prior=prior, cfg=self._copy_detector_cfg(self.cfg.detector_cfg))
        if committed_boundary_events:
            det.last_confirmed = int(committed_boundary_events[-1].frame_idx)
        return det

    def _build_segments(
        self,
        frames: Sequence[AcousticFrame],
        labels: Sequence[Label],
        provisional_events: Sequence[TimelineBoundaryEvent],
        committed_segments: Sequence[SegmentState],
    ) -> Tuple[Dict[int, int], List[SegmentState], Dict[int, Tuple[int, int]]]:
        frame_to_segment: Dict[int, int] = {}
        boundary_frames = set(int(ev.frame_idx) for ev in provisional_events)
        segments: List[SegmentState] = []
        boundary_lr: Dict[int, Tuple[int, int]] = {}

        if not frames:
            return frame_to_segment, segments, boundary_lr

        current_segment_id = self._starting_segment_id(committed_segments)
        current_start = int(frames[0].frame_idx)
        current_records: List[Tuple[int, Label]] = []

        def finalize(end_frame: int, end_closed: bool) -> SegmentState:
            emo_counter = Counter(int(lb[0]) for _, lb in current_records if lb[0] >= 0)
            lvl_counter = Counter(int(lb[1]) for _, lb in current_records if lb[1] >= 0)
            return SegmentState(
                segment_id=int(current_segment_id),
                start_frame=int(current_start),
                end_frame=int(end_frame),
                end_closed=bool(end_closed),
                status=EventStatus.PROVISIONAL,
                major_emotion_id=self._major_id(emo_counter),
                major_level_id=self._major_id(lvl_counter),
                emotion_hist=dict(emo_counter),
                level_hist=dict(lvl_counter),
                frame_count=len(current_records),
                confidence=self._segment_conf(emo_counter, len(current_records)),
            )

        for frame, label in zip(frames, labels):
            idx = int(frame.frame_idx)
            if idx in boundary_frames and idx > current_start and current_records:
                segments.append(finalize(idx - 1, True))
                boundary_lr[idx] = (int(current_segment_id), int(current_segment_id + 1))
                current_segment_id += 1
                current_start = idx
                current_records = []
            current_records.append((idx, label))
            frame_to_segment[idx] = int(current_segment_id)

        if current_records:
            segments.append(finalize(int(frames[-1].frame_idx), False))

        return frame_to_segment, segments, boundary_lr

    def refine_window(
        self,
        acoustic: AcousticSlice,
        committed_boundary_events: Sequence[TimelineBoundaryEvent],
        committed_segments: Sequence[SegmentState],
    ) -> FusedSlice:
        if not acoustic.frames:
            return FusedSlice(frame_begin=acoustic.frame_begin, frame_end=acoustic.frame_end)

        det = self._build_detector(acoustic.total_frames, committed_boundary_events)
        committed_anchor_frames = [int(ev.frame_idx) for ev in committed_boundary_events]
        tokens_pre = align_text_to_frames(self.text, max(1, acoustic.total_frames), committed_anchor_frames)
        chunk_tinfo = chunk_text_and_features(tokens_pre, acoustic.frame_begin, acoustic.frame_end + 1)

        if bool(self.cfg.dynamic_w_text):
            base_w_text = float(self.cfg.detector_cfg.w_text)
            sa = abs(float(chunk_tinfo["text_features"]["sentiment_score"]))
            dw = base_w_text + (float(self.cfg.w_text_max) - base_w_text) * min(1.0, sa / max(0.3, base_w_text))
            dw = max(base_w_text, min(float(self.cfg.w_text_max), dw))
            det.set_weights(1.0 - dw, dw)

        out = det.process_chunk(
            frame_start=int(acoustic.frame_begin),
            p_audio_chunk=[float(fr.boundary_prob) for fr in acoustic.frames],
        )

        provisional_events = [
            TimelineBoundaryEvent(
                boundary_id=i,
                frame_idx=int(ev.frame_idx),
                t_sec=float(ev.t_sec),
                confidence=float(ev.p_fused),
                status=EventStatus.PROVISIONAL,
                revision_id=0,
                p_audio=float(ev.p_audio),
                p_text=float(ev.p_text),
                p_fused=float(ev.p_fused),
            )
            for i, ev in enumerate(out.events)
        ]

        all_anchor_frames = committed_anchor_frames + [int(ev.frame_idx) for ev in provisional_events]
        tokens = align_text_to_frames(self.text, max(1, acoustic.total_frames), all_anchor_frames)

        emo_state = StreamEmotionState(
            smooth_win=int(self.cfg.smooth_win),
            hysteresis=int(self.cfg.emo_hysteresis),
            initial=self._seed_label(committed_segments),
        )

        refined_labels: List[Label] = []
        for fr in acoustic.frames:
            raw = (int(fr.emotion_id), int(fr.level_id))
            stable = emo_state.update(raw)
            refined_labels.append(stable if stable is not None else raw)

        if self._text_constraint is not None:
            segment_starts = [int(acoustic.frame_begin)]
            for ev in provisional_events:
                if acoustic.frame_begin < int(ev.frame_idx) <= acoustic.frame_end:
                    segment_starts.append(int(ev.frame_idx))
            segment_starts = sorted(set(segment_starts))
            for i, s in enumerate(segment_starts):
                e = segment_starts[i + 1] if i + 1 < len(segment_starts) else acoustic.frame_end + 1
                tinfo = chunk_text_and_features(tokens, s, e)
                for frame_idx in range(s, e):
                    local_idx = frame_idx - acoustic.frame_begin
                    if not (0 <= local_idx < len(refined_labels)):
                        continue
                    lb = refined_labels[local_idx]
                    refined_labels[local_idx] = self._text_constraint.apply(
                        type_id=int(lb[0]),
                        level_id=int(lb[1]),
                        chunk_text=str(tinfo["chunk_text"]),
                        text_features=dict(tinfo["text_features"]),
                    )

        frame_to_segment, segments, boundary_lr = self._build_segments(
            acoustic.frames,
            refined_labels,
            provisional_events,
            committed_segments,
        )

        for ev in provisional_events:
            if int(ev.frame_idx) in boundary_lr:
                ev.left_segment_id, ev.right_segment_id = boundary_lr[int(ev.frame_idx)]

        boundary_set = set(int(ev.frame_idx) for ev in provisional_events)
        fused_frames = []
        for fr, label in zip(acoustic.frames, refined_labels):
            fused_frames.append(
                FusedFrame(
                    frame_idx=int(fr.frame_idx),
                    emotion_id=int(label[0]),
                    level_id=int(label[1]),
                    boundary_prob=float(fr.boundary_prob),
                    boundary_flag=int(fr.frame_idx) in boundary_set,
                    segment_id=int(frame_to_segment.get(int(fr.frame_idx), -1)),
                    emotion_logits=tuple(fr.emotion_logits),
                    level_logits=tuple(fr.level_logits),
                    emotion_conf=float(fr.emotion_conf),
                    level_conf=float(fr.level_conf),
                    final_conf=max(float(fr.emotion_conf), float(fr.level_conf)),
                )
            )

        return FusedSlice(
            frame_begin=int(acoustic.frame_begin),
            frame_end=int(acoustic.frame_end),
            frames=fused_frames,
            boundary_events=provisional_events,
            segments=segments,
        )


class TimelineRuntime:
    def __init__(
        self,
        acoustic_adapter,
        text: str = "",
        cfg: Optional[TimelineRuntimeConfig] = None,
        fusion_adapter: Optional[SimpleFusionRefineAdapter] = None,
    ):
        self.cfg = cfg or TimelineRuntimeConfig()
        self.acoustic_adapter = acoustic_adapter
        self.text = str(text or "")
        self.timeline = TimelineBuffer()
        self.boundary_events: List[TimelineBoundaryEvent] = []
        self.segments: List[SegmentState] = []
        self.ingest = AudioIngestTracker(
            sample_rate=int(self.cfg.sample_rate),
            hop_length=int(self.cfg.hop_length),
            n_fft=int(self.cfg.n_fft),
        )
        adapter_type_map = tuple(getattr(acoustic_adapter, "type_map", ()) or ())
        adapter_n_types = int(getattr(acoustic_adapter, "n_types", 0) or 0)
        if not adapter_n_types and adapter_type_map:
            adapter_n_types = len(adapter_type_map)
        if adapter_n_types <= 0:
            adapter_n_types = 6
        self.fusion_adapter = fusion_adapter or SimpleFusionRefineAdapter(
            self.text,
            self.cfg,
            n_types=adapter_n_types,
        )
        self.infer_pass_id = 0
        self.end_of_stream = False

    def update_ingest_samples(self, n_samples: int, recv_ts_sec: float = 0.0) -> List[int]:
        del recv_ts_sec
        new_ingest_end, new_frames = self.ingest.update(int(n_samples))
        if new_frames:
            self.timeline.ensure_until(int(new_ingest_end), int(self.cfg.fps))
            for frame_idx in new_frames:
                rec = self.timeline.at(frame_idx)
                rec.audio_arrived = True
                rec.status = FrameStatus.UNREADY
            self.timeline.ingest_end = int(new_ingest_end)
        return new_frames

    def update_ingest_pcm(self, pcm_chunk, recv_ts_sec: float = 0.0, sample_rate: Optional[int] = None) -> List[int]:
        append = getattr(self.acoustic_adapter, "append_wav_chunk", None)
        if not callable(append):
            raise TypeError("acoustic_adapter does not support PCM chunk ingest")
        added_samples = int(append(pcm_chunk, sample_rate=sample_rate or self.cfg.sample_rate))
        if added_samples <= 0:
            return []
        return self.update_ingest_samples(added_samples, recv_ts_sec=recv_ts_sec)

    def mark_end_of_stream(self) -> None:
        mark = getattr(self.acoustic_adapter, "mark_end_of_stream", None)
        if callable(mark):
            mark()
        self.end_of_stream = True

    def _committed_boundary_events(self) -> List[TimelineBoundaryEvent]:
        return [ev for ev in self.boundary_events if ev.status == EventStatus.COMMITTED]

    def _committed_segments(self) -> List[SegmentState]:
        return [seg for seg in self.segments if seg.status == EventStatus.COMMITTED]

    def run_infer_tick(self, now_sec: float = 0.0) -> Optional[FusedSlice]:
        del now_sec
        if self.timeline.ingest_end < 0:
            return None

        write_begin = max(self.timeline.base_frame_idx, self.timeline.committed_end + 1)
        ctx_begin = max(self.timeline.base_frame_idx, write_begin - int(self.cfg.recompute_left_frames))
        write_end = int(self.timeline.ingest_end)
        if write_begin > write_end:
            return None

        acoustic = self.acoustic_adapter.infer_window(
            frame_ctx_begin=int(ctx_begin),
            frame_write_begin=int(write_begin),
            frame_write_end=int(write_end),
        )
        if not acoustic.frames:
            return None

        fused = self.fusion_adapter.refine_window(
            acoustic=acoustic,
            committed_boundary_events=self._committed_boundary_events(),
            committed_segments=self._committed_segments(),
        )

        self.timeline.ensure_until(int(fused.frame_end), int(self.cfg.fps))
        for fr in fused.frames:
            rec = self.timeline.at(int(fr.frame_idx))
            rec.inferred = True
            rec.status = FrameStatus.PROVISIONAL
            rec.emotion_logits = tuple(fr.emotion_logits)
            rec.level_logits = tuple(fr.level_logits)
            rec.boundary_prob = float(fr.boundary_prob)
            rec.emotion_id = int(fr.emotion_id)
            rec.level_id = int(fr.level_id)
            rec.boundary_flag = bool(fr.boundary_flag)
            rec.segment_id = int(fr.segment_id)
            rec.emotion_conf = float(fr.emotion_conf)
            rec.level_conf = float(fr.level_conf)
            rec.final_conf = float(fr.final_conf)
            rec.revision_id += 1
            rec.infer_pass_id = int(self.infer_pass_id)
            rec.source_begin_frame = int(acoustic.frame_ctx_begin)
            rec.source_end_frame = int(acoustic.frame_end)

        self.boundary_events = self._committed_boundary_events() + list(fused.boundary_events)
        self.timeline.inferred_end = max(int(self.timeline.inferred_end), int(fused.frame_end))
        self.infer_pass_id += 1
        self._refresh_boundary_flags(start_frame=write_begin)
        self._rebuild_segments()
        return fused

    def advance_commit_line(self, force_flush: bool = False) -> int:
        if bool(force_flush) or bool(self.end_of_stream):
            new_commit_end = int(self.timeline.inferred_end)
        else:
            new_commit_end = min(
                int(self.timeline.inferred_end),
                int(self.timeline.ingest_end) - int(self.cfg.stable_right_frames),
            )
        if new_commit_end <= int(self.timeline.committed_end):
            return int(self.timeline.committed_end)

        self.timeline.ensure_until(int(new_commit_end), int(self.cfg.fps))
        for frame_idx in range(int(self.timeline.committed_end) + 1, int(new_commit_end) + 1):
            rec = self.timeline.at(frame_idx)
            if rec.status == FrameStatus.PROVISIONAL:
                rec.status = FrameStatus.COMMITTED

        for ev in self.boundary_events:
            if ev.status == EventStatus.PROVISIONAL and int(ev.frame_idx) <= int(new_commit_end):
                ev.status = EventStatus.COMMITTED

        self.timeline.committed_end = int(new_commit_end)
        self._rebuild_segments()
        self._trim_history()
        return int(self.timeline.committed_end)

    def _refresh_boundary_flags(self, start_frame: int = 0) -> None:
        for rec in self.timeline.frames:
            if int(rec.frame_idx) >= int(start_frame):
                rec.boundary_flag = False
        for ev in self.boundary_events:
            if int(ev.frame_idx) < int(start_frame):
                continue
            if self.timeline.has_frame(int(ev.frame_idx)):
                self.timeline.at(int(ev.frame_idx)).boundary_flag = True

    def _rebuild_segments(self) -> None:
        inferred_frames = [rec for rec in self.timeline.frames if rec.inferred]
        if not inferred_frames:
            self.segments = []
            return

        inferred_frames.sort(key=lambda x: x.frame_idx)
        events_by_frame = {int(ev.frame_idx): ev for ev in self.boundary_events}
        boundary_frames = set(events_by_frame.keys())

        start_id = next((int(rec.segment_id) for rec in inferred_frames if rec.segment_id >= 0), 0)
        current_segment_id = int(start_id)
        current_start = int(inferred_frames[0].frame_idx)
        current_records: List[FrameRecord] = []
        segments: List[SegmentState] = []

        def finalize(end_frame: int, end_closed: bool) -> None:
            emo_counter = Counter(int(rec.emotion_id) for rec in current_records if rec.emotion_id >= 0)
            lvl_counter = Counter(int(rec.level_id) for rec in current_records if rec.level_id >= 0)
            status = (
                EventStatus.COMMITTED
                if bool(end_closed) and int(end_frame) <= int(self.timeline.committed_end)
                else EventStatus.PROVISIONAL
            )
            segments.append(
                SegmentState(
                    segment_id=int(current_segment_id),
                    start_frame=int(current_start),
                    end_frame=int(end_frame),
                    end_closed=bool(end_closed),
                    status=status,
                    major_emotion_id=int(emo_counter.most_common(1)[0][0]) if emo_counter else -1,
                    major_level_id=int(lvl_counter.most_common(1)[0][0]) if lvl_counter else -1,
                    emotion_hist=dict(emo_counter),
                    level_hist=dict(lvl_counter),
                    frame_count=len(current_records),
                    confidence=(
                        float(emo_counter.most_common(1)[0][1]) / float(max(1, len(current_records)))
                        if emo_counter
                        else 0.0
                    ),
                )
            )

        for rec in inferred_frames:
            idx = int(rec.frame_idx)
            if idx in boundary_frames and idx > current_start and current_records:
                finalize(idx - 1, True)
                current_segment_id += 1
                current_start = idx
                current_records = []
            rec.segment_id = int(current_segment_id)
            current_records.append(rec)

        if current_records:
            finalize(int(inferred_frames[-1].frame_idx), False)

        segment_by_start = {int(seg.start_frame): seg for seg in segments}
        segment_by_end = {int(seg.end_frame): seg for seg in segments}
        for ev in self.boundary_events:
            left = segment_by_end.get(int(ev.frame_idx) - 1)
            right = segment_by_start.get(int(ev.frame_idx))
            ev.left_segment_id = int(left.segment_id) if left is not None else -1
            ev.right_segment_id = int(right.segment_id) if right is not None else -1

        self.segments = segments

    def _trim_history(self) -> None:
        if not self.timeline.frames:
            return
        min_keep = int(self.timeline.committed_end) - int(self.cfg.history_keep_frames)
        if min_keep <= self.timeline.base_frame_idx:
            return
        trim_n = min_keep - self.timeline.base_frame_idx
        if trim_n <= 0:
            return
        self.timeline.frames = self.timeline.frames[trim_n:]
        self.timeline.base_frame_idx = min_keep
        self.boundary_events = [
            ev for ev in self.boundary_events
            if ev.status != EventStatus.COMMITTED or int(ev.frame_idx) >= int(self.timeline.base_frame_idx)
        ]
        self.segments = [
            seg for seg in self.segments
            if seg.status != EventStatus.COMMITTED or int(seg.end_frame) >= int(self.timeline.base_frame_idx)
        ]

    def _get_frame(self, frame_idx: int) -> Optional[FrameRecord]:
        if not self.timeline.has_frame(frame_idx):
            return None
        return self.timeline.at(frame_idx)

    def _latest_committed_before(self, frame_idx: int) -> Optional[FrameRecord]:
        for idx in range(min(int(frame_idx), int(self.timeline.committed_end)), self.timeline.base_frame_idx - 1, -1):
            rec = self._get_frame(idx)
            if rec is not None and rec.status == FrameStatus.COMMITTED:
                return rec
        return None

    def _segment_for_frame(self, frame_idx: int, committed_only: bool = False) -> Optional[SegmentState]:
        for seg in self.segments:
            if committed_only and seg.status != EventStatus.COMMITTED:
                continue
            if int(seg.start_frame) <= int(frame_idx) <= int(seg.end_frame):
                return seg
        return None

    def get_playback_view(self, play_frame_idx: int) -> PlaybackEmotionView:
        rec = self._get_frame(play_frame_idx)
        if rec is not None and rec.status == FrameStatus.COMMITTED:
            return PlaybackEmotionView(
                frame_idx=int(play_frame_idx),
                status=rec.status,
                emotion_id=int(rec.emotion_id),
                level_id=int(rec.level_id),
                confidence=float(rec.final_conf),
                segment_id=int(rec.segment_id),
                is_boundary=bool(rec.boundary_flag),
                source="committed",
            )
        if rec is not None and rec.status == FrameStatus.PROVISIONAL:
            return PlaybackEmotionView(
                frame_idx=int(play_frame_idx),
                status=rec.status,
                emotion_id=int(rec.emotion_id),
                level_id=int(rec.level_id),
                confidence=float(rec.final_conf),
                segment_id=int(rec.segment_id),
                is_boundary=bool(rec.boundary_flag),
                source="provisional",
            )
        hist = self._latest_committed_before(play_frame_idx)
        if hist is not None:
            return PlaybackEmotionView(
                frame_idx=int(play_frame_idx),
                status=hist.status,
                emotion_id=int(hist.emotion_id),
                level_id=int(hist.level_id),
                confidence=float(hist.final_conf),
                segment_id=int(hist.segment_id),
                is_boundary=False,
                source="history",
            )
        return PlaybackEmotionView(frame_idx=int(play_frame_idx), emotion_id=4, level_id=0, source="default")

    def get_rhythm_view(self, frame_idx: int) -> RhythmFrameView:
        rec = self._get_frame(frame_idx)
        if rec is not None and rec.status == FrameStatus.COMMITTED:
            return RhythmFrameView(
                frame_idx=int(frame_idx),
                status=rec.status,
                emotion_id=int(rec.emotion_id),
                level_id=int(rec.level_id),
                boundary_flag=bool(rec.boundary_flag),
                boundary_strength=float(rec.boundary_prob),
                source="committed",
            )
        hist = self._latest_committed_before(frame_idx)
        if hist is not None:
            return RhythmFrameView(
                frame_idx=int(frame_idx),
                status=hist.status,
                emotion_id=int(hist.emotion_id),
                level_id=int(hist.level_id),
                boundary_flag=False,
                boundary_strength=0.0,
                source="history",
            )
        return RhythmFrameView(frame_idx=int(frame_idx), emotion_id=4, level_id=0, source="default")

    def get_control_view(self, frame_idx: int) -> ControlStateView:
        rec = self._get_frame(frame_idx)
        if rec is not None and rec.status == FrameStatus.COMMITTED:
            seg = self._segment_for_frame(frame_idx, committed_only=True)
            return self._build_control_view(frame_idx, rec, seg, "committed")

        seg = self._segment_for_frame(frame_idx, committed_only=True)
        if seg is not None:
            hist = self._latest_committed_before(frame_idx)
            return self._build_control_view(frame_idx, hist, seg, "segment") if hist is not None else ControlStateView(
                frame_idx=int(frame_idx),
                emotion_id=seg.major_emotion_id,
                level_id=seg.major_level_id,
                confidence=float(seg.confidence),
                segment_id=int(seg.segment_id),
                segment_major_emotion=int(seg.major_emotion_id),
                segment_major_level=int(seg.major_level_id),
                segment_progress=0.0,
                source="segment",
            )

        hist = self._latest_committed_before(frame_idx)
        if hist is not None:
            prev_seg = self._segment_for_frame(hist.frame_idx, committed_only=True)
            return self._build_control_view(frame_idx, hist, prev_seg, "history")

        return ControlStateView(frame_idx=int(frame_idx), emotion_id=4, level_id=0, source="default")

    def _build_control_view(
        self,
        frame_idx: int,
        rec: Optional[FrameRecord],
        seg: Optional[SegmentState],
        source: str,
    ) -> ControlStateView:
        progress = 0.0
        if seg is not None and seg.end_closed and seg.end_frame >= seg.start_frame:
            progress = float(max(0, frame_idx - seg.start_frame)) / float(max(1, seg.end_frame - seg.start_frame + 1))
            progress = max(0.0, min(1.0, progress))
        return ControlStateView(
            frame_idx=int(frame_idx),
            status=rec.status if rec is not None else FrameStatus.UNSEEN,
            emotion_id=int(rec.emotion_id) if rec is not None else (seg.major_emotion_id if seg is not None else 4),
            level_id=int(rec.level_id) if rec is not None else (seg.major_level_id if seg is not None else 0),
            confidence=float(rec.final_conf) if rec is not None else (float(seg.confidence) if seg is not None else 0.0),
            segment_id=int(seg.segment_id) if seg is not None else (int(rec.segment_id) if rec is not None else -1),
            segment_major_emotion=int(seg.major_emotion_id) if seg is not None else -1,
            segment_major_level=int(seg.major_level_id) if seg is not None else -1,
            segment_progress=float(progress),
            is_boundary_on_this_frame=bool(rec.boundary_flag) if rec is not None else False,
            source=str(source),
        )

    def summary(self) -> Dict:
        status_counts = Counter(int(rec.status) for rec in self.timeline.frames)
        return {
            "base_frame_idx": int(self.timeline.base_frame_idx),
            "ingest_end": int(self.timeline.ingest_end),
            "inferred_end": int(self.timeline.inferred_end),
            "committed_end": int(self.timeline.committed_end),
            "end_of_stream": bool(self.end_of_stream),
            "num_frames": len(self.timeline.frames),
            "num_boundary_events": len(self.boundary_events),
            "num_segments": len(self.segments),
            "status_counts": dict(status_counts),
        }
