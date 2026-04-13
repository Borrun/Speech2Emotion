from collections import deque
from typing import List, Dict, Any, Tuple, Optional


LEVEL_TO_VALUE = [5, 18, 38, 68, 98, 130]

CPP_SYNC_DEFAULTS = {
    "thr_on": 0.12,
    "thr_off": 0.06,
    "confirm_win": 3,
    "min_gap": 5,
    "smooth_win": 3,
    "hysteresis": 2,
}


def apply_boundary_hold(frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Optional postprocess: hold previous state unless boundary==1.
    If your inference doesn't output boundary, this is a no-op.
    """
    if not frames:
        return frames

    out = []
    cur_type = frames[0]["type_id"]
    cur_lvl = frames[0]["level_id"]

    for f in frames:
        b = f.get("boundary", None)
        if b is None:
            out.append(f)
            continue
        if int(b) == 1:
            cur_type = f["type_id"]
            cur_lvl = f["level_id"]
        nf = dict(f)
        nf["type_id"] = cur_type
        nf["level_id"] = cur_lvl
        out.append(nf)
    return out


def decode_switch_points(
    boundary_p: List[float],
    thr_on: float = 0.78,
    thr_off: float = 0.60,
    confirm_win: int = 3,
    min_gap: int = 5,
) -> List[int]:
    """
    Offline decoding from per-frame boundary probability to switch frame indices.
    """
    events: List[int] = []
    peak_idx = None
    peak_p = -1.0
    last_event = -10**9

    for t, p in enumerate(boundary_p):
        p = float(p)

        if peak_idx is None:
            if p >= float(thr_on) and (t - last_event) >= int(min_gap):
                peak_idx = t
                peak_p = p
            continue

        if p > peak_p:
            peak_idx = t
            peak_p = p

        # confirm by lookahead window
        if t >= peak_idx + int(confirm_win):
            if (peak_idx - last_event) >= int(min_gap):
                events.append(int(peak_idx))
                last_event = int(peak_idx)
            peak_idx = None
            peak_p = -1.0
            continue

        # early confirm on clear fall
        if p < float(thr_off):
            if (peak_idx - last_event) >= int(min_gap):
                events.append(int(peak_idx))
                last_event = int(peak_idx)
            peak_idx = None
            peak_p = -1.0

    return events


class StreamingSwitchDecoder:
    """
    Stateful streaming decoder for chunked boundary probabilities.

    Example:
      dec = StreamingSwitchDecoder()
      events = dec.process_chunk(frame_start=120, boundary_p_chunk=[...])
    """
    def __init__(
        self,
        thr_on: float = 0.78,
        thr_off: float = 0.60,
        confirm_win: int = 3,
        min_gap: int = 5,
    ):
        self.thr_on = float(thr_on)
        self.thr_off = float(thr_off)
        self.confirm_win = int(confirm_win)
        self.min_gap = int(min_gap)

        self.peak_idx = None
        self.peak_p = -1.0
        self.last_event = -10**9

    def process_chunk(self, frame_start: int, boundary_p_chunk: List[float]) -> List[int]:
        """
        Returns absolute frame indices of confirmed switch points in this chunk.
        """
        events: List[int] = []
        for i, p_raw in enumerate(boundary_p_chunk):
            t = int(frame_start) + i
            p = float(p_raw)

            if self.peak_idx is None:
                if p >= self.thr_on and (t - self.last_event) >= self.min_gap:
                    self.peak_idx = t
                    self.peak_p = p
                continue

            if p > self.peak_p:
                self.peak_idx = t
                self.peak_p = p

            if t >= self.peak_idx + self.confirm_win:
                if (self.peak_idx - self.last_event) >= self.min_gap:
                    events.append(int(self.peak_idx))
                    self.last_event = int(self.peak_idx)
                self.peak_idx = None
                self.peak_p = -1.0
                continue

            if p < self.thr_off:
                if (self.peak_idx - self.last_event) >= self.min_gap:
                    events.append(int(self.peak_idx))
                    self.last_event = int(self.peak_idx)
                self.peak_idx = None
                self.peak_p = -1.0

        return events


def _majority_vote_cpp_style(window: List[Tuple[int, int]], fallback: Tuple[int, int]) -> Tuple[int, int]:
    if not window:
        return fallback

    counts: List[Dict[str, Any]] = []
    for label in window:
        found = False
        for row in counts:
            if row["label"] == label:
                row["count"] += 1
                found = True
                break
        if not found:
            counts.append({"label": label, "count": 1})

    best_n = -1
    best = fallback
    for row in counts:
        if int(row["count"]) > best_n:
            best_n = int(row["count"])
            best = row["label"]
    return best


def _argmax_index(xs: List[int], fallback: int = 0) -> int:
    if not xs:
        return int(fallback)
    best_idx = int(fallback)
    best_val = None
    for i, v in enumerate(xs):
        if best_val is None or int(v) > int(best_val):
            best_val = int(v)
            best_idx = int(i)
    return best_idx


def _label_value(level_id: int) -> int:
    lid = max(0, min(len(LEVEL_TO_VALUE) - 1, int(level_id)))
    return int(LEVEL_TO_VALUE[lid])


def _segments_from_timeline(timeline: List[Dict[str, Any]], fps: int, type_map: List[str]) -> List[Dict[str, Any]]:
    if not timeline:
        return []

    segs: List[Dict[str, Any]] = []
    start = 0
    cur_type = int(timeline[0].get("type_id", 0))
    cur_level = int(timeline[0].get("level_id", 0))

    def emit(end_exclusive: int):
        if end_exclusive <= start:
            return
        segs.append({
            "f0": int(start),
            "f1": int(end_exclusive),
            "t0": float(start) / float(max(1, fps)),
            "t1": float(end_exclusive) / float(max(1, fps)),
            "type_id": int(cur_type),
            "level_id": int(cur_level),
            "type": type_map[cur_type] if 0 <= cur_type < len(type_map) else str(cur_type),
            "value": _label_value(cur_level),
        })

    for i in range(1, len(timeline)):
        ty = int(timeline[i].get("type_id", 0))
        lv = int(timeline[i].get("level_id", 0))
        if ty != cur_type or lv != cur_level:
            emit(i)
            start = i
            cur_type = ty
            cur_level = lv
    emit(len(timeline))
    return segs


def apply_cpp_emotion_sync(
    frames: List[Dict[str, Any]],
    fps: int = 30,
    type_map: Optional[List[str]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Reproduce the cpp_emotion runtime smoothing pipeline:
      1. boundary hysteresis/confirm window
      2. majority-vote smoothing over (type_id, level_id)
      3. segment-majority commit + back-fill
    """
    if not frames:
        cfg = dict(CPP_SYNC_DEFAULTS)
        if config:
            cfg.update(config)
        return {"config": cfg, "frames": [], "boundary_frames": [], "segments": []}

    cfg = dict(CPP_SYNC_DEFAULTS)
    if config:
        cfg.update(config)

    n_types = 0
    for fr in frames:
        n_types = max(n_types, int(fr.get("type_id", -1)) + 1)
    type_map = list(type_map or [])
    if len(type_map) < n_types:
        type_map = list(type_map) + [f"type_{i}" for i in range(len(type_map), n_types)]

    thr_on = float(cfg["thr_on"])
    thr_off = float(cfg["thr_off"])
    confirm_win = int(cfg["confirm_win"])
    min_gap = int(cfg["min_gap"])
    smooth_win = max(1, int(cfg["smooth_win"]))
    hysteresis = max(1, int(cfg["hysteresis"]))

    last_confirmed_frame = -10000
    candidate_frame = -1
    candidate_peak = -1.0
    frames_since_on = 0

    window = deque(maxlen=smooth_win)
    current = (0, 0)
    candidate_label = (0, 0)
    cand_count = 0

    interval_type_cnt = [0] * max(1, len(type_map))
    interval_level_cnt = [0] * len(LEVEL_TO_VALUE)
    interval_start_idx = 0
    committed_type = -1
    committed_level = -1

    timeline: List[Dict[str, Any]] = []
    boundary_frames: List[int] = []

    for f, pred in enumerate(frames):
        boundary_triggered = False
        p = float(pred.get("boundary_p", 0.0))

        if candidate_frame < 0:
            if p >= thr_on:
                candidate_frame = f
                candidate_peak = p
                frames_since_on = 1
        else:
            if p > candidate_peak:
                candidate_frame = f
                candidate_peak = p
            frames_since_on += 1

            confirm = (frames_since_on >= confirm_win) or (p < thr_off)
            if confirm:
                if (candidate_frame - last_confirmed_frame) >= min_gap:
                    last_confirmed_frame = candidate_frame
                    boundary_triggered = True
                candidate_frame = -1
                candidate_peak = -1.0
                frames_since_on = 0

        raw_type = int(pred.get("type_id", 0))
        raw_level = int(pred.get("level_id", 0))
        if 0 <= raw_type < len(interval_type_cnt):
            interval_type_cnt[raw_type] += 1
        if 0 <= raw_level < len(interval_level_cnt):
            interval_level_cnt[raw_level] += 1

        raw_label = (raw_type, raw_level)
        window.append(raw_label)
        target = _majority_vote_cpp_style(list(window), current)

        if target == current:
            candidate_label = current
            cand_count = 0
        else:
            if target == candidate_label:
                cand_count += 1
            else:
                candidate_label = target
                cand_count = 1
            if cand_count >= hysteresis:
                current = target

        if committed_type >= 0:
            entry_type, entry_level = committed_type, committed_level
        else:
            entry_type, entry_level = current

        entry = {
            "i": int(pred.get("i", f)),
            "t": float(pred.get("t", float(f) / float(max(1, fps)))),
            "raw_type_id": raw_type,
            "raw_level_id": raw_level,
            "type_id": int(entry_type),
            "level_id": int(entry_level),
            "boundary": bool(boundary_triggered),
            "boundary_p": p,
        }
        timeline.append(entry)

        if boundary_triggered:
            seg_type = _argmax_index(interval_type_cnt, fallback=max(0, raw_type))
            seg_level = _argmax_index(interval_level_cnt, fallback=max(0, raw_level))

            committed_type = seg_type
            committed_level = seg_level
            end_idx = len(timeline)
            for i in range(interval_start_idx, end_idx):
                timeline[i]["type_id"] = int(seg_type)
                timeline[i]["level_id"] = int(seg_level)
            boundary_frames.append(int(pred.get("i", f)))

            interval_type_cnt = [0] * len(interval_type_cnt)
            interval_level_cnt = [0] * len(interval_level_cnt)
            interval_start_idx = end_idx

    segments = _segments_from_timeline(timeline, fps=int(fps), type_map=type_map)
    return {
        "config": {
            "thr_on": thr_on,
            "thr_off": thr_off,
            "confirm_win": confirm_win,
            "min_gap": min_gap,
            "smooth_win": smooth_win,
            "hysteresis": hysteresis,
        },
        "frames": timeline,
        "boundary_frames": boundary_frames,
        "segments": segments,
    }
