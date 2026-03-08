from typing import List, Dict, Any


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
