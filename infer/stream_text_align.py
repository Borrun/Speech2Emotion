from dataclasses import dataclass
from typing import Dict, List, Optional

from infer.postprocess import StreamingSwitchDecoder


PUNCT = set("，,。！？!?；;：:、")


@dataclass
class TokenSpan:
    token: str
    start_frame: int
    end_frame: int


class StreamingTextAligner:
    """
    Online monotonic token-frame aligner.

    Usage:
      aligner = StreamingTextAligner(text="你好，世界", fps=30, ...)
      out = aligner.process_chunk(frame_start=0, boundary_p_chunk=[...])  # returns finalized spans in this step
      ...
      tail = aligner.finalize(last_frame)
    """

    def __init__(
        self,
        text: str,
        fps: int = 30,
        thr_on: float = 0.74,
        thr_off: float = 0.50,
        confirm_win: int = 2,
        min_gap: int = 7,
        lookback: int = 4,
    ):
        self.fps = int(fps)
        self.tokens = self._tokenize(text)
        self.n_tokens = len(self.tokens)
        self.lookback = int(max(0, lookback))

        self.decoder = StreamingSwitchDecoder(
            thr_on=float(thr_on),
            thr_off=float(thr_off),
            confirm_win=int(confirm_win),
            min_gap=int(min_gap),
        )

        self._next_idx = 0
        self._seg_start = 0
        self._finalized: List[TokenSpan] = []

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        text = text or ""
        out = []
        for ch in text:
            if ch in ("\n", "\r", "\t"):
                continue
            if ch == " ":
                continue
            out.append(ch)
        return out

    @staticmethod
    def _weight(tok: str) -> float:
        if tok in PUNCT:
            return 1.6
        return 1.0

    def _alloc_tokens_for_segment(self, seg_start: int, seg_end: int) -> List[TokenSpan]:
        """
        Allocate a monotonic token block into [seg_start, seg_end).
        Heuristic: assign >=1 token per segment, more for longer segments and punctuation.
        """
        if self._next_idx >= self.n_tokens:
            return []
        seg_len = max(1, int(seg_end - seg_start))
        # estimate token count by duration (~4 frames / token baseline)
        est = max(1, int(round(seg_len / 4.0)))
        left = self.n_tokens - self._next_idx
        n_take = min(left, est)

        chosen = self.tokens[self._next_idx:self._next_idx + n_take]
        ws = [self._weight(t) for t in chosen]
        s = sum(ws) if sum(ws) > 0 else float(len(chosen))

        edges = [seg_start]
        c = 0.0
        for w in ws:
            c += w
            e = seg_start + int(round(seg_len * (c / s)))
            edges.append(min(seg_end, max(edges[-1] + 1, e)))
        edges[-1] = seg_end

        out: List[TokenSpan] = []
        for i, tok in enumerate(chosen):
            f0 = int(edges[i])
            f1 = int(edges[i + 1])
            if f1 <= f0:
                f1 = f0 + 1
            out.append(TokenSpan(token=tok, start_frame=f0, end_frame=f1))

        self._next_idx += n_take
        return out

    def _emit_until(self, boundary_frame: int) -> List[TokenSpan]:
        """
        Finalize a segment up to boundary (with small lookback compensation).
        """
        seg_end = max(self._seg_start + 1, int(boundary_frame) - self.lookback)
        if seg_end <= self._seg_start:
            seg_end = self._seg_start + 1
        spans = self._alloc_tokens_for_segment(self._seg_start, seg_end)
        self._seg_start = int(boundary_frame)
        self._finalized.extend(spans)
        return spans

    def process_chunk(self, frame_start: int, boundary_p_chunk: List[float]) -> Dict:
        events = self.decoder.process_chunk(frame_start=int(frame_start), boundary_p_chunk=boundary_p_chunk)
        new_spans: List[TokenSpan] = []
        for ev in events:
            new_spans.extend(self._emit_until(ev))
        return {
            "events": events,
            "new_spans": [s.__dict__ for s in new_spans],
            "next_token_index": self._next_idx,
        }

    def finalize(self, last_frame: int) -> List[Dict]:
        """
        Flush remaining tokens to the tail segment [seg_start, last_frame).
        """
        last_frame = max(self._seg_start + 1, int(last_frame))
        spans = self._alloc_tokens_for_segment(self._seg_start, last_frame)
        # if any token still remains, put all into 1-frame tails
        while self._next_idx < self.n_tokens:
            tok = self.tokens[self._next_idx]
            s = max(0, last_frame - (self.n_tokens - self._next_idx))
            e = s + 1
            spans.append(TokenSpan(token=tok, start_frame=s, end_frame=e))
            self._next_idx += 1
        self._finalized.extend(spans)
        return [s.__dict__ for s in spans]

    def all_spans(self) -> List[Dict]:
        return [s.__dict__ for s in self._finalized]

