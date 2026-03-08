import math
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


PUNCT_RE = re.compile(r"[，,。！？!?；;：:]")


POS_WORDS = {
    "开心", "高兴", "惊喜", "喜欢", "赞", "爱", "幸福", "激动", "愉快", "轻松",
    "happy", "great", "awesome", "love", "nice", "excellent",
}
NEG_WORDS = {
    "难过", "伤心", "愤怒", "生气", "害怕", "恐惧", "失望", "痛苦", "烦", "紧张",
    "sad", "angry", "fear", "upset", "bad", "terrible", "anxious",
}


@dataclass
class TokenTiming:
    token: str
    start_sec: float
    end_sec: float


class TextPrior:
    """
    Converts text-side boundary candidates into frame-domain priors.
    Mapping:
      frame = offset + scale * text_pos
    where text_pos in [0, 1].
    """
    def __init__(
        self,
        candidate_pos: Sequence[float],
        fps: int = 30,
        init_total_frames: int = 180,
        sigma_frames: float = 4.0,
    ):
        self.fps = int(fps)
        self.sigma_frames = max(1e-3, float(sigma_frames))

        pos = [float(x) for x in candidate_pos if 0.0 < float(x) < 1.0]
        self.candidate_pos = sorted(pos)

        self.offset = 0.0
        self.scale = max(1.0, float(init_total_frames))
        self._used = [False] * len(self.candidate_pos)

    def prob_at(self, frame_idx: int) -> float:
        if not self.candidate_pos:
            return 0.0
        x = float(frame_idx)
        s2 = self.sigma_frames * self.sigma_frames
        score = 0.0
        for p in self.candidate_pos:
            mu = self.offset + self.scale * p
            d2 = (x - mu) * (x - mu)
            score += math.exp(-0.5 * d2 / s2)
        return float(max(0.0, min(1.0, score)))

    def probs_for_range(self, frame_start: int, length: int) -> List[float]:
        return [self.prob_at(frame_start + i) for i in range(int(length))]

    def nearest_unmatched(self, frame_idx: int, max_dist: int = 45) -> Optional[Tuple[int, float, float]]:
        if not self.candidate_pos:
            return None
        best = None
        x = float(frame_idx)
        for i, p in enumerate(self.candidate_pos):
            if self._used[i]:
                continue
            mu = self.offset + self.scale * p
            d = abs(x - mu)
            if d <= float(max_dist):
                if best is None or d < best[0]:
                    best = (d, i, p)
        return None if best is None else (best[1], best[2], self.offset + self.scale * best[2])

    def register_boundary(self, frame_idx: int, lr_offset: float = 0.15, lr_scale: float = 0.05):
        hit = self.nearest_unmatched(frame_idx)
        if hit is None:
            return
        idx, pos, pred_frame = hit
        err = float(frame_idx) - float(pred_frame)

        self.offset += float(lr_offset) * err
        if pos > 1e-6:
            self.scale += float(lr_scale) * (err / pos)
            self.scale = max(20.0, self.scale)
        self._used[idx] = True


class TextPriorBuilder:
    """
    Build candidate boundary positions from:
      - punctuation boundaries
      - clause sentiment flips
      - optional token timing
    """
    def __init__(self, fps: int = 30):
        self.fps = int(fps)

    @staticmethod
    def _clause_score(clause: str) -> int:
        text = clause.lower()
        score = 0
        for w in POS_WORDS:
            if w in text:
                score += 1
        for w in NEG_WORDS:
            if w in text:
                score -= 1
        return score

    @staticmethod
    def _split_clauses(text: str) -> List[str]:
        if not text:
            return []
        parts = re.split(r"[，,。！？!?；;：:]", text)
        return [p.strip() for p in parts if p.strip()]

    def _from_punctuation(self, text: str) -> List[float]:
        if not text:
            return []
        n = max(1, len(text))
        pos = []
        for i, ch in enumerate(text):
            if PUNCT_RE.match(ch):
                p = (i + 1) / float(n)
                if 0.02 < p < 0.98:
                    pos.append(p)
        return pos

    def _from_sentiment_flip(self, text: str) -> List[float]:
        clauses = self._split_clauses(text)
        if len(clauses) < 2:
            return []
        lens = [len(c) for c in clauses]
        total = float(sum(lens))
        if total < 1:
            return []

        out = []
        acc = 0
        prev = self._clause_score(clauses[0])
        for i in range(1, len(clauses)):
            acc += lens[i - 1]
            cur = self._clause_score(clauses[i])
            if prev * cur < 0:
                p = acc / total
                if 0.02 < p < 0.98:
                    out.append(p)
            prev = cur
        return out

    def _from_token_timing(self, token_timing: Sequence[TokenTiming]) -> List[float]:
        if not token_timing:
            return []
        end = max(float(x.end_sec) for x in token_timing)
        if end <= 1e-6:
            return []

        out = []
        for x in token_timing:
            tok = (x.token or "").strip()
            if not tok:
                continue
            if PUNCT_RE.search(tok):
                p = float(x.end_sec) / end
                if 0.02 < p < 0.98:
                    out.append(p)
        return out

    @staticmethod
    def _dedup_sorted(xs: List[float], eps: float = 0.015) -> List[float]:
        xs = sorted(xs)
        out = []
        for x in xs:
            if not out or abs(out[-1] - x) > eps:
                out.append(x)
        return out

    def build(
        self,
        text: str,
        token_timing: Optional[Sequence[TokenTiming]] = None,
        total_sec_hint: Optional[float] = None,
    ) -> TextPrior:
        cands = []
        cands.extend(self._from_punctuation(text))
        cands.extend(self._from_sentiment_flip(text))
        if token_timing:
            cands.extend(self._from_token_timing(token_timing))
        cands = self._dedup_sorted(cands)

        init_total_frames = int(round(float(total_sec_hint) * self.fps)) if total_sec_hint else 180
        return TextPrior(
            candidate_pos=cands,
            fps=self.fps,
            init_total_frames=init_total_frames,
            sigma_frames=4.0,
        )

