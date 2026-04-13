from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

from .text_emotion import bilingual_sentiment_score


Label = Tuple[int, int]

PUNCT = set("，,。！？!?；;：:、")
POS_WORDS = {
    "开心", "高兴", "惊喜", "喜欢", "赞", "爱", "幸福", "激动", "愉快", "轻松",
    "happy", "great", "awesome", "love", "nice", "excellent",
}
NEG_WORDS = {
    "难过", "伤心", "愤怒", "生气", "害怕", "恐惧", "失望", "痛苦", "烦", "紧张",
    "sad", "angry", "fear", "upset", "bad", "terrible", "anxious",
}


def token_weight(ch: str) -> float:
    return 1.6 if ch in PUNCT else 1.0


def align_text_to_frames(text: str, total_frames: int, anchors: List[int]) -> List[Dict]:
    chars = [c for c in (text or "") if c not in ("\n", "\r", "\t", " ")]
    if not chars:
        return []

    ws = [token_weight(c) for c in chars]
    s = sum(ws) if sum(ws) > 0 else float(len(ws))
    edges = [0]
    c = 0.0
    for w in ws:
        c += w
        edges.append(int(round(total_frames * c / s)))

    edges[0] = 0
    edges[-1] = total_frames
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = min(total_frames, edges[i - 1] + 1)
    for i in range(len(edges) - 2, -1, -1):
        if edges[i] >= edges[i + 1]:
            edges[i] = max(0, edges[i + 1] - 1)

    punct_idx = [i + 1 for i, ch in enumerate(chars) if ch in PUNCT]
    for bi in punct_idx:
        pf = edges[bi]
        best = None
        for a in anchors:
            d = abs(int(a) - int(pf))
            if d <= 10 and (best is None or d < best[0]):
                best = (d, int(a))
        if best is not None:
            edges[bi] = best[1]

    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = min(total_frames, edges[i - 1] + 1)
    edges[-1] = total_frames

    out = []
    for i, ch in enumerate(chars):
        f0 = int(edges[i])
        f1 = int(edges[i + 1])
        if f1 <= f0:
            f1 = f0 + 1
        out.append({"token": ch, "f0": f0, "f1": f1})
    return out


def chunk_text_and_features(tokens: List[Dict], s: int, e: int) -> Dict:
    touched = []
    punct = 0
    for t in tokens:
        f0, f1 = int(t["f0"]), int(t["f1"])
        if max(f0, s) < min(f1, e):
            ch = str(t["token"])
            touched.append(ch)
            if ch in PUNCT:
                punct += 1

    txt = "".join(touched)
    low = txt.lower()
    pos_hits = sum(1 for w in POS_WORDS if w in low)
    neg_hits = sum(1 for w in NEG_WORDS if w in low)
    n_tok = len(touched)
    sentiment = bilingual_sentiment_score(txt) if txt.strip() else 0.0
    return {
        "chunk_text": txt,
        "text_features": {
            "n_tokens": n_tok,
            "punct_count": punct,
            "pos_hits": pos_hits,
            "neg_hits": neg_hits,
            "sentiment_score": sentiment,
        },
    }


@dataclass
class StreamEmotionState:
    smooth_win: int = 5
    hysteresis: int = 3
    initial: Optional[Label] = None

    def __post_init__(self):
        self._buf: Deque[Label] = deque(maxlen=max(1, int(self.smooth_win)))
        self.current: Optional[Label] = self.initial
        self._candidate: Optional[Label] = None
        self._cand_count: int = 0

    @staticmethod
    def _majority(buf: Deque[Label], fallback: Optional[Label]) -> Optional[Label]:
        if not buf:
            return fallback
        cnt = Counter(buf)
        ranked = cnt.most_common()
        top_n = ranked[0][1]
        ties = [x for x, n in ranked if n == top_n]
        if fallback in ties:
            return fallback
        return ties[0]

    def update(self, raw: Label) -> Optional[Label]:
        self._buf.append(raw)
        target = self._majority(self._buf, fallback=self.current)
        if target is None:
            return self.current

        if self.current is None:
            self.current = target
            self._candidate = None
            self._cand_count = 0
            return self.current

        if target == self.current:
            self._candidate = None
            self._cand_count = 0
            return self.current

        if target == self._candidate:
            self._cand_count += 1
        else:
            self._candidate = target
            self._cand_count = 1

        if self._cand_count >= max(1, int(self.hysteresis)):
            self.current = target
            self._candidate = None
            self._cand_count = 0

        return self.current


def segment_major(counter: Counter) -> Optional[Label]:
    if not counter:
        return None
    return counter.most_common(1)[0][0]
