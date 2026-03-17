"""
Text-based emotion constraints for streaming inference.

Layer 1 (in detector): dynamic w_text based on chunk sentiment magnitude.
Layer 2 (here):  rule-based hard override for clear sentiment/structural mismatches.
Layer 3 (here):  soft keyword-distribution blend when text signal is strong.

type_map assumed: ["happy", "sad", "angry", "fear", "calm", "confused"]
                    idx:       0       1       2       3       4          5

Bilingual support (Chinese + English):
  - langdetect  : language detection (fallback: CJK character heuristic)
  - SnowNLP     : Chinese sentiment score (0→1, converted to −1→1)
  - NRCLex      : English emotion distribution (8 NRC categories)
  All three are optional; keyword-based fallback is always available.
"""

import re
from typing import Dict, List, Tuple

# ── optional bilingual backends ──────────────────────────────────────────────
_HAS_LANGDETECT = False
_HAS_SNOWNLP    = False
_HAS_NRCLEX     = False

try:
    from langdetect import detect as _lang_detect, LangDetectException
    _HAS_LANGDETECT = True
except ImportError:
    pass

try:
    from snownlp import SnowNLP as _SnowNLP
    _HAS_SNOWNLP = True
except ImportError:
    pass

try:
    from nrclex import NRCLex as _NRCLex
    _HAS_NRCLEX = True
except ImportError:
    pass

QUESTION_RE = re.compile(r"[？?]")
EXCLAIM_RE  = re.compile(r"[！!]")

# ── emotion indices (must match TYPE_MAP in infer_file.py) ──────────────────
HAPPY, SAD, ANGRY, FEAR, CALM, CONFUSED = 0, 1, 2, 3, 4, 5
N_TYPES = 6

POSITIVE_TYPES = frozenset({HAPPY, CALM})
NEGATIVE_TYPES = frozenset({SAD, ANGRY, FEAR})

# ── keyword table ────────────────────────────────────────────────────────────
EMOTION_KEYWORDS: Dict[int, set] = {
    HAPPY: {
        "开心", "高兴", "惊喜", "喜欢", "赞", "爱", "幸福", "激动", "愉快",
        "太好了", "棒", "厉害", "哈哈", "嘻嘻", "太棒了", "真棒", "好极了",
        "happy", "great", "awesome", "love", "nice", "excellent", "wonderful",
        "yay", "haha", "wow",
    },
    SAD: {
        "难过", "伤心", "失望", "痛苦", "哭", "泪", "遗憾", "可惜", "心疼",
        "sad", "upset", "cry", "disappointed", "miss", "sorry", "regret",
    },
    ANGRY: {
        "愤怒", "生气", "烦", "讨厌", "气死", "可恶", "无语", "烦死了",
        "angry", "mad", "hate", "furious", "annoying", "ridiculous",
    },
    FEAR: {
        "害怕", "恐惧", "紧张", "担心", "慌", "怕", "不安", "心跳",
        "fear", "scared", "anxious", "nervous", "worried", "panic",
    },
    CALM: {
        "平静", "好的", "嗯", "没问题", "好", "没事", "可以", "了解", "明白",
        "calm", "ok", "fine", "alright", "sure", "yes", "okay",
    },
    CONFUSED: {
        "不知道", "疑惑", "奇怪", "搞不懂", "为什么", "怎么", "什么",
        "confused", "unsure", "why", "what", "how", "wonder", "huh",
    },
}


# NRC emotion category → (type_index, weight)
# primary emotion categories have weight 1.0; sentiment meta-categories 0.4
_NRC_TO_IDX: Dict[str, tuple] = {
    "joy":          (HAPPY,   1.0),
    "positive":     (HAPPY,   0.4),
    "trust":        (CALM,    1.0),
    "anticipation": (CALM,    0.6),
    "anger":        (ANGRY,   1.0),
    "disgust":      (ANGRY,   0.7),
    "sadness":      (SAD,     1.0),
    "negative":     (SAD,     0.4),
    "fear":         (FEAR,    1.0),
    "surprise":     (CONFUSED, 1.0),
}

# NRCLex v4: instantiate once, reload text per call
_nrc_instance = None


def _nrc_affect(text: str) -> dict:
    """Return NRCLex affect_frequencies dict, handling v4 API."""
    global _nrc_instance
    if not _HAS_NRCLEX:
        return {}
    try:
        if _nrc_instance is None:
            _nrc_instance = _NRCLex("placeholder")
        _nrc_instance.load_raw_text(text)
        return dict(_nrc_instance.affect_frequencies)
    except Exception:
        return {}


def _detect_lang(text: str) -> str:
    """Returns 'zh' or 'en'. Uses CJK heuristic; langdetect as secondary check."""
    cjk = sum(1 for c in (text or "") if "\u4e00" <= c <= "\u9fff")
    if cjk > len(text) * 0.15:
        return "zh"
    if _HAS_LANGDETECT and len(text.strip()) >= 6:
        try:
            code = _lang_detect(text)
            return "zh" if code.startswith("zh") else "en"
        except Exception:
            pass
    return "en"


def bilingual_sentiment_score(text: str) -> float:
    """
    Returns sentiment in [-1, 1].  Positive = positive sentiment.
    Chinese: SnowNLP (fallback: keyword counting).
    English: NRCLex positive/negative affect (fallback: keyword counting).
    """
    t = (text or "").strip()
    if not t:
        return 0.0

    lang = _detect_lang(t)

    if lang == "zh":
        low = t.lower()
        pos = sum(1 for w in EMOTION_KEYWORDS[HAPPY] | EMOTION_KEYWORDS[CALM] if w in low)
        neg = sum(1 for w in EMOTION_KEYWORDS[SAD] | EMOTION_KEYWORDS[ANGRY] | EMOTION_KEYWORDS[FEAR] if w in low)
        if pos + neg > 0:
            return (pos - neg) / float(pos + neg)
        # no keywords matched — SnowNLP as fallback (domain mismatch noted)
        if _HAS_SNOWNLP:
            try:
                return float(_SnowNLP(t).sentiments) * 2.0 - 1.0
            except Exception:
                pass
        return 0.0
    else:
        if _HAS_NRCLEX:
            af = _nrc_affect(t)
            if af:
                pos = float(af.get("positive", 0.0))
                neg = float(af.get("negative", 0.0))
                # also use primary emotion scores as sentiment signal
                pos += float(af.get("joy", 0.0)) + float(af.get("trust", 0.0))
                neg += float(af.get("sadness", 0.0)) + float(af.get("fear", 0.0)) + float(af.get("anger", 0.0))
                if pos + neg > 1e-6:
                    return (pos - neg) / (pos + neg)
        # keyword fallback
        low = t.lower()
        pos = sum(1 for w in EMOTION_KEYWORDS[HAPPY] | EMOTION_KEYWORDS[CALM] if w in low)
        neg = sum(1 for w in EMOTION_KEYWORDS[SAD] | EMOTION_KEYWORDS[ANGRY] | EMOTION_KEYWORDS[FEAR] if w in low)
        total = max(1, pos + neg)
        return (pos - neg) / float(total)


def text_emotion_distribution(text: str) -> List[float]:
    """
    Layer 3: emotion distribution over N_TYPES.
    English: NRCLex affect frequencies mapped to project type indices.
    Chinese / fallback: keyword counting.
    Punctuation structure adds secondary boosts for both languages.
    Returns a uniform prior when no signal is detected.
    """
    t = (text or "").lower()
    scores = [0.0] * N_TYPES

    lang = _detect_lang(text or "")
    if lang != "zh" and _HAS_NRCLEX:
        af = _nrc_affect(text)
        if af:
            for nrc_key, (type_idx, weight) in _NRC_TO_IDX.items():
                scores[type_idx] += float(af.get(nrc_key, 0.0)) * weight

    if sum(scores) < 1e-6:
        # keyword fallback (Chinese or NRCLex unavailable)
        for eid, kws in EMOTION_KEYWORDS.items():
            for kw in kws:
                if kw in t:
                    scores[eid] += 1.0

    # structural boosts (language-agnostic)
    if QUESTION_RE.search(text or ""):
        scores[CONFUSED] += 1.5
    if EXCLAIM_RE.search(text or ""):
        scores[HAPPY]  += 0.5
        scores[ANGRY]  += 0.3

    total = sum(scores)
    if total < 1e-6:
        return [1.0 / N_TYPES] * N_TYPES
    return [s / total for s in scores]


class TextEmotionConstraint:
    """
    Post-processes audio-model (type_id, level_id) with text-based constraints.

    Layer 2 — rule-based sentiment override:
      - Strong positive text  + negative emotion  →  happy
      - Strong negative text  + positive emotion  →  sad
      - Question marker + low-level emotion       →  confused

    Layer 3 — soft keyword-distribution blend:
      - Builds a dist over emotion types from chunk text keywords.
      - When text confidence exceeds text_conf_thr, blends the audio
        decision with the text distribution and takes the new argmax.

    Parameters
    ----------
    sentiment_thr : threshold on sentiment_score for Layer-2 flip  (default 0.3)
    text_conf_thr : min text-top probability to trigger Layer-3    (default 0.50)
    text_blend_w  : text weight in the Layer-3 blend               (default 0.35)
    """

    def __init__(
        self,
        sentiment_thr: float = 0.3,
        text_conf_thr: float = 0.50,
        text_blend_w: float  = 0.35,
        n_types: int = N_TYPES,
    ):
        self.sentiment_thr = float(sentiment_thr)
        self.text_conf_thr = float(text_conf_thr)
        self.text_blend_w  = float(text_blend_w)
        self.n_types       = int(n_types)

    def apply(
        self,
        type_id: int,
        level_id: int,
        chunk_text: str,
        text_features: dict,
    ) -> Tuple[int, int]:
        """
        Returns (constrained_type_id, level_id).
        level_id is currently passed through unchanged.
        """
        sentiment    = float(text_features.get("sentiment_score", 0.0))
        has_question = bool(QUESTION_RE.search(chunk_text or ""))

        # ── Layer 2: rule-based hard constraints ─────────────────────────────
        if sentiment > self.sentiment_thr and type_id in NEGATIVE_TYPES:
            # positive text sentiment clashes with negative audio emotion
            type_id = HAPPY

        elif sentiment < -self.sentiment_thr and type_id in POSITIVE_TYPES:
            # negative text sentiment clashes with positive audio emotion
            type_id = SAD

        elif has_question and level_id <= 2 and type_id != CONFUSED:
            # interrogative structure + low-intensity audio → confused
            type_id = CONFUSED

        # ── Layer 3: soft keyword-distribution blend ──────────────────────────
        dist     = text_emotion_distribution(chunk_text)
        text_top = int(max(range(self.n_types), key=lambda i: dist[i]))
        text_conf = dist[text_top]

        if text_conf >= self.text_conf_thr and text_top != type_id:
            # blend: audio contributes a unit spike; text contributes its distribution
            audio_w  = 1.0 - self.text_blend_w
            blended  = [self.text_blend_w * dist[i] for i in range(self.n_types)]
            blended[type_id] += audio_w
            type_id = int(max(range(self.n_types), key=lambda i: blended[i]))

        return type_id, level_id
