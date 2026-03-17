import argparse
import csv
import json
import os
import random
import sys
from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from online_emotion import DetectorConfig, OnlineBoundaryDetector, TextPriorBuilder, TextEmotionConstraint
from online_emotion.text_emotion import bilingual_sentiment_score


Label = Tuple[int, int]
PUNCT = set("，,。！？!?；;：:、")
POS_WORDS = {"开心", "高兴", "惊喜", "喜欢", "赞", "爱", "幸福", "激动", "愉快", "轻松", "happy", "great", "awesome", "love", "nice", "excellent"}
NEG_WORDS = {"难过", "伤心", "愤怒", "生气", "害怕", "恐惧", "失望", "痛苦", "烦", "紧张", "sad", "angry", "fear", "upset", "bad", "terrible", "anxious"}


def load_text_map(csv_path: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not csv_path or (not os.path.isfile(csv_path)):
        return out
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f)
        for row in rd:
            wav = str(row.get("wav", "") or "").strip()
            utt = str(row.get("utt_id", "") or "").strip()
            txt = str(row.get("transcription", "") or "").strip()
            if wav and txt:
                out[wav] = txt
            if utt and txt:
                out[utt + ".wav"] = txt
    return out


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

    def __post_init__(self):
        self._buf: Deque[Label] = deque(maxlen=max(1, int(self.smooth_win)))
        self.current: Optional[Label] = None
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


def refine_stable_with_lookahead(
    raw_labels: List[Label],
    causal_labels: List[Optional[Label]],
    lookahead: int,
    back: int = 4,
) -> List[Optional[Label]]:
    """
    Fixed-latency smoothing:
    stable_t = mode(raw[t-back : t+lookahead]) with tie-break by causal label.
    This lets each chunk output consider previous and possible future inputs.
    """
    n = len(raw_labels)
    out: List[Optional[Label]] = [None] * n
    la = max(0, int(lookahead))
    bk = max(0, int(back))
    for i in range(n):
        s = max(0, i - bk)
        e = min(n, i + la + 1)
        win = raw_labels[s:e]
        if not win:
            out[i] = causal_labels[i]
            continue
        cnt = Counter(win)
        top_n = cnt.most_common(1)[0][1]
        ties = [x for x, n0 in cnt.items() if n0 == top_n]
        fb = causal_labels[i]
        if fb in ties:
            out[i] = fb
        else:
            out[i] = ties[0]
    return out


def label_name(label: Optional[Label], type_map: List[str]) -> Dict:
    if label is None:
        return {"type_id": -1, "type": "", "level_id": -1}
    t, l = int(label[0]), int(label[1])
    name = type_map[t] if (0 <= t < len(type_map)) else str(t)
    return {"type_id": t, "type": name, "level_id": l}


def segment_major(counter: Counter) -> Optional[Label]:
    if not counter:
        return None
    return counter.most_common(1)[0][0]


def run_stream(
    pred: Dict,
    text: str,
    chunk_min: int,
    chunk_max: int,
    seed: int,
    smooth_win: int,
    emo_hysteresis: int,
    future_lookahead: int,
    cfg: DetectorConfig,
    # Layer 1: dynamic w_text
    dynamic_w_text: bool = False,
    w_text_max: float = 0.5,
    # Layer 2+3: text emotion constraint
    text_emotion: bool = False,
    sentiment_thr: float = 0.3,
    text_conf_thr: float = 0.50,
    text_blend_w: float = 0.35,
) -> Dict:
    wav = str(pred.get("wav", ""))
    fps = int(pred.get("fps", 30))
    duration = float(pred.get("duration", 0.0))
    frames = pred.get("frames", [])
    n_frames = len(frames)
    type_map = pred.get("type_map", []) if isinstance(pred.get("type_map", []), list) else []

    p_audio = [float(fr.get("boundary_p", 0.0)) for fr in frames]

    prior = TextPriorBuilder(fps=fps).build(
        text=text,
        token_timing=None,
        total_sec_hint=duration if duration > 0 else None,
    )
    detector = OnlineBoundaryDetector(text_prior=prior, cfg=cfg)

    emo_state = StreamEmotionState(smooth_win=smooth_win, hysteresis=emo_hysteresis)

    # Layer 1 base weights (restored each chunk when dynamic_w_text is off)
    base_w_audio = float(cfg.w_audio)
    base_w_text  = float(cfg.w_text)

    # Pre-compute a simple linear text alignment for per-chunk feature lookup.
    # (No anchor snapping here—anchors aren't known until after the full pass.)
    simple_tokens = align_text_to_frames(text=text, total_frames=max(1, n_frames), anchors=[])

    # Layer 2+3 constraint object (None when disabled)
    text_constraint = (
        TextEmotionConstraint(
            sentiment_thr=float(sentiment_thr),
            text_conf_thr=float(text_conf_thr),
            text_blend_w=float(text_blend_w),
        )
        if text_emotion else None
    )

    # Per-frame storage of the chunk's text info (needed for post-processing pass)
    frame_chunk_tinfo: List[Optional[Dict]] = [None] * n_frames

    random.seed(int(seed))
    idx = 0
    chunk_ranges: List[Tuple[int, int]] = []
    all_events: List[int] = []
    frame_states: List[Dict] = []
    raw_labels: List[Label] = []
    causal_labels: List[Optional[Label]] = []

    while idx < n_frames:
        chunk_n = random.randint(max(1, int(chunk_min)), max(1, int(chunk_max)))
        end = min(n_frames, idx + chunk_n)
        sub = p_audio[idx:end]

        # ── Layer 1: dynamic w_text based on chunk sentiment magnitude ────────
        chunk_tinfo_pre = chunk_text_and_features(tokens=simple_tokens, s=idx, e=end)
        if dynamic_w_text:
            sa  = abs(float(chunk_tinfo_pre["text_features"]["sentiment_score"]))
            dw  = base_w_text + (w_text_max - base_w_text) * min(1.0, sa / max(0.3, base_w_text))
            dw  = max(base_w_text, min(float(w_text_max), dw))
            detector.set_weights(1.0 - dw, dw)
        else:
            detector.set_weights(base_w_audio, base_w_text)

        out = detector.process_chunk(frame_start=idx, p_audio_chunk=sub)
        events = [int(ev.frame_idx) for ev in out.events]
        all_events.extend(events)
        event_set = set(events)
        chunk_ranges.append((idx, end))

        for f in range(idx, end):
            frame_chunk_tinfo[f] = chunk_tinfo_pre
            raw_label: Label = (int(frames[f].get("type_id", 0)), int(frames[f].get("level_id", 0)))
            stable = emo_state.update(raw_label)
            raw_labels.append(raw_label)
            causal_labels.append(stable)

            frame_states.append(
                {
                    "frame": f,
                    "t_sec": f / float(max(1, fps)),
                    "raw": label_name(raw_label, type_map),
                    "stable_causal": label_name(stable, type_map),
                    "stable": label_name(stable, type_map),
                    "segment_id": -1,
                    "boundary_event": f in event_set,
                }
            )

        idx = end

    # Non-causal correction with future context (fixed latency)
    refined = refine_stable_with_lookahead(
        raw_labels=raw_labels,
        causal_labels=causal_labels,
        lookahead=int(future_lookahead),
        back=4,
    )

    # ── Layers 2+3: text emotion constraint on refined labels ─────────────────
    if text_constraint is not None:
        for i, lb in enumerate(refined):
            if lb is None:
                continue
            tinfo = frame_chunk_tinfo[i]
            if not tinfo:
                continue
            new_type, new_level = text_constraint.apply(
                type_id=lb[0],
                level_id=lb[1],
                chunk_text=tinfo["chunk_text"],
                text_features=tinfo["text_features"],
            )
            if (new_type, new_level) != lb:
                refined[i] = (new_type, new_level)
                frame_states[i]["text_constrained"] = True

    for i, lb in enumerate(refined):
        frame_states[i]["stable"] = label_name(lb, type_map)

    # Rebuild segment ids and majors using boundary events + refined stable labels
    event_set_all = set(int(x) for x in all_events)
    seg_id = 0
    seg_start = 0
    seg_counter: Counter = Counter()
    seg_major_by_frame: List[Optional[Label]] = [None] * n_frames
    seg_id_by_frame: List[int] = [0] * n_frames
    seg_start_by_frame: List[int] = [0] * n_frames
    prev_major_by_frame: List[Optional[Label]] = [None] * n_frames
    prev_major: Optional[Label] = None
    for f in range(n_frames):
        if f in event_set_all and f > seg_start:
            prev_major = segment_major(seg_counter)
            seg_id += 1
            seg_start = f
            seg_counter = Counter()
        st = refined[f]
        if st is not None:
            seg_counter[st] += 1
        seg_major = segment_major(seg_counter)
        seg_major_by_frame[f] = seg_major
        seg_id_by_frame[f] = seg_id
        seg_start_by_frame[f] = seg_start
        prev_major_by_frame[f] = prev_major
        frame_states[f]["segment_id"] = seg_id

    tokens = align_text_to_frames(text=text, total_frames=max(1, n_frames), anchors=all_events)
    chunk_outputs: List[Dict] = []
    for s, e in chunk_ranges:
        events = [x for x in all_events if s <= int(x) < e]
        curf = e - 1
        tinfo = chunk_text_and_features(tokens=tokens, s=s, e=e)
        chunk_outputs.append(
            {
                "chunk_start": s,
                "chunk_end": e - 1,
                "chunk_frames": e - s,
                "boundary_events": events,
                "current_frame": frame_states[curf] if frame_states else None,
                "segment": {
                    "id": seg_id_by_frame[curf],
                    "start_frame": seg_start_by_frame[curf],
                    "major": label_name(seg_major_by_frame[curf], type_map),
                    "prev_major": label_name(prev_major_by_frame[curf], type_map),
                },
                **tinfo,
            }
        )

    final_major = seg_major_by_frame[-1] if seg_major_by_frame else None

    return {
        "wav": wav,
        "fps": fps,
        "n_frames": n_frames,
        "text": text,
        "stream_mode": {
            "text_once": True,
            "audio_streaming": True,
            "chunk_min": int(chunk_min),
            "chunk_max": int(chunk_max),
            "seed": int(seed),
            "emotion_smooth_win": int(smooth_win),
            "emotion_hysteresis": int(emo_hysteresis),
            "future_lookahead": int(future_lookahead),
            "text_fusion": {
                "dynamic_w_text": bool(dynamic_w_text),
                "w_text_max": float(w_text_max),
                "text_emotion": bool(text_emotion),
                "sentiment_thr": float(sentiment_thr),
                "text_conf_thr": float(text_conf_thr),
                "text_blend_w": float(text_blend_w),
            },
            "boundary_cfg": {
                "w_audio": float(cfg.w_audio),
                "w_text": float(cfg.w_text),
                "thr_on": float(cfg.thr_on),
                "thr_off": float(cfg.thr_off),
                "confirm_win": int(cfg.confirm_win),
                "min_gap": int(cfg.min_gap),
            },
        },
        "boundary_events": all_events,
        "chunks": chunk_outputs,
        "final_segment": {
            "id": seg_id_by_frame[-1] if seg_id_by_frame else 0,
            "start_frame": seg_start_by_frame[-1] if seg_start_by_frame else 0,
            "major": label_name(final_major, type_map),
        },
        "frames": frame_states,
        "tokens": tokens,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_json", default="", help="audio model json with frames/type/level/boundary_p")
    ap.add_argument("--wav", default="", help="optional: raw wav input, used when --pred_json is empty")
    ap.add_argument("--ckpt", default="", help="optional: model checkpoint for --wav mode")
    ap.add_argument("--device", default="cpu", help="device for --wav mode")
    ap.add_argument("--text", default="", help="text once input; if empty, try --text_csv by wav")
    ap.add_argument("--text_csv", default="./outputs/test_transcriptions_from_audio_chain.csv")
    ap.add_argument("--chunk_min", type=int, default=10)
    ap.add_argument("--chunk_max", type=int, default=15)
    ap.add_argument("--seed", type=int, default=7)

    ap.add_argument("--smooth_win", type=int, default=5)
    ap.add_argument("--emo_hysteresis", type=int, default=3)
    ap.add_argument("--future_lookahead", type=int, default=8, help="future frames used in fixed-latency refinement")

    ap.add_argument("--w_audio", type=float, default=0.8)
    ap.add_argument("--w_text", type=float, default=0.2)
    ap.add_argument("--thr_on", type=float, default=0.62)
    ap.add_argument("--thr_off", type=float, default=0.42)
    ap.add_argument("--confirm_win", type=int, default=3)
    ap.add_argument("--min_gap", type=int, default=5)

    # Text fusion layers
    ap.add_argument("--dynamic_w_text", action="store_true",
                    help="Layer 1: boost w_text when chunk sentiment is strong")
    ap.add_argument("--w_text_max", type=float, default=0.5,
                    help="Layer 1: max w_text when dynamic_w_text is on")
    ap.add_argument("--text_emotion", action="store_true",
                    help="Layer 2+3: apply text-based emotion type constraint")
    ap.add_argument("--sentiment_thr", type=float, default=0.3,
                    help="Layer 2: sentiment magnitude threshold for hard override")
    ap.add_argument("--text_conf_thr", type=float, default=0.50,
                    help="Layer 3: min text-distribution confidence to blend")
    ap.add_argument("--text_blend_w", type=float, default=0.35,
                    help="Layer 3: text weight in the soft blend")

    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.pred_json.strip():
        with open(args.pred_json, "r", encoding="utf-8") as f:
            pred = json.load(f)
    else:
        if not args.wav.strip() or not args.ckpt.strip():
            raise ValueError("Provide --pred_json, or provide both --wav and --ckpt.")
        from infer.infer_file import infer_one
        pred = infer_one(
            wav_path=args.wav,
            ckpt_path=args.ckpt,
            device=args.device,
            # audio chain decode params are independent from fusion detector params
            switch_thr_on=0.78,
            switch_thr_off=0.60,
            switch_confirm_win=3,
            switch_min_gap=5,
        )

    wav = str(pred.get("wav", ""))
    text = str(args.text or "").strip()
    if not text:
        text_map = load_text_map(args.text_csv)
        text = text_map.get(wav, "")

    cfg = DetectorConfig(
        fps=int(pred.get("fps", 30)),
        w_audio=float(args.w_audio),
        w_text=float(args.w_text),
        thr_on=float(args.thr_on),
        thr_off=float(args.thr_off),
        confirm_win=int(args.confirm_win),
        min_gap=int(args.min_gap),
    )

    res = run_stream(
        pred=pred,
        text=text,
        chunk_min=int(args.chunk_min),
        chunk_max=int(args.chunk_max),
        seed=int(args.seed),
        smooth_win=int(args.smooth_win),
        emo_hysteresis=int(args.emo_hysteresis),
        future_lookahead=int(args.future_lookahead),
        cfg=cfg,
        dynamic_w_text=bool(args.dynamic_w_text),
        w_text_max=float(args.w_text_max),
        text_emotion=bool(args.text_emotion),
        sentiment_thr=float(args.sentiment_thr),
        text_conf_thr=float(args.text_conf_thr),
        text_blend_w=float(args.text_blend_w),
    )

    out = args.out.strip()
    if not out:
        base = os.path.splitext(os.path.basename(args.pred_json))[0]
        out_dir = "./outputs/stream_online"
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, base + ".stream_online.json")
    else:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    print("wrote:", out)
    print("chunks:", len(res["chunks"]))
    print("boundary_events:", len(res["boundary_events"]))
    if res["chunks"]:
        print("last_segment:", res["chunks"][-1]["segment"])


if __name__ == "__main__":
    main()
