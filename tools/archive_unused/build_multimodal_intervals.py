import argparse
import csv
import json
import os
from typing import Dict, List, Tuple


LEVEL_THRESHOLDS = {
    5: (111.0, 150.0),
    4: (86.0, 110.0),
    3: (51.0, 85.0),
    2: (26.0, 50.0),
    1: (11.0, 25.0),
    0: (0.0, 10.0),
}
PUNCT = set("，,。！？!?；;：:、")


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def level_from_value(v: float) -> int:
    v = float(clamp(v, 0.0, 150.0))
    for lvl in (5, 4, 3, 2, 1, 0):
        lo, hi = LEVEL_THRESHOLDS[lvl]
        if lo <= v <= hi:
            return lvl
    return 0


def normalize_curve(curve, duration: float):
    out = []
    if not isinstance(curve, list):
        curve = []
    for p in curve:
        if not isinstance(p, dict):
            continue
        t = float(clamp(p.get("t", 0.0), 0.0, duration))
        ty = str(p.get("type", "calm")).strip().lower() or "calm"
        v = float(clamp(p.get("value", 60.0), 0.0, 150.0))
        out.append({"t": t, "type": ty, "value": v})
    out.sort(key=lambda x: x["t"])
    if not out:
        out = [{"t": 0.0, "type": "calm", "value": 60.0}, {"t": duration, "type": "calm", "value": 60.0}]
    else:
        if out[0]["t"] > 0:
            out.insert(0, {"t": 0.0, "type": out[0]["type"], "value": out[0]["value"]})
        if out[-1]["t"] < duration:
            out.append({"t": duration, "type": out[-1]["type"], "value": out[-1]["value"]})
    out[0]["t"] = 0.0
    out[-1]["t"] = duration
    return out


def tokenize_text(text: str) -> List[str]:
    return [c for c in (text or "") if c not in ("\n", "\r", "\t", " ")]


def token_weight(ch: str) -> float:
    return 1.6 if ch in PUNCT else 1.0


def align_tokens_to_frames(text: str, total_frames: int, anchors: List[int]) -> List[Dict]:
    toks = tokenize_text(text)
    if not toks:
        return []
    ws = [token_weight(t) for t in toks]
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

    # snap punctuation boundaries near anchors (emotion change points)
    punct_idx = [i + 1 for i, t in enumerate(toks) if t in PUNCT]
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
    for i, tok in enumerate(toks):
        f0 = int(edges[i])
        f1 = int(edges[i + 1])
        if f1 <= f0:
            f1 = f0 + 1
        out.append({"token": tok, "f0": f0, "f1": f1})
    return out


def overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def load_labels(path: str) -> Dict[str, Dict]:
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            wav = obj.get("wav")
            if wav:
                out[str(wav)] = obj
    return out


def load_text_map(csv_path: str) -> Dict[str, str]:
    out = {}
    if not os.path.isfile(csv_path):
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


def build_intervals_for_wav(wav: str, label_obj: Dict, text: str, fps: int = 30) -> Dict:
    duration = float(label_obj.get("duration", 0.0) or 0.0)
    duration = max(duration, 1e-6)
    n_frames = max(1, int(round(duration * fps)))

    curve = normalize_curve(label_obj.get("curve", []), duration)
    emotion_segments = []
    for i in range(len(curve) - 1):
        p = curve[i]
        q = curve[i + 1]
        s_sec = float(p["t"])
        e_sec = float(q["t"])
        s_f = max(0, min(n_frames, int(round(s_sec * fps))))
        e_f = max(s_f + 1, min(n_frames, int(round(e_sec * fps))))
        emotion_segments.append(
            {
                "seg_id": i,
                "start_sec": s_sec,
                "end_sec": e_sec,
                "start_frame": s_f,
                "end_frame": e_f,
                "emotion_type": p["type"],
                "emotion_value": float(p["value"]),
                "emotion_level": int(level_from_value(p["value"])),
            }
        )

    anchors = [int(round(float(c["t"]) * fps)) for c in curve[1:-1]]
    tokens = align_tokens_to_frames(text=text, total_frames=n_frames, anchors=anchors)

    intervals = []
    for seg in emotion_segments:
        s0 = int(seg["start_frame"])
        s1 = int(seg["end_frame"])
        touched = []
        first_tok = None
        last_tok = None
        for ti, tk in enumerate(tokens):
            ov = overlap(s0, s1, int(tk["f0"]), int(tk["f1"]))
            if ov > 0:
                touched.append(tk["token"])
                if first_tok is None:
                    first_tok = ti
                last_tok = ti

        obj = dict(seg)
        obj["text"] = "".join(touched)
        obj["token_start"] = first_tok if first_tok is not None else -1
        obj["token_end"] = last_tok if last_tok is not None else -1
        obj["n_tokens"] = len(touched)
        intervals.append(obj)

    return {
        "wav": wav,
        "fps": fps,
        "duration": duration,
        "text": text,
        "tokens": tokens,
        "intervals": intervals,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label_path", default="./annotater/labels_new.jsonl")
    ap.add_argument("--text_csv", default="./emotion_results.csv")
    ap.add_argument("--out_dir", default="./outputs/multimodal_intervals")
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    labels = load_labels(args.label_path)
    text_map = load_text_map(args.text_csv)

    agg_path = os.path.join(args.out_dir, "intervals.jsonl")
    n = 0
    with open(agg_path, "w", encoding="utf-8") as fa:
        for wav in sorted(labels.keys()):
            text = text_map.get(wav, str(labels[wav].get("text", "") or ""))
            obj = build_intervals_for_wav(
                wav=wav,
                label_obj=labels[wav],
                text=text,
                fps=int(args.fps),
            )
            out_one = os.path.join(args.out_dir, wav.replace(".wav", ".intervals.json"))
            with open(out_one, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
            fa.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n += 1
            if n % 10 == 0:
                print(f"[{n}/{len(labels)}] wrote {out_one}")

    print("wrote:", agg_path)
    print("total:", n)


if __name__ == "__main__":
    main()

