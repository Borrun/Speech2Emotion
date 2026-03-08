import argparse
import csv
import json
import os
import random
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from infer.stream_text_align import StreamingTextAligner


def load_label_text(label_path: str, wav_name: str) -> str:
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if str(obj.get("wav", "")) == wav_name:
                return str(obj.get("text", "") or "")
    return ""


def load_csv_text(csv_path: str, wav_name: str) -> str:
    if not csv_path or (not os.path.isfile(csv_path)):
        return ""
    try:
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            rd = csv.DictReader(f)
            for row in rd:
                wav = str(row.get("wav", "") or "").strip()
                utt = str(row.get("utt_id", "") or "").strip()
                if wav == wav_name or (utt and (utt + ".wav") == wav_name):
                    txt = str(row.get("transcription", "") or "").strip()
                    if txt:
                        return txt
    except Exception:
        return ""
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_json", required=True, help="./outputs/emotion_codes/utt_xxxx.json")
    ap.add_argument("--label_path", default="./annotater/labels_new.jsonl")
    ap.add_argument("--text_csv", default="./emotion_results.csv")
    ap.add_argument("--chunk_min", type=int, default=10)
    ap.add_argument("--chunk_max", type=int, default=15)
    ap.add_argument("--thr_on", type=float, default=0.74)
    ap.add_argument("--thr_off", type=float, default=0.50)
    ap.add_argument("--confirm_win", type=int, default=2)
    ap.add_argument("--min_gap", type=int, default=7)
    ap.add_argument("--lookback", type=int, default=4)
    ap.add_argument("--out", default="", help="optional output json path")
    args = ap.parse_args()

    with open(args.pred_json, "r", encoding="utf-8") as f:
        pred = json.load(f)
    wav = pred.get("wav", "")
    fps = int(pred.get("fps", 30))
    frames = pred.get("frames", [])
    bp = [float(x.get("boundary_p", 0.0)) for x in frames]
    text = load_csv_text(args.text_csv, wav_name=wav)
    if not text:
        text = load_label_text(args.label_path, wav_name=wav)
    if not text:
        print(f"[warn] empty text in labels for {wav}")

    aligner = StreamingTextAligner(
        text=text,
        fps=fps,
        thr_on=args.thr_on,
        thr_off=args.thr_off,
        confirm_win=args.confirm_win,
        min_gap=args.min_gap,
        lookback=args.lookback,
    )

    idx = 0
    events_all = []
    while idx < len(bp):
        n = random.randint(args.chunk_min, args.chunk_max)
        chunk = bp[idx: idx + n]
        out = aligner.process_chunk(frame_start=idx, boundary_p_chunk=chunk)
        events_all.extend(out["events"])
        idx += len(chunk)

    tail = aligner.finalize(last_frame=len(bp))
    spans = aligner.all_spans()

    res = {
        "wav": wav,
        "fps": fps,
        "n_frames": len(bp),
        "text": text,
        "events": events_all,
        "token_spans": spans,
        "tail_spans": tail,
        "params": {
            "thr_on": args.thr_on,
            "thr_off": args.thr_off,
            "confirm_win": args.confirm_win,
            "min_gap": args.min_gap,
            "lookback": args.lookback,
            "chunk_min": args.chunk_min,
            "chunk_max": args.chunk_max,
        },
    }

    out_path = args.out.strip()
    if not out_path:
        base = os.path.splitext(os.path.basename(args.pred_json))[0]
        out_dir = "./outputs/alignment_view"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, base + ".stream_align.json")
    else:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    print("wrote:", out_path)
    print("events:", len(events_all), "tokens:", len(spans))


if __name__ == "__main__":
    main()
