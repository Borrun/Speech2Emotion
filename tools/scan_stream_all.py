"""
Batch-run stream_online_emotion over all train+test files.

Usage
-----
# Baseline (no text fusion):
  python tools/scan_stream_all.py --tag baseline

# All text fusion layers on:
  python tools/scan_stream_all.py --tag tf --dynamic_w_text --text_emotion

Output
------
  outputs/stream_online/{stem}.stream_online.{tag}.json   (one per file)
  outputs/stream_online/stream_online_manifest_{tag}.json (file list)
"""

import argparse
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stream_online_emotion import run_stream, load_text_map
from online_emotion import DetectorConfig

TRAIN_PRED_DIR  = "./outputs/emotion_codes_train"
TEST_PRED_DIR   = "./outputs/test_emotion_codes"
TRAIN_TEXT_CSV  = "./outputs/train_transcriptions_from_labels.csv"
TEST_TEXT_CSV   = "./outputs/test_transcriptions_from_audio_chain.csv"
OUT_DIR         = "./outputs/stream_online"


def collect_pred_files():
    rows = []
    for d, csv in [(TRAIN_PRED_DIR, TRAIN_TEXT_CSV), (TEST_PRED_DIR, TEST_TEXT_CSV)]:
        if not os.path.isdir(d):
            print(f"[warn] pred dir not found: {d}")
            continue
        text_map = load_text_map(csv)
        for fn in sorted(os.listdir(d)):
            if fn.lower().endswith(".json"):
                rows.append((os.path.join(d, fn), text_map, fn))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="baseline",
                    help="suffix for output files, e.g. 'baseline' or 'tf'")
    # streaming params
    ap.add_argument("--chunk_min",       type=int,   default=10)
    ap.add_argument("--chunk_max",       type=int,   default=15)
    ap.add_argument("--seed",            type=int,   default=7)
    ap.add_argument("--smooth_win",      type=int,   default=5)
    ap.add_argument("--emo_hysteresis",  type=int,   default=3)
    ap.add_argument("--future_lookahead",type=int,   default=8)
    # boundary cfg
    ap.add_argument("--w_audio",    type=float, default=0.8)
    ap.add_argument("--w_text",     type=float, default=0.2)
    ap.add_argument("--thr_on",     type=float, default=0.62)
    ap.add_argument("--thr_off",    type=float, default=0.42)
    ap.add_argument("--confirm_win",type=int,   default=3)
    ap.add_argument("--min_gap",    type=int,   default=5)
    # text fusion layers
    ap.add_argument("--dynamic_w_text",  action="store_true")
    ap.add_argument("--w_text_max",      type=float, default=0.5)
    ap.add_argument("--text_emotion",    action="store_true")
    ap.add_argument("--sentiment_thr",   type=float, default=0.3)
    ap.add_argument("--text_conf_thr",   type=float, default=0.50)
    ap.add_argument("--text_blend_w",    type=float, default=0.35)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    cfg = DetectorConfig(
        w_audio=args.w_audio, w_text=args.w_text,
        thr_on=args.thr_on, thr_off=args.thr_off,
        confirm_win=args.confirm_win, min_gap=args.min_gap,
    )

    rows = collect_pred_files()
    if not rows:
        print("[error] No prediction files found. Check TRAIN_PRED_DIR / TEST_PRED_DIR.")
        return

    manifest_files = []
    for pred_path, text_map, fn in rows:
        with open(pred_path, "r", encoding="utf-8") as f:
            pred = json.load(f)

        wav  = str(pred.get("wav", fn.replace(".json", ".wav")))
        text = text_map.get(wav, "")

        res = run_stream(
            pred=pred, text=text,
            chunk_min=args.chunk_min, chunk_max=args.chunk_max,
            seed=args.seed, smooth_win=args.smooth_win,
            emo_hysteresis=args.emo_hysteresis,
            future_lookahead=args.future_lookahead,
            cfg=cfg,
            dynamic_w_text=args.dynamic_w_text, w_text_max=args.w_text_max,
            text_emotion=args.text_emotion,
            sentiment_thr=args.sentiment_thr,
            text_conf_thr=args.text_conf_thr,
            text_blend_w=args.text_blend_w,
        )

        stem      = os.path.splitext(fn)[0]
        out_name  = f"{stem}.stream_online.{args.tag}.json"
        out_path  = os.path.join(OUT_DIR, out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)

        manifest_files.append(out_name)
        constrained = sum(1 for fr in res["frames"] if fr.get("text_constrained"))
        print(f"[{args.tag}] {wav}  chunks={len(res['chunks'])}  "
              f"events={len(res['boundary_events'])}  text_constrained_frames={constrained}")

    manifest = {"files": manifest_files}
    manifest_path = os.path.join(OUT_DIR, f"stream_online_manifest_{args.tag}.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\nwrote {len(manifest_files)} files → {OUT_DIR}")
    print(f"manifest → {manifest_path}")


if __name__ == "__main__":
    main()
