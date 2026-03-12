import argparse
import csv
import json
import os
import sys
from typing import Dict, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from online_emotion import DetectorConfig, OnlineBoundaryDetector, TextPriorBuilder


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


def chunked_indices(total: int, chunk: int) -> List[int]:
    idx = 0
    out = []
    while idx < total:
        out.append(idx)
        idx += chunk
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", default="./outputs/test_emotion_codes")
    ap.add_argument("--text_csv", default="./test/transcriptions.csv")
    ap.add_argument("--out_dir", default="./outputs/test_emotion_codes_stream_text")
    ap.add_argument("--chunk_frames", type=int, default=12)
    ap.add_argument("--w_audio", type=float, default=0.65)
    ap.add_argument("--w_text", type=float, default=0.35)
    ap.add_argument("--thr_on", type=float, default=0.58)
    ap.add_argument("--thr_off", type=float, default=0.45)
    ap.add_argument("--confirm_win", type=int, default=4)
    ap.add_argument("--min_gap", type=int, default=6)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    text_map = load_text_map(args.text_csv)

    files = [x for x in os.listdir(args.pred_dir) if x.lower().endswith(".json")]
    files.sort()
    if not files:
        raise RuntimeError(f"No prediction files found in {args.pred_dir}")

    summary = []
    for fn in files:
        src = os.path.join(args.pred_dir, fn)
        with open(src, "r", encoding="utf-8") as f:
            obj = json.load(f)

        wav = str(obj.get("wav", fn.replace(".json", ".wav")))
        fps = int(obj.get("fps", 30))
        dur = float(obj.get("duration", 0.0))
        frames = obj.get("frames", [])
        text = text_map.get(wav, "")

        p_audio = []
        for fr in frames:
            p_audio.append(float(fr.get("boundary_p", 0.0)))

        prior = TextPriorBuilder(fps=fps).build(
            text=text,
            token_timing=None,
            total_sec_hint=dur if dur > 0 else None,
        )
        det = OnlineBoundaryDetector(
            text_prior=prior,
            cfg=DetectorConfig(
                fps=fps,
                w_audio=float(args.w_audio),
                w_text=float(args.w_text),
                thr_on=float(args.thr_on),
                thr_off=float(args.thr_off),
                confirm_win=int(args.confirm_win),
                min_gap=int(args.min_gap),
            ),
        )

        events = []
        fused_probs = [0.0] * len(p_audio)
        text_probs = [0.0] * len(p_audio)
        for s in chunked_indices(len(p_audio), max(1, int(args.chunk_frames))):
            e = min(len(p_audio), s + int(args.chunk_frames))
            chunk = p_audio[s:e]
            res = det.process_chunk(frame_start=s, p_audio_chunk=chunk)
            for i, p in enumerate(res.fused_probs):
                fused_probs[s + i] = float(p)
            for i, p in enumerate(res.text_probs):
                text_probs[s + i] = float(p)
            for ev in res.events:
                events.append(int(ev.frame_idx))

        out = dict(obj)
        out["text"] = text
        out["switch_frames_audio"] = [int(x) for x in obj.get("switch_frames", [])]
        out["switch_frames"] = events
        out["switch_times"] = [float(x) / float(max(1, fps)) for x in events]
        out["stream_mode"] = {
            "text_once": True,
            "audio_streaming": True,
            "chunk_frames": int(args.chunk_frames),
            "cfg": {
                "w_audio": float(args.w_audio),
                "w_text": float(args.w_text),
                "thr_on": float(args.thr_on),
                "thr_off": float(args.thr_off),
                "confirm_win": int(args.confirm_win),
                "min_gap": int(args.min_gap),
            },
        }
        out["fused_boundary_p"] = fused_probs
        out["text_boundary_p"] = text_probs

        dst = os.path.join(args.out_dir, fn)
        with open(dst, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

        summary.append(
            {
                "wav": wav,
                "n_frames": len(p_audio),
                "switch_audio": len(out["switch_frames_audio"]),
                "switch_fused": len(events),
                "text_len": len(text),
            }
        )
        print(
            f"ok: {wav} audio={len(out['switch_frames_audio'])} "
            f"fused={len(events)} text_len={len(text)}"
        )

    summary_path = os.path.join(args.out_dir, "stream_fusion_summary.jsonl")
    with open(summary_path, "w", encoding="utf-8") as f:
        for row in summary:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("wrote:", summary_path)


if __name__ == "__main__":
    main()
