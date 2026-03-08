import argparse
import csv
import json
import os
import sys
from typing import Dict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from infer.interval_decode import decode_intervals


def load_text_map(csv_path: str) -> Dict[str, str]:
    out = {}
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", default="./outputs/emotion_codes")
    ap.add_argument("--text_csv", default="./emotion_results.csv")
    ap.add_argument("--out_dir", default="./outputs/interval_predictions")
    ap.add_argument("--min_seg_len", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    txt_map = load_text_map(args.text_csv)
    files = [x for x in os.listdir(args.pred_dir) if x.lower().endswith(".json")]
    files.sort()
    if not files:
        raise RuntimeError(f"No prediction json in {args.pred_dir}")

    out_jsonl = os.path.join(args.out_dir, "interval_predictions.jsonl")
    n = 0
    with open(out_jsonl, "w", encoding="utf-8") as fj:
        for fn in files:
            path = os.path.join(args.pred_dir, fn)
            with open(path, "r", encoding="utf-8") as f:
                pred = json.load(f)
            wav = str(pred.get("wav", fn.replace(".json", ".wav")))
            text = txt_map.get(wav, "")

            obj = decode_intervals(pred_obj=pred, text=text, min_seg_len=int(args.min_seg_len))
            out_one = os.path.join(args.out_dir, wav.replace(".wav", ".interval_pred.json"))
            with open(out_one, "w", encoding="utf-8") as fo:
                json.dump(obj, fo, ensure_ascii=False, indent=2)
            fj.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n += 1
            if n % 10 == 0:
                print(f"[{n}/{len(files)}] wrote {out_one}")

    print("wrote:", out_jsonl)
    print("total:", n)


if __name__ == "__main__":
    main()

