import argparse
import os
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav_dir", default="./annotater/wavs")
    ap.add_argument("--pred_dir", default="./outputs/emotion_codes")
    ap.add_argument("--label_path", default="./annotater/labels_new.jsonl")
    ap.add_argument("--out_dir", default="./outputs/alignment_view")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    pred_files = [x for x in os.listdir(args.pred_dir) if x.lower().endswith(".json")]
    pred_files.sort()
    if not pred_files:
        raise RuntimeError(f"No prediction json in: {args.pred_dir}")

    script = os.path.join(os.path.dirname(__file__), "make_text_alignment_view.py")
    n_ok = 0
    for fn in pred_files:
        wav = fn[:-5] + ".wav"
        wav_path = os.path.join(args.wav_dir, wav)
        pred_path = os.path.join(args.pred_dir, fn)
        if not os.path.isfile(wav_path):
            print(f"[skip] missing wav: {wav_path}")
            continue
        cmd = [
            args.python,
            script,
            "--wav",
            wav_path,
            "--pred_json",
            pred_path,
            "--label_path",
            args.label_path,
            "--out_dir",
            args.out_dir,
        ]
        subprocess.run(cmd, check=True)
        n_ok += 1
        if n_ok % 5 == 0:
            print(f"[{n_ok}/{len(pred_files)}] done")
    print(f"DONE: generated {n_ok} alignment views in {args.out_dir}")


if __name__ == "__main__":
    main()

