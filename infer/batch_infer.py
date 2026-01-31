import os
import argparse
import json

# allow imports when running as script from repo root
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from infer.infer_file import infer_one


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    wavs = [x for x in os.listdir(args.wav_dir) if x.lower().endswith(".wav")]
    wavs.sort()

    for w in wavs:
        in_path = os.path.join(args.wav_dir, w)
        out_path = os.path.join(args.out_dir, w.replace(".wav", ".json"))
        obj = infer_one(in_path, args.ckpt, device=args.device)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        print("ok:", out_path)


if __name__ == "__main__":
    main()
