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

    # 递归收集所有 wav，保留相对于 wav_dir 的路径（含子目录）
    wav_rel_paths = []
    for root, _, files in os.walk(args.wav_dir):
        for name in sorted(files):
            if name.lower().endswith(".wav"):
                rel = os.path.relpath(os.path.join(root, name), args.wav_dir)
                wav_rel_paths.append(rel.replace("\\", "/"))
    wav_rel_paths.sort()

    # wav_dir 相对于 server wav root（annotater/wavs）的子目录名
    subdir = os.path.basename(os.path.normpath(args.wav_dir))

    for rel in wav_rel_paths:
        in_path = os.path.join(args.wav_dir, rel)
        out_name = rel.replace("/", "_").replace("\\", "_").replace(".wav", ".json")
        out_path = os.path.join(args.out_dir, out_name)
        obj = infer_one(in_path, args.ckpt, device=args.device)
        # 覆盖 wav 字段为带子目录的相对路径，server 才能找到音频
        obj["wav"] = subdir + "/" + rel
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        print("ok:", out_path)


if __name__ == "__main__":
    main()
