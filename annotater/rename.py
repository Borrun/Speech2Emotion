#!/usr/bin/env python3
from pathlib import Path
import re

def main():
    wav_dir = Path("/Users/Zhuanz1/Speech2Emotion/wavs")
    prefix = "utt_"
    digits = 4

    if not wav_dir.is_dir():
        raise SystemExit(f"Directory not found: {wav_dir}")

    # 1) 找到当前已有 utt_XXXX.wav 的最大编号
    max_id = 0
    pat = re.compile(rf"^{re.escape(prefix)}(\d+)\.wav$", re.IGNORECASE)

    for p in wav_dir.iterdir():
        if not p.is_file():
            continue
        m = pat.match(p.name)
        if m:
            max_id = max(max_id, int(m.group(1)))

    start = max_id + 1
    print(f"Existing max utt id = {max_id}, will start from {start:0{digits}d}")

    # 2) 只挑 web_*.wav
    web_files = sorted(
        [p for p in wav_dir.iterdir()
         if p.is_file() and p.suffix.lower() == ".wav" and p.name.startswith("web_")],
        key=lambda p: p.name.lower()
    )

    if not web_files:
        print("No web_*.wav files found. Nothing to do.")
        return

    # 3) 规划新名字
    planned = []
    idx = start
    for p in web_files:
        new_name = f"{prefix}{idx:0{digits}d}.wav"
        dst = wav_dir / new_name

        # 防覆盖：目标已存在就退出
        if dst.exists():
            raise SystemExit(f"ABORT: target already exists: {dst.name}")

        planned.append((p, dst))
        idx += 1

    # 4) 执行改名
    print(f"Renaming {len(planned)} files:")
    for src, dst in planned:
        print(f"  {src.name} -> {dst.name}")
        src.rename(dst)

    print("\nDone.")

if __name__ == "__main__":
    main()