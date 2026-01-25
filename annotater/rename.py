#!/usr/bin/env python3
import os
import csv
from pathlib import Path

def main():
    # ====== 修改这里即可 ======
    base_dir = Path(__file__).resolve().parent           # .../annotator
    wav_dir = base_dir / "wavs"                          # .../annotator/wavs
    prefix = "utt_"
    start = 1
    digits = 4
    map_csv = base_dir / "rename_map.csv"                # .../annotator/rename_map.csv
    # =========================

    if not wav_dir.is_dir():
        raise SystemExit(f"Directory not found: {wav_dir}")

    files = [p for p in wav_dir.iterdir() if p.is_file() and p.suffix.lower() == ".wav"]
    files.sort(key=lambda p: p.name.lower())

    if not files:
        print(f"No wav files found in: {wav_dir}")
        return

    # Plan new names
    planned = []
    idx = start
    for p in files:
        new_name = f"{prefix}{idx:0{digits}d}.wav"
        planned.append((p, wav_dir / new_name))
        idx += 1

    # Print plan
    print(f"Found {len(files)} wav files in {wav_dir}")
    for src, dst in planned:
        print(f"  {src.name}  ->  {dst.name}")

    # Write mapping CSV
    with open(map_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["old_name", "new_name"])
        for src, dst in planned:
            w.writerow([src.name, dst.name])

    # Two-phase rename to avoid overwriting collisions
    tmp_pairs = []
    for i, (src, dst) in enumerate(planned):
        tmp = wav_dir / f".__tmp__{i:06d}__.wav"
        tmp_pairs.append((src, tmp, dst))

    for src, tmp, _ in tmp_pairs:
        os.rename(src, tmp)

    for _, tmp, dst in tmp_pairs:
        os.rename(tmp, dst)

    print(f"\nDone. Renamed files and wrote mapping to: {map_csv}")

if __name__ == "__main__":
    main()
