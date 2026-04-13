import os
import json
import argparse
from typing import Dict, Any, List

import torch

# allow imports when running as script from repo root
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from infer.postprocess import apply_cpp_emotion_sync, decode_switch_points
from infer.window_infer import DEFAULT_TYPE_MAP, WindowedAcousticAdapter

TYPE_MAP = list(DEFAULT_TYPE_MAP)


@torch.no_grad()
def infer_one(
    wav_path: str,
    ckpt_path: str,
    device: str = "cpu",
    switch_thr_on: float = 0.78,
    switch_thr_off: float = 0.60,
    switch_confirm_win: int = 3,
    switch_min_gap: int = 5,
) -> Dict[str, Any]:
    adapter = WindowedAcousticAdapter(wav_path=wav_path, ckpt_path=ckpt_path, device=device)
    acoustic = adapter.infer_all()

    T = int(acoustic.total_frames)
    frames: List[Dict[str, Any]] = []
    boundary_probs: List[float] = []
    for fr in acoustic.frames:
        frame_idx = int(fr.frame_idx)
        f = {
            "i": frame_idx,
            "t": float(frame_idx) / float(adapter.fps),
            "type_id": int(fr.emotion_id),
            "level_id": int(fr.level_id),
        }
        if bool(adapter.use_boundary_head):
            bp = float(fr.boundary_prob)
            f["boundary_p"] = bp
            boundary_probs.append(bp)
        frames.append(f)

    switch_frames: List[int] = []
    switch_times: List[float] = []
    if boundary_probs:
        switch_frames = decode_switch_points(
            boundary_p=boundary_probs,
            thr_on=float(switch_thr_on),
            thr_off=float(switch_thr_off),
            confirm_win=int(switch_confirm_win),
            min_gap=int(switch_min_gap),
        )
        switch_times = [float(i) / 30.0 for i in switch_frames]

    cpp_sync = apply_cpp_emotion_sync(
        frames=frames,
        fps=int(adapter.fps),
        type_map=list(adapter.type_map or TYPE_MAP),
    )

    return {
        "wav": os.path.basename(wav_path),
        "sample_rate": int(adapter.sample_rate),
        "fps": int(adapter.fps),
        "duration": float(adapter.duration),
        "type_map": list(adapter.type_map or TYPE_MAP),
        "switch_params": {
            "thr_on": float(switch_thr_on),
            "thr_off": float(switch_thr_off),
            "confirm_win": int(switch_confirm_win),
            "min_gap": int(switch_min_gap),
        },
        "switch_frames": switch_frames,
        "switch_times": switch_times,
        "frames": frames,
        "cpp_sync": cpp_sync,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--switch_thr_on", type=float, default=0.78)
    ap.add_argument("--switch_thr_off", type=float, default=0.60)
    ap.add_argument("--switch_confirm_win", type=int, default=3)
    ap.add_argument("--switch_min_gap", type=int, default=5)
    args = ap.parse_args()

    obj = infer_one(
        args.wav,
        args.ckpt,
        device=args.device,
        switch_thr_on=args.switch_thr_on,
        switch_thr_off=args.switch_thr_off,
        switch_confirm_win=args.switch_confirm_win,
        switch_min_gap=args.switch_min_gap,
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print("wrote:", args.out)


if __name__ == "__main__":
    main()
