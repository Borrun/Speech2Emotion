import os
import json
import argparse
from typing import Dict, Any, List

import torch
import torchaudio

# allow imports when running as script from repo root
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.featurizer import CausalLogMelFeaturizer
from models.model_emotion_tcn import EmotionTCN
from infer.postprocess import decode_switch_points

TYPE_MAP = ["happy", "sad", "angry", "fear", "calm", "confused"]


def load_wav(path: str, sr: int = 16000) -> torch.Tensor:
    wav, s = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if s != sr:
        wav = torchaudio.functional.resample(wav, s, sr)
    return wav  # [1, N]


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
    ckpt = torch.load(ckpt_path, map_location="cpu")
    mcfg = ckpt.get("cfg", {})
    use_bnd = bool(mcfg.get("use_boundary_head", True))

    model = EmotionTCN(
        n_mels=80,
        channels=int(mcfg.get("channels", 128)),
        layers=int(mcfg.get("layers", 6)),
        dropout=float(mcfg.get("dropout", 0.1)),
        use_boundary_head=use_bnd,
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    feat = CausalLogMelFeaturizer(sample_rate=16000, n_mels=80, hop_sec=1.0 / 30.0, win_sec=0.05).to(device)

    wav = load_wav(wav_path, sr=16000).to(device)
    dur = wav.size(1) / 16000.0

    mel = feat(wav).unsqueeze(0)  # [1, T, M]
    out = model(mel)
    type_logits = out["type"][0]  # [T,6]
    lvl_logits = out["lvl"][0]    # [T,6]
    bnd_logits = out["bnd"][0] if out["bnd"] is not None else None

    T = int(type_logits.size(0))
    frames: List[Dict[str, Any]] = []
    boundary_probs: List[float] = []
    for i in range(T):
        t = i / 30.0
        ty = int(type_logits[i].argmax().item())
        lv = int(lvl_logits[i].argmax().item())
        f = {"i": i, "t": float(t), "type_id": ty, "level_id": lv}
        if bnd_logits is not None:
            bp = float(torch.sigmoid(bnd_logits[i]).item())
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

    return {
        "wav": os.path.basename(wav_path),
        "sample_rate": 16000,
        "fps": 30,
        "duration": float(dur),
        "type_map": TYPE_MAP,
        "switch_params": {
            "thr_on": float(switch_thr_on),
            "thr_off": float(switch_thr_off),
            "confirm_win": int(switch_confirm_win),
            "min_gap": int(switch_min_gap),
        },
        "switch_frames": switch_frames,
        "switch_times": switch_times,
        "frames": frames,
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
