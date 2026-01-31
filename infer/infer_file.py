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

TYPE_MAP = ["happy", "sad", "angry", "fear", "calm", "confused"]


def load_wav(path: str, sr: int = 16000) -> torch.Tensor:
    wav, s = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if s != sr:
        wav = torchaudio.functional.resample(wav, s, sr)
    return wav  # [1, N]


@torch.no_grad()
def infer_one(wav_path: str, ckpt_path: str, device: str = "cpu") -> Dict[str, Any]:
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
    for i in range(T):
        t = i / 30.0
        ty = int(type_logits[i].argmax().item())
        lv = int(lvl_logits[i].argmax().item())
        f = {"i": i, "t": float(t), "type_id": ty, "level_id": lv}
        if bnd_logits is not None:
            f["boundary_p"] = float(torch.sigmoid(bnd_logits[i]).item())
        frames.append(f)

    return {
        "wav": os.path.basename(wav_path),
        "sample_rate": 16000,
        "fps": 30,
        "duration": float(dur),
        "type_map": TYPE_MAP,
        "frames": frames,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    obj = infer_one(args.wav, args.ckpt, device=args.device)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print("wrote:", args.out)


if __name__ == "__main__":
    main()
