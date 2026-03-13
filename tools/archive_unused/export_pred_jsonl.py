#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export model predictions for unlabeled wavs to pred.jsonl.

Key points:
- Use same featurizer as training (CausalLogMelFeaturizer @ 16k, hop=1/30, win=50ms).
- Decode with boundary head: p_bnd > post_thr -> candidate boundaries
- Enforce minimum segment length using "peak picking" within min_len window (keeps highest-prob boundary).
- Segment-wise label by averaging logits inside segment (type/lvl).
- Output JSONL lines that include:
  - wav, fps(30), duration, n_frames
  - segments (frame indices)
  - boundaries (frame indices)
  - curve (time/value/type) compatible with annotater labels.jsonl
  - per-frame type/lvl arrays optionally (dump_frames)

Run example:
python tools/export_pred_jsonl.py \
  --wav_dir ./wavs \
  --label_path ./annotater/labels.jsonl \
  --ckpt ./outputs/ckpt_7types_fixbnd_cap30_wb12/best.pt \
  --out_path ./pred.jsonl \
  --post_thr 0.60 \
  --post_min_len 3
"""

import os
import json
import argparse
from typing import Dict, List, Tuple, Optional

import torch
import torchaudio

# allow imports when running from repo root
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from train.emotion_data import ALLOWED_TYPES  # must be 7-types in your repo
try:
    from train.emotion_data import LEVEL_THRESHOLDS  # optional
except Exception:
    LEVEL_THRESHOLDS = None

from models.featurizer import CausalLogMelFeaturizer
from models.model_emotion_tcn import EmotionTCN


def _load_labeled_wavs(label_path: str) -> set:
    labeled = set()
    if not label_path or not os.path.isfile(label_path):
        return labeled
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            w = obj.get("wav", None)
            if w:
                labeled.add(str(w))
    return labeled


def _list_wavs(wav_dir: str) -> List[str]:
    xs = []
    for name in os.listdir(wav_dir):
        if name.lower().endswith(".wav"):
            xs.append(name)
    xs.sort()
    return xs


def _level_to_value_mid(level: int) -> float:
    """
    Convert discrete level 0..5 -> a representative value in 0..150 for annotater curve.
    Prefer midpoints of LEVEL_THRESHOLDS if available.
    """
    level = int(level)
    level = max(0, min(5, level))
    if isinstance(LEVEL_THRESHOLDS, dict) and level in LEVEL_THRESHOLDS:
        lo, hi = LEVEL_THRESHOLDS[level]
        return float(lo + hi) / 2.0
    # fallback (roughly midpoints of your original bins)
    fallback = {0: 5.0, 1: 18.0, 2: 38.0, 3: 68.0, 4: 98.0, 5: 130.0}
    return float(fallback[level])


def _load_wav_16k(path: str, target_sr: int = 16000) -> Tuple[torch.Tensor, float]:
    wav, sr = torchaudio.load(path)  # [C, N]
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    duration = wav.size(1) / float(target_sr)
    return wav, float(duration)


def _pick_boundaries_peak(probs: List[float], thr: float, min_len: int) -> List[int]:
    """
    candidates = {t | probs[t] > thr, t>=1}
    then peak-pick within min_len window:
      if a new candidate arrives within <min_len frames of last kept boundary,
      keep the one with larger prob.
    """
    T = len(probs)
    if T <= 1:
        return []

    cand = [t for t in range(1, T) if probs[t] > thr]
    if not cand:
        return []

    kept = []
    for t in cand:
        if not kept:
            kept.append(t)
            continue
        if t - kept[-1] < min_len:
            # within window: keep higher peak
            if probs[t] > probs[kept[-1]]:
                kept[-1] = t
        else:
            kept.append(t)

    # ensure last segment length >= min_len
    if kept and (T - kept[-1] < min_len):
        kept.pop(-1)

    # ensure first segment length >= min_len (rare, but guard)
    if kept and kept[0] < min_len:
        # drop too-early boundary; (alternatively could shift to argmax in [1,min_len])
        kept.pop(0)

    return kept


def _segments_from_boundaries(T: int, boundaries: List[int]) -> List[Tuple[int, int]]:
    segs = []
    s = 0
    for b in boundaries:
        b = int(b)
        b = max(0, min(T, b))
        if b > s:
            segs.append((s, b))
            s = b
    if s < T:
        segs.append((s, T))
    if not segs:
        segs = [(0, T)]
    return segs


def _segment_label_meanlogits(type_logits: torch.Tensor, lvl_logits: torch.Tensor, s: int, e: int) -> Tuple[int, int]:
    """
    type_logits: [T, n_types], lvl_logits: [T, n_levels]
    """
    ty = int(type_logits[s:e].mean(dim=0).argmax().item())
    lv = int(lvl_logits[s:e].mean(dim=0).argmax().item())
    return ty, lv


def _build_curve_from_segments(
    segments: List[Dict],
    fps: int,
    duration: float,
) -> List[Dict]:
    """
    Build annotater-compatible step curve points:
    points at each segment start time, plus final point at duration.
    """
    if not segments:
        # default calm L0
        return [{"t": 0.0, "type": "calm", "value": 5.0}, {"t": duration, "type": "calm", "value": 5.0}]

    curve = []
    for seg in segments:
        f0 = int(seg["f0"])
        t0 = f0 / float(fps)
        curve.append({"t": float(t0), "type": seg["type"], "value": float(seg["value"])})
    # ensure final endpoint
    last = segments[-1]
    curve.append({"t": float(duration), "type": last["type"], "value": float(last["value"])})
    return curve


@torch.no_grad()
def predict_one(
    model: torch.nn.Module,
    feat: CausalLogMelFeaturizer,
    wav_path: str,
    fps: int,
    post_thr: float,
    post_min_len: int,
    device: str,
    dump_frames: bool = False,
) -> Dict:
    wav, duration = _load_wav_16k(wav_path, target_sr=16000)
    mel = feat(wav)  # [T, M]
    T = int(mel.size(0))
    if T <= 0:
        raise RuntimeError(f"Empty mel for {wav_path}")

    x = mel.unsqueeze(0).to(device)  # [1,T,M]
    out = model(x)
    type_logits = out["type"][0].detach().cpu()  # [T,n_types]
    lvl_logits = out["lvl"][0].detach().cpu()    # [T,n_levels]
    bnd_logits = out.get("bnd", None)
    if bnd_logits is None:
        raise RuntimeError("Checkpoint/model has no boundary head (bnd).")
    bnd_prob = torch.sigmoid(bnd_logits[0]).detach().cpu().tolist()  # len=T

    boundaries = _pick_boundaries_peak(bnd_prob, thr=float(post_thr), min_len=int(post_min_len))
    seg_ranges = _segments_from_boundaries(T, boundaries)

    segments = []
    for (s, e) in seg_ranges:
        ty_id, lv_id = _segment_label_meanlogits(type_logits, lvl_logits, s, e)
        ty_id = max(0, min(len(ALLOWED_TYPES) - 1, ty_id))
        lv_id = max(0, min(5, lv_id))
        ty = ALLOWED_TYPES[ty_id]
        val = _level_to_value_mid(lv_id)

        segments.append({
            "f0": int(s),
            "f1": int(e),
            "type_id": int(ty_id),
            "lvl": int(lv_id),
            "type": str(ty),
            "value": float(val),
        })

    # optional per-frame expansion (30fps aligned)
    frame_type = None
    frame_lvl = None
    if dump_frames:
        frame_type = [0] * T
        frame_lvl = [0] * T
        for seg in segments:
            for t in range(seg["f0"], seg["f1"]):
                frame_type[t] = int(seg["type_id"])
                frame_lvl[t] = int(seg["lvl"])

    result = {
        "wav": os.path.basename(wav_path),
        "fps": int(fps),
        "duration": float(duration),
        "n_frames": int(T),
        "post_thr": float(post_thr),
        "post_min_len": int(post_min_len),
        "boundaries": [int(b) for b in boundaries],  # frame index t where change starts
        "segments": segments,
        # annotater-compatible curve
        "curve": _build_curve_from_segments(segments, fps=int(fps), duration=float(duration)),
    }
    if dump_frames:
        result["frame_type"] = frame_type
        result["frame_lvl"] = frame_lvl
    return result


def load_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("cfg", {})

    n_types = int(cfg.get("n_types", len(ALLOWED_TYPES)))
    n_levels = int(cfg.get("n_levels", 6))
    use_bnd = bool(cfg.get("use_boundary_head", True))

    model = EmotionTCN(
        n_mels=int(cfg.get("n_mels", 80)),  # if you saved n_mels; else default 80
        channels=int(cfg.get("channels", 128)),
        layers=int(cfg.get("layers", 6)),
        dropout=float(cfg.get("dropout", 0.1)),
        n_types=n_types,
        n_levels=n_levels,
        use_boundary_head=use_bnd,
    ).to(device)

    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav_dir", required=True)
    ap.add_argument("--label_path", default="", help="annotater/labels.jsonl (used to exclude labeled wavs)")
    ap.add_argument("--only_unlabeled", action="store_true", help="if set, export only wavs NOT in label_path")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--post_thr", type=float, default=0.60)
    ap.add_argument("--post_min_len", type=int, default=3)
    ap.add_argument("--dump_frames", action="store_true", help="also dump per-frame type/lvl arrays (bigger jsonl)")
    ap.add_argument("--device", default="", help="cuda/cpu; default auto")
    args = ap.parse_args()

    device = args.device.strip() if args.device.strip() else ("cuda" if torch.cuda.is_available() else "cpu")

    labeled = _load_labeled_wavs(args.label_path) if args.label_path else set()
    wavs = _list_wavs(args.wav_dir)

    if args.only_unlabeled and labeled:
        targets = [w for w in wavs if w not in labeled]
    else:
        targets = wavs

    if not targets:
        print("No wavs to export. Check --wav_dir / --label_path / --only_unlabeled")
        return

    model, mcfg = load_model(args.ckpt, device=device)

    # featurizer must match training: 16k, hop=1/30, win=0.05, n_mels=80
    fps = int(mcfg.get("fps", 30))
    hop_sec = float(mcfg.get("hop_sec", 1.0 / 30.0))
    win_sec = float(mcfg.get("win_sec", 0.05))
    n_mels = int(mcfg.get("n_mels", 80))

    feat = CausalLogMelFeaturizer(
        sample_rate=16000,
        n_mels=n_mels,
        hop_sec=hop_sec,
        win_sec=win_sec,
    )

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)

    n_ok = 0
    with open(args.out_path, "w", encoding="utf-8") as f:
        for i, wav_name in enumerate(targets):
            wav_path = os.path.join(args.wav_dir, wav_name)
            try:
                obj = predict_one(
                    model=model,
                    feat=feat,
                    wav_path=wav_path,
                    fps=fps,
                    post_thr=float(args.post_thr),
                    post_min_len=int(args.post_min_len),
                    device=device,
                    dump_frames=bool(args.dump_frames),
                )
                obj["source"] = "model_pred"
                obj["ckpt"] = os.path.abspath(args.ckpt)
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                n_ok += 1
            except Exception as e:
                print(f"[WARN] failed {wav_name}: {e}")

            if (i + 1) % 5 == 0:
                print(f"[{i+1}/{len(targets)}] exported={n_ok}")

    print(f"DONE: wrote {n_ok}/{len(targets)} lines -> {args.out_path}")


if __name__ == "__main__":
    main()
