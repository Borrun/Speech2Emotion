"""
Export EmotionTCN to ONNX and verify numerical accuracy.

Usage:
    python tools/export_onnx.py \
        --ckpt outputs/ckpt_bnd_v1/best.pt \
        --out  outputs/emotion_tcn.onnx

The exported model has:
    Input:  mel     float32 [B, T, 80]   (dynamic B and T)
    Output: type_logits  float32 [B, T, 7]
            level_logits float32 [B, T, 6]
            boundary_logits float32 [B, T]
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.model_emotion_tcn import EmotionTCN


class EmotionTCNExport(torch.nn.Module):
    """Wrapper that returns a flat tuple (ONNX doesn't support dict outputs)."""

    def __init__(self, model: EmotionTCN):
        super().__init__()
        self.model = model

    def forward(self, mel: torch.Tensor):
        out = self.model(mel)
        type_logits  = out["type"]   # [B, T, 7]
        level_logits = out["lvl"]    # [B, T, 6]
        bnd_logits   = out["bnd"]    # [B, T]
        return type_logits, level_logits, bnd_logits


def load_model(ckpt_path: str) -> EmotionTCN:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg  = ckpt.get("cfg", {})
    model = EmotionTCN(
        n_mels=80,
        channels=int(cfg.get("channels", 128)),
        layers=int(cfg.get("layers", 6)),
        dropout=0.0,                          # dropout=0 at export time
        n_types=int(cfg.get("n_types", 7)),
        n_levels=int(cfg.get("n_levels", 6)),
        use_boundary_head=bool(cfg.get("use_boundary_head", True)),
    )
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model


def export(ckpt_path: str, out_path: str) -> None:
    print(f"Loading checkpoint: {ckpt_path}")
    model  = load_model(ckpt_path)
    export_model = EmotionTCNExport(model)

    dummy = torch.zeros(1, 300, 80)   # [B=1, T=300, M=80]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    print(f"Exporting to: {out_path}")
    torch.onnx.export(
        export_model,
        dummy,
        out_path,
        input_names=["mel"],
        output_names=["type_logits", "level_logits", "boundary_logits"],
        dynamic_axes={
            "mel":              {0: "batch", 1: "T"},
            "type_logits":      {0: "batch", 1: "T"},
            "level_logits":     {0: "batch", 1: "T"},
            "boundary_logits":  {0: "batch", 1: "T"},
        },
        opset_version=17,
    )
    print("Export done.")


def verify(ckpt_path: str, onnx_path: str) -> None:
    try:
        import onnxruntime as ort
    except ImportError:
        print("[skip verify] onnxruntime not installed. Run: pip install onnxruntime")
        return

    print("\nVerifying numerical accuracy ...")
    model = load_model(ckpt_path)

    rng   = np.random.default_rng(42)
    mel_np = rng.standard_normal((1, 150, 80)).astype(np.float32)
    mel_t  = torch.from_numpy(mel_np)

    with torch.no_grad():
        out = model(mel_t)
        pt_type = out["type"].numpy()
        pt_lvl  = out["lvl"].numpy()
        pt_bnd  = out["bnd"].numpy()

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_type, ort_lvl, ort_bnd = sess.run(None, {"mel": mel_np})

    for name, a, b in [
        ("type_logits",     pt_type, ort_type),
        ("level_logits",    pt_lvl,  ort_lvl),
        ("boundary_logits", pt_bnd,  ort_bnd),
    ]:
        max_diff = float(np.abs(a - b).max())
        print(f"  {name:<22}  max_abs_diff = {max_diff:.2e}  {'OK' if max_diff < 1e-4 else 'WARN'}")

    # spot-check: argmax agreement
    type_match = (pt_type.argmax(-1) == ort_type.argmax(-1)).all()
    lvl_match  = (pt_lvl.argmax(-1)  == ort_lvl.argmax(-1)).all()
    print(f"  type argmax match: {type_match}   level argmax match: {lvl_match}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/ckpt_bnd_v1/best.pt")
    ap.add_argument("--out",  default="outputs/emotion_tcn.onnx")
    ap.add_argument("--no_verify", action="store_true")
    args = ap.parse_args()

    export(args.ckpt, args.out)
    if not args.no_verify:
        verify(args.ckpt, args.out)

    size_kb = os.path.getsize(args.out) / 1024
    print(f"\nFile: {args.out}  ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
