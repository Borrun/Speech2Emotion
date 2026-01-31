import os
import argparse
import torch
from torch.utils.data import DataLoader

# allow imports when running as script from repo root
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from train.emotion_data import DataConfig, EmotionSeqDataset, collate
from models.model_emotion_tcn import EmotionTCN
from train.train_emotion_tcn import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav_dir", required=True)
    ap.add_argument("--label_path", required=True)
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = DataConfig(wav_dir=args.wav_dir, label_path=args.label_path)
    ds = EmotionSeqDataset(cfg)
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    mcfg = ckpt.get("cfg", {})
    model = EmotionTCN(
        n_mels=cfg.n_mels,
        channels=int(mcfg.get("channels", 128)),
        layers=int(mcfg.get("layers", 6)),
        dropout=float(mcfg.get("dropout", 0.1)),
        use_boundary_head=bool(mcfg.get("use_boundary_head", True)),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)

    metrics = evaluate(model, loader, device)
    print(metrics)


if __name__ == "__main__":
    main()
