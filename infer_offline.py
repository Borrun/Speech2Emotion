import json
import yaml
import torch
import numpy as np

from audiokey.data import DataConfig, TargetConfig, AugConfig, AudioKeyDataset
from audiokey.model_tcn import TCNScoreNet
from audiokey.postprocess import select_key_frames_from_scores


def main():
    cfg = yaml.safe_load(open("configs/config.yaml", "r", encoding="utf-8"))

    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")

    ckpt_path = "ckpt/best.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg_ckpt = ckpt.get("cfg", cfg)

    # build model
    model = TCNScoreNet(
        n_mels=80,
        channels=int(cfg_ckpt["model"]["tcn_channels"]),
        layers=int(cfg_ckpt["model"]["tcn_layers"]),
        dropout=float(cfg_ckpt["model"]["dropout"])
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # choose one wav to test
    wav_name = "utt_0001.wav"

    # load one sample through dataset featurizer (no augmentation)
    data_cfg = DataConfig(**cfg["data"])
    target_cfg = TargetConfig(**cfg.get("target", {}))
    aug_cfg = AugConfig(**cfg.get("augment", {}))
    aug_cfg.enable = False

    ds = AudioKeyDataset(data_cfg, target_cfg, aug_cfg)
    # find item by name
    idx = None
    for i, it in enumerate(ds.items):
        if it["wav"] == wav_name:
            idx = i
            break
    if idx is None:
        raise ValueError(f"wav not found in labels.jsonl: {wav_name}")

    sample = ds[idx]
    mel = sample["mel"].unsqueeze(0).to(device)  # [1, T, M]

    with torch.no_grad():
        logits = model(mel)[0]                    # [T]
        scores = torch.sigmoid(logits).cpu().numpy()

    print("score max:", float(scores.max()), "mean:", float(scores.mean()))

    # step_hz = 1/hop_sec
    step_hz = 1.0 / float(cfg["data"]["hop_sec"])

    # postprocess (dynamic thresholding)
    post = cfg.get("post", {})
    key_frames, chosen = select_key_frames_from_scores(
        scores=scores,
        step_hz=step_hz,
        fps=int(cfg["data"]["fps"]),
        max_events=int(post.get("max_events", 3)),
        min_gap_sec=float(post.get("min_gap_sec", 0.8)),
        dynamic=bool(post.get("dynamic", True)),
        tau=float(post.get("tau_global", 0.30)),
        tau_min=float(post.get("tau_min", 0.20)),
        tau_frac_of_max=float(post.get("tau_frac_of_max", 0.75)),
    )

    out = {"wav": wav_name, "fps": int(cfg["data"]["fps"]), "key_frames": key_frames}
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
