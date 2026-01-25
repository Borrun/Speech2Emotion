import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader

from audiokey.data import DataConfig, TargetConfig, AugConfig, AudioKeyDataset, collate_fn
from audiokey.model_tcn import TCNScoreNet
from audiokey.postprocess import select_key_frames_from_scores
from audiokey.metrics import match_events, prf
from audiokey.utils import to_device

def main(cfg_path="configs/config.yaml", ckpt_path="ckpt/best.pt"):
    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")

    # disable aug for eval
    aug = dict(cfg["augment"])
    aug["enable"] = False

    ds = AudioKeyDataset(
        DataConfig(**cfg["data"]),
        TargetConfig(**cfg["target"]),
        AugConfig(**aug)
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = TCNScoreNet(
        channels=cfg["model"]["tcn_channels"],
        layers=cfg["model"]["tcn_layers"],
        dropout=cfg["model"]["dropout"]
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    step_hz = int(round(1.0 / cfg["data"]["hop_sec"]))

    TP=FP=FN=0
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            logits = model(batch["mel"])[0]     # [T]
            scores = torch.sigmoid(logits).cpu().numpy()

            gt = batch["key_frames"][0]
            pred, _ = select_key_frames_from_scores(
                scores,
                step_hz=step_hz,
                fps=cfg["data"]["fps"],
                tau=cfg["post"]["tau_global"],
                max_events=cfg["post"]["max_events"],
                min_gap_sec=cfg["post"]["min_gap_sec"]
            )
            tp, fp, fn = match_events(pred, gt, fps=cfg["data"]["fps"], tol_sec=cfg["eval"]["tol_sec"])
            TP += tp; FP += fp; FN += fn

    P,R,F = prf(TP,FP,FN)
    print(f"TP={TP} FP={FP} FN={FN} | P={P:.3f} R={R:.3f} F={F:.3f}")

if __name__ == "__main__":
    main()
