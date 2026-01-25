import os
import json
import yaml
import numpy as np
import torch

from audiokey.data import DataConfig, TargetConfig, AugConfig, AudioKeyDataset
from audiokey.model_tcn import TCNScoreNet
from audiokey.postprocess import select_key_frames_from_scores


def match_events(pred_frames, gt_frames, fps=30, tol_sec=0.15):
    """
    Greedy matching within tolerance window.
    Return tp, fp, fn.
    """
    tol = int(round(tol_sec * fps))
    pred = sorted(pred_frames)
    gt = sorted(gt_frames)
    used = [False] * len(gt)
    tp = 0
    for p in pred:
        ok_j = -1
        best = 10**9
        for j, g in enumerate(gt):
            if used[j]:
                continue
            d = abs(p - g)
            if d <= tol and d < best:
                best = d
                ok_j = j
        if ok_j >= 0:
            used[ok_j] = True
            tp += 1
    fp = len(pred) - tp
    fn = len(gt) - tp
    return tp, fp, fn


def prf(tp, fp, fn):
    p = tp / (tp + fp + 1e-8)
    r = tp / (tp + fn + 1e-8)
    f = 2 * p * r / (p + r + 1e-8)
    return p, r, f


def main():
    cfg = yaml.safe_load(open("configs/config.yaml", "r", encoding="utf-8"))
    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")

    # load checkpoint
    ckpt_path = "ckpt/best.pt"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

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

    # dataset (no augmentation for inference)
    data_cfg = DataConfig(**cfg["data"])
    target_cfg = TargetConfig(**cfg.get("target", {}))
    aug_cfg = AugConfig(**cfg.get("augment", {}))
    aug_cfg.enable = False

    ds = AudioKeyDataset(data_cfg, target_cfg, aug_cfg)

    # postprocess config
    post = cfg.get("post", {})
    fps = int(cfg["data"]["fps"])
    hop_sec = float(cfg["data"]["hop_sec"])
    step_hz = 1.0 / hop_sec

    max_events = int(post.get("max_events", 3))
    min_gap_sec = float(post.get("min_gap_sec", 0.8))

    dynamic = bool(post.get("dynamic", True))
    tau = float(post.get("tau_global", 0.30))
    tau_min = float(post.get("tau_min", 0.20))
    tau_frac_of_max = float(post.get("tau_frac_of_max", 0.75))

    tol_sec = float(cfg.get("eval", {}).get("tol_sec", 0.15))

    # eval stats
    TP = FP = FN = 0

    # output file
    out_path = "predictions.jsonl"
    fout = open(out_path, "w", encoding="utf-8")

    print("=== Infer All ===")
    print(f"ckpt: {ckpt_path}")
    print(f"device: {device}")
    print(f"items: {len(ds)}")
    print(f"post: dynamic={dynamic} min_gap_sec={min_gap_sec} max_events={max_events}")
    print(f"post: tau={tau} tau_min={tau_min} tau_frac_of_max={tau_frac_of_max}")
    print(f"match tol: {tol_sec}s")
    print("")

    for i in range(len(ds)):
        sample = ds[i]
        wav_name = sample["wav"]
        gt = sample["key_frames"]

        mel = sample["mel"].unsqueeze(0).to(device)  # [1, T, M]

        with torch.no_grad():
            logits = model(mel)[0]  # [T]
            scores = torch.sigmoid(logits).cpu().numpy().astype(np.float32)

        score_max = float(scores.max()) if scores.size > 0 else 0.0
        score_mean = float(scores.mean()) if scores.size > 0 else 0.0

        pred, chosen = select_key_frames_from_scores(
            scores=scores,
            step_hz=step_hz,
            fps=fps,
            max_events=max_events,
            min_gap_sec=min_gap_sec,
            dynamic=dynamic,
            tau=tau,
            tau_min=tau_min,
            tau_frac_of_max=tau_frac_of_max
        )

        tp, fp, fn = match_events(pred, gt, fps=fps, tol_sec=tol_sec)
        TP += tp
        FP += fp
        FN += fn

        row = {
            "wav": wav_name,
            "fps": fps,
            "gt_key_frames": gt,
            "pred_key_frames": pred,
            "score_max": score_max,
            "score_mean": score_mean,
        }
        fout.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(f"[{i:02d}] {wav_name} | max={score_max:.3f} mean={score_mean:.3f} | GT={gt} | Pred={pred}")

    fout.close()

    P, R, F = prf(TP, FP, FN)
    print("")
    print("=== Summary ===")
    print(f"TP={TP} FP={FP} FN={FN}")
    print(f"P={P:.3f} R={R:.3f} F={F:.3f}")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
