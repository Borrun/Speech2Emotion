import os
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader

from audiokey.data import DataConfig, TargetConfig, AugConfig, AudioKeyDataset, collate_fn
from audiokey.model_tcn import TCNScoreNet
from audiokey.utils import set_seed, to_device


def focal_bce_with_logits(logits, targets, alpha=0.75, gamma=2.0):
    """
    logits, targets: [N]
    alpha: weight for positive class (higher => emphasize positives)
    gamma: focusing parameter (higher => focus hard examples)
    """
    targets = targets.float()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    loss = ce * ((1.0 - p_t) ** gamma)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return (alpha_t * loss).mean()


def main(cfg_path="configs/config.yaml"):
    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))

    # ---- force type casting (avoid YAML parsing as strings) ----
    cfg["train"]["lr"] = float(cfg["train"]["lr"])
    cfg["train"]["weight_decay"] = float(cfg["train"]["weight_decay"])
    cfg["train"]["batch_size"] = int(cfg["train"]["batch_size"])
    cfg["train"]["epochs"] = int(cfg["train"]["epochs"])
    cfg["train"]["grad_clip"] = float(cfg["train"]["grad_clip"])
    cfg["train"]["val_ratio"] = float(cfg["train"]["val_ratio"])
    # -----------------------------------------------------------

    set_seed(int(cfg["train"]["seed"]))
    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")

    ds = AudioKeyDataset(
        DataConfig(**cfg["data"]),
        TargetConfig(**cfg.get("target", {})),
        AugConfig(**cfg.get("augment", {}))
    )

    n_val = max(1, int(len(ds) * cfg["train"]["val_ratio"]))
    n_train = len(ds) - n_val
    tr, va = random_split(ds, [n_train, n_val])

    tr_loader = DataLoader(
        tr,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn
    )
    va_loader = DataLoader(
        va,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn
    )

    if cfg["model"]["backbone"] != "tcn":
        raise ValueError("This train.py currently supports TCN backbone only.")

    model = TCNScoreNet(
        n_mels=80,
        channels=int(cfg["model"]["tcn_channels"]),
        layers=int(cfg["model"]["tcn_layers"]),
        dropout=float(cfg["model"]["dropout"])
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"]
    )

    use_amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val = 1e9
    os.makedirs("ckpt", exist_ok=True)

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        tr_loss = 0.0

        for batch in tr_loader:
            batch = to_device(batch, device)
            mel = batch["mel"]        # [B, T, M]
            y = batch["target"]       # [B, T]
            mask = batch["mask"]      # [B, T] bool

            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(mel)   # [B, T]
                loss = focal_bce_with_logits(
                    logits[mask],
                    y[mask],
                    alpha=0.75,
                    gamma=2.0
                )

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            scaler.step(opt)
            scaler.update()

            tr_loss += float(loss.item())

        tr_loss /= max(1, len(tr_loader))

        # val
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for batch in va_loader:
                batch = to_device(batch, device)
                logits = model(batch["mel"])
                loss = focal_bce_with_logits(
                    logits[batch["mask"]],
                    batch["target"][batch["mask"]],
                    alpha=0.75,
                    gamma=2.0
                )
                va_loss += float(loss.item())
        va_loss /= max(1, len(va_loader))

        print(f"Epoch {epoch:03d} | train {tr_loss:.4f} | val {va_loss:.4f}")

        if va_loss < best_val:
            best_val = va_loss
            torch.save({"model": model.state_dict(), "cfg": cfg}, "ckpt/best.pt")
            print("  saved: ckpt/best.pt")


if __name__ == "__main__":
    main()
