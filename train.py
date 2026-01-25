import os, yaml
import torch
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader

from audiokey.data import DataConfig, TargetConfig, AugConfig, AudioKeyDataset, collate_fn
from audiokey.model_tcn import TCNScoreNet
from audiokey.utils import set_seed, to_device

def main(cfg_path="configs/config.yaml"):
    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))

    set_seed(cfg["train"]["seed"])
    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")

    ds = AudioKeyDataset(
        DataConfig(**cfg["data"]),
        TargetConfig(**cfg["target"]),
        AugConfig(**cfg["augment"])
    )

    n_val = max(1, int(len(ds) * cfg["train"]["val_ratio"]))
    n_train = len(ds) - n_val
    tr, va = random_split(ds, [n_train, n_val])

    tr_loader = DataLoader(tr, batch_size=cfg["train"]["batch_size"], shuffle=True, collate_fn=collate_fn)
    va_loader = DataLoader(va, batch_size=cfg["train"]["batch_size"], shuffle=False, collate_fn=collate_fn)

    # model
    if cfg["model"]["backbone"] == "tcn":
        model = TCNScoreNet(
            n_mels=80,
            channels=cfg["model"]["tcn_channels"],
            layers=cfg["model"]["tcn_layers"],
            dropout=cfg["model"]["dropout"]
        )
    else:
        raise ValueError("For wavlm backbone use a separate wavlm train variant or adapt dataset to wav input.")

    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg["train"]["amp"]) and device.type == "cuda")

    pos_w = torch.tensor(cfg["target"]["pos_weight"], device=device)

    best_val = 1e9
    os.makedirs("ckpt", exist_ok=True)

    for epoch in range(1, cfg["train"]["epochs"]+1):
        model.train()
        tr_loss = 0.0

        for batch in tr_loader:
            batch = to_device(batch, device)
            mel = batch["mel"]          # [B, T, M]
            y = batch["target"]         # [B, T]
            mask = batch["mask"]        # [B, T] bool

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                logits = model(mel)     # [B, T]
                # BCE with mask
                loss = F.binary_cross_entropy_with_logits(
                    logits[mask], y[mask],
                    pos_weight=pos_w
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
                loss = F.binary_cross_entropy_with_logits(
                    logits[batch["mask"]],
                    batch["target"][batch["mask"]],
                    pos_weight=pos_w
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
