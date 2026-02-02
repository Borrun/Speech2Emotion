import os
import argparse
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# allow imports when running as script from repo root
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from train.emotion_data import DataConfig, EmotionSeqDataset, collate, ALLOWED_TYPES
from models.model_emotion_tcn import EmotionTCN


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_indices(n: int, val_ratio: float, seed: int):
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_val = max(1, int(round(n * val_ratio)))
    return idx[n_val:], idx[:n_val]


@torch.no_grad()
def evaluate(model, loader, device: str):
    """
    Returns:
      acc_type, acc_lvl, f1_bnd (best over threshold sweep),
      thr_bnd, p_bnd, r_bnd,
      loss_type, loss_lvl, loss_bnd
    """
    model.eval()
    ce = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
    bce = nn.BCEWithLogitsLoss(reduction="sum")

    tot_frames = 0
    correct_type = 0
    correct_lvl = 0
    loss_type = 0.0
    loss_lvl = 0.0
    loss_bnd = 0.0

    # collect boundary probabilities & GT to sweep thresholds
    bnd_probs_all = []
    bnd_gt_all = []

    for batch in loader:
        mel = batch["mel"].to(device)
        y_type = batch["y_type"].to(device)
        y_lvl = batch["y_lvl"].to(device)
        y_bnd = batch["y_bnd"].to(device)
        mask = batch["mask"].to(device)

        out = model(mel)
        type_logits = out["type"]   # [B,T,n_types]
        lvl_logits = out["lvl"]     # [B,T,n_levels]
        bnd_logits = out["bnd"]     # [B,T] or None

        n_types = int(type_logits.size(-1))
        n_levels = int(lvl_logits.size(-1))

        flat_mask = mask.view(-1)

        loss_type += float(ce(type_logits.reshape(-1, n_types), y_type.view(-1)).item())
        loss_lvl += float(ce(lvl_logits.reshape(-1, n_levels), y_lvl.view(-1)).item())
        if bnd_logits is not None:
            loss_bnd += float(bce(bnd_logits.view(-1)[flat_mask], y_bnd.view(-1)[flat_mask]).item())

        pred_type = type_logits.argmax(dim=-1)
        pred_lvl = lvl_logits.argmax(dim=-1)

        tot_frames += int(mask.sum().item())
        correct_type += int(((pred_type == y_type) & mask).sum().item())
        correct_lvl += int(((pred_lvl == y_lvl) & mask).sum().item())

        if bnd_logits is not None:
            prob = torch.sigmoid(bnd_logits)[mask].detach().float().cpu()
            gt = (y_bnd > 0.5)[mask].detach().bool().cpu()
            bnd_probs_all.append(prob)
            bnd_gt_all.append(gt)

    best_f1 = 0.0
    best_thr = 0.5
    best_p = 0.0
    best_r = 0.0

    if len(bnd_probs_all) > 0:
        probs = torch.cat(bnd_probs_all, dim=0)  # [N]
        gt = torch.cat(bnd_gt_all, dim=0)        # [N] bool

        # sweep thresholds: 0.10..0.90 step 0.02
        for thr_i in range(10, 91, 2):
            thr = thr_i / 100.0
            pred = probs > thr

            tp = int((pred & gt).sum().item())
            fp = int((pred & ~gt).sum().item())
            fn = int((~pred & gt).sum().item())

            p = tp / (tp + fp + 1e-8)
            r = tp / (tp + fn + 1e-8)
            f1 = 2 * p * r / (p + r + 1e-8)

            if f1 > best_f1:
                best_f1 = f1
                best_thr = thr
                best_p = p
                best_r = r

    return {
        "acc_type": correct_type / max(1, tot_frames),
        "acc_lvl": correct_lvl / max(1, tot_frames),
        "f1_bnd": best_f1,
        "thr_bnd": best_thr,
        "p_bnd": best_p,
        "r_bnd": best_r,
        "loss_type": loss_type / max(1, tot_frames),
        "loss_lvl": loss_lvl / max(1, tot_frames),
        "loss_bnd": loss_bnd / max(1, tot_frames),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav_dir", required=True)
    ap.add_argument("--label_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--use_boundary_head", action="store_true")
    ap.add_argument("--no_boundary_head", action="store_true")
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--w_type", type=float, default=1.0)
    ap.add_argument("--w_lvl", type=float, default=0.7)
    ap.add_argument("--w_bnd", type=float, default=0.3)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    # ✅ 新增：pos_weight 上限，防止边界 FP 爆炸（你现在就是 70+ 的典型）
    ap.add_argument("--pos_weight_cap", type=float, default=20.0)

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    use_bnd = True
    if args.no_boundary_head:
        use_bnd = False
    if args.use_boundary_head:
        use_bnd = True

    cfg = DataConfig(wav_dir=args.wav_dir, label_path=args.label_path)
    ds = EmotionSeqDataset(cfg)
    tr_idx, val_idx = split_indices(len(ds), args.val_ratio, args.seed)

    tr_ds = torch.utils.data.Subset(ds, tr_idx)
    val_ds = torch.utils.data.Subset(ds, val_idx)

    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate)

    n_types = len(ALLOWED_TYPES)   # 7
    n_levels = 6                   # 0..5

    model = EmotionTCN(
        n_mels=cfg.n_mels,
        channels=args.channels,
        layers=args.layers,
        dropout=args.dropout,
        n_types=n_types,
        n_levels=n_levels,
        use_boundary_head=use_bnd,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ce_type = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=args.label_smoothing)
    ce_lvl = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=args.label_smoothing)

    # boundary imbalance with clip
    pos_ratio = float(ds.bnd_pos_ratio)
    raw_pos_weight = (1.0 - pos_ratio) / max(pos_ratio, 1e-6)
    clipped_pos_weight = min(raw_pos_weight, float(args.pos_weight_cap))
    pos_weight = torch.tensor([clipped_pos_weight], device=device)
    bce_bnd = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print(f"[bnd] pos_ratio={pos_ratio:.6f} raw_pos_weight={raw_pos_weight:.2f} cap={args.pos_weight_cap:.2f} used_pos_weight={clipped_pos_weight:.2f}")

    best = -1.0

    for ep in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        nb = 0

        for batch in tr_loader:
            mel = batch["mel"].to(device)
            y_type = batch["y_type"].to(device)
            y_lvl = batch["y_lvl"].to(device)
            y_bnd = batch["y_bnd"].to(device)
            mask = batch["mask"].to(device)

            out = model(mel)
            type_logits = out["type"]  # [B,T,n_types]
            lvl_logits = out["lvl"]    # [B,T,n_levels]
            bnd_logits = out["bnd"]    # [B,T] or None

            loss_type = ce_type(type_logits.reshape(-1, n_types), y_type.view(-1))
            loss_lvl = ce_lvl(lvl_logits.reshape(-1, n_levels), y_lvl.view(-1))

            loss = args.w_type * loss_type + args.w_lvl * loss_lvl

            if bnd_logits is not None:
                flat_mask = mask.view(-1)
                loss_bnd = bce_bnd(bnd_logits.view(-1)[flat_mask], y_bnd.view(-1)[flat_mask])
                loss = loss + args.w_bnd * loss_bnd

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            running += float(loss.item())
            nb += 1

        val = evaluate(model, val_loader, device)
        score = val["acc_type"] + val["acc_lvl"] + (val["f1_bnd"] if use_bnd else 0.0)

        if score > best:
            best = score
            ckpt = {
                "epoch": ep,
                "model": model.state_dict(),
                "cfg": {
                    "wav_dir": args.wav_dir,
                    "label_path": args.label_path,
                    "channels": args.channels,
                    "layers": args.layers,
                    "dropout": args.dropout,
                    "use_boundary_head": use_bnd,
                    "n_types": n_types,
                    "n_levels": n_levels,
                    "allowed_types": list(ALLOWED_TYPES),
                    "pos_weight_cap": float(args.pos_weight_cap),
                    "pos_weight_used": float(clipped_pos_weight),
                },
                "pos_ratio": float(pos_ratio),
                "raw_pos_weight": float(raw_pos_weight),
            }
            torch.save(ckpt, os.path.join(args.out_dir, "best.pt"))

        print(
            f"[ep {ep:03d}] train_loss={running/max(1,nb):.4f} "
            f"val_acc_type={val['acc_type']:.3f} val_acc_lvl={val['acc_lvl']:.3f} "
            f"val_f1_bnd={val['f1_bnd']:.3f} thr={val.get('thr_bnd', 0.5):.2f} best={best:.3f}"
        )

    print("DONE:", os.path.join(args.out_dir, "best.pt"))


if __name__ == "__main__":
    main()
