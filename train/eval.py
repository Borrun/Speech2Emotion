import os
import argparse
from typing import List

import torch
from torch.utils.data import DataLoader

# allow imports when running as script from repo root
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from train.emotion_data import DataConfig, EmotionSeqDataset, collate, ALLOWED_TYPES
from models.model_emotion_tcn import EmotionTCN
from train.train_emotion_tcn import evaluate


def _boundaries_from_bndprob(probs: List[float], thr: float) -> List[int]:
    """Return boundary indices t where boundary at t (1..T-1)."""
    b = []
    for t in range(1, len(probs)):
        if probs[t] > thr:
            b.append(t)
    return b


def _enforce_min_gap(boundaries: List[int], T: int, min_len: int) -> List[int]:
    """
    Given boundary indices (sorted), enforce minimum segment length by removing
    boundaries that are too close. Greedy: keep earlier boundary, drop later if gap < min_len.
    """
    if min_len <= 1:
        return boundaries

    kept = []
    last = 0  # segment start
    for t in boundaries:
        if t - last >= min_len:
            kept.append(t)
            last = t
        else:
            # drop this boundary
            continue

    # also ensure last segment has length >= min_len by dropping the last boundary if needed
    if kept:
        if T - kept[-1] < min_len:
            kept.pop(-1)
    return kept


def _segments_from_boundaries(T: int, boundaries: List[int]):
    """Convert boundary indices into segments [s,e) covering 0..T"""
    segs = []
    s = 0
    for t in boundaries:
        segs.append((s, t))
        s = t
    segs.append((s, T))
    return segs


def _segment_label_from_logits(type_logits, lvl_logits, s: int, e: int):
    """
    type_logits: [T,n_types], lvl_logits: [T,n_levels]
    Use mean logits over the segment then argmax.
    """
    seg_type = type_logits[s:e].mean(dim=0).argmax().item()
    seg_lvl = lvl_logits[s:e].mean(dim=0).argmax().item()
    return int(seg_type), int(seg_lvl)


def _boundary_counts(pred_states, gt_states):
    T = min(len(pred_states), len(gt_states))
    if T <= 1:
        return 0, 0, 0
    tp = fp = fn = 0
    for t in range(1, T):
        p = (pred_states[t] != pred_states[t - 1])
        g = (gt_states[t] != gt_states[t - 1])
        if p and g:
            tp += 1
        elif p and (not g):
            fp += 1
        elif (not p) and g:
            fn += 1
    return tp, fp, fn


@torch.no_grad()
def eval_post_bnd_guided(model, loader, device: str, min_len: int = 5, thr: float = None):
    """
    Postprocess using boundary head:
      - boundary = sigmoid(bnd_logits) > thr
      - enforce min_len gap between boundaries
      - segment-wise label by averaging type/lvl logits
    Then compute boundary F1 against GT boundary induced by GT (type,lvl) changes.
    """
    model.eval()

    TP = FP = FN = 0

    for batch in loader:
        mel = batch["mel"].to(device)
        y_type = batch["y_type"]
        y_lvl = batch["y_lvl"]
        mask = batch["mask"]

        out = model(mel)
        type_logits = out["type"].detach().cpu()   # [B,T,n_types]
        lvl_logits = out["lvl"].detach().cpu()     # [B,T,n_levels]
        bnd_logits = out.get("bnd", None)
        if bnd_logits is None:
            continue
        bnd_prob = torch.sigmoid(bnd_logits).detach().cpu()  # [B,T]

        B = type_logits.size(0)
        for i in range(B):
            T = int(mask[i].sum().item())
            if T <= 1:
                continue

            # build GT states from labels
            gtype = y_type[i, :T].tolist()
            glvl = y_lvl[i, :T].tolist()
            gt_states = []
            for t in range(T):
                if gtype[t] == -100 or glvl[t] == -100:
                    break
                gt_states.append((int(gtype[t]), int(glvl[t])))
            if len(gt_states) <= 1:
                continue
            T = len(gt_states)

            probs = bnd_prob[i, :T].tolist()

            use_thr = float(thr) if thr is not None else 0.7
            boundaries = _boundaries_from_bndprob(probs, use_thr)
            boundaries = _enforce_min_gap(boundaries, T, min_len=min_len)
            segs = _segments_from_boundaries(T, boundaries)

            # create pred state per frame from segments
            p_states = [None] * T
            for (s, e) in segs:
                ty, lv = _segment_label_from_logits(type_logits[i, :T], lvl_logits[i, :T], s, e)
                for t in range(s, e):
                    p_states[t] = (ty, lv)

            tp, fp, fn = _boundary_counts(p_states, gt_states)
            TP += tp
            FP += fp
            FN += fn

    p = TP / (TP + FP + 1e-8)
    r = TP / (TP + FN + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)
    return float(f1), float(p), float(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav_dir", required=True)
    ap.add_argument("--label_path", required=True)
    ap.add_argument("--ckpt", required=True)

    ap.add_argument("--post_min_len", type=int, default=5)
    ap.add_argument("--post_thr", type=float, default=-1.0, help="boundary prob threshold; -1 means use thr_bnd from evaluate() if available else 0.7")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = DataConfig(wav_dir=args.wav_dir, label_path=args.label_path)
    ds = EmotionSeqDataset(cfg)
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    mcfg = ckpt.get("cfg", {})

    n_types = int(mcfg.get("n_types", len(ALLOWED_TYPES)))
    n_levels = int(mcfg.get("n_levels", 6))

    model = EmotionTCN(
        n_mels=cfg.n_mels,
        channels=int(mcfg.get("channels", 128)),
        layers=int(mcfg.get("layers", 6)),
        dropout=float(mcfg.get("dropout", 0.1)),
        n_types=n_types,
        n_levels=n_levels,
        use_boundary_head=bool(mcfg.get("use_boundary_head", True)),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)

    metrics = evaluate(model, loader, device)

    if args.post_thr >= 0:
        thr = float(args.post_thr)
    else:
        thr = float(metrics.get("thr_bnd", 0.7))

    f1_post, p_post, r_post = eval_post_bnd_guided(
        model, loader, device, min_len=int(args.post_min_len), thr=thr
    )
    metrics["post_min_len"] = int(args.post_min_len)
    metrics["post_thr"] = float(thr)
    metrics["f1_bnd_post_bnd_min5"] = f1_post
    metrics["p_bnd_post_bnd_min5"] = p_post
    metrics["r_bnd_post_bnd_min5"] = r_post

    print(metrics)


if __name__ == "__main__":
    main()
