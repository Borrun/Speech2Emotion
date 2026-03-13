import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from infer.postprocess import decode_switch_points


LEVEL_THRESHOLDS = {
    5: (111.0, 150.0),
    4: (86.0, 110.0),
    3: (51.0, 85.0),
    2: (26.0, 50.0),
    1: (11.0, 25.0),
    0: (0.0, 10.0),
}


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def norm_type(t: str) -> str:
    t = (t or "").strip().lower()
    return t if t else "calm"


def level_from_value(v: float) -> int:
    v = float(clamp(v, 0.0, 150.0))
    for lvl in (5, 4, 3, 2, 1, 0):
        lo, hi = LEVEL_THRESHOLDS[lvl]
        if lo <= v <= hi:
            return lvl
    return 0


def normalize_curve(curve, duration: float):
    out = []
    if not isinstance(curve, list):
        curve = []
    for p in curve:
        if not isinstance(p, dict):
            continue
        t = float(clamp(p.get("t", 0.0), 0.0, duration))
        ty = norm_type(p.get("type", "calm"))
        v = float(clamp(p.get("value", 60.0), 0.0, 150.0))
        out.append({"t": t, "type": ty, "value": v})
    out.sort(key=lambda x: x["t"])

    dedup = []
    for p in out:
        if dedup and abs(dedup[-1]["t"] - p["t"]) < 1e-9:
            dedup[-1] = p
        else:
            dedup.append(p)
    out = dedup

    if not out:
        out = [{"t": 0.0, "type": "calm", "value": 60.0}, {"t": duration, "type": "calm", "value": 60.0}]
    else:
        if out[0]["t"] > 0.0:
            out.insert(0, {"t": 0.0, "type": out[0]["type"], "value": out[0]["value"]})
        if out[-1]["t"] < duration:
            out.append({"t": duration, "type": out[-1]["type"], "value": out[-1]["value"]})
    out[0]["t"] = 0.0
    out[-1]["t"] = duration
    return out


def state_at(curve, t: float) -> Tuple[str, int]:
    if t <= curve[0]["t"]:
        ty = curve[0]["type"]
        lv = level_from_value(curve[0]["value"])
        return ty, lv
    for i in range(len(curve) - 1):
        if curve[i]["t"] <= t < curve[i + 1]["t"]:
            ty = curve[i]["type"]
            lv = level_from_value(curve[i]["value"])
            return ty, lv
    ty = curve[-1]["type"]
    lv = level_from_value(curve[-1]["value"])
    return ty, lv


def gt_boundaries_from_label(label_obj: Dict, fps: int, n_frames_hint: int = 0) -> List[int]:
    duration = float(label_obj.get("duration", 0.0) or 0.0)
    curve = normalize_curve(label_obj.get("curve", []), duration=max(duration, 1e-6))
    n_frames = int(round(duration * fps))
    if n_frames_hint > 0:
        n_frames = n_frames_hint
    n_frames = max(1, n_frames)

    prev = None
    bnd = []
    for i in range(n_frames):
        t = i / float(fps)
        st = state_at(curve, t)
        if prev is not None and st != prev:
            bnd.append(i)
        prev = st
    return bnd


def match_f1(pred: List[int], gt: List[int], tol: int) -> Tuple[int, int, int]:
    pred = sorted(int(x) for x in pred)
    gt = sorted(int(x) for x in gt)
    used = [False] * len(gt)

    tp = 0
    fp = 0
    for p in pred:
        best_j = -1
        best_d = 10**9
        for j, g in enumerate(gt):
            if used[j]:
                continue
            d = abs(p - g)
            if d <= tol and d < best_d:
                best_d = d
                best_j = j
        if best_j >= 0:
            used[best_j] = True
            tp += 1
        else:
            fp += 1
    fn = len(gt) - tp
    return tp, fp, fn


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


@dataclass
class Score:
    f1: float
    p: float
    r: float
    tp: int
    fp: int
    fn: int


def load_labels(label_path: str) -> Dict[str, Dict]:
    labels = {}
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            wav = obj.get("wav")
            if wav:
                labels[str(wav)] = obj
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", default="./outputs/emotion_codes")
    ap.add_argument("--label_path", default="./annotater/labels_new.jsonl")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--tol", type=int, default=5, help="Boundary match tolerance in frames")
    ap.add_argument("--thr_on", default="0.74,0.78,0.82,0.86")
    ap.add_argument("--thr_off", default="0.50,0.55,0.60,0.65")
    ap.add_argument("--confirm_win", default="2,3,4,5")
    ap.add_argument("--min_gap", default="4,5,6,7,8")
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    labels = load_labels(args.label_path)
    pred_files = [x for x in os.listdir(args.pred_dir) if x.lower().endswith(".json")]
    pred_files.sort()
    if not pred_files:
        raise RuntimeError(f"No prediction json found in: {args.pred_dir}")

    dataset = []
    for fn in pred_files:
        wav = fn[:-5] + ".wav" if fn.lower().endswith(".json") else fn
        if wav not in labels:
            continue
        path = os.path.join(args.pred_dir, fn)
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        frames = obj.get("frames", [])
        bp = []
        for fr in frames:
            if "boundary_p" in fr:
                bp.append(float(fr["boundary_p"]))
        if not bp:
            continue
        gt = gt_boundaries_from_label(labels[wav], fps=int(args.fps), n_frames_hint=len(bp))
        dataset.append((wav, bp, gt))

    if not dataset:
        raise RuntimeError("No usable data: make sure prediction json includes boundary_p and wav names match labels.")

    on_list = parse_float_list(args.thr_on)
    off_list = parse_float_list(args.thr_off)
    cw_list = parse_int_list(args.confirm_win)
    mg_list = parse_int_list(args.min_gap)

    results = []
    for on in on_list:
        for off in off_list:
            if off >= on:
                continue
            for cw in cw_list:
                for mg in mg_list:
                    TP = FP = FN = 0
                    for _, bp, gt in dataset:
                        pred = decode_switch_points(
                            bp,
                            thr_on=on,
                            thr_off=off,
                            confirm_win=cw,
                            min_gap=mg,
                        )
                        tp, fp, fn = match_f1(pred, gt, tol=int(args.tol))
                        TP += tp
                        FP += fp
                        FN += fn
                    p = TP / (TP + FP + 1e-8)
                    r = TP / (TP + FN + 1e-8)
                    f1 = 2 * p * r / (p + r + 1e-8)
                    results.append(
                        {
                            "thr_on": on,
                            "thr_off": off,
                            "confirm_win": cw,
                            "min_gap": mg,
                            "score": Score(f1=f1, p=p, r=r, tp=TP, fp=FP, fn=FN),
                        }
                    )

    results.sort(
        key=lambda x: (
            x["score"].f1,
            x["score"].p,
            x["score"].r,
        ),
        reverse=True,
    )

    print(f"[data] usable_wavs={len(dataset)} tol={args.tol}fps={args.fps}")
    print("[best]")
    b = results[0]
    print(
        f"thr_on={b['thr_on']:.2f} thr_off={b['thr_off']:.2f} "
        f"confirm_win={b['confirm_win']} min_gap={b['min_gap']} "
        f"F1={b['score'].f1:.4f} P={b['score'].p:.4f} R={b['score'].r:.4f} "
        f"(TP={b['score'].tp} FP={b['score'].fp} FN={b['score'].fn})"
    )

    print(f"[top{args.topk}]")
    for i, x in enumerate(results[: int(args.topk)], 1):
        s = x["score"]
        print(
            f"{i:02d}. on={x['thr_on']:.2f} off={x['thr_off']:.2f} "
            f"cw={x['confirm_win']} mg={x['min_gap']} "
            f"F1={s.f1:.4f} P={s.p:.4f} R={s.r:.4f}"
        )


if __name__ == "__main__":
    main()
