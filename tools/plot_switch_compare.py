import argparse
import json
import os
from typing import Dict, List, Tuple

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
        ty = str(p.get("type", "calm")).strip().lower() or "calm"
        v = float(clamp(p.get("value", 60.0), 0.0, 150.0))
        out.append({"t": t, "type": ty, "value": v})
    out.sort(key=lambda x: x["t"])

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
        return curve[0]["type"], level_from_value(curve[0]["value"])
    for i in range(len(curve) - 1):
        if curve[i]["t"] <= t < curve[i + 1]["t"]:
            return curve[i]["type"], level_from_value(curve[i]["value"])
    return curve[-1]["type"], level_from_value(curve[-1]["value"])


def gt_boundaries_from_label(label_obj: Dict, fps: int, n_frames_hint: int) -> List[int]:
    duration = float(label_obj.get("duration", 0.0) or 0.0)
    curve = normalize_curve(label_obj.get("curve", []), duration=max(duration, 1e-6))
    n_frames = max(1, int(n_frames_hint))
    bnd = []
    prev = None
    for i in range(n_frames):
        t = i / float(fps)
        st = state_at(curve, t)
        if prev is not None and st != prev:
            bnd.append(i)
        prev = st
    return bnd


def match_tags(pred: List[int], gt: List[int], tol: int):
    pred = sorted(int(x) for x in pred)
    gt = sorted(int(x) for x in gt)
    used = [False] * len(gt)
    tp = []
    fp = []
    for p in pred:
        best_j = -1
        best_d = 10**9
        for j, g in enumerate(gt):
            if used[j]:
                continue
            d = abs(p - g)
            if d <= tol and d < best_d:
                best_j = j
                best_d = d
        if best_j >= 0:
            used[best_j] = True
            tp.append(p)
        else:
            fp.append(p)
    fn = [g for j, g in enumerate(gt) if not used[j]]
    return tp, fp, fn


def load_labels(path: str) -> Dict[str, Dict]:
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            wav = obj.get("wav")
            if wav:
                out[str(wav)] = obj
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", default="./outputs/emotion_codes")
    ap.add_argument("--label_path", default="./annotater/labels_new.jsonl")
    ap.add_argument("--wav", required=True, help="e.g. utt_0001.wav")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--tol", type=int, default=5)
    ap.add_argument("--out", default="", help="png path, default: ./outputs/plots/<wav>.png")
    args = ap.parse_args()

    labels = load_labels(args.label_path)
    if args.wav not in labels:
        raise RuntimeError(f"wav not in labels: {args.wav}")

    pred_json = os.path.join(args.pred_dir, args.wav.replace(".wav", ".json"))
    if not os.path.isfile(pred_json):
        raise RuntimeError(f"prediction json not found: {pred_json}")

    with open(pred_json, "r", encoding="utf-8") as f:
        pred = json.load(f)

    frames = pred.get("frames", [])
    if not frames:
        raise RuntimeError("prediction has no frames")
    boundary_p = [float(x.get("boundary_p", 0.0)) for x in frames]
    pred_switch = [int(x) for x in pred.get("switch_frames", [])]

    gt_switch = gt_boundaries_from_label(labels[args.wav], fps=int(args.fps), n_frames_hint=len(frames))
    tp, fp, fn = match_tags(pred_switch, gt_switch, tol=int(args.tol))

    t = [i / float(args.fps) for i in range(len(frames))]
    thr_on = float(pred.get("switch_params", {}).get("thr_on", 0.74))
    thr_off = float(pred.get("switch_params", {}).get("thr_off", 0.50))

    out_path = args.out.strip()
    if not out_path:
        out_dir = "./outputs/plots"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, args.wav.replace(".wav", ".png"))
    else:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # prefer matplotlib if available; otherwise fallback to pure SVG output.
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(14, 5))
        ax.plot(t, boundary_p, color="#1f77b4", linewidth=1.3, label="boundary_p")
        ax.axhline(thr_on, color="#ff7f0e", linestyle="--", linewidth=1.0, label=f"thr_on={thr_on:.2f}")
        ax.axhline(thr_off, color="#bcbd22", linestyle="--", linewidth=1.0, label=f"thr_off={thr_off:.2f}")

        for i, fidx in enumerate(gt_switch):
            ax.axvline(fidx / float(args.fps), color="#2ca02c", alpha=0.25, linewidth=1.3, label="GT switch" if i == 0 else None)
        for i, fidx in enumerate(tp):
            ax.axvline(fidx / float(args.fps), color="#17becf", alpha=0.55, linewidth=1.6, label="Pred TP" if i == 0 else None)
        for i, fidx in enumerate(fp):
            ax.axvline(fidx / float(args.fps), color="#d62728", alpha=0.6, linewidth=1.4, label="Pred FP" if i == 0 else None)
        for i, fidx in enumerate(fn):
            ax.axvline(fidx / float(args.fps), color="#9467bd", alpha=0.55, linewidth=1.4, label="GT Miss(FN)" if i == 0 else None)

        ax.set_title(
            f"{args.wav} | TP={len(tp)} FP={len(fp)} FN={len(fn)} | "
            f"P={len(tp)/(len(tp)+len(fp)+1e-8):.3f} "
            f"R={len(tp)/(len(tp)+len(fn)+1e-8):.3f}"
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Boundary Probability")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.2)
        ax.legend(loc="upper right")
        plt.tight_layout()
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
    except Exception:
        # fallback: write simple SVG that requires no third-party package
        if out_path.lower().endswith(".png"):
            out_path = out_path[:-4] + ".svg"
        w, h = 1400, 440
        ml, mr, mt, mb = 70, 30, 50, 60
        pw = w - ml - mr
        ph = h - mt - mb
        t_end = max(1e-6, t[-1] if t else 1.0)

        def x_of(ts: float) -> float:
            return ml + pw * (ts / t_end)

        def y_of(p: float) -> float:
            p = max(0.0, min(1.0, p))
            return mt + ph * (1.0 - p)

        pts = " ".join(f"{x_of(tt):.2f},{y_of(pp):.2f}" for tt, pp in zip(t, boundary_p))
        lines = []
        lines.append(f'<line x1="{ml}" y1="{y_of(thr_on):.2f}" x2="{ml+pw}" y2="{y_of(thr_on):.2f}" stroke="#ff7f0e" stroke-width="1.2" stroke-dasharray="6 4"/>')
        lines.append(f'<line x1="{ml}" y1="{y_of(thr_off):.2f}" x2="{ml+pw}" y2="{y_of(thr_off):.2f}" stroke="#bcbd22" stroke-width="1.2" stroke-dasharray="6 4"/>')
        for fi in gt_switch:
            xx = x_of(fi / float(args.fps))
            lines.append(f'<line x1="{xx:.2f}" y1="{mt}" x2="{xx:.2f}" y2="{mt+ph}" stroke="#2ca02c" stroke-width="1.0" opacity="0.35"/>')
        for fi in tp:
            xx = x_of(fi / float(args.fps))
            lines.append(f'<line x1="{xx:.2f}" y1="{mt}" x2="{xx:.2f}" y2="{mt+ph}" stroke="#17becf" stroke-width="1.4" opacity="0.75"/>')
        for fi in fp:
            xx = x_of(fi / float(args.fps))
            lines.append(f'<line x1="{xx:.2f}" y1="{mt}" x2="{xx:.2f}" y2="{mt+ph}" stroke="#d62728" stroke-width="1.4" opacity="0.75"/>')
        for fi in fn:
            xx = x_of(fi / float(args.fps))
            lines.append(f'<line x1="{xx:.2f}" y1="{mt}" x2="{xx:.2f}" y2="{mt+ph}" stroke="#9467bd" stroke-width="1.3" opacity="0.7"/>')

        title = (
            f"{args.wav} | TP={len(tp)} FP={len(fp)} FN={len(fn)} | "
            f"P={len(tp)/(len(tp)+len(fp)+1e-8):.3f} "
            f"R={len(tp)/(len(tp)+len(fn)+1e-8):.3f}"
        )
        svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">
<rect x="0" y="0" width="{w}" height="{h}" fill="white"/>
<text x="{ml}" y="26" font-size="16" font-family="Arial">{title}</text>
<rect x="{ml}" y="{mt}" width="{pw}" height="{ph}" fill="none" stroke="#777" stroke-width="1"/>
<polyline fill="none" stroke="#1f77b4" stroke-width="1.3" points="{pts}"/>
{''.join(lines)}
<text x="{ml}" y="{h-20}" font-size="12" font-family="Arial">Blue=boundary_p, Green=GT, Cyan=TP, Red=FP, Purple=FN, Orange/Olive=thresholds</text>
</svg>
"""
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(svg)

    print("wrote:", out_path)


if __name__ == "__main__":
    main()
