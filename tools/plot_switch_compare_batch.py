import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


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


def parse_int_list(xs: List) -> List[int]:
    return [int(x) for x in xs]


def match_counts(pred: List[int], gt: List[int], tol: int) -> Tuple[int, int, int]:
    pred = sorted(pred)
    gt = sorted(gt)
    used = [False] * len(gt)
    tp = 0
    fp = 0
    for p in pred:
        best = -1
        best_d = 10**9
        for j, g in enumerate(gt):
            if used[j]:
                continue
            d = abs(p - g)
            if d <= tol and d < best_d:
                best = j
                best_d = d
        if best >= 0:
            used[best] = True
            tp += 1
        else:
            fp += 1
    fn = len(gt) - tp
    return tp, fp, fn


def gt_boundaries_from_curve(label_obj: Dict, fps: int, n_frames: int) -> List[int]:
    # lightweight: use curve change points as boundary approximation
    curve = label_obj.get("curve", [])
    out = []
    if not isinstance(curve, list):
        return out
    for i in range(1, len(curve) - 1):
        t = float(curve[i].get("t", 0.0))
        fi = int(round(t * fps))
        fi = max(1, min(n_frames - 1, fi))
        out.append(fi)
    # dedup
    out = sorted(set(out))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", default="./outputs/emotion_codes")
    ap.add_argument("--label_path", default="./annotater/labels_new.jsonl")
    ap.add_argument("--plot_dir", default="./outputs/plots")
    ap.add_argument("--tol", type=int, default=5)
    ap.add_argument("--worst_k", type=int, default=8)
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    labels = load_labels(args.label_path)
    pred_files = [x for x in os.listdir(args.pred_dir) if x.lower().endswith(".json")]
    pred_files.sort()
    if not pred_files:
        raise RuntimeError(f"no prediction json in {args.pred_dir}")

    rows = []
    for fn in pred_files:
        wav = fn.replace(".json", ".wav")
        if wav not in labels:
            continue
        p = os.path.join(args.pred_dir, fn)
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
        frames = obj.get("frames", [])
        if not frames:
            continue
        pred_sw = parse_int_list(obj.get("switch_frames", []))
        gt_sw = gt_boundaries_from_curve(labels[wav], fps=30, n_frames=len(frames))
        tp, fp, fnc = match_counts(pred_sw, gt_sw, tol=int(args.tol))
        f1 = 2 * tp / (2 * tp + fp + fnc + 1e-8)
        rows.append((wav, f1, tp, fp, fnc))

    rows.sort(key=lambda x: x[1])  # worst first
    picked = rows[: max(1, int(args.worst_k))]
    os.makedirs(args.plot_dir, exist_ok=True)

    script = os.path.join(os.path.dirname(__file__), "plot_switch_compare.py")
    for wav, f1, tp, fp, fnc in picked:
        out_png = os.path.join(args.plot_dir, wav.replace(".wav", ".svg"))
        cmd = [
            args.python,
            script,
            "--pred_dir",
            args.pred_dir,
            "--label_path",
            args.label_path,
            "--wav",
            wav,
            "--tol",
            str(args.tol),
            "--out",
            out_png,
        ]
        subprocess.run(cmd, check=True)
        print(f"[plot] {wav} f1={f1:.3f} tp={tp} fp={fp} fn={fnc} -> {out_png}")


if __name__ == "__main__":
    main()
