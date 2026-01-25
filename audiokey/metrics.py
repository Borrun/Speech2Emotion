def match_events(pred_frames, gt_frames, fps=30, tol_sec=0.15):
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
    f = 2*p*r / (p + r + 1e-8)
    return p, r, f
