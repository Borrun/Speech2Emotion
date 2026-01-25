import numpy as np

def _local_peaks(scores, tau):
    """Find local maxima above tau. Return list of (idx, score) sorted by score desc."""
    peaks = []
    n = len(scores)
    if n < 3:
        return peaks
    for i in range(1, n - 1):
        if scores[i] >= tau and scores[i] >= scores[i - 1] and scores[i] >= scores[i + 1]:
            peaks.append((i, float(scores[i])))
    peaks.sort(key=lambda x: x[1], reverse=True)
    return peaks

def select_key_frames_from_scores(
    scores,
    step_hz,
    fps=30,
    max_events=3,
    min_gap_sec=0.8,
    dynamic=True,
    tau=0.30,            # fallback if dynamic=False
    tau_min=0.20,        # floor threshold
    tau_frac_of_max=0.75 # dynamic threshold: tau=max(tau_min, frac*max)
):
    """
    Convert per-step scores -> <=max_events key frames (30 fps).
    - Peak pick (local maxima)
    - NMS via min_gap_sec
    - Dynamic thresholding to handle score scale variation

    Returns:
      key_frames: list[int]
      chosen: list[(frame30, score)]
    """
    scores = np.asarray(scores, dtype=np.float32)
    if scores.size < 3:
        return [], []

    if dynamic:
        smax = float(scores.max())
        tau_used = max(float(tau_min), float(tau_frac_of_max) * smax)
    else:
        tau_used = float(tau)

    peaks = _local_peaks(scores, tau_used)

    def step_to_sec(k): return k / float(step_hz)
    def step_to_frame30(k): return int(round(step_to_sec(k) * float(fps)))

    chosen = []
    for k, s in peaks:
        t = step_to_sec(k)
        ok = True
        for (kf, _) in chosen:
            if abs((kf / float(fps)) - t) < float(min_gap_sec):
                ok = False
                break
        if ok:
            chosen.append((step_to_frame30(k), s))
        if len(chosen) >= int(max_events):
            break

    chosen.sort(key=lambda x: x[0])
    key_frames = [kf for (kf, _) in chosen]
    return key_frames, chosen
