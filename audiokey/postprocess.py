import numpy as np

def _local_peaks(scores, tau):
    # scores: 1D numpy
    peaks = []
    for i in range(1, len(scores)-1):
        if scores[i] >= tau and scores[i] >= scores[i-1] and scores[i] >= scores[i+1]:
            peaks.append((i, float(scores[i])))
    peaks.sort(key=lambda x: x[1], reverse=True)
    return peaks

def select_key_frames_from_scores(
    scores,                # 1D array over steps
    step_hz,               # e.g. 50 for hop=0.02
    fps=30,
    tau=0.65,
    max_events=3,
    min_gap_sec=0.60
):
    peaks = _local_peaks(scores, tau)
    chosen = []

    def sec_from_step(k): return k / float(step_hz)
    def frame30_from_step(k): return int(round(sec_from_step(k) * fps))

    for k, s in peaks:
        t = sec_from_step(k)
        ok = True
        for (kf, _) in chosen:
            if abs((kf / float(fps)) - t) < min_gap_sec:
                ok = False
                break
        if ok:
            chosen.append((frame30_from_step(k), s))
        if len(chosen) >= max_events:
            break

    chosen.sort(key=lambda x: x[0])
    key_frames = [kf for (kf, _) in chosen]
    return key_frames, chosen
