import math
import random
from typing import List

from online_emotion import DetectorConfig, OnlineBoundaryDetector, TextPriorBuilder


def gaussian_peaks(n: int, centers: List[int], sigma: float = 2.5, noise: float = 0.08) -> List[float]:
    out = []
    s2 = sigma * sigma
    for t in range(n):
        v = 0.0
        for c in centers:
            d2 = (t - c) * (t - c)
            v += math.exp(-0.5 * d2 / s2)
        v += random.uniform(0.0, noise)
        out.append(max(0.0, min(1.0, v)))
    return out


def main():
    fps = 30
    text = "今天心情很好，但是看到结果有点失望，不过最后还是很开心。"
    total_sec_hint = 6.0

    prior = TextPriorBuilder(fps=fps).build(
        text=text,
        token_timing=None,
        total_sec_hint=total_sec_hint,
    )
    det = OnlineBoundaryDetector(
        text_prior=prior,
        cfg=DetectorConfig(
            fps=fps,
            w_audio=0.65,
            w_text=0.35,
            thr_on=0.58,
            thr_off=0.45,
            confirm_win=4,
            min_gap=6,
        ),
    )

    # synthetic audio boundary probabilities (ground-truth switch around these frames)
    T = int(round(total_sec_hint * fps))
    p_audio = gaussian_peaks(T, centers=[48, 95, 146], sigma=2.2, noise=0.06)

    # stream in chunk sizes 10~15 frames
    idx = 0
    events = []
    while idx < T:
        chunk = random.randint(10, 15)
        sub = p_audio[idx: idx + chunk]
        res = det.process_chunk(frame_start=idx, p_audio_chunk=sub)
        if res.events:
            events.extend(res.events)
            for e in res.events:
                print(
                    f"[event] frame={e.frame_idx:3d} t={e.t_sec:5.2f}s "
                    f"p_audio={e.p_audio:.3f} p_text={e.p_text:.3f} p_fused={e.p_fused:.3f}"
                )
        idx += len(sub)

    print(f"total_events={len(events)}")


if __name__ == "__main__":
    main()

