import torch

def frames30_to_steps(frames30, fps=30, hop_sec=0.02):
    steps = []
    for f in frames30:
        t = f / float(fps)
        k = int(round(t / hop_sec))
        steps.append(k)
    return steps

def build_sparse_targets(T, key_frames_30, fps=30, hop_sec=0.02, max_events=3):
    """
    Sparse one-hot targets on feature steps.
    For each keyframe, mark the nearest feature step as 1.0; others 0.0.
    """
    y = torch.zeros(T, dtype=torch.float32)
    steps = frames30_to_steps(key_frames_30, fps=fps, hop_sec=hop_sec)

    for k in steps[:max_events]:
        k = int(k)
        if k < 0:
            continue
        if k >= T:
            continue
        y[k] = 1.0
    return y
