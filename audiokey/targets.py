import torch
import math

def frames30_to_steps(frames30, fps=30, hop_sec=0.02):
    steps = []
    for f in frames30:
        t = f / float(fps)
        k = int(round(t / hop_sec))
        steps.append(k)
    return steps

def gaussian_1d(T, k, sigma):
    t = torch.arange(T).float()
    y = torch.exp(-0.5 * ((t - float(k)) / float(sigma)) ** 2)
    y = y / (y.max() + 1e-8)  # peak=1
    return y

def build_heatmap(T, key_frames_30, fps=30, hop_sec=0.02, sigma_steps=2.0):
    """
    Return y shape [T], values in [0,1], with gaussian peaks at each key.
    Combine peaks by max (not sum) to keep target bounded and stable.
    """
    y = torch.zeros(T)
    steps = frames30_to_steps(key_frames_30, fps=fps, hop_sec=hop_sec)
    for k in steps[:3]:
        k = max(0, min(int(k), T-1))
        y = torch.maximum(y, gaussian_1d(T, k, sigma_steps))
    return y
