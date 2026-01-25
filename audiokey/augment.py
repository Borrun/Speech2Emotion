import random
import torch
import torchaudio

def random_gain(wav, gain_db=6.0):
    g = random.uniform(-gain_db, gain_db)
    amp = 10 ** (g / 20.0)
    return wav * amp

def add_noise(wav, snr_db_min=20, snr_db_max=35):
    snr = random.uniform(snr_db_min, snr_db_max)
    noise = torch.randn_like(wav)
    # match power
    sig_pow = wav.pow(2).mean().clamp(min=1e-8)
    noi_pow = noise.pow(2).mean().clamp(min=1e-8)
    scale = (sig_pow / (noi_pow * (10 ** (snr / 10.0)))).sqrt()
    return wav + noise * scale

def specaugment(mel, time_mask_min=5, time_mask_max=12, freq_mask_min=6, freq_mask_max=12):
    # mel: [T, M]
    T, M = mel.shape
    out = mel.clone()

    # time mask
    w = random.randint(time_mask_min, min(time_mask_max, max(time_mask_min, T-1)))
    t0 = random.randint(0, max(0, T - w))
    out[t0:t0+w, :] = 0

    # freq mask
    f = random.randint(freq_mask_min, min(freq_mask_max, max(freq_mask_min, M-1)))
    f0 = random.randint(0, max(0, M - f))
    out[:, f0:f0+f] = 0
    return out
