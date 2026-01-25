import os, json
from dataclasses import dataclass
from typing import List, Dict, Any
import torch
import torchaudio

from .targets import build_heatmap
from .augment import random_gain, add_noise, specaugment

@dataclass
class DataConfig:
    wav_dir: str
    label_path: str
    sample_rate: int = 16000
    fps: int = 30
    hop_sec: float = 0.02
    max_events: int = 3

@dataclass
class TargetConfig:
    sigma_steps: float = 2.0
    pos_weight: float = 4.0

@dataclass
class AugConfig:
    enable: bool = True
    gain_db: float = 6.0
    noise_prob: float = 0.7
    snr_db_min: int = 20
    snr_db_max: int = 35
    specaug_prob: float = 0.5
    time_mask_min: int = 5
    time_mask_max: int = 12
    freq_mask_min: int = 6
    freq_mask_max: int = 12

class LogMelFeaturizer(torch.nn.Module):
    def __init__(self, sample_rate=16000, n_mels=80, hop_sec=0.02, win_sec=0.05):
        super().__init__()
        hop_length = int(sample_rate * hop_sec)
        n_fft = int(sample_rate * win_sec)
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=40,
            f_max=min(7600, sample_rate//2 - 100),
            power=2.0,
        )

    def forward(self, wav):  # wav [1, N]
        x = self.melspec(wav)                 # [1, M, T]
        x = torch.clamp(x, min=1e-10).log()   # log-mel
        x = x.squeeze(0).transpose(0, 1)      # [T, M]
        return x

class AudioKeyDataset(torch.utils.data.Dataset):
    def __init__(self, data_cfg: DataConfig, target_cfg: TargetConfig, aug_cfg: AugConfig):
        self.data_cfg = data_cfg
        self.target_cfg = target_cfg
        self.aug_cfg = aug_cfg
        self.items = []
        with open(data_cfg.label_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self.items.append(obj)

        self.feat = LogMelFeaturizer(
            sample_rate=data_cfg.sample_rate, hop_sec=data_cfg.hop_sec
        )

    def __len__(self):
        return len(self.items)

    def _load_wav(self, wav_name: str):
        path = os.path.join(self.data_cfg.wav_dir, wav_name)
        wav, sr = torchaudio.load(path)  # [C, N]
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.data_cfg.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.data_cfg.sample_rate)
        return wav  # [1, N]

    def __getitem__(self, idx):
        obj = self.items[idx]
        wav_name = obj["wav"]
        key_frames = obj.get("key_frames", [])[:self.data_cfg.max_events]

        wav = self._load_wav(wav_name)

        # waveform-level aug (safe)
        if self.aug_cfg.enable:
            wav = random_gain(wav, self.aug_cfg.gain_db)
            import random
            if random.random() < self.aug_cfg.noise_prob:
                wav = add_noise(wav, self.aug_cfg.snr_db_min, self.aug_cfg.snr_db_max)

        mel = self.feat(wav)  # [T, M]

        # feature-level aug (safe)
        if self.aug_cfg.enable:
            import random
            if random.random() < self.aug_cfg.specaug_prob:
                mel = specaugment(
                    mel,
                    self.aug_cfg.time_mask_min, self.aug_cfg.time_mask_max,
                    self.aug_cfg.freq_mask_min, self.aug_cfg.freq_mask_max
                )

        T = mel.size(0)
        y = build_heatmap(
            T, key_frames,
            fps=self.data_cfg.fps,
            hop_sec=self.data_cfg.hop_sec,
            sigma_steps=self.target_cfg.sigma_steps
        )  # [T]

        return {
            "wav": wav_name,
            "mel": mel,        # [T, M]
            "target": y,       # [T]
            "key_frames": key_frames
        }

def collate_fn(batch: List[Dict[str, Any]]):
    # pad mel and target to max T in batch
    maxT = max(x["mel"].size(0) for x in batch)
    M = batch[0]["mel"].size(1)

    mel_pad = torch.zeros(len(batch), maxT, M)
    tgt_pad = torch.zeros(len(batch), maxT)
    attn_mask = torch.zeros(len(batch), maxT, dtype=torch.bool)

    wav_names = []
    key_frames = []

    for i, x in enumerate(batch):
        T = x["mel"].size(0)
        mel_pad[i, :T] = x["mel"]
        tgt_pad[i, :T] = x["target"]
        attn_mask[i, :T] = True
        wav_names.append(x["wav"])
        key_frames.append(x["key_frames"])

    return {
        "wav": wav_names,
        "mel": mel_pad,          # [B, T, M]
        "target": tgt_pad,       # [B, T]
        "mask": attn_mask,       # [B, T] True for valid
        "key_frames": key_frames
    }
