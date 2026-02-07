import os
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
import torchaudio

# allow "from models..." when running as: python train/train_emotion_tcn.py
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.featurizer import CausalLogMelFeaturizer


# ✅ 7 类：新增 action
ALLOWED_TYPES = ["happy", "sad", "angry", "fear", "calm", "confused", "action"]
TYPE2ID = {t: i for i, t in enumerate(ALLOWED_TYPES)}
ID2TYPE = {i: t for t, i in TYPE2ID.items()}

LEVEL_THRESHOLDS = {
    5: (111.0, 150.0),
    4: (86.0, 110.0),
    3: (51.0, 85.0),
    2: (26.0, 50.0),
    1: (11.0, 25.0),
    0: (0.0, 10.0),
}


def clamp(x: float, lo: float, hi: float) -> float:
    try:
        x = float(x)
    except Exception:
        x = lo
    return max(lo, min(hi, x))


def norm_type(t: str) -> str:
    t = (t or "").strip().lower()
    return t if t in TYPE2ID else "calm"


def level_from_value(v: float) -> int:
    v = float(clamp(v, 0.0, 150.0))
    for lvl in (5, 4, 3, 2, 1, 0):
        lo, hi = LEVEL_THRESHOLDS[lvl]
        if lo <= v <= hi:
            return lvl
    return 0


def normalize_curve(curve: Any, duration: float) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(curve, list):
        curve = []
    for p in curve:
        if not isinstance(p, dict):
            continue
        t = float(p.get("t", 0.0))
        ty = norm_type(p.get("type", "calm"))
        v = float(clamp(p.get("value", 60.0), 0.0, 150.0))
        t = float(clamp(t, 0.0, duration))
        out.append({"t": t, "type": ty, "value": v})

    out.sort(key=lambda x: x["t"])
    # dedup same t: keep last
    dedup: List[Dict[str, Any]] = []
    for p in out:
        if dedup and abs(dedup[-1]["t"] - p["t"]) < 1e-9:
            dedup[-1] = p
        else:
            dedup.append(p)
    out = dedup

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


def type_at(curve: List[Dict[str, Any]], t: float) -> str:
    if t <= curve[0]["t"]:
        return curve[0]["type"]
    for i in range(len(curve) - 1):
        if curve[i]["t"] <= t < curve[i + 1]["t"]:
            return curve[i]["type"]
    return curve[-1]["type"]


def value_at_step(curve: List[Dict[str, Any]], t: float) -> float:
    if t <= curve[0]["t"]:
        return float(curve[0]["value"])
    for i in range(len(curve) - 1):
        if curve[i]["t"] <= t < curve[i + 1]["t"]:
            return float(curve[i]["value"])
    return float(curve[-1]["value"])


def sample_30fps_targets(curve: Any, duration: float, fps: int = 30) -> Tuple[torch.Tensor, torch.Tensor]:
    duration = float(duration)
    fps = int(fps)
    n = max(1, int(round(duration * fps)))
    curve_n = normalize_curve(curve, duration)

    y_type = torch.empty(n, dtype=torch.long)
    y_lvl = torch.empty(n, dtype=torch.long)
    for i in range(n):
        t = i / float(fps)
        ty = type_at(curve_n, t)
        v = value_at_step(curve_n, t)
        y_type[i] = TYPE2ID.get(ty, TYPE2ID["calm"])
        y_lvl[i] = int(level_from_value(v))
    return y_type, y_lvl


@dataclass
class DataConfig:
    wav_dir: str
    label_path: str
    sample_rate: int = 16000
    fps: int = 30
    win_sec: float = 0.05
    hop_sec: float = 1.0 / 30.0
    n_mels: int = 80


class EmotionSeqDataset(torch.utils.data.Dataset):
    """
    Reads annotater/labels_new.jsonl
    returns:
      mel: [T, M]
      y_type: [T]
      y_lvl: [T]
      y_bnd: [T] 0/1 if (type or lvl) changes from previous frame
    """
    def __init__(self, cfg: DataConfig):
        self.cfg = cfg
        self.items: List[Dict[str, Any]] = []
        with open(cfg.label_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                wav = obj.get("wav", None)
                curve = obj.get("curve", None)
                if not wav or not isinstance(curve, list) or len(curve) < 2:
                    continue
                self.items.append(obj)

        self.feat = CausalLogMelFeaturizer(
            sample_rate=cfg.sample_rate,
            n_mels=cfg.n_mels,
            hop_sec=cfg.hop_sec,
            win_sec=cfg.win_sec,
        )

        self._bnd_pos_ratio = 0.02
        self._estimate_boundary_ratio()

    @property
    def bnd_pos_ratio(self) -> float:
        return float(self._bnd_pos_ratio)

    def _estimate_boundary_ratio(self):
        tot = 0
        pos = 0
        for obj in self.items:
            wav_name = obj["wav"]
            wav_path = os.path.join(self.cfg.wav_dir, wav_name)
            if not os.path.isfile(wav_path):
                continue
            wav, sr = torchaudio.load(wav_path)
            if wav.size(0) > 1:
                wav = wav.mean(dim=0, keepdim=True)
            if sr != self.cfg.sample_rate:
                wav = torchaudio.functional.resample(wav, sr, self.cfg.sample_rate)
            duration = obj.get("duration", None)
            if duration is None:
                duration = wav.size(1) / float(self.cfg.sample_rate)
            y_type, y_lvl = sample_30fps_targets(obj.get("curve", []), float(duration), fps=self.cfg.fps)
            bnd = torch.zeros_like(y_type, dtype=torch.float32)
            bnd[1:] = ((y_type[1:] != y_type[:-1]) | (y_lvl[1:] != y_lvl[:-1])).float()
            tot += int(bnd.numel())
            pos += int(bnd.sum().item())
        if tot > 0:
            self._bnd_pos_ratio = max(1e-4, pos / float(tot))

    def __len__(self):
        return len(self.items)

    def _load_wav(self, wav_name: str) -> Tuple[torch.Tensor, float]:
        path = os.path.join(self.cfg.wav_dir, wav_name)
        wav, sr = torchaudio.load(path)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.cfg.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.cfg.sample_rate)
        duration = wav.size(1) / float(self.cfg.sample_rate)
        return wav, float(duration)

    def __getitem__(self, idx):
        obj = self.items[idx]
        wav_name = obj["wav"]
        wav, dur_audio = self._load_wav(wav_name)

        duration = obj.get("duration", None)
        duration = float(duration) if duration is not None else float(dur_audio)

        y_type, y_lvl = sample_30fps_targets(obj.get("curve", []), duration, fps=self.cfg.fps)

        mel = self.feat(wav)  # [T, M]

        # align by truncation (rounding differences at tail)
        T = min(int(mel.size(0)), int(y_type.numel()))
        mel = mel[:T]
        y_type = y_type[:T]
        y_lvl = y_lvl[:T]

        y_bnd = torch.zeros(T, dtype=torch.float32)
        if T > 1:
            y_bnd[1:] = ((y_type[1:] != y_type[:-1]) | (y_lvl[1:] != y_lvl[:-1])).float()

        return {"wav": wav_name, "mel": mel, "y_type": y_type, "y_lvl": y_lvl, "y_bnd": y_bnd, "duration": duration}


def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    B = len(batch)
    maxT = max(x["mel"].size(0) for x in batch)
    M = batch[0]["mel"].size(1)

    mel = torch.zeros(B, maxT, M, dtype=torch.float32)
    y_type = torch.full((B, maxT), -100, dtype=torch.long)
    y_lvl = torch.full((B, maxT), -100, dtype=torch.long)
    y_bnd = torch.zeros(B, maxT, dtype=torch.float32)
    mask = torch.zeros(B, maxT, dtype=torch.bool)

    wavs = []
    durations = []

    for i, x in enumerate(batch):
        T = x["mel"].size(0)
        mel[i, :T] = x["mel"]
        y_type[i, :T] = x["y_type"]
        y_lvl[i, :T] = x["y_lvl"]
        y_bnd[i, :T] = x["y_bnd"]
        mask[i, :T] = True
        wavs.append(x["wav"])
        durations.append(float(x["duration"]))

    return {"wav": wavs, "mel": mel, "y_type": y_type, "y_lvl": y_lvl, "y_bnd": y_bnd, "mask": mask, "duration": durations}
