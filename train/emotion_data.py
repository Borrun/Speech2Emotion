import os
import json
import random
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torchaudio

# allow "from models..." when running as: python train/train_emotion_tcn.py
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.featurizer import CausalLogMelFeaturizer


# 10 类：5 个基础情绪 + 5 个 *_confused 复合情绪
ALLOWED_TYPES = [
    "happy",
    "sad",
    "angry",
    "fear",
    "calm",
    "happy_confused",
    "sad_confused",
    "angry_confused",
    "fear_confused",
    "calm_confused",
]
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


@dataclass
class AugConfig:
    # 变速：从列表中随机选一个 factor，每个以 p_speed 概率施加
    speed_factors: List[float] = field(default_factory=lambda: [0.9, 0.95, 1.05, 1.1])
    p_speed: float = 0.5
    # 加噪：以 p_noise 概率施加高斯白噪，SNR 从 [snr_min, snr_max] 均匀采样
    p_noise: float = 0.5
    snr_min_db: float = 15.0
    snr_max_db: float = 30.0


def speed_perturb(wav: torch.Tensor, sr: int, factor: float) -> torch.Tensor:
    """
    变速：factor > 1 加速（时长变短），factor < 1 减速（时长变长）。
    原理：将 wav 视为以 sr*factor 采样，重采样回 sr。
    """
    if abs(factor - 1.0) < 1e-6:
        return wav
    orig_len = wav.size(1)
    new_len = max(1, int(round(orig_len / factor)))
    return torchaudio.functional.resample(wav, orig_len, new_len)


def add_noise(wav: torch.Tensor, snr_db: float) -> torch.Tensor:
    """加高斯白噪，按 SNR(dB) 控制强度。"""
    signal_rms = wav.pow(2).mean().sqrt().clamp(min=1e-9)
    noise_rms = signal_rms / (10.0 ** (snr_db / 20.0))
    noise = torch.randn_like(wav) * noise_rms
    return wav + noise


def clamp(x: float, lo: float, hi: float) -> float:
    try:
        x = float(x)
    except Exception:
        x = lo
    return max(lo, min(hi, x))


def load_wav(path: str, sr: int = 16000) -> torch.Tensor:
    wav = None
    sample_rate = None

    try:
        wav, sample_rate = torchaudio.load(path)
    except Exception:
        try:
            import soundfile as sf

            data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
            wav = torch.from_numpy(data.T)
        except Exception:
            try:
                from scipy.io import wavfile
                from scipy.io.wavfile import WavFileWarning

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", WavFileWarning)
                    sample_rate, data = wavfile.read(path)
                wav = torch.from_numpy(data)
                if wav.dim() == 1:
                    wav = wav.unsqueeze(0)
                else:
                    wav = wav.transpose(0, 1)
                if wav.dtype.is_floating_point:
                    wav = wav.to(torch.float32)
                elif wav.dtype == torch.uint8:
                    wav = (wav.to(torch.float32) - 128.0) / 128.0
                elif wav.dtype == torch.int16:
                    wav = wav.to(torch.float32) / 32768.0
                elif wav.dtype == torch.int32:
                    wav = wav.to(torch.float32) / 2147483648.0
                else:
                    wav = wav.to(torch.float32)
            except Exception:
                import wave

                with wave.open(path, "rb") as wf:
                    sample_rate = int(wf.getframerate())
                    n_channels = int(wf.getnchannels())
                    sampwidth = int(wf.getsampwidth())
                    n_frames = int(wf.getnframes())
                    pcm = wf.readframes(n_frames)

                if sampwidth == 1:
                    dtype = torch.uint8
                    scale = 128.0
                    offset = 128.0
                elif sampwidth == 2:
                    dtype = torch.int16
                    scale = 32768.0
                    offset = 0.0
                elif sampwidth == 4:
                    dtype = torch.int32
                    scale = 2147483648.0
                    offset = 0.0
                else:
                    raise RuntimeError(f"unsupported wav sample width: {sampwidth}")

                raw = torch.frombuffer(pcm, dtype=dtype)
                if n_channels > 1:
                    raw = raw.view(-1, n_channels).transpose(0, 1)
                else:
                    raw = raw.view(1, -1)
                wav = raw.to(torch.float32)
                if sampwidth == 1:
                    wav = (wav - offset) / scale
                else:
                    wav = wav / scale

    if wav is None or sample_rate is None:
        raise RuntimeError(f"failed to load wav: {path}")
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if int(sample_rate) != int(sr):
        wav = torchaudio.functional.resample(wav, int(sample_rate), int(sr))
    return wav.contiguous()


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


def scale_curve(curve: List[Dict[str, Any]], factor: float) -> List[Dict[str, Any]]:
    """将 curve 中所有时间戳除以 factor（配合变速使用）。"""
    return [{"t": p["t"] / factor, "type": p["type"], "value": p["value"]} for p in curve]


@dataclass
class DataConfig:
    wav_dir: str
    label_path: str
    sample_rate: int = 16000
    fps: int = 30
    win_sec: float = 0.05
    hop_sec: float = 1.0 / 30.0
    n_mels: int = 80


def load_items(label_path: str) -> List[Dict[str, Any]]:
    """从 jsonl 读取合法条目列表。"""
    items: List[Dict[str, Any]] = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            wav = obj.get("wav", None)
            curve = obj.get("curve", None)
            if not wav or not isinstance(curve, list) or len(curve) < 2:
                continue
            items.append(obj)
    return items


class EmotionSeqDataset(torch.utils.data.Dataset):
    """
    参数：
      cfg         : DataConfig
      items       : 条目列表（由外部按 train/val 切分后传入）
      aug_config  : AugConfig 或 None（None 表示不增广）
    """
    def __init__(
        self,
        cfg: DataConfig,
        items: List[Dict[str, Any]],
        aug_config: Optional[AugConfig] = None,
        estimate_bnd_ratio: bool = True,
    ):
        self.cfg = cfg
        self.items = items
        self.aug = aug_config

        self.feat = CausalLogMelFeaturizer(
            sample_rate=cfg.sample_rate,
            n_mels=cfg.n_mels,
            hop_sec=cfg.hop_sec,
            win_sec=cfg.win_sec,
        )

        self._bnd_pos_ratio = 0.02
        if estimate_bnd_ratio:
            self._estimate_boundary_ratio()

    @property
    def bnd_pos_ratio(self) -> float:
        return float(self._bnd_pos_ratio)

    def _estimate_boundary_ratio(self):
        """用 jsonl 中的 duration 字段估算边界比例，完全不加载音频。"""
        tot = 0
        pos = 0
        for obj in self.items:
            duration = obj.get("duration", None)
            if duration is None:
                continue  # 无 duration 跳过，避免 OOM
            y_type, y_lvl = sample_30fps_targets(obj.get("curve", []), float(duration), fps=self.cfg.fps)
            bnd = torch.zeros_like(y_type, dtype=torch.float32)
            bnd[1:] = ((y_type[1:] != y_type[:-1]) | (y_lvl[1:] != y_lvl[:-1])).float()
            tot += int(bnd.numel())
            pos += int(bnd.sum().item())
        if tot > 0:
            self._bnd_pos_ratio = max(1e-4, pos / float(tot))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        obj = self.items[idx]
        wav_name = obj["wav"]
        path = os.path.join(self.cfg.wav_dir, wav_name)
        wav = load_wav(path, sr=self.cfg.sample_rate)

        duration = obj.get("duration", None)
        duration = float(duration) if duration is not None else wav.size(1) / float(self.cfg.sample_rate)
        curve = obj.get("curve", [])

        # --- 增广 ---
        if self.aug is not None:
            # 变速
            if self.aug.speed_factors and random.random() < self.aug.p_speed:
                factor = random.choice(self.aug.speed_factors)
                wav = speed_perturb(wav, self.cfg.sample_rate, factor)
                duration = duration / factor
                curve = scale_curve(normalize_curve(curve, obj.get("duration", duration) * factor), factor)

            # 加噪
            if random.random() < self.aug.p_noise:
                snr = random.uniform(self.aug.snr_min_db, self.aug.snr_max_db)
                wav = add_noise(wav, snr)

        y_type, y_lvl = sample_30fps_targets(curve, duration, fps=self.cfg.fps)
        mel = self.feat(wav)  # [T, M]

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
