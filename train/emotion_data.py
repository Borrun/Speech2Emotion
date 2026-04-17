import os
import json
import random
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
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


# ---------------------------------------------------------------------------
# 情感亲缘距离表（基于唤醒度 / 效价维度）
# 值域 [0, 1]：0 = 完全相同，1 = 完全对立
# 仅定义基础 5 类之间的距离，confused 变体继承对应基础类的距离
# ---------------------------------------------------------------------------
#                   happy  sad   angry  fear  calm
_BASE_DISTANCE = [
    [0.0,  0.9,  0.7,  0.8,  0.5],   # happy
    [0.9,  0.0,  0.6,  0.5,  0.4],   # sad
    [0.7,  0.6,  0.0,  0.4,  0.8],   # angry
    [0.8,  0.5,  0.4,  0.0,  0.7],   # fear
    [0.5,  0.4,  0.8,  0.7,  0.0],   # calm
]


class AffinityAwareLoss(nn.Module):
    """
    混淆矩阵感知的 type 分类损失。

    将 hard one-hot 标签松弛为软标签，概率按类别间语义距离分配：
      - 兄弟类（如 angry ↔ angry_confused）：分配 sibling_share
      - 情感近邻（如 angry ↔ fear）：按距离反比分配
      - 远距离类（如 angry ↔ calm）：几乎不分配

    相比均匀 label_smoothing 的优势：
      标准 smoothing：所有非目标类均分 → 不区分 angry_confused 和 calm 的错误代价
      本方案：语义近的类多分 → "错得近"比"错得远"惩罚小
    """

    def __init__(
        self,
        num_classes: int = 10,
        sibling_share: float = 0.08,
        base_smooth: float = 0.02,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        soft = self._build_soft_targets(num_classes, sibling_share, base_smooth)
        self.register_buffer("soft_targets", soft)

    @staticmethod
    def _build_soft_targets(
        num_classes: int, sibling_share: float, base_smooth: float
    ) -> torch.Tensor:
        """构建 [num_classes, num_classes] 的软标签查找表。"""
        n_base = 5
        dist = torch.tensor(_BASE_DISTANCE, dtype=torch.float32)

        targets = torch.zeros(num_classes, num_classes)
        for c in range(num_classes):
            base_c = c % n_base
            is_confused_c = c >= n_base

            # 兄弟类（base ↔ confused 变体）
            sibling = (c + n_base) % (2 * n_base)
            targets[c, sibling] = sibling_share

            # 其他类按距离反比分配
            for j in range(num_classes):
                if j == c or j == sibling:
                    continue
                base_j = j % n_base
                d = dist[base_c, base_j]
                # 同为 confused 或同为 base → 稍近；跨组 → 稍远
                is_confused_j = j >= n_base
                if is_confused_c == is_confused_j:
                    d = d * 0.9  # 同组内稍近
                # 亲缘度 = (1 - distance)，归一化后乘以 base_smooth
                targets[c, j] = (1.0 - d) * base_smooth

            # 主类别：剩余概率
            targets[c, c] = 0.0  # 先置零
            targets[c, c] = 1.0 - targets[c].sum()

        return targets

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits:  [N, C] 或 [B, T, C]
        targets: [N] 或 [B, T]，值域 0~C-1 或 ignore_index
        """
        if logits.dim() == 3:
            B, T, C = logits.shape
            logits = logits.reshape(-1, C)
            targets = targets.reshape(-1)
        elif logits.dim() != 2:
            raise ValueError(f"expected 2D or 3D logits, got {logits.dim()}D")

        valid = targets != self.ignore_index
        if not valid.any():
            return logits.sum() * 0.0

        logits = logits[valid]
        targets = targets[valid]

        soft = self.soft_targets[targets]  # [N_valid, C]
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(soft * log_probs).sum(dim=-1).mean()
        return loss

    def extra_repr(self) -> str:
        return f"ignore_index={self.ignore_index}"


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
    注意：必须用采样率值（如 16000）而非样本长度，否则 torchaudio 会分配巨型 filter 导致 OOM。
    """
    if abs(factor - 1.0) < 1e-6:
        return wav
    return torchaudio.functional.resample(wav, int(sr * factor), sr)


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
