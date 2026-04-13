import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch
import torchaudio

from models.featurizer import CausalLogMelFeaturizer
from models.model_emotion_tcn import EmotionTCN


DEFAULT_TYPE_MAP = (
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
)


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


def samples_to_ready_frame_count(total_samples: int, n_fft: int, hop_length: int) -> int:
    if int(total_samples) < int(n_fft):
        return 0
    return 1 + (int(total_samples) - int(n_fft)) // int(hop_length)


def resolve_type_map(mcfg: Dict, n_types: int) -> Tuple[str, ...]:
    allowed = list(mcfg.get("allowed_types", []) or [])
    if len(allowed) >= int(n_types):
        return tuple(str(x) for x in allowed[: int(n_types)])

    fallback = list(DEFAULT_TYPE_MAP)
    if len(fallback) < int(n_types):
        fallback.extend(f"type_{i}" for i in range(len(fallback), int(n_types)))
    return tuple(fallback[: int(n_types)])


def normalize_audio_tensor(wav: torch.Tensor) -> torch.Tensor:
    wav = torch.as_tensor(wav)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    elif wav.dim() != 2:
        raise ValueError("audio chunk must be [N] or [C, N]")

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

    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav.contiguous()


def load_model_bundle(
    ckpt_path: str,
    device: str,
    sample_rate: int,
    n_mels: int,
    hop_sec: float,
    win_sec: float,
) -> Dict[str, object]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    mcfg = ckpt.get("cfg", {})
    n_types = int(mcfg.get("n_types", len(DEFAULT_TYPE_MAP)))
    n_levels = int(mcfg.get("n_levels", 6))
    use_boundary_head = bool(mcfg.get("use_boundary_head", True))
    type_map = resolve_type_map(mcfg, n_types)

    model = EmotionTCN(
        n_mels=int(n_mels),
        channels=int(mcfg.get("channels", 128)),
        layers=int(mcfg.get("layers", 6)),
        dropout=float(mcfg.get("dropout", 0.1)),
        n_types=n_types,
        n_levels=n_levels,
        use_boundary_head=use_boundary_head,
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    featurizer = CausalLogMelFeaturizer(
        sample_rate=int(sample_rate),
        n_mels=int(n_mels),
        hop_sec=float(hop_sec),
        win_sec=float(win_sec),
    ).to(device)

    return {
        "mcfg": mcfg,
        "model": model,
        "featurizer": featurizer,
        "n_types": n_types,
        "n_levels": n_levels,
        "use_boundary_head": use_boundary_head,
        "type_map": type_map,
    }


@dataclass
class AcousticFrame:
    frame_idx: int
    emotion_id: int
    level_id: int
    boundary_prob: float
    emotion_logits: Tuple[float, ...] = ()
    level_logits: Tuple[float, ...] = ()
    emotion_conf: float = 0.0
    level_conf: float = 0.0


@dataclass
class AcousticSlice:
    frame_ctx_begin: int
    frame_begin: int
    frame_end: int
    total_frames: int
    type_map: Tuple[str, ...] = ()
    frames: List[AcousticFrame] = field(default_factory=list)


class WindowedAcousticAdapter:
    def __init__(
        self,
        wav_path: str,
        ckpt_path: str,
        device: str = "cpu",
        sample_rate: int = 16000,
        fps: int = 30,
        n_mels: int = 80,
        hop_sec: float = 1.0 / 30.0,
        win_sec: float = 0.05,
    ):
        self.wav_path = os.path.abspath(wav_path)
        self.ckpt_path = os.path.abspath(ckpt_path)
        self.device = str(device)
        self.sample_rate = int(sample_rate)
        self.fps = int(fps)
        self.n_mels = int(n_mels)
        self.hop_sec = float(hop_sec)
        self.win_sec = float(win_sec)
        self.hop_length = int(round(self.sample_rate * self.hop_sec))
        self.n_fft = int(round(self.sample_rate * self.win_sec))

        bundle = load_model_bundle(
            ckpt_path=self.ckpt_path,
            device=self.device,
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
            hop_sec=self.hop_sec,
            win_sec=self.win_sec,
        )
        self.n_types = int(bundle["n_types"])
        self.n_levels = int(bundle["n_levels"])
        self.use_boundary_head = bool(bundle["use_boundary_head"])
        self.type_map = tuple(bundle["type_map"])
        self.model = bundle["model"]
        self.featurizer = bundle["featurizer"]

        self.wav = load_wav(self.wav_path, sr=self.sample_rate)
        self.total_samples = int(self.wav.size(1))
        self.duration = float(self.total_samples) / float(max(1, self.sample_rate))
        self.total_frames = samples_to_ready_frame_count(self.total_samples, self.n_fft, self.hop_length)

    def _frame_sample_begin(self, frame_idx: int) -> int:
        return int(frame_idx) * int(self.hop_length)

    def _frame_sample_end_exclusive(self, frame_idx: int) -> int:
        return int(frame_idx) * int(self.hop_length) + int(self.n_fft)

    def _slice_wav_for_frames(self, frame_ctx_begin: int, frame_end: int) -> torch.Tensor:
        sample_begin = self._frame_sample_begin(frame_ctx_begin)
        sample_end = self._frame_sample_end_exclusive(frame_end)
        return self.wav[:, sample_begin:sample_end].contiguous()

    @torch.no_grad()
    def infer_window(self, frame_ctx_begin: int, frame_write_begin: int, frame_write_end: int) -> AcousticSlice:
        if self.total_frames <= 0:
            return AcousticSlice(
                frame_ctx_begin=int(frame_ctx_begin),
                frame_begin=int(frame_write_begin),
                frame_end=int(frame_write_begin) - 1,
                total_frames=0,
                type_map=self.type_map,
                frames=[],
            )

        ctx_begin = max(0, int(frame_ctx_begin))
        write_begin = max(0, int(frame_write_begin))
        write_end = min(int(frame_write_end), int(self.total_frames) - 1)
        if write_begin > write_end:
            return AcousticSlice(
                frame_ctx_begin=ctx_begin,
                frame_begin=write_begin,
                frame_end=write_begin - 1,
                total_frames=int(self.total_frames),
                type_map=self.type_map,
                frames=[],
            )

        ctx_begin = min(ctx_begin, write_begin)
        wav_slice = self._slice_wav_for_frames(ctx_begin, write_end).to(self.device)
        mel = self.featurizer(wav_slice).unsqueeze(0)
        out = self.model(mel)
        type_logits = out["type"][0]
        level_logits = out["lvl"][0]
        bnd_logits = out["bnd"][0] if out["bnd"] is not None else None

        type_prob = torch.softmax(type_logits, dim=-1)
        level_prob = torch.softmax(level_logits, dim=-1)
        frames: List[AcousticFrame] = []

        for frame_idx in range(write_begin, write_end + 1):
            local_idx = int(frame_idx) - int(ctx_begin)
            type_row = type_logits[local_idx]
            level_row = level_logits[local_idx]
            emotion_id = int(type_row.argmax().item())
            level_id = int(level_row.argmax().item())
            boundary_prob = float(torch.sigmoid(bnd_logits[local_idx]).item()) if bnd_logits is not None else 0.0
            frames.append(
                AcousticFrame(
                    frame_idx=int(frame_idx),
                    emotion_id=emotion_id,
                    level_id=level_id,
                    boundary_prob=boundary_prob,
                    emotion_logits=tuple(float(x) for x in type_row.detach().cpu().tolist()),
                    level_logits=tuple(float(x) for x in level_row.detach().cpu().tolist()),
                    emotion_conf=float(type_prob[local_idx, emotion_id].item()),
                    level_conf=float(level_prob[local_idx, level_id].item()),
                )
            )

        return AcousticSlice(
            frame_ctx_begin=ctx_begin,
            frame_begin=write_begin,
            frame_end=write_end,
            total_frames=int(self.total_frames),
            type_map=self.type_map,
            frames=frames,
        )

    def infer_all(self) -> AcousticSlice:
        return self.infer_window(
            frame_ctx_begin=0,
            frame_write_begin=0,
            frame_write_end=max(-1, int(self.total_frames) - 1),
        )


class StreamingWindowedAcousticAdapter:
    def __init__(
        self,
        ckpt_path: str,
        device: str = "cpu",
        sample_rate: int = 16000,
        fps: int = 30,
        n_mels: int = 80,
        hop_sec: float = 1.0 / 30.0,
        win_sec: float = 0.05,
    ):
        self.ckpt_path = os.path.abspath(ckpt_path)
        self.device = str(device)
        self.sample_rate = int(sample_rate)
        self.fps = int(fps)
        self.n_mels = int(n_mels)
        self.hop_sec = float(hop_sec)
        self.win_sec = float(win_sec)
        self.hop_length = int(round(self.sample_rate * self.hop_sec))
        self.n_fft = int(round(self.sample_rate * self.win_sec))

        bundle = load_model_bundle(
            ckpt_path=self.ckpt_path,
            device=self.device,
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
            hop_sec=self.hop_sec,
            win_sec=self.win_sec,
        )
        self.n_types = int(bundle["n_types"])
        self.n_levels = int(bundle["n_levels"])
        self.use_boundary_head = bool(bundle["use_boundary_head"])
        self.type_map = tuple(bundle["type_map"])
        self.model = bundle["model"]
        self.featurizer = bundle["featurizer"]

        self.received_wav = torch.zeros((1, 0), dtype=torch.float32)
        self.total_samples = 0
        self.total_frames = 0
        self.duration = 0.0
        self.end_of_stream = False

    def append_wav_chunk(self, wav_chunk: torch.Tensor, sample_rate: int = None) -> int:
        if wav_chunk is None:
            return 0
        wav = normalize_audio_tensor(wav_chunk)
        src_rate = int(sample_rate or self.sample_rate)
        if src_rate != int(self.sample_rate):
            wav = torchaudio.functional.resample(wav, src_rate, int(self.sample_rate))
        if wav.numel() <= 0 or int(wav.size(1)) <= 0:
            return 0

        self.received_wav = torch.cat([self.received_wav, wav.cpu()], dim=1)
        added = int(wav.size(1))
        self.total_samples = int(self.received_wav.size(1))
        self.total_frames = samples_to_ready_frame_count(self.total_samples, self.n_fft, self.hop_length)
        self.duration = float(self.total_samples) / float(max(1, self.sample_rate))
        return added

    def mark_end_of_stream(self) -> None:
        self.end_of_stream = True

    def _slice_wav_for_frames(self, frame_ctx_begin: int, frame_end: int) -> torch.Tensor:
        sample_begin = int(frame_ctx_begin) * int(self.hop_length)
        sample_end = int(frame_end) * int(self.hop_length) + int(self.n_fft)
        return self.received_wav[:, sample_begin:sample_end].contiguous()

    @torch.no_grad()
    def infer_window(self, frame_ctx_begin: int, frame_write_begin: int, frame_write_end: int) -> AcousticSlice:
        if self.total_frames <= 0:
            return AcousticSlice(
                frame_ctx_begin=int(frame_ctx_begin),
                frame_begin=int(frame_write_begin),
                frame_end=int(frame_write_begin) - 1,
                total_frames=0,
                type_map=self.type_map,
                frames=[],
            )

        ctx_begin = max(0, int(frame_ctx_begin))
        write_begin = max(0, int(frame_write_begin))
        write_end = min(int(frame_write_end), int(self.total_frames) - 1)
        if write_begin > write_end:
            return AcousticSlice(
                frame_ctx_begin=ctx_begin,
                frame_begin=write_begin,
                frame_end=write_begin - 1,
                total_frames=int(self.total_frames),
                type_map=self.type_map,
                frames=[],
            )

        ctx_begin = min(ctx_begin, write_begin)
        wav_slice = self._slice_wav_for_frames(ctx_begin, write_end).to(self.device)
        mel = self.featurizer(wav_slice).unsqueeze(0)
        out = self.model(mel)
        type_logits = out["type"][0]
        level_logits = out["lvl"][0]
        bnd_logits = out["bnd"][0] if out["bnd"] is not None else None

        type_prob = torch.softmax(type_logits, dim=-1)
        level_prob = torch.softmax(level_logits, dim=-1)
        frames: List[AcousticFrame] = []

        for frame_idx in range(write_begin, write_end + 1):
            local_idx = int(frame_idx) - int(ctx_begin)
            type_row = type_logits[local_idx]
            level_row = level_logits[local_idx]
            emotion_id = int(type_row.argmax().item())
            level_id = int(level_row.argmax().item())
            boundary_prob = float(torch.sigmoid(bnd_logits[local_idx]).item()) if bnd_logits is not None else 0.0
            frames.append(
                AcousticFrame(
                    frame_idx=int(frame_idx),
                    emotion_id=emotion_id,
                    level_id=level_id,
                    boundary_prob=boundary_prob,
                    emotion_logits=tuple(float(x) for x in type_row.detach().cpu().tolist()),
                    level_logits=tuple(float(x) for x in level_row.detach().cpu().tolist()),
                    emotion_conf=float(type_prob[local_idx, emotion_id].item()),
                    level_conf=float(level_prob[local_idx, level_id].item()),
                )
            )

        return AcousticSlice(
            frame_ctx_begin=ctx_begin,
            frame_begin=write_begin,
            frame_end=write_end,
            total_frames=int(self.total_frames),
            type_map=self.type_map,
            frames=frames,
        )
