import torch
import torchaudio


class CausalLogMelFeaturizer(torch.nn.Module):
    """
    Causal log-mel:
      - center=False prevents future leakage
      - hop_sec is set to 1/30 to align with 30fps output
      - win_sec=0.05 provides ~50ms time span (fits 50ms latency budget)
    """
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        hop_sec: float = 1.0 / 30.0,
        win_sec: float = 0.05,
        f_min: int = 40,
        f_max: int = 7600,
    ):
        super().__init__()
        hop_length = int(round(sample_rate * float(hop_sec)))
        n_fft = int(round(sample_rate * float(win_sec)))
        f_max = min(int(f_max), sample_rate // 2 - 100)

        self.sample_rate = int(sample_rate)
        self.hop_sec = float(hop_sec)
        self.win_sec = float(win_sec)

        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=int(n_mels),
            f_min=float(f_min),
            f_max=float(f_max),
            power=2.0,
            center=False,  # IMPORTANT: causal
        )

    def forward(self, wav_1xn: torch.Tensor) -> torch.Tensor:
        """
        wav_1xn: [1, N]
        return:  [T, M]
        """
        if wav_1xn.dim() != 2 or wav_1xn.size(0) != 1:
            raise ValueError("wav must be [1, N] mono")
        x = self.melspec(wav_1xn)               # [1, M, T]
        x = torch.clamp(x, min=1e-10).log()
        x = x.squeeze(0).transpose(0, 1)        # [T, M]
        return x
