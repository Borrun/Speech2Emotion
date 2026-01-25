import time
import numpy as np
import torch
import torchaudio

from .data import LogMelFeaturizer
from .postprocess import _local_peaks

class StreamKeyDetector:
    """
    输入：不断 append 的音频chunk + 当前播放进度
    输出：到点 emit(frame30)
    关键特性：
      - ahead不确定，自适应 online / two-stage / global-like
      - committed 区域不回滚
      - max 3 events
    """
    def __init__(self, model, sr=16000, fps=30, hop_sec=0.02,
                 margin_sec=0.30, min_gap_sec=0.60, confirm_sec=0.20,
                 tau_global=0.65, tau_online=0.85, max_events=3,
                 ahead_lo=0.5, ahead_hi=1.5, window_sec=6.0, device="cpu"):
        self.model = model
        self.model.eval()
        self.sr = sr
        self.fps = fps
        self.hop_sec = hop_sec
        self.step_hz = int(round(1.0 / hop_sec))
        self.margin_sec = margin_sec
        self.min_gap_sec = min_gap_sec
        self.confirm_sec = confirm_sec
        self.tau_global = tau_global
        self.tau_online = tau_online
        self.max_events = max_events
        self.ahead_lo = ahead_lo
        self.ahead_hi = ahead_hi
        self.window_sec = window_sec
        self.device = device

        self.feat = LogMelFeaturizer(sample_rate=sr, hop_sec=hop_sec).to(device)

        self.full_audio = torch.zeros(1, 0)  # [1, N]
        self.recv_samples = 0
        self.play_samples = 0

        self.score_series = np.zeros((0,), dtype=np.float32)  # per feature step
        self.planned = []   # list of dict: {frame30, score, committed, fired}
        self.fired_count = 0

    def _sec(self, samples): return samples / float(self.sr)
    def _frame30_sec(self, frame30): return frame30 / float(self.fps)
    def _sec_to_frame30(self, t): return int(round(t * self.fps))
    def _sec_to_step(self, t): return int(round(t / self.hop_sec))
    def _step_to_sec(self, k): return k / float(self.step_hz)

    def append_chunk(self, chunk_wav_1xn: torch.Tensor):
        if chunk_wav_1xn.dim() != 2 or chunk_wav_1xn.size(0) != 1:
            raise ValueError("chunk must be [1, N]")
        self.full_audio = torch.cat([self.full_audio, chunk_wav_1xn.cpu()], dim=1)
        self.recv_samples = self.full_audio.size(1)

    def set_playhead_sec(self, t_play):
        self.play_samples = int(round(t_play * self.sr))

    def _infer_update_scores(self):
        t_recv = self._sec(self.recv_samples)
        t0 = max(0.0, t_recv - self.window_sec)
        s0 = int(round(t0 * self.sr))
        wav = self.full_audio[:, s0:self.recv_samples].to(self.device)  # [1, Nwin]
        mel = self.feat(wav).unsqueeze(0)  # [1, T, M]

        with torch.no_grad():
            logits = self.model(mel)[0]   # [T]
            scores = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)

        k0 = self._sec_to_step(t0)
        need = k0 + len(scores)
        if len(self.score_series) < need:
            pad = np.zeros((need - len(self.score_series),), dtype=np.float32)
            self.score_series = np.concatenate([self.score_series, pad], axis=0)
        self.score_series[k0:need] = scores

    def _commit_before(self, t_commit):
        for p in self.planned:
            if (not p["committed"]) and self._frame30_sec(p["frame30"]) <= t_commit:
                p["committed"] = True

    def _nms_ok(self, frame30, chosen):
        t = self._frame30_sec(frame30)
        for q in chosen:
            if abs(self._frame30_sec(q["frame30"]) - t) < self.min_gap_sec:
                return False
        return True

    def _plan_future(self, tau, t_play, t_recv):
        if self.fired_count >= self.max_events:
            return

        t_start = t_play + self.margin_sec
        t_end = t_recv
        k_start = self._sec_to_step(t_start)
        k_end = self._sec_to_step(t_end)
        k_start = max(1, k_start)
        k_end = min(len(self.score_series)-1, k_end)
        if k_end <= k_start + 2:
            return

        seg = self.score_series  # global
        peaks = []
        # local peaks in [k_start, k_end)
        for i in range(k_start, k_end):
            s = float(seg[i])
            if s < tau: 
                continue
            if seg[i] >= seg[i-1] and seg[i] >= seg[i+1]:
                peaks.append((i, s))
        peaks.sort(key=lambda x: x[1], reverse=True)
        peaks = peaks[:20]

        committed = [p for p in self.planned if p["committed"]]
        flexible = [p for p in self.planned if not p["committed"]]

        candidates = []
        for k, s in peaks:
            t = self._step_to_sec(k)
            f = self._sec_to_frame30(t)
            candidates.append({"frame30": f, "score": s, "committed": False, "fired": False})

        pool = flexible + candidates
        pool.sort(key=lambda p: p["score"], reverse=True)

        new_flex = []
        for p in pool:
            if len(committed) + len(new_flex) >= self.max_events:
                break
            if not self._nms_ok(p["frame30"], committed + new_flex):
                continue
            new_flex.append(p)

        self.planned = committed + new_flex
        self.planned.sort(key=lambda p: p["frame30"])

    def _emit_due(self, t_play):
        emitted = []
        for p in self.planned:
            if self.fired_count >= self.max_events:
                break
            if p["fired"] or (not p["committed"]):
                continue
            if self._frame30_sec(p["frame30"]) <= t_play:
                p["fired"] = True
                self.fired_count += 1
                emitted.append((p["frame30"], p["score"]))
        return emitted

    def update(self):
        """
        Call periodically (e.g., every 50~100ms or when chunk arrives).
        Return list of emitted (frame30, score)
        """
        t_recv = self._sec(self.recv_samples)
        t_play = self._sec(self.play_samples)
        t_ahead = t_recv - t_play

        if t_ahead >= self.ahead_hi:
            tau = self.tau_global
        elif t_ahead >= self.ahead_lo:
            tau = self.tau_global
        else:
            tau = self.tau_online

        self._infer_update_scores()

        t_commit = t_play + self.margin_sec
        self._commit_before(t_commit)
        self._plan_future(tau, t_play, t_recv)
        emitted = self._emit_due(t_play)
        return emitted
