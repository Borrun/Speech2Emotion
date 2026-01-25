import yaml, time
import torch
import torchaudio

from audiokey.model_tcn import TCNScoreNet
from audiokey.stream import StreamKeyDetector

def main(cfg_path="configs/config.yaml", ckpt_path="ckpt/best.pt", wav_path="wavs/utt_0001.wav"):
    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
    device = "cuda" if (cfg["train"]["device"] == "cuda" and torch.cuda.is_available()) else "cpu"

    model = TCNScoreNet(
        channels=cfg["model"]["tcn_channels"],
        layers=cfg["model"]["tcn_layers"],
        dropout=cfg["model"]["dropout"]
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    det = StreamKeyDetector(
        model=model,
        sr=cfg["data"]["sample_rate"],
        fps=cfg["data"]["fps"],
        hop_sec=cfg["data"]["hop_sec"],
        margin_sec=cfg["post"]["margin_sec"],
        min_gap_sec=cfg["post"]["min_gap_sec"],
        confirm_sec=cfg["post"]["confirm_sec"],
        tau_global=cfg["post"]["tau_global"],
        tau_online=cfg["post"]["tau_online"],
        max_events=cfg["post"]["max_events"],
        device=device
    )

    wav, sr = torchaudio.load(wav_path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != cfg["data"]["sample_rate"]:
        wav = torchaudio.functional.resample(wav, sr, cfg["data"]["sample_rate"])

    # simulate: chunk arrive faster than playback
    chunk_sec = 0.2
    chunk_n = int(chunk_sec * cfg["data"]["sample_rate"])

    t_play = 0.0
    play_rate = 1.0

    N = wav.size(1)
    idx = 0
    while idx < N or t_play < (N / cfg["data"]["sample_rate"]):
        # push chunks quickly (e.g., 3x playback speed)
        for _ in range(3):
            if idx < N:
                det.append_chunk(wav[:, idx:idx+chunk_n])
                idx += chunk_n

        # update playhead
        det.set_playhead_sec(t_play)
        emitted = det.update()
        for f, s in emitted:
            print(f"EMIT frame={f}  score={s:.3f}")

        time.sleep(0.05)
        t_play += 0.05 * play_rate

if __name__ == "__main__":
    main()
