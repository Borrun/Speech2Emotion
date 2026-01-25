import yaml, json, os
import torch
import torchaudio
import numpy as np

from audiokey.data import LogMelFeaturizer
from audiokey.model_tcn import TCNScoreNet
from audiokey.postprocess import select_key_frames_from_scores

def main(cfg_path="configs/config.yaml", ckpt_path="ckpt/best.pt", wav_path="wavs/utt_0001.wav"):
    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")

    # model
    model = TCNScoreNet(
        channels=cfg["model"]["tcn_channels"],
        layers=cfg["model"]["tcn_layers"],
        dropout=cfg["model"]["dropout"]
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # load wav
    wav, sr = torchaudio.load(wav_path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != cfg["data"]["sample_rate"]:
        wav = torchaudio.functional.resample(wav, sr, cfg["data"]["sample_rate"])

    feat = LogMelFeaturizer(sample_rate=cfg["data"]["sample_rate"], hop_sec=cfg["data"]["hop_sec"])
    mel = feat(wav).unsqueeze(0).to(device)  # [1, T, M]

    with torch.no_grad():
        logits = model(mel)[0]
        scores = torch.sigmoid(logits).cpu().numpy()

    step_hz = int(round(1.0 / cfg["data"]["hop_sec"]))
    key_frames, debug = select_key_frames_from_scores(
        scores,
        step_hz=step_hz,
        fps=cfg["data"]["fps"],
        tau=cfg["post"]["tau_global"],
        max_events=cfg["post"]["max_events"],
        min_gap_sec=cfg["post"]["min_gap_sec"]
    )

    out = {"wav": os.path.basename(wav_path), "fps": cfg["data"]["fps"], "key_frames": key_frames}
    print(json.dumps(out, ensure_ascii=False))
    # debug contains (frame30, score)
    # print(debug)

if __name__ == "__main__":
    main()
