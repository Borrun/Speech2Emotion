# Speech2Emotion (Discrete Emotion Codes for TTS Audio)

This repo turns external TTS-generated 16kHz wav into a 30fps discrete emotion code stream:
(type_id, level_id), where:
- type_id in 6 classes: happy/sad/angry/fear/calm/confused
- level_id in 0..5

## Folder layout
- wavs/: input wavs (16kHz)
- annotater/: labeling UI + labels.jsonl (supervision)
- train/: training scripts
- models/: model + featurizer
- infer/: offline inference to generate emotion_codes
- runtime/: playback sync + IPC stubs
- outputs/: checkpoints + generated emotion codes

## Setup
Python deps:
- torch
- torchaudio
- numpy

Example:
pip install torch torchaudio numpy

## Step 1: Label
cd annotater
python app.py
Open UI, label wavs, produces annotater/labels.jsonl

## Step 2: Train
From repo root:

python train/train_emotion_tcn.py \
  --wav_dir ./wavs \
  --label_path ./annotater/labels.jsonl \
  --out_dir ./outputs/ckpt

## Step 3: Offline infer (recommended for TTS)
python infer/batch_infer.py \
  --wav_dir ./wavs \
  --ckpt ./outputs/ckpt/best.pt \
  --out_dir ./outputs/emotion_codes

## Step 4: Runtime sync
python runtime/player_sync.py \
  --wav ./wavs/utt_0001.wav \
  --code ./outputs/emotion_codes/utt_0001.json

This prints 30fps codes at the correct timestamps (replace print with your local actuator calls).

## Notes
- Feature hop = 1/30 sec and STFT center=False to preserve causality.
- With tiny dataset (~40 wavs), start with strong regularization and early stopping.
