# Emotion Code Schema (Discrete)

## Goal
For each 16kHz wav (TTS generated), produce a 30fps discrete emotion code sequence:
(type_id, level_id)

## Definitions
- fps: 30
- type_id: int in [0..5]
- level_id: int in [0..5]

### type_id mapping (MUST MATCH annotater/app.py)
0: happy
1: sad
2: angry
3: fear
4: calm
5: confused

### level_id mapping
0..5 are discrete strength levels.

If you annotate with value in [0..150], level_id is derived by thresholds:
5: [111..150]
4: [86..110]
3: [51..85]
2: [26..50]
1: [11..25]
0: [0..10]

## Output file format
outputs/emotion_codes/<wav_id>.json

{
  "wav": "utt_0001.wav",
  "sample_rate": 16000,
  "fps": 30,
  "duration": 6.98,
  "type_map": ["happy","sad","angry","fear","calm","confused"],
  "frames": [
    {"i": 0, "t": 0.000, "type_id": 4, "level_id": 0},
    {"i": 1, "t": 0.033, "type_id": 4, "level_id": 0}
  ]
}

## Notes
- The model is trained causally (no future lookahead).
- Feature hop is 1/30 sec to align model steps with output frames.
- The annotater produces step-wise curves with 1-frame jumps; model may optionally include a boundary head for stability.
