# Online Emotion Boundary Detection (Text One-Shot + Audio Streaming)

This module is a standalone design for your deployment shape:
- Text arrives once at session start.
- Audio arrives in chunks (10-15 frames each).
- Goal: detect emotion switch time points online.

It does not modify `annotater/labels_new.jsonl`.

## Core Idea

1. Build text-side boundary prior once (`TextPriorBuilder`).
2. Consume streaming audio boundary probabilities chunk by chunk.
3. Fuse audio + text prior with hysteresis and confirmation window.
4. Emit stable boundary events with minimal delay.
5. Adapt text-audio alignment online after each confirmed event.

## API

```python
from online_emotion import TextPriorBuilder, OnlineBoundaryDetector, DetectorConfig

prior = TextPriorBuilder(fps=30).build(
    text="完整输入文本",
    token_timing=None,     # optional
    total_sec_hint=8.0,    # optional
)

det = OnlineBoundaryDetector(
    text_prior=prior,
    cfg=DetectorConfig(
        fps=30, w_audio=0.65, w_text=0.35,
        thr_on=0.58, thr_off=0.45,
        confirm_win=4, min_gap=6,
    ),
)

# called repeatedly
result = det.process_chunk(frame_start=chunk_start_idx, p_audio_chunk=[...])
events = result.events
```

`p_audio_chunk` is boundary probability per frame, which can come from:
- your existing boundary head (`sigmoid(bnd_logits)`), or
- any other streaming model.

## Suggested Production Defaults

- `fps=30`
- `thr_on=0.58`, `thr_off=0.45`
- `confirm_win=3~5` frames
- `min_gap=6` frames (200ms)
- `w_audio:w_text = 0.65:0.35`

## Run Demo

```bash
python -m online_emotion.examples.simulate_stream
```

## Next Integration Step

Use your real model output per chunk:
- read chunk wav -> model -> `p_audio_chunk`
- feed detector
- events -> downstream emotion state machine / actuator scheduler

