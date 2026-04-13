import argparse
import csv
import json
import math
import os
import sys
from collections import Counter
from typing import Dict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from infer.window_infer import (
    StreamingWindowedAcousticAdapter,
    load_wav,
    samples_to_ready_frame_count,
)
from online_emotion import DetectorConfig
from online_emotion.runtime_utils import align_text_to_frames, chunk_text_and_features
from online_emotion.timeline_runtime import (
    EventStatus,
    FrameStatus,
    PredJsonAcousticAdapter,
    SegmentState,
    TimelineRuntime,
    TimelineRuntimeConfig,
)


def load_text_map(csv_path: str) -> Dict[str, str]:
    out = {}
    if not csv_path or (not os.path.isfile(csv_path)):
        return out
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f)
        for row in rd:
            wav = str(row.get("wav", "") or "").strip()
            utt = str(row.get("utt_id", "") or "").strip()
            txt = str(row.get("transcription", "") or "").strip()
            if wav and txt:
                out[wav] = txt
            if utt and txt:
                out[utt + ".wav"] = txt
    return out


def label_name(type_id: int, level_id: int, type_map) -> Dict[str, object]:
    tid = int(type_id)
    lid = int(level_id)
    name = ""
    if 0 <= tid < len(type_map):
        name = str(type_map[tid])
    return {"type_id": tid, "type": name, "level_id": lid}


def find_segment(segments, frame_idx: int):
    for seg in segments:
        if int(seg.start_frame) <= int(frame_idx) <= int(seg.end_frame):
            return seg
    return None


def build_stream_online_compat(
    runtime: TimelineRuntime,
    wav_name: str,
    text: str,
    cfg: TimelineRuntimeConfig,
    chunk_ranges,
    mode: str,
    type_map,
):
    total_frames = max(0, int(runtime.timeline.inferred_end) + 1)
    boundary_events = sorted(int(ev.frame_idx) for ev in runtime.boundary_events)
    tokens = align_text_to_frames(text=text, total_frames=max(1, total_frames), anchors=boundary_events)

    frames = []
    for frame_idx in range(total_frames):
        rec = runtime.timeline.at(frame_idx)
        label = label_name(rec.emotion_id, rec.level_id, type_map)
        frames.append(
            {
                "frame": int(frame_idx),
                "t_sec": float(frame_idx) / float(max(1, cfg.fps)),
                "raw": dict(label),
                "stable_causal": dict(label),
                "stable": dict(label),
                "segment_id": int(rec.segment_id),
                "boundary_event": bool(rec.boundary_flag),
            }
        )

    chunks = []
    segments = list(runtime.segments)
    for chunk_start, chunk_end in chunk_ranges:
        if chunk_end < chunk_start or total_frames <= 0:
            continue
        curf = max(0, min(total_frames - 1, int(chunk_end)))
        seg = find_segment(segments, curf)
        prev_major = {"type_id": -1, "type": "", "level_id": -1}
        if seg is not None:
            prev_seg = None
            for cand in segments:
                if int(cand.segment_id) == int(seg.segment_id) - 1:
                    prev_seg = cand
                    break
            if prev_seg is not None:
                prev_major = label_name(prev_seg.major_emotion_id, prev_seg.major_level_id, type_map)
        tinfo = chunk_text_and_features(tokens=tokens, s=int(chunk_start), e=int(chunk_end) + 1)
        chunks.append(
            {
                "chunk_start": int(chunk_start),
                "chunk_end": int(chunk_end),
                "chunk_frames": int(chunk_end) - int(chunk_start) + 1,
                "boundary_events": [int(x) for x in boundary_events if int(chunk_start) <= int(x) <= int(chunk_end)],
                "current_frame": frames[curf] if frames else None,
                "segment": {
                    "id": int(seg.segment_id) if seg is not None else -1,
                    "start_frame": int(seg.start_frame) if seg is not None else -1,
                    "major": (
                        label_name(seg.major_emotion_id, seg.major_level_id, type_map)
                        if seg is not None
                        else {"type_id": -1, "type": "", "level_id": -1}
                    ),
                    "prev_major": prev_major,
                },
                **tinfo,
            }
        )

    final_seg = segments[-1] if segments else None
    chunk_lengths = [int(e) - int(s) + 1 for s, e in chunk_ranges if int(e) >= int(s)]
    return {
        "wav": wav_name,
        "fps": int(cfg.fps),
        "n_frames": int(total_frames),
        "text": str(text or ""),
        "stream_mode": {
            "text_once": True,
            "audio_streaming": True,
            "chunk_min": min(chunk_lengths) if chunk_lengths else 0,
            "chunk_max": max(chunk_lengths) if chunk_lengths else 0,
            "seed": 0,
            "emotion_smooth_win": int(cfg.smooth_win),
            "emotion_hysteresis": int(cfg.emo_hysteresis),
            "future_lookahead": int(cfg.stable_right_frames),
            "text_fusion": {
                "dynamic_w_text": bool(cfg.dynamic_w_text),
                "w_text_max": float(cfg.w_text_max),
                "text_emotion": bool(cfg.text_emotion),
                "sentiment_thr": float(cfg.sentiment_thr),
                "text_conf_thr": float(cfg.text_conf_thr),
                "text_blend_w": float(cfg.text_blend_w),
            },
            "boundary_cfg": {
                "w_audio": float(cfg.detector_cfg.w_audio),
                "w_text": float(cfg.detector_cfg.w_text),
                "thr_on": float(cfg.detector_cfg.thr_on),
                "thr_off": float(cfg.detector_cfg.thr_off),
                "confirm_win": int(cfg.detector_cfg.confirm_win),
                "min_gap": int(cfg.detector_cfg.min_gap),
            },
            "timeline_runtime_mode": str(mode),
        },
        "boundary_events": boundary_events,
        "chunks": chunks,
        "final_segment": {
            "id": int(final_seg.segment_id) if final_seg is not None else -1,
            "start_frame": int(final_seg.start_frame) if final_seg is not None else -1,
            "major": (
                label_name(final_seg.major_emotion_id, final_seg.major_level_id, type_map)
                if final_seg is not None
                else {"type_id": -1, "type": "", "level_id": -1}
            ),
        },
        "frames": frames,
        "tokens": tokens,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_json", default="")
    ap.add_argument("--wav", default="")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--text", default="")
    ap.add_argument("--text_csv", default="./outputs/test_transcriptions_from_audio_chain.csv")
    ap.add_argument("--ingest_ratio", type=float, default=1.8, help="ingest speed / playback speed")
    ap.add_argument("--wall_tick_ms", type=float, default=33.333)
    ap.add_argument("--infer_tick_ms", type=int, default=300)
    ap.add_argument("--recompute_left_frames", type=int, default=120)
    ap.add_argument("--stable_right_frames", type=int, default=42)
    ap.add_argument("--history_keep_frames", type=int, default=900)
    ap.add_argument("--revise_keep_frames", type=int, default=300)
    ap.add_argument("--w_audio", type=float, default=0.8)
    ap.add_argument("--w_text", type=float, default=0.2)
    ap.add_argument("--thr_on", type=float, default=0.62)
    ap.add_argument("--thr_off", type=float, default=0.42)
    ap.add_argument("--confirm_win", type=int, default=3)
    ap.add_argument("--min_gap", type=int, default=5)
    ap.add_argument("--smooth_win", type=int, default=5)
    ap.add_argument("--emo_hysteresis", type=int, default=3)
    ap.add_argument("--text_emotion", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--compat_out", default="")
    ap.add_argument("--compat_manifest", default="")
    args = ap.parse_args()

    pred_json = str(args.pred_json or "").strip()
    wav_path = str(args.wav or "").strip()
    ckpt_path = str(args.ckpt or "").strip()
    if not pred_json and not (wav_path and ckpt_path):
        raise SystemExit("need either --pred_json or both --wav and --ckpt")
    if pred_json and (wav_path or ckpt_path):
        raise SystemExit("use either --pred_json or --wav/--ckpt, not both")

    if pred_json:
        adapter = PredJsonAcousticAdapter.from_path(pred_json)
        with open(pred_json, "r", encoding="utf-8") as f:
            pred = json.load(f)
        mode = "pred_json"
        wav_name = str(pred.get("wav", "") or "")
        total_frames = len(pred.get("frames", []))
        total_samples = int(round(float(pred.get("duration", 0.0)) * float(adapter.sample_rate)))
        sample_rate = int(pred.get("sample_rate", adapter.sample_rate))
        fps = int(pred.get("fps", adapter.fps))
    else:
        adapter = StreamingWindowedAcousticAdapter(ckpt_path=ckpt_path, device=args.device)
        source_wav = load_wav(wav_path, sr=adapter.sample_rate)
        pred = {
            "wav": os.path.basename(wav_path),
            "sample_rate": int(adapter.sample_rate),
            "fps": int(adapter.fps),
            "duration": float(source_wav.size(1)) / float(adapter.sample_rate),
            "frames": [
                None
            ] * int(samples_to_ready_frame_count(int(source_wav.size(1)), int(adapter.n_fft), int(adapter.hop_length))),
        }
        mode = "wav_ckpt_streaming"
        wav_name = os.path.basename(wav_path)
        total_frames = int(len(pred["frames"]))
        total_samples = int(source_wav.size(1))
        sample_rate = int(adapter.sample_rate)
        fps = int(adapter.fps)

    text = str(args.text or "").strip()
    if not text:
        text_map = load_text_map(args.text_csv)
        text = text_map.get(wav_name, "")

    cfg = TimelineRuntimeConfig(
        sample_rate=sample_rate,
        fps=fps,
        infer_tick_ms=int(args.infer_tick_ms),
        recompute_left_frames=int(args.recompute_left_frames),
        stable_right_frames=int(args.stable_right_frames),
        history_keep_frames=int(args.history_keep_frames),
        revise_keep_frames=int(args.revise_keep_frames),
        smooth_win=int(args.smooth_win),
        emo_hysteresis=int(args.emo_hysteresis),
        text_emotion=bool(args.text_emotion),
        detector_cfg=DetectorConfig(
            fps=fps,
            w_audio=float(args.w_audio),
            w_text=float(args.w_text),
            thr_on=float(args.thr_on),
            thr_off=float(args.thr_off),
            confirm_win=int(args.confirm_win),
            min_gap=int(args.min_gap),
        ),
    )
    runtime = TimelineRuntime(acoustic_adapter=adapter, text=text, cfg=cfg)

    wall_tick_sec = float(args.wall_tick_ms) / 1000.0
    infer_tick_sec = float(args.infer_tick_ms) / 1000.0

    now_sec = 0.0
    next_infer_sec = 0.0
    sent_samples = 0
    last_play_frame = -1
    playback_sources = Counter()
    rhythm_sources = Counter()
    control_sources = Counter()
    trace = []
    chunk_ranges = []

    max_loops = max(1, int((float(pred.get("duration", 0.0)) + 10.0) / max(1e-3, wall_tick_sec))) * 4
    loops = 0
    while loops < max_loops:
        loops += 1

        target_samples = min(
            total_samples,
            int(math.floor(now_sec * float(args.ingest_ratio) * float(cfg.sample_rate))),
        )
        if target_samples > sent_samples:
            if pred_json:
                runtime.update_ingest_samples(target_samples - sent_samples, recv_ts_sec=now_sec)
            else:
                chunk = source_wav[:, sent_samples:target_samples]
                runtime.update_ingest_pcm(chunk, recv_ts_sec=now_sec, sample_rate=sample_rate)
            sent_samples = target_samples
            if sent_samples >= total_samples:
                runtime.mark_end_of_stream()

        while next_infer_sec <= now_sec + 1e-9:
            fused = runtime.run_infer_tick(now_sec=now_sec)
            if fused is not None:
                chunk_ranges.append((int(fused.frame_begin), int(fused.frame_end)))
            runtime.advance_commit_line()
            next_infer_sec += infer_tick_sec

        if runtime.end_of_stream and runtime.timeline.inferred_end >= total_frames - 1:
            runtime.advance_commit_line(force_flush=True)

        play_frame = min(total_frames - 1, int(math.floor(now_sec * float(cfg.fps))))
        if play_frame >= 0 and play_frame != last_play_frame:
            pv = runtime.get_playback_view(play_frame)
            rv = runtime.get_rhythm_view(play_frame)
            cv = runtime.get_control_view(play_frame)
            playback_sources[pv.source] += 1
            rhythm_sources[rv.source] += 1
            control_sources[cv.source] += 1
            if len(trace) < 40:
                trace.append(
                    {
                        "frame_idx": int(play_frame),
                        "playback": {
                            "source": pv.source,
                            "emotion_id": pv.emotion_id,
                            "level_id": pv.level_id,
                            "segment_id": pv.segment_id,
                            "is_boundary": pv.is_boundary,
                        },
                        "rhythm": {
                            "source": rv.source,
                            "emotion_id": rv.emotion_id,
                            "level_id": rv.level_id,
                            "boundary_flag": rv.boundary_flag,
                        },
                        "control": {
                            "source": cv.source,
                            "emotion_id": cv.emotion_id,
                            "level_id": cv.level_id,
                            "segment_id": cv.segment_id,
                            "segment_major_emotion": cv.segment_major_emotion,
                        },
                    }
                )
            last_play_frame = play_frame

        if (
            runtime.end_of_stream
            and sent_samples >= total_samples
            and runtime.timeline.committed_end >= total_frames - 1
            and last_play_frame >= total_frames - 1
        ):
            break

        now_sec += wall_tick_sec

    result = {
        "mode": mode,
        "pred_json": os.path.abspath(pred_json) if pred_json else "",
        "wav": wav_name,
        "ckpt": os.path.abspath(ckpt_path) if ckpt_path else "",
        "text": text,
        "config": {
            "ingest_ratio": float(args.ingest_ratio),
            "wall_tick_ms": float(args.wall_tick_ms),
            "infer_tick_ms": int(args.infer_tick_ms),
            "recompute_left_frames": int(args.recompute_left_frames),
            "stable_right_frames": int(args.stable_right_frames),
            "device": str(args.device),
        },
        "runtime": runtime.summary(),
        "fallback_counts": {
            "playback": dict(playback_sources),
            "rhythm": dict(rhythm_sources),
            "control": dict(control_sources),
        },
        "boundary_events": [
            {
                "frame_idx": int(ev.frame_idx),
                "status": int(ev.status),
                "confidence": float(ev.confidence),
                "left_segment_id": int(ev.left_segment_id),
                "right_segment_id": int(ev.right_segment_id),
            }
            for ev in runtime.boundary_events
        ],
        "segments": [
            {
                "segment_id": int(seg.segment_id),
                "start_frame": int(seg.start_frame),
                "end_frame": int(seg.end_frame),
                "end_closed": bool(seg.end_closed),
                "status": int(seg.status),
                "major_emotion_id": int(seg.major_emotion_id),
                "major_level_id": int(seg.major_level_id),
                "frame_count": int(seg.frame_count),
                "confidence": float(seg.confidence),
            }
            for seg in runtime.segments
        ],
        "trace": trace,
    }

    compat_out = str(args.compat_out or "").strip()
    if compat_out:
        type_map = tuple(getattr(adapter, "type_map", ()) or ())
        compat_obj = build_stream_online_compat(
            runtime=runtime,
            wav_name=wav_name,
            text=text,
            cfg=cfg,
            chunk_ranges=chunk_ranges,
            mode=mode,
            type_map=type_map,
        )
        compat_dir = os.path.dirname(compat_out)
        if compat_dir:
            os.makedirs(compat_dir, exist_ok=True)
        with open(compat_out, "w", encoding="utf-8") as f:
            json.dump(compat_obj, f, ensure_ascii=False, indent=2)
        compat_manifest = str(args.compat_manifest or "").strip()
        if compat_manifest:
            manifest_dir = os.path.dirname(compat_manifest)
            if manifest_dir:
                os.makedirs(manifest_dir, exist_ok=True)
            manifest_obj = {"files": [os.path.basename(compat_out)]}
            with open(compat_manifest, "w", encoding="utf-8") as f:
                json.dump(manifest_obj, f, ensure_ascii=False, indent=2)

    out = str(args.out or "").strip()
    if out:
        out_dir = os.path.dirname(out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print("wrote:", out)
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
