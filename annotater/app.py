import os
import json
import math
from flask import Flask, jsonify, request, send_from_directory, abort
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

AUDIO_DIR = os.environ.get("AUDIO_DIR") or str(Path("/home/bor/works/AudioKey/wavs"))
LABEL_PATH = os.path.join(BASE_DIR, "labels.jsonl")

DEFAULT_FPS = 100  # 16kHz / 160-sample hop = 100 fps
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_HOP_SAMPLES = 160  # 10ms hop at 16kHz

ALLOWED_TYPES = ["happy", "sad", "angry", "fear", "calm", "confused"]

DEFAULT_LEVEL_FRAMES = 15
LEAD_IN_FRAMES = 5

LEVEL_INTENSITY_MAPPING = {0: 5.0, 1: 18.0, 2: 38.0, 3: 68.0, 4: 98.0, 5: 130.0}
EMOTION_LEVEL_THRESHOLDS = {
    5: (111.0, 150.0), 4: (86.0, 110.0), 3: (51.0, 85.0),
    2: (26.0, 50.0),   1: (11.0, 25.0),  0: (0.0, 10.0), 
}

DECAY_LEVELS = {
    "happy": {5: [3, 0], 4: [2, 0], 3: [1, 0], 2: [0], 1: [0]},
    "angry": {5: [3, 1, 0], 4: [2, 0], 3: [1, 0], 2: [0], 1: [0]},
    "sad":   {5: [3, 1, 0], 4: [2, 0], 3: [1, 0], 2: [0], 1: [0]},
    "fear":  {5: [3, 1, 0], 4: [2, 0], 3: [1, 0], 2: [0], 1: [0]},
    "calm": {},
    "confused": {5: [0], 4: [0], 3: [0], 2: [0], 1: [0]},
}

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def level_from_value(v: float) -> int:
    v = float(clamp(v, 0, 150))
    for lvl in (5, 4, 3, 2, 1, 0):
        lo, hi = EMOTION_LEVEL_THRESHOLDS[lvl]
        if lo <= v <= hi:
            return lvl
    return 0

def snap_to_level_center(v: float) -> float:
    lvl = level_from_value(v)
    return float(LEVEL_INTENSITY_MAPPING.get(lvl, 68.0))

def normalize_curve(curve, duration: float):
    duration = float(duration)
    if not curve or len(curve) < 2:
        v = snap_to_level_center(68.0)
        return [
            {"t": 0.0, "type": "calm", "value": float(v)},
            {"t": float(duration), "type": "calm", "value": float(v)},
        ]
    curve = list(curve)
    curve.sort(key=lambda p: float(p.get("t", 0.0)))
    curve[0]["t"] = 0.0
    curve[-1]["t"] = float(duration)
    for p in curve:
        ty = str(p.get("type", "calm"))
        p["type"] = ty if ty in ALLOWED_TYPES else "calm"
        p["value"] = float(snap_to_level_center(float(p.get("value", 68.0))))
        p["t"] = float(p.get("t", 0.0))
    return curve

def type_at(curve, t: float) -> str:
    t = float(t)
    if not curve or len(curve) < 2:
        return "calm"
    for i in range(len(curve) - 1):
        if float(curve[i]["t"]) <= t < float(curve[i + 1]["t"]):
            ty = curve[i].get("type", "calm")
            return ty if ty in ALLOWED_TYPES else "calm"
    ty = curve[-2].get("type", "calm")
    return ty if ty in ALLOWED_TYPES else "calm"

def value_at_step(curve, t: float) -> float:
    t = float(t)
    if not curve or len(curve) < 2:
        return 68.0
    for i in range(len(curve) - 1):
        if float(curve[i]["t"]) <= t < float(curve[i + 1]["t"]):
            return float(curve[i].get("value", 68.0))
    return float(curve[-2].get("value", 68.0))

def load_labels():
    labels = {}
    if os.path.exists(LABEL_PATH):
        with open(LABEL_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    wav = obj.get("wav")
                    if wav:
                        labels[wav] = obj
                except Exception:
                    continue
    return labels

def upsert_label(obj):
    labels = load_labels()
    labels[obj["wav"]] = obj
    tmp = LABEL_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for _, v in labels.items():
            f.write(json.dumps(v, ensure_ascii=False) + "\n")
    os.replace(tmp, LABEL_PATH)

def sample_frames(curve, duration, fps):
    duration = float(duration)
    fps = int(fps)
    n = max(1, int(round(duration * fps)))
    curve = normalize_curve(curve, duration=duration)
    frames = []
    for i in range(n):
        t = i / fps
        ty = type_at(curve, t)
        v = value_at_step(curve, t)
        frames.append({"i": i, "t": float(t), "type": ty, "value": float(snap_to_level_center(v))})
    return frames

def curve_to_segments(curve, duration, sample_rate=DEFAULT_SAMPLE_RATE):
    duration = float(duration)
    curve = normalize_curve(curve, duration=duration)
    segs = []
    for i in range(max(0, len(curve) - 1)):
        t0 = float(curve[i]["t"])
        t1 = float(curve[i + 1]["t"])
        if t1 <= t0:
            continue
        ty = type_at(curve, t0)
        v = float(snap_to_level_center(value_at_step(curve, t0)))
        lvl = int(level_from_value(v))
        s0 = int(round(t0 * sample_rate))
        s1 = int(round(t1 * sample_rate))
        segs.append({"t0": t0, "t1": t1, "s0": s0, "s1": s1, "type": ty, "value": v, "level": lvl})
    return segs

def sample_hop_frames(curve, duration, sample_rate=DEFAULT_SAMPLE_RATE, hop_samples=DEFAULT_HOP_SAMPLES):
    duration = float(duration)
    curve = normalize_curve(curve, duration=duration)
    total_samples = int(round(duration * sample_rate))
    n = max(1, int(math.ceil(total_samples / float(hop_samples))))
    frames = []
    for i in range(n):
        s = i * hop_samples
        t = s / float(sample_rate)
        ty = type_at(curve, t)
        v = float(snap_to_level_center(value_at_step(curve, t)))
        lvl = int(level_from_value(v))
        frames.append({"i": i, "t": float(t), "sample": int(s), "type": ty, "value": v, "level": lvl})
    return frames

def transition_frames_from_curve(curve, fps: int):
    """Internal change points only (exclude endpoints)."""
    fps = int(fps)
    if not curve or len(curve) < 3:
        return []
    out = []
    for i in range(1, len(curve)-1):
        out.append(int(round(float(curve[i]["t"]) * fps)))
    return out

def transition_hop_frames_from_curve(curve, sample_rate=DEFAULT_SAMPLE_RATE, hop_samples=DEFAULT_HOP_SAMPLES):
    hop_fps = float(sample_rate) / float(hop_samples)
    if not curve or len(curve) < 3:
        return []
    out = []
    for i in range(1, len(curve)-1):
        out.append(int(round(float(curve[i]["t"]) * hop_fps)))
    return out

app = Flask(__name__)

@app.after_request
def add_no_cache_headers(resp):
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp

@app.get("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")

@app.get("/api/files")
def api_files():
    if not os.path.isdir(AUDIO_DIR):
        return jsonify({"error": f"Missing wavs dir: {AUDIO_DIR}"}), 400

    labels = load_labels()
    files = [fn for fn in os.listdir(AUDIO_DIR) if fn.lower().endswith(".wav")]
    files.sort()

    out = []
    for fn in files:
        obj = labels.get(fn, {})
        out.append({
            "wav": fn,
            "fps": int(obj.get("fps", DEFAULT_FPS)),
            "labeled": fn in labels
        })
    return jsonify(out)

@app.get("/api/label/<path:wav>")
def api_label(wav):
    labels = load_labels()
    obj = labels.get(wav, {})
    return jsonify({
        "wav": wav,
        "curve": obj.get("curve", None),
        "fps": obj.get("fps", DEFAULT_FPS),
        "duration": obj.get("duration", None),
        "text": obj.get("text", ""),
    })

@app.post("/api/save")
def api_save():
    obj = request.get_json(force=True, silent=False)
    wav = obj.get("wav")
    curve = obj.get("curve")
    fps = int(obj.get("fps", DEFAULT_FPS))
    duration = obj.get("duration", None)
    text = obj.get("text", "")

    if not wav or not isinstance(wav, str):
        return jsonify({"error": "wav required"}), 400
    if duration is None:
        return jsonify({"error": "duration required (open wav once in UI)"}), 400

    if curve is not None:
        if not isinstance(curve, list):
            return jsonify({"error": "curve must be list"}), 400
        for p in curve:
            if not isinstance(p, dict):
                return jsonify({"error": "curve points must be objects"}), 400
            t = p.get("t")
            ty = p.get("type")
            v = p.get("value")
            if t is None or v is None or ty is None:
                return jsonify({"error": "each point needs t/type/value"}), 400
            if str(ty) not in ALLOWED_TYPES:
                return jsonify({"error": f"invalid type: {ty}"}), 400

    curve = normalize_curve(curve, duration=float(duration))

    rec = {
        "wav": wav,
        "fps": fps,
        "duration": float(duration),
        "curve": curve,
        "text": str(text) if text is not None else "",
    }
    upsert_label(rec)
    return jsonify({"ok": True})

def generate_decay_curve(base_type: str, base_level: int, fps: int, duration_s: float):
    fps = int(fps)
    duration_s = float(duration_s)
    total_frames = max(1, int(round(duration_s * fps)))

    base_type = base_type if base_type in ALLOWED_TYPES else "calm"
    base_level = int(clamp(base_level, 0, 5))

    frames = []
    for _ in range(LEAD_IN_FRAMES):
        frames.append({"type": "calm", "level": 0})
    for _ in range(DEFAULT_LEVEL_FRAMES):
        frames.append({"type": base_type, "level": base_level})
    decay = DECAY_LEVELS.get(base_type, {}).get(base_level, [])
    for lvl in decay:
        for _ in range(DEFAULT_LEVEL_FRAMES):
            frames.append({"type": base_type, "level": int(lvl)})
    while len(frames) < total_frames:
        frames.append({"type": "calm", "level": 0})

    curve = []
    def add_point(frame_i, ty, lvl):
        t = frame_i / float(fps)
        curve.append({"t": float(t), "type": ty, "value": float(LEVEL_INTENSITY_MAPPING[int(lvl)])})

    prev = frames[0]
    add_point(0, prev["type"], prev["level"])
    for i in range(1, total_frames):
        cur = frames[i]
        if cur["type"] != prev["type"] or cur["level"] != prev["level"]:
            add_point(i, cur["type"], cur["level"])
            prev = cur

    curve.append({"t": float(total_frames / float(fps)), "type": prev["type"], "value": float(LEVEL_INTENSITY_MAPPING[int(prev["level"])])})
    return normalize_curve(curve, duration=float(total_frames / float(fps)))

@app.post("/api/generate")
def api_generate():
    obj = request.get_json(force=True, silent=False)
    wav = obj.get("wav")
    fps = int(obj.get("fps", DEFAULT_FPS))
    duration = obj.get("duration", None)
    base = obj.get("base", "calm")
    triggers = obj.get("triggers", [])

    if not wav:
        return jsonify({"error": "wav required"}), 400
    if duration is None:
        return jsonify({"error": "duration required"}), 400

    base_level = 3
    curve = generate_decay_curve(base_type=str(base), base_level=base_level, fps=fps, duration_s=float(duration))

    out = {"wav": wav, "fps": fps, "duration": float(duration), "base": base, "triggers": triggers, "curve": curve}
    upsert_label(out)
    return jsonify({"ok": True, "curve": curve})

@app.get("/api/export/<path:wav>")
def api_export(wav):
    labels = load_labels()
    obj = labels.get(wav)
    if not obj:
        return jsonify({"error": "No label for wav"}), 404

    duration = obj.get("duration", None)
    if duration is None:
        return jsonify({"error": "No duration saved yet (open the wav once in UI)"}), 400

    fps = int(obj.get("fps", DEFAULT_FPS))
    curve = obj.get("curve", [])
    text = obj.get("text", "")

    # UI-fps frames
    frames = sample_frames(curve, duration=float(duration), fps=fps)
    total_frames = max(1, int(round(float(duration) * fps)))
    trans_frames = transition_frames_from_curve(normalize_curve(curve, float(duration)), fps=fps)

    # 16k hop frames
    hop_frames = sample_hop_frames(curve, duration=float(duration), sample_rate=DEFAULT_SAMPLE_RATE, hop_samples=DEFAULT_HOP_SAMPLES)
    total_samples = int(round(float(duration) * DEFAULT_SAMPLE_RATE))
    total_hop_frames = max(1, int(math.ceil(total_samples / float(DEFAULT_HOP_SAMPLES))))
    trans_hop_frames = transition_hop_frames_from_curve(normalize_curve(curve, float(duration)), sample_rate=DEFAULT_SAMPLE_RATE, hop_samples=DEFAULT_HOP_SAMPLES)

    segments = curve_to_segments(curve, duration=float(duration), sample_rate=DEFAULT_SAMPLE_RATE)

    return jsonify({
        "wav": wav,
        "text": text,
        "fps": fps,
        "duration": float(duration),

        "total_frames": total_frames,
        "transition_frames": trans_frames,
        "frames": frames,

        "sample_rate": DEFAULT_SAMPLE_RATE,
        "hop_samples": DEFAULT_HOP_SAMPLES,
        "hop_fps": float(DEFAULT_SAMPLE_RATE) / float(DEFAULT_HOP_SAMPLES),
        "total_hop_frames": total_hop_frames,
        "transition_hop_frames": trans_hop_frames,
        "hop_frames": hop_frames,

        "segments": segments,
    })

@app.get("/audio/<path:filename>")
def audio(filename):
    full = os.path.join(AUDIO_DIR, filename)
    if not os.path.isfile(full):
        abort(404)
    return send_from_directory(AUDIO_DIR, filename, as_attachment=False)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=7860, debug=False, use_reloader=False)
