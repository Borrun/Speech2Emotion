import os
import json
from flask import Flask, jsonify, request, send_from_directory, abort
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 改成你本机 wav 目录
AUDIO_DIR = os.environ.get("AUDIO_DIR") or str(Path("/home/borrun/Speech2Emotion/wavs"))

LABEL_PATH = os.path.join(BASE_DIR, "labels.jsonl")
DEFAULT_FPS = 30

# + action: 表示需要行为表现
ALLOWED_TYPES = ["happy", "sad", "angry", "fear", "calm", "confused", "action"]

# 6-level intensity mapping (from config.cpp)
LEVEL_INTENSITY_MAPPING = {0: 5.0, 1: 18.0, 2: 38.0, 3: 68.0, 4: 98.0, 5: 130.0}
EMOTION_LEVEL_THRESHOLDS = {
    5: (111.0, 150.0), 4: (86.0, 110.0), 3: (51.0, 85.0),
    2: (26.0, 50.0),   1: (11.0, 25.0),  0: (0.0, 10.0),
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
    return float(LEVEL_INTENSITY_MAPPING.get(lvl, 5.0))

def norm_type(t):
    t = (t or "").strip().lower()
    return t if t in ALLOWED_TYPES else "calm"

def normalize_base(base):
    """
    base: {"type": str, "value": float}
    默认 calm L0
    """
    if not isinstance(base, dict):
        base = {}
    ty = norm_type(base.get("type", "calm"))
    v = base.get("value", 5.0)
    v = float(snap_to_level_center(v))
    return {"type": ty, "value": v}

def normalize_triggers(triggers, duration=None):
    # 目前保留接口兼容，但不再用于自动生成衰变曲线
    if not isinstance(triggers, list):
        return []
    out = []
    for tr in triggers:
        if not isinstance(tr, dict):
            continue
        t = tr.get("t", None)
        ty = tr.get("type", None)
        v = tr.get("value", None)
        if t is None or ty is None or v is None:
            continue
        t = float(t)
        if duration is not None:
            t = clamp(t, 0.0, float(duration))
        out.append({"t": t, "type": norm_type(ty), "value": float(snap_to_level_center(v))})
    out.sort(key=lambda x: x["t"])
    return out

def normalize_curve(curve, duration=None):
    if not isinstance(curve, list):
        curve = []
    out = []
    for p in curve:
        if not isinstance(p, dict):
            continue
        t = p.get("t", 0.0)
        v = p.get("value", 5.0)
        ty = p.get("type", "calm")
        if duration is not None:
            t = clamp(t, 0.0, duration)
        else:
            t = clamp(t, 0.0, 1e9)
        out.append({"t": float(t), "type": norm_type(ty), "value": float(snap_to_level_center(v))})

    out.sort(key=lambda x: x["t"])

    # dedup same t: keep last
    dedup = []
    for p in out:
        if dedup and abs(dedup[-1]["t"] - p["t"]) < 1e-9:
            dedup[-1] = p
        else:
            dedup.append(p)
    out = dedup

    if duration is not None:
        if not out:
            # 默认 calm + L0（value=5）
            out = [
                {"t": 0.0, "type": "calm", "value": 5.0},
                {"t": float(duration), "type": "calm", "value": 5.0},
            ]
        # force endpoints
        out[0]["t"] = 0.0
        out[-1]["t"] = float(duration)

    return out

def type_at(curve, t: float) -> str:
    t = float(t)
    if not curve or len(curve) < 2:
        return "calm"
    for i in range(len(curve) - 1):
        if float(curve[i]["t"]) <= t < float(curve[i + 1]["t"]):
            return norm_type(curve[i].get("type", "calm"))
    return norm_type(curve[-2].get("type", "calm"))

def value_at_step(curve, t: float) -> float:
    t = float(t)
    if not curve or len(curve) < 2:
        return 5.0
    for i in range(len(curve) - 1):
        if float(curve[i]["t"]) <= t < float(curve[i + 1]["t"]):
            return float(curve[i].get("value", 5.0))
    return float(curve[-2].get("value", 5.0))

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

def transition_frames_from_curve(curve, fps: int):
    """Internal change points only (exclude endpoints)."""
    fps = int(fps)
    if not curve or len(curve) < 3:
        return []
    out = []
    for i in range(1, len(curve)-1):
        out.append(int(round(float(curve[i]["t"]) * fps)))
    return out

# ========= Removed auto-decay generation =========
def generate_curve(base, triggers, duration, default_tau=0.5):
    """
    Manual-annotation mode (no auto decay).

    This endpoint is kept for backward compatibility, but it now generates a
    constant staircase curve over the whole clip using `base` only.
    """
    duration = float(duration)
    base = normalize_base(base)

    fps = int(DEFAULT_FPS)
    dur_frames = max(1, int(round(duration * fps)))
    dur_t = dur_frames / float(fps)

    curve = [
        {"t": 0.0, "type": base["type"], "value": base["value"]},
        {"t": float(dur_t), "type": base["type"], "value": base["value"]},
    ]
    return normalize_curve(curve, duration=float(dur_t))

# ================= Flask app =================
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

@app.post("/api/generate")
def api_generate():
    # 保留接口兼容，但不再生成衰变/trigger 曲线：只生成 base 常量曲线
    data = request.get_json(force=True)
    wav = data.get("wav")
    if not wav or not isinstance(wav, str):
        return jsonify({"error": "Missing wav"}), 400

    fps = int(data.get("fps", DEFAULT_FPS))
    duration = data.get("duration", None)
    if duration is None:
        return jsonify({"error": "Missing duration"}), 400
    duration = float(duration)

    base = normalize_base(data.get("base", {}))
    triggers = normalize_triggers(data.get("triggers", []), duration=duration)
    default_tau = float(clamp(data.get("default_tau", 0.5), 0.0, 10.0))

    curve = generate_curve(base, triggers, duration, default_tau=default_tau)

    out = {
        "wav": wav,
        "fps": fps,
        "duration": duration,
        "base": base,
        "triggers": triggers,
        "curve": curve,
    }
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

    frames = sample_frames(curve, duration=float(duration), fps=fps)
    total_frames = max(1, int(round(float(duration) * fps)))
    trans_frames = transition_frames_from_curve(normalize_curve(curve, float(duration)), fps=fps)

    return jsonify({
        "wav": wav,
        "text": text,
        "fps": fps,
        "duration": float(duration),
        "total_frames": total_frames,
        "transition_frames": trans_frames,
        "frames": frames,
        "curve": normalize_curve(curve, float(duration)),
    })

@app.get("/audio/<path:filename>")
def audio(filename):
    full = os.path.join(AUDIO_DIR, filename)
    if not os.path.isfile(full):
        abort(404)
    return send_from_directory(AUDIO_DIR, filename, as_attachment=False)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=7860, debug=False, use_reloader=False)
