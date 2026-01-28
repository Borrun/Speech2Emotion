import os
import json
from flask import Flask, jsonify, request, send_from_directory, abort
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 改成你本机 wav 目录
AUDIO_DIR = os.environ.get("AUDIO_DIR") or str(Path("/home/borrun/Speech2Emotion/wavs"))

LABEL_PATH = os.path.join(BASE_DIR, "labels.jsonl")
DEFAULT_FPS = 30

ALLOWED_TYPES = ["happy", "sad", "angry", "fear", "calm", "confused"]

DEFAULT_LEVEL_FRAMES = 15
LEAD_IN_FRAMES = 5

# 6-level intensity mapping (from config.cpp)
LEVEL_INTENSITY_MAPPING = {0: 5.0, 1: 18.0, 2: 38.0, 3: 68.0, 4: 98.0, 5: 130.0}
EMOTION_LEVEL_THRESHOLDS = {
    5: (111.0, 150.0), 4: (86.0, 110.0), 3: (51.0, 85.0),
    2: (26.0, 50.0),   1: (11.0, 25.0),  0: (0.0, 10.0),
}

# decay level sequences (durations are DEFAULT_LEVEL_FRAMES each; from config.cpp decay_paths)
DECAY_LEVELS = {
    "happy": {5: [3, 0], 4: [2, 0], 3: [1, 0], 2: [0], 1: [0]},
    "angry": {5: [3, 1, 0], 4: [2, 0], 3: [1, 0], 2: [0], 1: [0]},
    "sad":   {5: [3, 1, 0], 4: [2, 0], 3: [1, 0], 2: [0], 1: [0]},
    "fear":  {5: [3, 1, 0], 4: [2, 0], 3: [1, 0], 2: [0], 1: [0]},
    # calm: no decay needed
    "calm": {},
    # confused isn't in config.cpp; fallback: decay straight to 0
    "confused": {5: [0], 4: [0], 3: [0], 2: [0], 1: [0]},
}

def level_from_value(v: float) -> int:
    v = float(clamp(v, 0, 150))
    for lvl in (5, 4, 3, 2, 1, 0):
        lo, hi = EMOTION_LEVEL_THRESHOLDS[lvl]
        if lo <= v <= hi:
            return lvl
    return 0

def value_for_level(level: int, base_value: float) -> float:
    if int(level) <= 0:
        return float(base_value)
    return float(LEVEL_INTENSITY_MAPPING.get(int(level), base_value))


app = Flask(__name__)


def load_labels():
    labels = {}
    if os.path.exists(LABEL_PATH):
        with open(LABEL_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                wav = obj.get("wav")
                if wav:
                    labels[wav] = obj
    return labels


def upsert_label(obj):
    labels = load_labels()
    labels[obj["wav"]] = obj
    with open(LABEL_PATH, "w", encoding="utf-8") as f:
        for wav in sorted(labels.keys()):
            f.write(json.dumps(labels[wav], ensure_ascii=False) + "\n")


def clamp(x, lo, hi):
    try:
        x = float(x)
    except Exception:
        x = lo
    return max(lo, min(hi, x))



def snap_to_level_center(v):
    """Quantize any value to the center (midpoint) of its emotion level bucket.
    This is the global '阶级粘滞/量化' rule used everywhere in backend.
    """
    v = clamp(v, 0, 150)
    for lvl in (5, 4, 3, 2, 1, 0):
        lo, hi = EMOTION_LEVEL_THRESHOLDS[lvl]
        if lo <= v <= hi:
            return (lo + hi) / 2.0
    return 5.0


def norm_type(t):
    t = (t or "").strip().lower()
    return t if t in ALLOWED_TYPES else "calm"


def normalize_base(base):
    if not isinstance(base, dict):
        base = {}
    return {
        "type": norm_type(base.get("type", "calm")),
        "value": float(snap_to_level_center(base.get("value", 60))),
    }


def normalize_triggers(triggers, duration=None):
    out = []
    if not isinstance(triggers, list):
        return out

    for tr in triggers:
        if not isinstance(tr, dict):
            continue
        kind = (tr.get("kind") or "").strip().lower()
        if kind not in ("anchor", "switch", "spike"):
            continue

        t = tr.get("t", 0.0)
        if duration is not None:
            t = clamp(t, 0.0, duration)
        else:
            t = clamp(t, 0.0, 1e9)

        if kind == "anchor":
            out.append({
                "kind": "anchor",
                "t": float(t),
                "type": norm_type(tr.get("type", "calm")),
                "value": float(clamp(tr.get("value", 60), 0, 150)),
            })
        elif kind == "switch":
            out.append({
                "kind": "switch",
                "t": float(t),
                "type": norm_type(tr.get("type", "calm")),
            })
        else:
            out.append({
                "kind": "spike",
                "t": float(t),
                "value": float(clamp(tr.get("value", 120), 0, 150)),
                "tau": float(clamp(tr.get("tau", 0.5), 0.0, 10.0)),
            })

    out.sort(key=lambda x: (x["t"], {"anchor": 0, "switch": 1, "spike": 2}[x["kind"]]))
    return out


def normalize_curve(curve, duration=None):
    if not isinstance(curve, list):
        curve = []
    out = []
    for p in curve:
        if not isinstance(p, dict):
            continue
        t = p.get("t", 0.0)
        v = p.get("value", 60)
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
            out = [
                {"t": 0.0, "type": "calm", "value": 60.0},
                {"t": float(duration), "type": "calm", "value": 60.0},
            ]
        else:
            if out[0]["t"] > 0.0:
                out.insert(0, {"t": 0.0, "type": out[0]["type"], "value": out[0]["value"]})
            if out[-1]["t"] < float(duration):
                out.append({"t": float(duration), "type": out[-1]["type"], "value": out[-1]["value"]})

        # enforce endpoints exact
        out[0]["t"] = 0.0
        out[-1]["t"] = float(duration)

    return out


def value_at(curve, t):
    """linear interpolate value at time t; type not used here"""
    if not curve:
        return 60.0
    if t <= curve[0]["t"]:
        return curve[0]["value"]
    if t >= curve[-1]["t"]:
        return curve[-1]["value"]
    for i in range(len(curve) - 1):
        p0 = curve[i]
        p1 = curve[i + 1]
        if p0["t"] <= t <= p1["t"]:
            if abs(p1["t"] - p0["t"]) < 1e-9:
                return p0["value"]
            a = (t - p0["t"]) / (p1["t"] - p0["t"])
            return (1 - a) * p0["value"] + a * p1["value"]
    return curve[-1]["value"]



def value_at_step(curve, t):
    """Step-hold value at time t (left-constant), for staircase curves."""
    if not curve:
        return 60.0
    if t <= curve[0]["t"]:
        return curve[0]["value"]
    for i in range(len(curve) - 1):
        if curve[i]["t"] <= t < curve[i + 1]["t"]:
            return curve[i]["value"]
    return curve[-1]["value"]

def type_at(curve, t):
    """piecewise constant: take left point type"""
    if not curve:
        return "calm"
    if t <= curve[0]["t"]:
        return curve[0]["type"]
    for i in range(len(curve) - 1):
        if curve[i]["t"] <= t < curve[i + 1]["t"]:
            return curve[i]["type"]
    return curve[-1]["type"]


def generate_curve(base, triggers, duration, default_tau=0.5):
    """
    Trigger-only staircase curve generator aligned to the FRAME GRID.

    Semantics (fps=DEFAULT_FPS):
      - Start ramp LEAD_IN_FRAMES before trigger, reaching target at trigger frame
      - Hold each level for DEFAULT_LEVEL_FRAMES
      - Decay through pre-defined level sequence (DECAY_LEVELS[type][start_level])
      - Finally settle back to base emotion (base type/value)

    IMPORTANT: All generated control points land exactly on frame times k/fps.
               Vertical "steps" are represented by two points on adjacent frames
               (k-1)/fps and k/fps to avoid duplicate timestamps in the UI.
    """
    duration = float(duration)
    base = normalize_base(base)
    triggers = normalize_triggers(triggers, duration=duration)

    fps = int(DEFAULT_FPS)
    dur_frames = max(0, int(round(duration * fps)))
    hold_frames = int(DEFAULT_LEVEL_FRAMES)
    lead_in_frames = int(LEAD_IN_FRAMES)

    def k_of(t: float) -> int:
        return int(round(float(t) * fps))

    def t_of(k: int) -> float:
        return float(clamp(k, 0, dur_frames)) / fps

    curve = [{"t": 0.0, "type": base["type"], "value": base["value"]}]

    def add_point_k(k: int, ty: str, val: float):
        curve.append({
            "t": t_of(k),
            "type": norm_type(ty),
            "value": float(clamp(val, 0, 150)),
        })

    def add_step_k(k: int, ty: str, new_val: float):
        """
        Represent a vertical step into (k) by adding a point at (k-1) with old value,
        then a point at (k) with new value. This keeps all timestamps unique.
        """
        k = int(clamp(k, 0, dur_frames))
        if not curve:
            add_point_k(k, ty, new_val)
            return

        prev = curve[-1]
        prev_k = k_of(prev["t"])

        # ensure (k-1) exists and is >= prev time
        if k > 0:
            k0 = k - 1
            if k0 > prev_k:
                add_point_k(k0, prev["type"], prev["value"])

        # now the step at k
        if k >= prev_k:
            add_point_k(k, ty, new_val)

    for i, tr in enumerate(triggers):
        k_tr = k_of(tr["t"])
        k_tr = int(clamp(k_tr, 0, dur_frames))
        k_next = k_of(triggers[i + 1]["t"]) if i + 1 < len(triggers) else dur_frames
        k_next = int(clamp(k_next, 0, dur_frames))

        # Determine trigger target (type + level)
        t_tr = t_of(k_tr)
        if tr["kind"] == "anchor":
            ty = norm_type(tr.get("type", type_at(curve, t_tr)))
            target_val_raw = float(tr.get("value", value_at(curve, t_tr)))
        elif tr["kind"] == "switch":
            ty = norm_type(tr.get("type", type_at(curve, t_tr)))
            target_val_raw = float(value_at(curve, t_tr))  # keep current
        else:  # spike
            ty = type_at(curve, t_tr)  # inherit type (spike has no type)
            target_val_raw = float(tr.get("value", value_at(curve, t_tr)))

        target_level = level_from_value(target_val_raw)
        target_val = value_for_level(target_level, base["value"])

        # 1) lead-in ramp: point at (k_tr - LEAD_IN_FRAMES) with current curve state
        k_ramp0 = max(0, k_tr - lead_in_frames)
        if t_of(k_ramp0) > curve[-1]["t"] + 1e-9:
            add_point_k(k_ramp0, type_at(curve, t_of(k_ramp0)), value_at(curve, t_of(k_ramp0)))

        # 2) reach target at trigger frame
        add_point_k(k_tr, ty, target_val)

        # no room before next trigger
        if k_tr >= k_next:
            curve = normalize_curve(curve, duration=duration)
            continue

        # 3) hold target for DEFAULT_LEVEL_FRAMES
        k_hold_end = k_tr + hold_frames
        if k_hold_end >= k_next:
            curve = normalize_curve(curve, duration=duration)
            continue
        add_point_k(k_hold_end, ty, target_val)

        # 4) decay sequence; each level holds DEFAULT_LEVEL_FRAMES
        if target_level > 0 and ty != "calm":
            levels = DECAY_LEVELS.get(ty, {}).get(target_level, [0])
            k_cur = k_hold_end

            for lvl in levels:
                lvl = int(lvl)
                val_lvl = value_for_level(lvl, base["value"])

                # step into this level at k_cur
                if k_cur < k_next:
                    add_step_k(k_cur, ty, val_lvl)
                else:
                    break

                # hold this level for DEFAULT_LEVEL_FRAMES
                k_end = k_cur + hold_frames
                if k_end < k_next:
                    add_point_k(k_end, ty, val_lvl)
                    k_cur = k_end
                else:
                    break

            # 5) settle to base at k_cur
            if k_cur < k_next:
                add_step_k(k_cur, base["type"], base["value"])

        curve = normalize_curve(curve, duration=duration)

    # ensure end point at duration
    curve.append({"t": duration, "type": curve[-1]["type"], "value": curve[-1]["value"]})
    return normalize_curve(curve, duration=duration)



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

    payload = []
    for fn in files:
        obj = labels.get(fn, {})
        curve = obj.get("curve", [])
        labeled = isinstance(curve, list) and len(curve) >= 2
        payload.append({
            "wav": fn,
            "labeled": bool(labeled),
            "fps": int(obj.get("fps", DEFAULT_FPS)),
        })
    return jsonify(payload)


@app.get("/api/label/<path:wav>")
def api_get_label(wav):
    labels = load_labels()
    obj = labels.get(wav, {})
    return jsonify({
        "wav": wav,
        "fps": int(obj.get("fps", DEFAULT_FPS)),
        "duration": obj.get("duration", None),
        "base": obj.get("base", {"type": "calm", "value": 60}),
        "triggers": obj.get("triggers", []),
        "curve": obj.get("curve", []),
    })


@app.post("/api/label")
def api_post_label():
    data = request.get_json(force=True)
    wav = data.get("wav")
    if not wav or not isinstance(wav, str):
        return jsonify({"error": "Missing wav"}), 400

    fps = data.get("fps", DEFAULT_FPS)
    try:
        fps = int(fps)
    except Exception:
        fps = DEFAULT_FPS

    duration = data.get("duration", None)
    if duration is not None:
        try:
            duration = float(duration)
        except Exception:
            duration = None

    base = normalize_base(data.get("base", {}))

    triggers = data.get("triggers", [])
    triggers = normalize_triggers(triggers, duration=duration)

    curve = data.get("curve", [])
    curve = normalize_curve(curve, duration=duration) if duration is not None else normalize_curve(curve, None)

    out = {
        "wav": wav,
        "fps": fps,
        "duration": duration,
        "base": base,
        "triggers": triggers,
        "curve": curve,
    }
    upsert_label(out)
    return jsonify({"ok": True})


@app.post("/api/generate")
def api_generate():
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
    frames = sample_frames(curve, duration=float(duration), fps=fps)
    return jsonify({"wav": wav, "fps": fps, "duration": float(duration), "frames": frames})


@app.get("/audio/<path:filename>")
def audio(filename):
    full = os.path.join(AUDIO_DIR, filename)
    if not os.path.isfile(full):
        abort(404)
    return send_from_directory(AUDIO_DIR, filename, as_attachment=False)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=7860, debug=True)