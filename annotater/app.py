import os
import json
from flask import Flask, jsonify, request, send_from_directory, abort
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 改成你本机 wav 目录
AUDIO_DIR = str(Path("/Users/Zhuanz1/Speech2Emotion/wavs"))

LABEL_PATH = os.path.join(BASE_DIR, "labels.jsonl")
DEFAULT_FPS = 30

ALLOWED_TYPES = ["happy", "sad", "angry", "fear", "calm", "confused"]

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


def norm_type(t):
    t = (t or "").strip().lower()
    return t if t in ALLOWED_TYPES else "calm"


def normalize_base(base):
    if not isinstance(base, dict):
        base = {}
    return {
        "type": norm_type(base.get("type", "calm")),
        "value": float(clamp(base.get("value", 60), 0, 150)),
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
        out.append({"t": float(t), "type": norm_type(ty), "value": float(clamp(v, 0, 150))})

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
    triggers:
      - anchor(t,type,value): hard point
      - switch(t,type): change type at t, value inherited from current curve
      - spike(t,value,tau): insert peak and decay back to baseline at t+tau
    rule:
      - start with base at t=0
      - process triggers in time order; anchors override both type/value at t
      - ensure end at duration
    """
    duration = float(duration)
    base = normalize_base(base)
    triggers = normalize_triggers(triggers, duration=duration)

    # start curve with base endpoints (we'll refine)
    curve = [
        {"t": 0.0, "type": base["type"], "value": base["value"]},
        {"t": duration, "type": base["type"], "value": base["value"]},
    ]

    def insert_point(t, ty=None, val=None):
        ty0 = type_at(curve, t) if ty is None else norm_type(ty)
        val0 = value_at(curve, t) if val is None else float(clamp(val, 0, 150))
        curve.append({"t": float(t), "type": ty0, "value": val0})

    for tr in triggers:
        t = float(tr["t"])
        if tr["kind"] == "anchor":
            insert_point(t, tr["type"], tr["value"])
        elif tr["kind"] == "switch":
            # change type at t, keep current value
            insert_point(t, tr["type"], None)
        else:  # spike
            tau = float(tr.get("tau", default_tau))
            tau = max(0.0, tau)
            peak = float(tr["value"])
            base_val = value_at(curve, t)
            ty = type_at(curve, t)
            insert_point(t, ty, peak)
            insert_point(min(duration, t + tau), ty, base_val)

        curve = normalize_curve(curve, duration=duration)

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
        v = value_at(curve, t)
        frames.append({"i": i, "t": float(t), "type": ty, "value": float(clamp(v, 0, 150))})
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