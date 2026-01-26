import os
import json
from flask import Flask, jsonify, request, send_from_directory, abort
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 你原来写死的目录，按需改
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


def clamp(v, lo, hi):
    try:
        v = float(v)
    except Exception:
        v = lo
    return max(lo, min(hi, v))


def norm_type(t):
    t = (t or "").strip().lower()
    return t if t in ALLOWED_TYPES else "calm"


def normalize_curve(curve, duration=None):
    """
    curve: list of {t,type,value}
    - clamp t to [0,duration] if duration provided
    - clamp value to [0,150]
    - ensure type in allowed
    - sort by t
    - remove duplicate t by keeping last
    - ensure at least two points when duration provided: t=0 and t=duration
    """
    if not isinstance(curve, list):
        curve = []

    out = []
    for p in curve:
        if not isinstance(p, dict):
            continue
        t = clamp(p.get("t", 0.0), 0.0, duration if duration is not None else 1e9)
        v = clamp(p.get("value", 0.0), 0.0, 150.0)
        ty = norm_type(p.get("type", "calm"))
        out.append({"t": float(t), "type": ty, "value": float(v)})

    out.sort(key=lambda x: x["t"])

    # dedup by t, keep last
    dedup = []
    last_t = None
    for p in out:
        if last_t is not None and abs(p["t"] - last_t) < 1e-9:
            dedup[-1] = p
        else:
            dedup.append(p)
            last_t = p["t"]

    out = dedup

    if duration is not None:
        if len(out) == 0:
            out = [
                {"t": 0.0, "type": "calm", "value": 50.0},
                {"t": float(duration), "type": "calm", "value": 50.0},
            ]
        else:
            if out[0]["t"] > 0.0:
                out.insert(0, {"t": 0.0, "type": out[0]["type"], "value": out[0]["value"]})
            if out[-1]["t"] < float(duration):
                out.append({"t": float(duration), "type": out[-1]["type"], "value": out[-1]["value"]})

    return out


def generate_curve(base, events, duration, fps=DEFAULT_FPS, spike_tau=0.5):
    """
    base: {type,value}
    events: list of {t, kind: "switch"|"spike", type?, value?}
    rule:
      - start at t=0 with base
      - switch: at t insert new type, keep value
      - spike: at t insert value=spikeValue (type unchanged), and at t+tau insert back to previous value
      - always add end at duration
    """
    duration = float(duration)
    fps = int(fps) if isinstance(fps, (int, float, str)) else DEFAULT_FPS
    spike_tau = float(spike_tau)

    base_type = norm_type((base or {}).get("type", "calm"))
    base_val = clamp((base or {}).get("value", 60), 0, 150)

    evs = []
    if isinstance(events, list):
        for e in events:
            if not isinstance(e, dict):
                continue
            t = clamp(e.get("t", 0.0), 0.0, duration)
            kind = (e.get("kind") or "").strip().lower()
            if kind not in ("switch", "spike"):
                continue
            if kind == "switch":
                evs.append({"t": float(t), "kind": "switch", "type": norm_type(e.get("type", base_type))})
            else:
                evs.append({"t": float(t), "kind": "spike", "value": clamp(e.get("value", base_val), 0, 150)})
    evs.sort(key=lambda x: x["t"])

    curve = [{"t": 0.0, "type": base_type, "value": float(base_val)}]
    cur_type = base_type
    cur_val = float(base_val)

    for e in evs:
        t = float(e["t"])
        if e["kind"] == "switch":
            cur_type = norm_type(e.get("type", cur_type))
            curve.append({"t": t, "type": cur_type, "value": cur_val})
        else:
            spike_v = float(e["value"])
            curve.append({"t": t, "type": cur_type, "value": spike_v})
            back_t = min(duration, t + spike_tau)
            curve.append({"t": back_t, "type": cur_type, "value": cur_val})

    curve.append({"t": duration, "type": cur_type, "value": cur_val})
    return normalize_curve(curve, duration=duration)


def sample_frames_from_curve(curve, duration, fps):
    """
    Output list per frame:
      {"i": i, "t": t, "type": type, "value": value}
    type: piecewise constant (from left point)
    value: linear interpolation between neighbor points
    """
    duration = float(duration)
    fps = int(fps)
    n = max(1, int(round(duration * fps)))
    if not curve:
        curve = [{"t": 0.0, "type": "calm", "value": 50.0}, {"t": duration, "type": "calm", "value": 50.0}]

    curve = normalize_curve(curve, duration=duration)

    frames = []
    j = 0
    for i in range(n):
        t = i / fps
        while j + 1 < len(curve) and curve[j + 1]["t"] <= t:
            j += 1

        p0 = curve[j]
        p1 = curve[j + 1] if j + 1 < len(curve) else p0

        ty = p0["type"]
        if abs(p1["t"] - p0["t"]) < 1e-9:
            v = p0["value"]
        else:
            alpha = (t - p0["t"]) / (p1["t"] - p0["t"])
            v = (1 - alpha) * p0["value"] + alpha * p1["value"]

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
        curve = obj.get("curve", None)
        labeled = isinstance(curve, list) and len(curve) >= 2
        payload.append({
            "wav": fn,
            "labeled": bool(labeled),
            "fps": int(obj.get("fps", DEFAULT_FPS)),
        })
    return jsonify(payload)


@app.get("/api/curve/<path:wav>")
def api_get_curve(wav):
    labels = load_labels()
    obj = labels.get(wav, {})
    return jsonify({
        "wav": wav,
        "fps": int(obj.get("fps", DEFAULT_FPS)),
        "base": obj.get("base", {"type": "calm", "value": 60}),
        "events": obj.get("events", []),
        "curve": obj.get("curve", []),
        "duration": obj.get("duration", None),
    })


@app.post("/api/curve")
def api_post_curve():
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

    base = data.get("base", {"type": "calm", "value": 60})
    if not isinstance(base, dict):
        base = {"type": "calm", "value": 60}
    base = {"type": norm_type(base.get("type", "calm")), "value": float(clamp(base.get("value", 60), 0, 150))}

    events = data.get("events", [])
    if not isinstance(events, list):
        events = []

    curve = data.get("curve", [])
    if duration is not None:
        curve = normalize_curve(curve, duration=duration)
    else:
        curve = normalize_curve(curve, duration=None)

    labels = load_labels()
    old = labels.get(wav, {})
    out = {
        "wav": wav,
        "fps": fps,
        "duration": duration if duration is not None else old.get("duration", None),
        "base": base,
        "events": events,
        "curve": curve,
    }
    upsert_label(out)
    return jsonify({"ok": True})


@app.post("/api/generate_curve")
def api_generate_curve():
    data = request.get_json(force=True)
    wav = data.get("wav")
    duration = data.get("duration", None)
    if not wav or not isinstance(wav, str):
        return jsonify({"error": "Missing wav"}), 400
    if duration is None:
        return jsonify({"error": "Missing duration"}), 400
    try:
        duration = float(duration)
    except Exception:
        return jsonify({"error": "Bad duration"}), 400

    fps = int(data.get("fps", DEFAULT_FPS))
    spike_tau = float(data.get("spike_tau", 0.5))
    base = data.get("base", {"type": "calm", "value": 60})
    events = data.get("events", [])

    curve = generate_curve(base, events, duration=duration, fps=fps, spike_tau=spike_tau)

    labels = load_labels()
    old = labels.get(wav, {})
    out = {
        "wav": wav,
        "fps": fps,
        "duration": duration,
        "base": {"type": norm_type((base or {}).get("type", "calm")), "value": float(clamp((base or {}).get("value", 60), 0, 150))},
        "events": events if isinstance(events, list) else [],
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
    frames = sample_frames_from_curve(curve, duration=duration, fps=fps)
    return jsonify({"wav": wav, "fps": fps, "duration": float(duration), "frames": frames})


@app.get("/audio/<path:filename>")
def audio(filename):
    full = os.path.join(AUDIO_DIR, filename)
    if not os.path.isfile(full):
        abort(404)
    return send_from_directory(AUDIO_DIR, filename, as_attachment=False)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=7860, debug=True)