import os
import json
from flask import Flask, jsonify, request, send_from_directory, abort

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(BASE_DIR, "wavs")
LABEL_PATH = os.path.join(BASE_DIR, "labels.jsonl")
DEFAULT_FPS = 30
MAX_KEYS = 3

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

def normalize_key_frames_keep_order(key_frames):
    """Keep slot order. Remove invalids. Deduplicate while preserving first occurrence. Limit MAX_KEYS."""
    out = []
    seen = set()
    if not isinstance(key_frames, list):
        return out
    for x in key_frames:
        try:
            v = int(x)
        except Exception:
            continue
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
        if len(out) >= MAX_KEYS:
            break
    return out

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
        obj = labels.get(fn)
        if obj is None:
            payload.append({
                "wav": fn,
                "labeled": False,
                "key_frames": [],
                "fps": DEFAULT_FPS,
            })
        else:
            kf = normalize_key_frames_keep_order(obj.get("key_frames", []))
            payload.append({
                "wav": fn,
                "labeled": True,
                "key_frames": kf,
                "fps": int(obj.get("fps", DEFAULT_FPS)),
            })
    return jsonify(payload)

@app.get("/api/label/<path:wav>")
def api_get_label(wav):
    labels = load_labels()
    obj = labels.get(wav)
    if not obj:
        return jsonify({"wav": wav, "fps": DEFAULT_FPS, "key_frames": []})
    kf = normalize_key_frames_keep_order(obj.get("key_frames", []))
    return jsonify({"wav": wav, "fps": int(obj.get("fps", DEFAULT_FPS)), "key_frames": kf})

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

    key_frames = data.get("key_frames", None)
    if key_frames is None or not isinstance(key_frames, list):
        return jsonify({"error": "Missing key_frames(list)"}), 400

    frames = normalize_key_frames_keep_order(key_frames)
    upsert_label({"wav": wav, "fps": fps, "key_frames": frames})
    return jsonify({"ok": True})

@app.get("/audio/<path:filename>")
def audio(filename):
    full = os.path.join(AUDIO_DIR, filename)
    if not os.path.isfile(full):
        abort(404)
    return send_from_directory(AUDIO_DIR, filename, as_attachment=False)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=7860, debug=True)
