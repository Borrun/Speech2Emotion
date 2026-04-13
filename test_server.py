#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from pathlib import Path
from typing import Dict, Any

from flask import Flask, jsonify, request, send_from_directory, abort, Response

from infer.postprocess import apply_cpp_emotion_sync

# -----------------------------------------------------------------------------
# Paths / Config
# -----------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))

DEFAULT_PRED_DIR = Path(REPO_ROOT) / "outputs" / "test_emotion_codes_10cls"
if not DEFAULT_PRED_DIR.exists():
    DEFAULT_PRED_DIR = Path(REPO_ROOT) / "outputs" / "test_emotion_codes"

DEFAULT_WAV_ROOTS = [
    Path(REPO_ROOT) / "test",
    Path(REPO_ROOT) / "annotater" / "wavs",
    Path(REPO_ROOT) / "wavs",
]
DATA_DIR = Path(REPO_ROOT) / "annotater" / "data"
ALLOWED_TYPES = [
    "happy", "sad", "angry", "fear", "calm",
    "happy_confused", "sad_confused", "angry_confused", "fear_confused", "calm_confused",
]
LEVEL_INTENSITY_MAPPING = {0: 5, 1: 18, 2: 38, 3: 68, 4: 98, 5: 130}

WAV_DIR = os.environ.get("WAV_DIR") or str(DEFAULT_WAV_ROOTS[0])
LABEL_PATH = os.environ.get("LABEL_PATH") or str(Path(REPO_ROOT) / "annotater" / "labels_new.jsonl")
PRED_PATH = os.environ.get("PRED_PATH") or str(DEFAULT_PRED_DIR)

DEFAULT_FPS = int(os.environ.get("FPS") or 30)

# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------
app = Flask(__name__, static_folder=None)
_emo_cache = {}

# Simple CORS (for local dev / separate frontend)
@app.after_request
def add_cors(resp: Response):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    resp.headers["Cache-Control"] = "no-cache"
    return resp


def _read_jsonl_as_map(path: str) -> Dict[str, Dict[str, Any]]:
    mp: Dict[str, Dict[str, Any]] = {}
    if not path or (not os.path.isfile(path)):
        return mp
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            wav = obj.get("wav")
            if wav:
                mp[str(wav)] = obj
    return mp


def _read_pred_as_map(path: str) -> Dict[str, Dict[str, Any]]:
    if not path:
        return {}
    if os.path.isfile(path):
        return _read_jsonl_as_map(path)

    mp: Dict[str, Dict[str, Any]] = {}
    if not os.path.isdir(path):
        return mp

    for root, _, files in os.walk(path):
        for name in files:
            if not name.lower().endswith(".json"):
                continue
            full = os.path.join(root, name)
            try:
                with open(full, "r", encoding="utf-8") as f:
                    obj = json.load(f)
            except Exception:
                continue
            wav = str(obj.get("wav") or (name[:-5] + ".wav"))
            mp[wav] = obj
    return mp


def _list_wavs(wav_dir: str):
    if not os.path.isdir(wav_dir):
        return []
    xs = []
    for root, _, files in os.walk(wav_dir):
        for n in files:
            if not n.lower().endswith(".wav"):
                continue
            rel = os.path.relpath(os.path.join(root, n), wav_dir)
            xs.append(rel.replace("\\", "/"))
    xs.sort()
    return xs


def _wav_roots():
    env_root = os.environ.get("WAV_DIR")
    if env_root:
        roots = [Path(env_root)]
    else:
        roots = list(DEFAULT_WAV_ROOTS)
    out = []
    seen = set()
    for root in roots:
        root = Path(root)
        key = str(root.resolve()) if root.exists() else str(root)
        if key in seen:
            continue
        seen.add(key)
        out.append(root)
    return out


def _list_all_wavs():
    names = set()
    for root in _wav_roots():
        if root.is_dir():
            names.update(_list_wavs(str(root)))
    return sorted(names)


def _resolve_wav_path(wav: str):
    wav = _safe_wav_name(wav)
    for root in _wav_roots():
        full = root / wav
        if full.is_file():
            return root, full
    return None, None


def emotion_txt_for_type(etype: str) -> Path:
    return DATA_DIR / f"{etype}.txt"


def load_emotion_template(etype: str):
    if etype in _emo_cache:
        return _emo_cache[etype]

    txt_path = emotion_txt_for_type(etype)
    blocks = {}
    max_block = None

    if not txt_path.is_file():
        _emo_cache[etype] = {"frames": {}, "max_frame": None, "path": str(txt_path)}
        return _emo_cache[etype]

    cur = {}
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()

            if len(parts) == 1 + 64:
                try:
                    bi = int(float(parts[0]))
                except Exception:
                    continue
                sid = "0"
                num_parts = parts[1:]
            elif len(parts) >= 2 + 64:
                try:
                    bi = int(float(parts[0]))
                except Exception:
                    continue
                sid = parts[1]
                num_parts = parts[2:2 + 64]
            else:
                continue

            nums = []
            ok = True
            for s in num_parts:
                try:
                    nums.append(float(s))
                except Exception:
                    ok = False
                    break
            if not ok or len(nums) != 64:
                continue

            pts = [[nums[i], nums[i + 1]] for i in range(0, 64, 2)]
            key = (bi, sid)
            cur.setdefault(key, []).append(pts)

            if max_block is None or bi > max_block:
                max_block = bi

            if len(cur[key]) == 4 and bi not in blocks:
                blocks[bi] = cur[key][:4]

    _emo_cache[etype] = {"frames": blocks, "max_frame": max_block, "path": str(txt_path)}
    return _emo_cache[etype]


def _safe_wav_name(name: str) -> str:
    # prevent path traversal
    name = str(name or "").replace("\\", "/")
    name = os.path.normpath(name).replace("\\", "/")
    if name in {"", ".", ".."} or name.startswith("../") or os.path.isabs(name):
        raise ValueError("bad wav name")
    return name


def _ensure_paths():
    if not any(root.is_dir() for root in _wav_roots()):
        roots = ", ".join(str(root) for root in _wav_roots())
        raise RuntimeError(f"no wav roots found: {roots}")


# -----------------------------------------------------------------------------
# API
# -----------------------------------------------------------------------------
@app.route("/api/files", methods=["GET"])
def api_files():
    """
    Unified list:
      [{wav, fps, labeled, predicted}]
    """
    _ensure_paths()
    labels = _read_jsonl_as_map(LABEL_PATH)
    preds = _read_pred_as_map(PRED_PATH)

    out = []
    wav_names = set(_list_all_wavs())
    wav_names.update(labels.keys())
    wav_names.update(preds.keys())
    for wav in sorted(wav_names):
        out.append({
            "wav": wav,
            "fps": int(preds.get(wav, {}).get("fps", DEFAULT_FPS)),
            "labeled": bool(wav in labels),
            "predicted": bool(wav in preds),
        })
    return jsonify(out)


@app.route("/api/pred_files", methods=["GET"])
def api_pred_files():
    """
    Return list of predicted wav entries (from pred.jsonl):
      [{wav, fps, n_frames, duration}]  (best effort)
    """
    preds = _read_pred_as_map(PRED_PATH)
    items = []
    for wav, obj in sorted(preds.items(), key=lambda x: x[0]):
        items.append({
            "wav": wav,
            "fps": int(obj.get("fps", DEFAULT_FPS)),
            "n_frames": int(obj.get("n_frames", 0) or 0),
            "duration": float(obj.get("duration", 0.0) or 0.0),
        })
    return jsonify(items)


@app.route("/api/pred/<path:wav>", methods=["GET"])
def api_pred_one(wav: str):
    """
    Get prediction object for a wav. Expected to include:
      {wav,fps,duration,n_frames,boundaries,segments,curve,...}
    """
    try:
        wav = _safe_wav_name(wav)
    except Exception:
        abort(400)

    preds = _read_pred_as_map(PRED_PATH)
    obj = preds.get(wav)
    if obj is None:
        abort(404, description=f"no prediction for {wav}. Did you export pred.jsonl?")
    if "cpp_sync" not in obj:
        obj = dict(obj)
        obj["cpp_sync"] = apply_cpp_emotion_sync(
            frames=obj.get("frames", []),
            fps=int(obj.get("fps", DEFAULT_FPS)),
            type_map=obj.get("type_map", []),
        )
    return jsonify(obj)


@app.route("/api/label/<path:wav>", methods=["GET"])
def api_label_one(wav: str):
    """
    Pass-through to read annotater labels.jsonl for a wav.
    If absent, return empty curve (frontend can init default).
    """
    try:
        wav = _safe_wav_name(wav)
    except Exception:
        abort(400)

    labels = _read_jsonl_as_map(LABEL_PATH)
    obj = labels.get(wav)
    if obj is None:
        return jsonify({"wav": wav, "fps": DEFAULT_FPS, "curve": [], "text": ""})
    return jsonify(obj)


@app.route("/api/emo_shape/<etype>", methods=["GET"])
def api_emo_shape(etype: str):
    etype = (etype or "").strip().lower()
    if etype not in ALLOWED_TYPES:
        abort(400, description=f"bad emotion type: {etype}")

    frame = request.args.get("frame", default=None, type=int)
    level = request.args.get("level", default=None, type=int)

    if frame is None and level is None:
        meta = load_emotion_template(etype)
        return jsonify({
            "type": etype,
            "available": bool(meta["frames"]),
            "max_frame": meta["max_frame"],
            "path": meta["path"],
        })

    if frame is None:
        if level not in LEVEL_INTENSITY_MAPPING:
            abort(400, description=f"bad level: {level}")
        frame = int(LEVEL_INTENSITY_MAPPING[level])

    meta = load_emotion_template(etype)
    frames = meta.get("frames", {})
    if not frames:
        abort(404, description=f"missing template txt: {meta.get('path')}")

    if frame not in frames:
        frame = min(frames.keys(), key=lambda x: abs(x - frame))

    return jsonify({"type": etype, "frame": frame, "groups": frames[frame]})


@app.route("/audio/<path:wav>", methods=["GET", "HEAD"])
def audio(wav: str):
    """
    Serve wav file.
    """
    _ensure_paths()
    try:
        wav = _safe_wav_name(wav)
    except Exception:
        abort(400)

    root, full = _resolve_wav_path(wav)
    if root is None or full is None:
        abort(404)
    # send_from_directory handles range/headers ok for wav
    rel = os.path.relpath(str(full), str(root)).replace("\\", "/")
    return send_from_directory(str(root), rel, mimetype="audio/x-wav", as_attachment=False)


# Optional: serve a local test page if you want
@app.route("/", methods=["GET"])
def index():
    """
    If you place test/index.html next to this server, it will be served.
    Otherwise, just show a help page.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    idx = os.path.join(here, "test_index.html")
    if os.path.isfile(idx):
        return send_from_directory(here, "test_index.html")
    return (
        "<h3>Speech2Emotion Test Server</h3>"
        "<p>Endpoints:</p>"
        "<ul>"
        "<li>/api/files</li>"
        "<li>/api/pred_files</li>"
        "<li>/api/pred/&lt;wav&gt;</li>"
        "<li>/api/label/&lt;wav&gt;</li>"
        "<li>/audio/&lt;wav&gt;</li>"
        "</ul>"
        f"<p>WAV_DIR={WAV_DIR}</p>"
        f"<p>WAV_ROOTS={[str(p) for p in _wav_roots()]}</p>"
        f"<p>LABEL_PATH={LABEL_PATH}</p>"
        f"<p>PRED_PATH={PRED_PATH}</p>"
    )


def main():
    _ensure_paths()
    host = os.environ.get("HOST") or "0.0.0.0"
    port = int(os.environ.get("PORT") or 7861)

    print("[config]")
    print("  WAV_DIR   =", WAV_DIR)
    print("  WAV_ROOTS =", [str(p) for p in _wav_roots()])
    print("  LABEL_PATH=", LABEL_PATH)
    print("  PRED_PATH =", PRED_PATH)
    print(f"listening on http://{host}:{port}")
    app.run(host=host, port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
