#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from pathlib import Path
from typing import Dict, Any

from flask import Flask, jsonify, request, send_from_directory, abort, Response

# -----------------------------------------------------------------------------
# Paths / Config
# -----------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

WAV_DIR = os.environ.get("WAV_DIR") or str(Path(REPO_ROOT) / "wavs")
LABEL_PATH = os.environ.get("LABEL_PATH") or str(Path(REPO_ROOT) / "annotater" / "labels.jsonl")
PRED_PATH = os.environ.get("PRED_PATH") or str(Path(REPO_ROOT) / "pred.jsonl")

DEFAULT_FPS = int(os.environ.get("FPS") or 30)

# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------
app = Flask(__name__, static_folder=None)

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


def _list_wavs(wav_dir: str):
    if not os.path.isdir(wav_dir):
        return []
    xs = []
    for n in os.listdir(wav_dir):
        if n.lower().endswith(".wav"):
            xs.append(n)
    xs.sort()
    return xs


def _safe_wav_name(name: str) -> str:
    # prevent path traversal
    name = os.path.basename(name)
    if "/" in name or "\\" in name or name.startswith("."):
        raise ValueError("bad wav name")
    return name


def _ensure_paths():
    if not os.path.isdir(WAV_DIR):
        raise RuntimeError(f"WAV_DIR not found: {WAV_DIR}")


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
    preds = _read_jsonl_as_map(PRED_PATH)

    out = []
    for wav in _list_wavs(WAV_DIR):
        out.append({
            "wav": wav,
            "fps": int(DEFAULT_FPS),
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
    preds = _read_jsonl_as_map(PRED_PATH)
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

    preds = _read_jsonl_as_map(PRED_PATH)
    obj = preds.get(wav)
    if obj is None:
        abort(404, description=f"no prediction for {wav}. Did you export pred.jsonl?")
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

    full = os.path.join(WAV_DIR, wav)
    if not os.path.isfile(full):
        abort(404)
    # send_from_directory handles range/headers ok for wav
    return send_from_directory(WAV_DIR, wav, mimetype="audio/x-wav", as_attachment=False)


# Optional: serve a local test page if you want
@app.route("/", methods=["GET"])
def index():
    """
    If you place test/index.html next to this server, it will be served.
    Otherwise, just show a help page.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    idx = os.path.join(here, "index.html")
    if os.path.isfile(idx):
        return send_from_directory(here, "index.html")
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
        f"<p>LABEL_PATH={LABEL_PATH}</p>"
        f"<p>PRED_PATH={PRED_PATH}</p>"
    )


def main():
    _ensure_paths()
    host = os.environ.get("HOST") or "0.0.0.0"
    port = int(os.environ.get("PORT") or 7861)

    print("[config]")
    print("  WAV_DIR   =", WAV_DIR)
    print("  LABEL_PATH=", LABEL_PATH)
    print("  PRED_PATH =", PRED_PATH)
    print(f"listening on http://{host}:{port}")
    app.run(host=host, port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
