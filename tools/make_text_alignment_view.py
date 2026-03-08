import argparse
import csv
import html
import json
import os
import wave
from typing import Dict, List, Tuple


PUNCT = set("，,。！？!?；;：:、")


def read_wav_mono(path: str) -> Tuple[int, List[float]]:
    try:
        with wave.open(path, "rb") as wf:
            ch = wf.getnchannels()
            sw = wf.getsampwidth()
            sr = wf.getframerate()
            n = wf.getnframes()
            raw = wf.readframes(n)

        if sw == 1:
            vals = [((b - 128) / 128.0) for b in raw]
        elif sw == 2:
            import struct
            vals_i = struct.unpack("<" + "h" * (len(raw) // 2), raw)
            vals = [x / 32768.0 for x in vals_i]
        elif sw == 4:
            import struct
            vals_i = struct.unpack("<" + "i" * (len(raw) // 4), raw)
            vals = [x / 2147483648.0 for x in vals_i]
        else:
            raise RuntimeError(f"Unsupported sample width: {sw}")

        if ch > 1:
            mono = []
            for i in range(0, len(vals), ch):
                mono.append(sum(vals[i:i + ch]) / float(ch))
            vals = mono
        return sr, vals
    except Exception:
        import torchaudio

        wav, sr = torchaudio.load(path)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        vals = wav.squeeze(0).detach().cpu().tolist()
        return int(sr), [float(x) for x in vals]


def compute_peaks(samples: List[float], n_bins: int) -> List[float]:
    n = len(samples)
    if n <= 0:
        return [0.0] * max(1, n_bins)
    n_bins = max(1, n_bins)
    out = []
    for b in range(n_bins):
        s = int(b * n / n_bins)
        e = int((b + 1) * n / n_bins)
        if e <= s:
            e = min(n, s + 1)
        m = 0.0
        for x in samples[s:e]:
            ax = abs(float(x))
            if ax > m:
                m = ax
        out.append(min(1.0, m))
    return out


def load_label_text(label_path: str, wav_name: str) -> str:
    if not os.path.isfile(label_path):
        return ""
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if str(obj.get("wav", "")) == wav_name:
                return str(obj.get("text", "") or "")
    return ""


def load_csv_text(csv_path: str, wav_name: str) -> str:
    if not csv_path or (not os.path.isfile(csv_path)):
        return ""
    try:
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            rd = csv.DictReader(f)
            for row in rd:
                wav = str(row.get("wav", "") or "").strip()
                utt = str(row.get("utt_id", "") or "").strip()
                if wav == wav_name or (utt and (utt + ".wav") == wav_name):
                    txt = str(row.get("transcription", "") or "").strip()
                    if txt:
                        return txt
    except Exception:
        return ""
    return ""


def token_weight(ch: str) -> float:
    if ch.isspace():
        return 0.2
    if ch in PUNCT:
        return 1.6
    return 1.0


def align_text_to_frames(text: str, total_frames: int, switch_frames: List[int]) -> List[Dict]:
    text = text or ""
    chars = [c for c in text if c != "\n" and c != "\r"]
    chars = [c for c in chars if c.strip() != ""]
    if not chars:
        return []

    w = [token_weight(c) for c in chars]
    s = sum(w) if sum(w) > 0 else float(len(chars))
    cum = 0.0
    edges = [0]
    for wi in w:
        cum += wi
        edges.append(int(round(total_frames * cum / s)))

    # enforce monotonic edges
    edges[0] = 0
    edges[-1] = total_frames
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = min(total_frames, edges[i - 1] + 1)
    for i in range(len(edges) - 2, -1, -1):
        if edges[i] >= edges[i + 1]:
            edges[i] = max(0, edges[i + 1] - 1)

    # snap punctuation boundaries near switch points (helps TTS sentence turns)
    punct_bound_idx = [i + 1 for i, c in enumerate(chars) if c in PUNCT]
    for bi in punct_bound_idx:
        pf = edges[bi]
        best = None
        for sf in switch_frames:
            d = abs(int(sf) - int(pf))
            if d <= 10 and (best is None or d < best[0]):
                best = (d, int(sf))
        if best is not None:
            edges[bi] = best[1]

    # re-fix after snapping
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = min(total_frames, edges[i - 1] + 1)
    edges[-1] = total_frames

    tokens = []
    for i, ch in enumerate(chars):
        f0 = int(edges[i])
        f1 = int(edges[i + 1])
        if f1 <= f0:
            f1 = min(total_frames, f0 + 1)
        tokens.append({"token": ch, "f0": f0, "f1": f1})
    return tokens


def build_frame_to_token(tokens: List[Dict], total_frames: int) -> List[int]:
    arr = [-1] * total_frames
    for i, t in enumerate(tokens):
        f0 = max(0, min(total_frames, int(t["f0"])))
        f1 = max(0, min(total_frames, int(t["f1"])))
        for f in range(f0, f1):
            arr[f] = i
    return arr


def token_spans_to_tokens(spans: List[Dict]) -> List[Dict]:
    out = []
    for s in spans:
        if not isinstance(s, dict):
            continue
        tok = str(s.get("token", ""))
        f0 = int(s.get("start_frame", 0))
        f1 = int(s.get("end_frame", f0 + 1))
        if f1 <= f0:
            f1 = f0 + 1
        out.append({"token": tok, "f0": f0, "f1": f1})
    out.sort(key=lambda x: (x["f0"], x["f1"]))
    return out


def render_html(
    wav_rel: str,
    wav_name: str,
    fps: int,
    duration: float,
    peaks: List[float],
    switch_frames: List[int],
    tokens: List[Dict],
    frame_to_token: List[int],
    out_html: str,
):
    token_json = json.dumps(tokens, ensure_ascii=False)
    peaks_json = json.dumps(peaks, ensure_ascii=False)
    switch_json = json.dumps(switch_frames, ensure_ascii=False)
    frame_map_json = json.dumps(frame_to_token, ensure_ascii=False)

    title = html.escape(f"Text-Frame Alignment | {wav_name}")
    wav_esc = html.escape(wav_rel).replace("\\", "/")
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    body {{ font-family: system-ui, -apple-system, sans-serif; margin:0; background:#f7f8fb; color:#222; }}
    .wrap {{ max-width: 1200px; margin: 18px auto; padding: 0 16px; }}
    .card {{ background:#fff; border:1px solid #e6e8ef; border-radius:14px; padding:14px; margin-bottom:14px; }}
    .row {{ display:flex; gap:14px; align-items:center; flex-wrap:wrap; }}
    .pill {{ padding:3px 10px; border-radius:999px; background:#eef2ff; color:#3a4a8f; font-size:12px; }}
    #wave {{ width:100%; height:180px; border:1px solid #e5e7eb; border-radius:10px; background:#fbfcff; }}
    #textBar {{ position:relative; width:100%; height:78px; border:1px solid #e5e7eb; border-radius:10px; overflow:hidden; background:#fff; }}
    .tok {{ position:absolute; top:8px; height:60px; font-size:14px; border-right:1px solid #f0f1f5; display:flex; align-items:center; justify-content:center; user-select:none; }}
    .tok.active {{ background:#ffefc2; font-weight:700; }}
    .tok.punct {{ color:#8a4a00; }}
    .legend {{ font-size:12px; color:#666; display:flex; gap:10px; flex-wrap:wrap; }}
    .dot {{ width:10px; height:10px; border-radius:50%; display:inline-block; margin-right:4px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <div class="row">
        <b>{html.escape(wav_name)}</b>
        <span class="pill">fps={fps}</span>
        <span class="pill">duration={duration:.3f}s</span>
        <span class="pill">switches={len(switch_frames)}</span>
        <span class="pill">tokens={len(tokens)}</span>
      </div>
      <div style="margin-top:10px">
        <audio id="audio" src="{wav_esc}" controls preload="metadata" style="width:100%"></audio>
      </div>
    </div>

    <div class="card">
      <canvas id="wave"></canvas>
      <div style="height:8px"></div>
      <div id="textBar"></div>
      <div style="height:8px"></div>
      <div class="legend">
        <span><span class="dot" style="background:#4f46e5"></span>波形</span>
        <span><span class="dot" style="background:#d946ef"></span>切换点</span>
        <span><span class="dot" style="background:#ffefc2"></span>当前字</span>
      </div>
    </div>
  </div>

<script>
const FPS = {fps};
const DURATION = {duration};
const PEAKS = {peaks_json};
const SWITCH = {switch_json};
const TOKENS = {token_json};
const FRAME2TOK = {frame_map_json};

const audio = document.getElementById("audio");
const cv = document.getElementById("wave");
const bar = document.getElementById("textBar");

function resizeCanvas() {{
  const rect = cv.getBoundingClientRect();
  cv.width = Math.max(600, Math.floor(rect.width));
  cv.height = Math.max(120, Math.floor(rect.height));
}}

function drawWave(curT=0) {{
  const ctx = cv.getContext("2d");
  const w = cv.width, h = cv.height;
  ctx.clearRect(0,0,w,h);
  ctx.fillStyle = "#fbfcff";
  ctx.fillRect(0,0,w,h);
  ctx.strokeStyle = "#4f46e5";
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  for(let i=0;i<PEAKS.length;i++) {{
    const x = i * w / Math.max(1, PEAKS.length-1);
    const y = h * 0.5;
    const a = Math.max(0, Math.min(1, PEAKS[i]));
    const dy = a * (h * 0.45);
    ctx.moveTo(x, y - dy);
    ctx.lineTo(x, y + dy);
  }}
  ctx.stroke();

  // switch lines
  ctx.strokeStyle = "#d946ef";
  ctx.lineWidth = 1.1;
  for(const f of SWITCH) {{
    const x = (f / (DURATION * FPS)) * w;
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
  }}

  // playhead
  const px = (curT / Math.max(1e-6, DURATION)) * w;
  ctx.strokeStyle = "#ef4444";
  ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, h); ctx.stroke();
}}

function buildTextBar() {{
  bar.innerHTML = "";
  for(let i=0;i<TOKENS.length;i++) {{
    const t = TOKENS[i];
    const div = document.createElement("div");
    div.className = "tok" + ((/[，,。！？!?；;：:、]/.test(t.token)) ? " punct" : "");
    const left = 100 * (t.f0 / Math.max(1, DURATION*FPS));
    const width = 100 * ((t.f1 - t.f0) / Math.max(1, DURATION*FPS));
    div.style.left = left + "%";
    div.style.width = Math.max(0.25, width) + "%";
    div.textContent = t.token;
    div.title = `${{t.token}} [${{t.f0}}, ${{t.f1}})`;
    bar.appendChild(div);
  }}
}}

function highlightByTime(tSec) {{
  const frame = Math.max(0, Math.min(FRAME2TOK.length-1, Math.floor(tSec * FPS)));
  const idx = FRAME2TOK[frame];
  const nodes = bar.querySelectorAll(".tok");
  nodes.forEach((n, i) => {{
    if(i === idx) n.classList.add("active");
    else n.classList.remove("active");
  }});
}}

window.addEventListener("resize", () => {{ resizeCanvas(); drawWave(audio.currentTime || 0); }});
audio.addEventListener("timeupdate", () => {{
  drawWave(audio.currentTime || 0);
  highlightByTime(audio.currentTime || 0);
}});
audio.addEventListener("loadedmetadata", () => {{
  drawWave(0);
}});

resizeCanvas();
buildTextBar();
drawWave(0);
</script>
</body>
</html>
"""
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html_text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True, help="path to wav file")
    ap.add_argument("--pred_json", required=True, help="outputs/emotion_codes/*.json")
    ap.add_argument("--align_json", default="", help="optional stream align json; if provided, token_spans are used")
    ap.add_argument("--label_path", default="./annotater/labels_new.jsonl")
    ap.add_argument("--text_csv", default="./emotion_results.csv", help="optional csv with wav/transcription columns")
    ap.add_argument("--text", default="", help="optional text override")
    ap.add_argument("--out_dir", default="./outputs/alignment_view")
    args = ap.parse_args()

    wav_path = os.path.abspath(args.wav)
    pred_path = os.path.abspath(args.pred_json)
    wav_name = os.path.basename(wav_path)
    os.makedirs(args.out_dir, exist_ok=True)

    with open(pred_path, "r", encoding="utf-8") as f:
        pred = json.load(f)

    fps = int(pred.get("fps", 30))
    n_frames = int(len(pred.get("frames", [])))
    duration = float(pred.get("duration", 0.0))
    switch_frames = [int(x) for x in pred.get("switch_frames", [])]

    if args.text.strip():
        text = args.text.strip()
    else:
        text = load_csv_text(args.text_csv, wav_name=wav_name)
        if not text:
            text = load_label_text(args.label_path, wav_name)

    tokens = None
    if args.align_json.strip():
        with open(args.align_json, "r", encoding="utf-8") as f:
            aj = json.load(f)
        if not text:
            text = str(aj.get("text", "") or "")
        if not switch_frames:
            switch_frames = [int(x) for x in aj.get("events", [])]
        tokens = token_spans_to_tokens(aj.get("token_spans", []))
    if not tokens:
        tokens = align_text_to_frames(text=text, total_frames=max(1, n_frames), switch_frames=switch_frames)

    frame_to_token = build_frame_to_token(tokens, max(1, n_frames))

    sr, samples = read_wav_mono(wav_path)
    if duration <= 0 and sr > 0:
        duration = len(samples) / float(sr)
    peaks = compute_peaks(samples, n_bins=max(600, int(max(1.0, duration) * 120)))

    # write alignment json
    out_json = os.path.join(args.out_dir, wav_name.replace(".wav", ".align.json"))
    align_obj = {
        "wav": wav_name,
        "fps": fps,
        "duration": duration,
        "text": text,
        "switch_frames": switch_frames,
        "tokens": tokens,
        "frame_to_token": frame_to_token,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(align_obj, f, ensure_ascii=False, indent=2)

    # html path
    out_html = os.path.join(args.out_dir, wav_name.replace(".wav", ".align.html"))
    wav_rel = os.path.relpath(wav_path, start=os.path.abspath(args.out_dir))
    render_html(
        wav_rel=wav_rel,
        wav_name=wav_name,
        fps=fps,
        duration=duration,
        peaks=peaks,
        switch_frames=switch_frames,
        tokens=tokens,
        frame_to_token=frame_to_token,
        out_html=out_html,
    )
    print("wrote:", out_json)
    print("wrote:", out_html)


if __name__ == "__main__":
    main()
