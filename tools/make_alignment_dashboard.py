import argparse
import base64
import csv
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
            raise RuntimeError(f"unsupported sample width: {sw}")

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


def compute_peaks(samples: List[float], n_bins: int = 1200) -> List[float]:
    n = len(samples)
    if n <= 0:
        return [0.0] * n_bins
    out = []
    for i in range(n_bins):
        s = int(i * n / n_bins)
        e = int((i + 1) * n / n_bins)
        if e <= s:
            e = min(n, s + 1)
        m = 0.0
        for x in samples[s:e]:
            ax = abs(float(x))
            if ax > m:
                m = ax
        out.append(min(1.0, m))
    return out


def token_weight(ch: str) -> float:
    if ch in PUNCT:
        return 1.6
    return 1.0


def align_text_to_frames(text: str, total_frames: int, switch_frames: List[int]) -> List[Dict]:
    chars = [c for c in (text or "") if c not in ("\n", "\r", "\t", " ")]
    if not chars:
        return []
    ws = [token_weight(c) for c in chars]
    s = sum(ws) if sum(ws) > 0 else float(len(ws))
    edges = [0]
    c = 0.0
    for w in ws:
        c += w
        edges.append(int(round(total_frames * c / s)))
    edges[0] = 0
    edges[-1] = total_frames
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = min(total_frames, edges[i - 1] + 1)
    for i in range(len(edges) - 2, -1, -1):
        if edges[i] >= edges[i + 1]:
            edges[i] = max(0, edges[i + 1] - 1)

    punct_idx = [i + 1 for i, ch in enumerate(chars) if ch in PUNCT]
    for bi in punct_idx:
        pf = edges[bi]
        best = None
        for sf in switch_frames:
            d = abs(int(sf) - int(pf))
            if d <= 10 and (best is None or d < best[0]):
                best = (d, int(sf))
        if best is not None:
            edges[bi] = best[1]

    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = min(total_frames, edges[i - 1] + 1)
    edges[-1] = total_frames

    out = []
    for i, ch in enumerate(chars):
        f0 = int(edges[i])
        f1 = int(edges[i + 1])
        if f1 <= f0:
            f1 = f0 + 1
        out.append({"token": ch, "f0": f0, "f1": f1})
    return out


def build_frame_to_token(tokens: List[Dict], total_frames: int) -> List[int]:
    out = [-1] * total_frames
    for i, t in enumerate(tokens):
        f0 = max(0, min(total_frames, int(t["f0"])))
        f1 = max(0, min(total_frames, int(t["f1"])))
        for f in range(f0, f1):
            out[f] = i
    return out


def load_csv_text_map(path: str) -> Dict[str, str]:
    out = {}
    if not os.path.isfile(path):
        return out
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
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


def render_dashboard_html(out_html: str, items: List[Dict]):
    data_inline = json.dumps({"items": items}, ensure_ascii=False)
    data_b64 = base64.b64encode(data_inline.encode("utf-8")).decode("ascii")
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Alignment Dashboard</title>
  <style>
    body {{ margin:0; font-family: system-ui, -apple-system, sans-serif; background:#f7f8fb; color:#222; }}
    .wrap {{ display:grid; grid-template-columns:320px 1fr; height:100vh; }}
    .left {{ border-right:1px solid #dfe3ec; background:#fff; overflow:auto; }}
    .left .hdr {{ position:sticky; top:0; background:#fff; z-index:2; padding:12px; border-bottom:1px solid #eef1f6; }}
    .left input {{ width:100%; padding:8px 10px; border:1px solid #d7dce6; border-radius:8px; }}
    #list .item {{ padding:10px 12px; border-bottom:1px solid #f2f4f8; cursor:pointer; font-size:13px; }}
    #list .item:hover {{ background:#f7fbff; }}
    #list .item.active {{ background:#ecf3ff; border-left:4px solid #3b82f6; padding-left:8px; }}
    .right {{ padding:14px; overflow:auto; }}
    .card {{ background:#fff; border:1px solid #e2e8f0; border-radius:12px; padding:12px; margin-bottom:12px; }}
    .row {{ display:flex; gap:12px; align-items:center; flex-wrap:wrap; }}
    .pill {{ padding:2px 9px; border-radius:999px; background:#eef2ff; color:#334e9a; font-size:12px; }}
    #wave {{ width:100%; height:200px; border:1px solid #e5e7eb; border-radius:10px; background:#fcfcff; }}
    #textBar {{ position:relative; width:100%; height:80px; border:1px solid #e5e7eb; border-radius:10px; overflow:hidden; }}
    .tok {{ position:absolute; top:8px; height:64px; border-right:1px solid #f3f4f6; display:flex; align-items:center; justify-content:center; font-size:14px; }}
    .tok.active {{ background:#ffefc2; font-weight:700; }}
    .tok.punct {{ color:#975a16; }}
    #meta {{ font-size:13px; color:#4b5563; }}
    #intervals {{ max-height: 260px; overflow:auto; border:1px solid #e5e7eb; border-radius:10px; }}
    .seg {{ padding:8px 10px; border-bottom:1px solid #f3f4f8; font-size:13px; }}
    .seg.active {{ background:#fff4d6; }}
    .seg .t {{ color:#374151; font-weight:600; }}
    .seg .x {{ color:#6b7280; margin-top:3px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="left">
      <div class="hdr">
        <input id="q" placeholder="搜索 wav..."/>
      </div>
      <div id="list"></div>
    </div>
    <div class="right">
      <div class="card">
        <div class="row">
          <b id="name">-</b>
          <span class="pill" id="fps">fps=30</span>
          <span class="pill" id="dur">duration=0</span>
          <span class="pill" id="sw">switches=0</span>
          <span class="pill" id="tk">tokens=0</span>
        </div>
        <div id="meta" style="margin-top:8px">-</div>
        <div style="margin-top:10px">
          <audio id="audio" controls preload="metadata" style="width:100%"></audio>
        </div>
      </div>
      <div class="card">
        <canvas id="wave"></canvas>
        <div style="height:8px"></div>
        <div id="textBar"></div>
      </div>
      <div class="card">
        <b>区间标注（情绪 + 文本 + 音频）</b>
        <div style="height:8px"></div>
        <div id="intervals"></div>
      </div>
    </div>
  </div>

<script>
let DATA = [];
let CUR = null;
const q = document.getElementById('q');
const list = document.getElementById('list');
const audio = document.getElementById('audio');
const nameEl = document.getElementById('name');
const fpsEl = document.getElementById('fps');
const durEl = document.getElementById('dur');
const swEl = document.getElementById('sw');
const tkEl = document.getElementById('tk');
const metaEl = document.getElementById('meta');
const cv = document.getElementById('wave');
const bar = document.getElementById('textBar');
const intervalsEl = document.getElementById('intervals');

function resizeCanvas() {{
  const r = cv.getBoundingClientRect();
  cv.width = Math.max(800, Math.floor(r.width));
  cv.height = Math.max(140, Math.floor(r.height));
}}

function drawWave(t=0) {{
  if(!CUR) return;
  const peaks = CUR.peaks || [];
  const sw = CUR.switch_frames || [];
  const w = cv.width, h = cv.height;
  const ctx = cv.getContext('2d');
  ctx.clearRect(0,0,w,h);
  ctx.fillStyle = '#fcfcff';
  ctx.fillRect(0,0,w,h);
  ctx.strokeStyle = '#4f46e5';
  ctx.lineWidth = 1.0;
  ctx.beginPath();
  for(let i=0;i<peaks.length;i++) {{
    const x = i * w / Math.max(1, peaks.length-1);
    const y = h * 0.5;
    const dy = Math.max(0, Math.min(1, peaks[i])) * h * 0.45;
    ctx.moveTo(x, y-dy); ctx.lineTo(x, y+dy);
  }}
  ctx.stroke();
  ctx.strokeStyle = '#d946ef'; ctx.lineWidth = 1.1;
  for(const f of sw) {{
    const x = (f / Math.max(1, CUR.n_frames)) * w;
    ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,h); ctx.stroke();
  }}
  const px = (t / Math.max(1e-6, CUR.duration)) * w;
  ctx.strokeStyle = '#ef4444'; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(px,0); ctx.lineTo(px,h); ctx.stroke();
}}

function buildTextBar() {{
  bar.innerHTML = '';
  if(!CUR) return;
  const total = Math.max(1, CUR.n_frames);
  for(let i=0;i<CUR.tokens.length;i++) {{
    const t = CUR.tokens[i];
    const d = document.createElement('div');
    const isP = /[，,。！？!?；;：:、]/.test(t.token);
    d.className = 'tok' + (isP ? ' punct' : '');
    d.style.left = (100 * t.f0 / total) + '%';
    d.style.width = Math.max(0.25, 100 * (t.f1 - t.f0) / total) + '%';
    d.textContent = t.token;
    bar.appendChild(d);
  }}
}}

function highlight(tsec) {{
  if(!CUR) return;
  const frame = Math.max(0, Math.min(CUR.frame_to_token.length-1, Math.floor(tsec * CUR.fps)));
  const ti = CUR.frame_to_token[frame];
  const nodes = bar.querySelectorAll('.tok');
  nodes.forEach((n,i) => n.classList.toggle('active', i===ti));
}}

function renderIntervals() {{
  intervalsEl.innerHTML = '';
  if(!CUR) return;
  const xs = CUR.intervals || [];
  for(let i=0;i<xs.length;i++) {{
    const s = xs[i];
    const d = document.createElement('div');
    d.className = 'seg';
    d.dataset.i = String(i);
    d.innerHTML = `<div class="t">[${{s.start_sec.toFixed(2)}}s ~ ${{s.end_sec.toFixed(2)}}s]  ${{s.emotion_type}} / L${{s.emotion_level}}</div>
                   <div class="x">${{s.text || '(空)'}} </div>`;
    d.onclick = () => {{
      audio.currentTime = s.start_sec;
      drawWave(audio.currentTime || 0);
      highlight(audio.currentTime || 0);
      highlightInterval(audio.currentTime || 0);
    }};
    intervalsEl.appendChild(d);
  }}
}}

function highlightInterval(tsec) {{
  if(!CUR) return;
  const xs = CUR.intervals || [];
  let idx = -1;
  for(let i=0;i<xs.length;i++) {{
    if(tsec >= xs[i].start_sec && tsec < xs[i].end_sec) {{
      idx = i; break;
    }}
  }}
  const nodes = intervalsEl.querySelectorAll('.seg');
  nodes.forEach((n,i) => n.classList.toggle('active', i===idx));
}}

function selectItem(item) {{
  CUR = item;
  nameEl.textContent = item.wav;
  fpsEl.textContent = 'fps=' + item.fps;
  durEl.textContent = 'duration=' + item.duration.toFixed(3) + 's';
  swEl.textContent = 'switches=' + item.switch_frames.length;
  tkEl.textContent = 'tokens=' + item.tokens.length;
  metaEl.textContent = item.text || '(no text)';
  audio.src = item.wav_rel;
  buildTextBar();
  renderIntervals();
  drawWave(0);
  [...list.querySelectorAll('.item')].forEach(n => n.classList.toggle('active', n.dataset.wav === item.wav));
}}

function renderList() {{
  const key = (q.value || '').toLowerCase().trim();
  list.innerHTML = '';
  const xs = DATA.filter(x => !key || x.wav.toLowerCase().includes(key));
  for(const it of xs) {{
    const d = document.createElement('div');
    d.className = 'item';
    d.dataset.wav = it.wav;
    d.textContent = it.wav + '  (' + it.switch_frames.length + ' switch)';
    d.onclick = () => selectItem(it);
    list.appendChild(d);
  }}
  if(xs.length && !CUR) selectItem(xs[0]);
}}

q.addEventListener('input', renderList);
audio.addEventListener('timeupdate', () => {{
  drawWave(audio.currentTime || 0);
  highlight(audio.currentTime || 0);
  highlightInterval(audio.currentTime || 0);
}});
window.addEventListener('resize', () => {{ resizeCanvas(); drawWave(audio.currentTime||0); }});

const DATA_B64 = "{data_b64}";
const INLINE = JSON.parse(decodeURIComponent(escape(atob(DATA_B64))));
DATA = INLINE.items || [];
resizeCanvas();
renderList();
</script>
</body>
</html>
"""
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav_dir", default="./annotater/wavs")
    ap.add_argument("--pred_dir", default="./outputs/emotion_codes")
    ap.add_argument("--text_csv", default="./emotion_results.csv")
    ap.add_argument("--intervals_dir", default="./outputs/multimodal_intervals")
    ap.add_argument("--out_dir", default="./outputs/alignment_view")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    text_map = load_csv_text_map(args.text_csv)
    pred_files = [x for x in os.listdir(args.pred_dir) if x.lower().endswith(".json")]
    pred_files.sort()

    items = []
    for fn in pred_files:
        pred_path = os.path.join(args.pred_dir, fn)
        with open(pred_path, "r", encoding="utf-8") as f:
            pred = json.load(f)
        wav = str(pred.get("wav", fn.replace(".json", ".wav")))
        wav_path = os.path.join(args.wav_dir, wav)
        if not os.path.isfile(wav_path):
            continue
        frames = pred.get("frames", [])
        n_frames = len(frames)
        fps = int(pred.get("fps", 30))
        duration = float(pred.get("duration", 0.0))
        switch_frames = [int(x) for x in pred.get("switch_frames", [])]
        text = text_map.get(wav, "")

        tokens = align_text_to_frames(text, max(1, n_frames), switch_frames)
        frame_to_token = build_frame_to_token(tokens, max(1, n_frames))
        intervals = []
        cand = [
            os.path.join(args.intervals_dir, wav.replace(".wav", ".intervals.json")),
            os.path.join(args.intervals_dir, wav.replace(".wav", ".interval_pred.json")),
        ]
        for int_path in cand:
            if os.path.isfile(int_path):
                try:
                    with open(int_path, "r", encoding="utf-8") as f:
                        intervals = json.load(f).get("intervals", [])
                    if intervals:
                        break
                except Exception:
                    intervals = []
        sr, samples = read_wav_mono(wav_path)
        if duration <= 0 and sr > 0:
            duration = len(samples) / float(sr)
        peaks = compute_peaks(samples, n_bins=1200)

        wav_rel = os.path.relpath(wav_path, start=os.path.abspath(args.out_dir)).replace("\\", "/")
        items.append(
            {
                "wav": wav,
                "wav_rel": wav_rel,
                "fps": fps,
                "duration": duration,
                "n_frames": n_frames,
                "switch_frames": switch_frames,
                "text": text,
                "tokens": tokens,
                "frame_to_token": frame_to_token,
                "peaks": peaks,
                "intervals": intervals,
            }
        )

    out_json = os.path.join(args.out_dir, "dashboard_data.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"items": items}, f, ensure_ascii=False)
    out_html = os.path.join(args.out_dir, "dashboard.html")
    render_dashboard_html(out_html=out_html, items=items)

    print("wrote:", out_json)
    print("wrote:", out_html)
    print("items:", len(items))


if __name__ == "__main__":
    main()
