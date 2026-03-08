from typing import Dict, List, Tuple


PUNCT = set("，,。！？!?；;：:、")


def _tokenize(text: str) -> List[str]:
    return [c for c in (text or "") if c not in ("\n", "\r", "\t", " ")]


def _token_weight(ch: str) -> float:
    return 1.6 if ch in PUNCT else 1.0


def align_tokens_to_frames(text: str, total_frames: int, anchors: List[int]) -> List[Dict]:
    toks = _tokenize(text)
    if not toks:
        return []
    ws = [_token_weight(t) for t in toks]
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

    punct_idx = [i + 1 for i, t in enumerate(toks) if t in PUNCT]
    for bi in punct_idx:
        pf = edges[bi]
        best = None
        for a in anchors:
            d = abs(int(a) - int(pf))
            if d <= 10 and (best is None or d < best[0]):
                best = (d, int(a))
        if best is not None:
            edges[bi] = best[1]
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = min(total_frames, edges[i - 1] + 1)
    edges[-1] = total_frames

    out = []
    for i, tok in enumerate(toks):
        f0 = int(edges[i])
        f1 = int(edges[i + 1])
        if f1 <= f0:
            f1 = f0 + 1
        out.append({"token": tok, "f0": f0, "f1": f1})
    return out


def build_segments(n_frames: int, switch_frames: List[int], min_len: int = 1) -> List[Tuple[int, int]]:
    n_frames = max(1, int(n_frames))
    xs = sorted(set(int(x) for x in switch_frames if 0 < int(x) < n_frames))
    segs = []
    s = 0
    for x in xs:
        if x - s < int(min_len):
            continue
        segs.append((s, x))
        s = x
    if n_frames - s >= int(min_len):
        segs.append((s, n_frames))
    elif segs:
        a, _ = segs[-1]
        segs[-1] = (a, n_frames)
    else:
        segs = [(0, n_frames)]
    return segs


def _majority(values: List[int]) -> Tuple[int, float]:
    if not values:
        return 0, 0.0
    cnt = {}
    for v in values:
        cnt[int(v)] = cnt.get(int(v), 0) + 1
    best_k = None
    best_c = -1
    for k, c in cnt.items():
        if c > best_c:
            best_k = k
            best_c = c
    conf = best_c / float(len(values))
    return int(best_k), float(conf)


def decode_intervals(
    pred_obj: Dict,
    text: str = "",
    min_seg_len: int = 1,
) -> Dict:
    frames = pred_obj.get("frames", [])
    fps = int(pred_obj.get("fps", 30))
    duration = float(pred_obj.get("duration", 0.0))
    type_map = pred_obj.get("type_map", [])
    n_frames = len(frames)
    switch_frames = [int(x) for x in pred_obj.get("switch_frames", [])]

    segs = build_segments(n_frames=n_frames, switch_frames=switch_frames, min_len=min_seg_len)
    tokens = align_tokens_to_frames(text=text, total_frames=max(1, n_frames), anchors=switch_frames)

    intervals = []
    for i, (s, e) in enumerate(segs):
        seg = frames[s:e]
        type_ids = [int(x.get("type_id", 0)) for x in seg]
        lvl_ids = [int(x.get("level_id", 0)) for x in seg]
        tid, tconf = _majority(type_ids)
        lid, lconf = _majority(lvl_ids)
        tname = type_map[tid] if (isinstance(type_map, list) and 0 <= tid < len(type_map)) else str(tid)

        txt = []
        tok_s = None
        tok_e = None
        for ti, tk in enumerate(tokens):
            ov = max(0, min(e, int(tk["f1"])) - max(s, int(tk["f0"])))
            if ov > 0:
                txt.append(tk["token"])
                if tok_s is None:
                    tok_s = ti
                tok_e = ti

        intervals.append(
            {
                "seg_id": i,
                "start_frame": int(s),
                "end_frame": int(e),
                "start_sec": float(s / float(fps)),
                "end_sec": float(e / float(fps)),
                "emotion_type_id": int(tid),
                "emotion_type": tname,
                "emotion_type_conf": float(tconf),
                "emotion_level": int(lid),
                "emotion_level_conf": float(lconf),
                "text": "".join(txt),
                "token_start": int(tok_s) if tok_s is not None else -1,
                "token_end": int(tok_e) if tok_e is not None else -1,
                "n_tokens": len(txt),
            }
        )

    return {
        "wav": pred_obj.get("wav", ""),
        "fps": fps,
        "duration": duration,
        "n_frames": n_frames,
        "text": text,
        "switch_frames": switch_frames,
        "tokens": tokens,
        "intervals": intervals,
    }

