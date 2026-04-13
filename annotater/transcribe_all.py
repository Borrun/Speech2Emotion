"""
批量转写 wavs/ 下所有音频，写入 labels_new.jsonl。
- 已有 text 的条目跳过（除非加 --overwrite）
- 已有 curve 等标注数据完整保留
用法：
    python transcribe_all.py
    python transcribe_all.py --overwrite   # 强制重新转写所有
"""
import os, json, argparse

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(BASE_DIR, "wavs")
LABEL_PATH = os.path.join(BASE_DIR, "labels_new.jsonl")

def load_labels():
    labels = {}
    if os.path.exists(LABEL_PATH):
        with open(LABEL_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    obj = json.loads(line)
                    if obj.get("wav"):
                        labels[obj["wav"]] = obj
                except Exception:
                    pass
    return labels

def save_labels(labels: dict):
    tmp = LABEL_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for v in labels.values():
            f.write(json.dumps(v, ensure_ascii=False) + "\n")
    os.replace(tmp, LABEL_PATH)

def collect_wavs():
    wavs = []
    for folder in sorted(os.listdir(AUDIO_DIR)):
        fp = os.path.join(AUDIO_DIR, folder)
        if not os.path.isdir(fp):
            continue
        for fn in sorted(os.listdir(fp)):
            if fn.lower().endswith(".wav"):
                wavs.append((f"{folder}/{fn}", os.path.join(fp, fn)))
    return wavs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true", help="重新转写已有文本的条目")
    args = parser.parse_args()

    print("加载 FunASR paraformer-zh ...")
    from funasr import AutoModel
    model = AutoModel(
        model="paraformer-zh",
        vad_model="fsmn-vad",
        punc_model="ct-punc",
        disable_update=True,
    )
    print("模型加载完成\n")

    labels = load_labels()
    wavs   = collect_wavs()
    total  = len(wavs)

    skipped = 0
    done    = 0
    failed  = 0

    for i, (wav_key, wav_path) in enumerate(wavs, 1):
        existing = labels.get(wav_key, {})
        if existing.get("text") and not args.overwrite:
            skipped += 1
            print(f"[{i}/{total}] 跳过（已有文本）: {wav_key}")
            continue

        print(f"[{i}/{total}] 转写: {wav_key} ...", end=" ", flush=True)
        try:
            res = model.generate(input=wav_path, batch_size_s=300)
            text = res[0].get("text", "").strip() if res else ""
            existing["wav"]  = wav_key
            existing.setdefault("fps", 30)
            existing["text"] = text
            labels[wav_key]  = existing
            save_labels(labels)
            done += 1
            print(f"OK  →  {text[:60]}{'...' if len(text)>60 else ''}")
        except Exception as e:
            failed += 1
            print(f"失败: {e}")

    print(f"\n完成 {done}，跳过 {skipped}，失败 {failed}，共 {total} 个文件")

if __name__ == "__main__":
    main()
