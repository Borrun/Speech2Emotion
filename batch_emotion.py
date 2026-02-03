from funasr import AutoModel
import json
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import re

# ---------------- 配置 ----------------
AUDIO_DIR = Path("/Users/Zhuanz1/Speech2Emotion/wavs")
SCP_PATH = AUDIO_DIR / "wav.scp"
OUTPUT_JSON = Path("/Users/Zhuanz1/Speech2Emotion/emotion_results.json")
OUTPUT_CSV = Path("/Users/Zhuanz1/Speech2Emotion/emotion_results.csv")
OUTPUT_PLOT = Path("/Users/Zhuanz1/Speech2Emotion/emotion_distribution.png")

BATCH_SIZE = 8
DEVICE = "mps"  # 或 "cpu"
HUB = "hf"      # SenseVoiceSmall 用 hf 更稳定
# ---------------- 配置 ----------------

def generate_wav_scp_if_needed():
    if not SCP_PATH.exists():
        print("wav.scp 不存在，正在自动生成...")
        wav_files = sorted(AUDIO_DIR.glob("*.wav"))
        if not wav_files:
            raise FileNotFoundError(f"文件夹 {AUDIO_DIR} 中没有找到任何 .wav 文件！")
        with open(SCP_PATH, "w", encoding="utf-8") as f:
            for wav in wav_files:
                utt_id = wav.stem
                f.write(f"{utt_id} {wav.absolute()}\n")
        print(f"已生成 wav.scp，包含 {len(wav_files)} 条音频")

def main():
    generate_wav_scp_if_needed()

    print("加载 SenseVoiceSmall 模型（已下载成功，去掉 VAD 避免注册问题）...")
    try:
        model = AutoModel(
            model="FunAudioLLM/SenseVoiceSmall",  # HF 官方 repo，已下载成功
            hub=HUB,
            device=DEVICE,
            trust_remote_code=True,
            disable_update=True,
            # 去掉 vad_model，避免 'fsmn-vad is not registered' 错误
        )
        print("✅ 模型加载成功！")
    except Exception as e:
        print(f"❌ 模型加载失败：{e}")
        return

    print(f"开始批量推理（{DEVICE}, batch_size={BATCH_SIZE}）...")

    res = model.generate(
        input=str(SCP_PATH),
        batch_size=BATCH_SIZE,
        language="auto",      # 自动检测中文
        use_itn=True,         # 数字转写
        # 无需 merge_vad，SenseVoiceSmall 自带 VAD
    )

    results = []
    emotions = []

    print("处理 SenseVoice 输出（提取 <|EMOTION|> token）...")
    emotion_map = {
        "happy": "happy", "sad": "sad", "angry": "angry", "neutral": "neutral",
        "fear": "fearful", "fearful": "fearful", "disgust": "disgusted", 
        "surprise": "surprised", "surprised": "surprised", "unk": "unknown"
    }

    for item in tqdm(res, desc="保存结果", unit="audio"):
        utt_id = item.get("key", item.get("utt", "unknown"))
        text = item.get("text", "")
        
        # SenseVoice 输出示例：'<|zh|><|neutral|>你好世界' 或 '<|HAPPY|>hello'
        emo_match = re.search(r'<\|([A-Z]+)\|>', text, re.IGNORECASE)
        emo_raw = emo_match.group(1).lower() if emo_match else "unknown"
        emo = emotion_map.get(emo_raw, emo_raw if emo_raw != "unk" else "unknown")
        
        # 清理文本：移除所有 <|token|>
        clean_text = re.sub(r'<\|[^>]+>', '', text).strip()
        
        # SenseVoice 是 hard label，score 设为 1.0（后期可加置信度）
        score = 1.0

        results.append({
            "utt_id": utt_id,
            "wav": f"{utt_id}.wav",
            "emotion": emo,
            "score": score,
            "transcription": clean_text[:200] + "..." if len(clean_text) > 200 else clean_text
        })
        emotions.append(emo)

    # 调试输出
    print("\n🔍 调试 - 前 3 条完整原始输出：")
    for i, item in enumerate(res[:3], 1):
        print(f"第{i}条 raw: key='{item.get('key')}', text='{item.get('text', '')[:100]}...'")
        print(f"   → emotion='{results[i-1]['emotion']}', text='{results[i-1]['transcription']}'")

    # 保存 JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✅ JSON: {OUTPUT_JSON}")

    # 保存 CSV
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"✅ CSV: {OUTPUT_CSV}")

    # 情绪分布统计 + 可视化
    if emotions:
        emo_counts = Counter(emotions)
        total = len(emotions)
        print("\n📊 情绪分布统计：")
        for emo, cnt in sorted(emo_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {emo:12} : {cnt:2d} ({cnt/total*100:.1f}%)")

        # 柱状图
        plt.figure(figsize=(10, 6))
        labels, counts = zip(*emo_counts.most_common())
        plt.bar(labels, counts, color='skyblue', alpha=0.7)
        plt.title(f"SenseVoiceSmall Emotion Distribution\n({total} utterances)")
        plt.xlabel("Emotion")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ 分布图: {OUTPUT_PLOT}")

    print("\n🎉 全部完成！检查 JSON/CSV 中的 transcription，验证情绪是否匹配内容。")
    print("如果 emotion 都是 'unknown'，说明音频中性或需调整阈值。")

if __name__ == "__main__":
    main()