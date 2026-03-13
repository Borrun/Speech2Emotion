# Streaming Minimal Runtime (Speech2Emotion)

## 目标
仅保留以下能力：
- 音频 chunk + 文本一次输入的流式边界检测（online fusion）
- 流式情绪在线状态输出（平滑 + 迟滞 + 段主情绪）
- 对齐看板展示
- 标注工具（annotater）

## 当前保留的核心目录
- `annotater/` 标注工具
- `models/` 主模型与特征
- `infer/` 音频推理与边界解码
- `online_emotion/` 文本先验与在线融合检测器
- `tools/`
  - `stream_online_emotion.py` 新增：流式情绪+边界在线输出
  - `stream_fuse_predictions_with_text.py` 融合批处理
  - `export_interval_predictions.py` 区间导出
  - `make_alignment_dashboard.py` 看板数据生成
  - `sim_stream_text_align.py` 流式对齐模拟
  - `archive_unused/` 历史脚本归档
- `outputs/`（已裁剪）
  - `ckpt_bnd_v1/`
  - `test_emotion_codes/`
  - `test_emotion_codes_stream_text_tuned_audio_textsrc/`
  - `test_interval_predictions_stream_text_tuned_audio_textsrc/`
  - `test_alignment_view/`
  - `test_alignment_view_stream_text_tuned_audio_textsrc/`
  - `alignment_view/`
  - `stream_online/`
  - `test_transcriptions_from_audio_chain.csv`

## 最小落地形态
输入：
- 文本一次输入（text once）
- 音频每 10~15 帧一包（chunk streaming）

每包输出：
- 当前帧情绪 `type/level`（raw + stable）
- 是否触发新边界事件
- 当前段 ID 与段主情绪（major emotion）

## 运行示例
```bash
python tools/stream_online_emotion.py \
  --pred_json outputs/test_emotion_codes/web_20260309.122708.683.json \
  --text_csv outputs/test_transcriptions_from_audio_chain.csv \
  --chunk_min 10 --chunk_max 15 \
  --smooth_win 5 --emo_hysteresis 3 \
  --w_audio 0.8 --w_text 0.2 \
  --thr_on 0.62 --thr_off 0.42 --confirm_win 3 --min_gap 5 \
  --out outputs/stream_online/web_20260309.122708.683.stream_online.json
```

输出 JSON 关键字段：
- `chunks[]`：每个 chunk 的在线摘要
- `frames[]`：逐帧 raw/stable 情绪与 segment 状态
- `boundary_events[]`：融合边界事件帧
- `final_segment`：最终段状态
