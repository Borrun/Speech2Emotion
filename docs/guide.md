# Speech2Emotion 训练 / 测试 / 调试指南

---

## 一、环境准备

```bash
# 依赖
pip install torch torchaudio pyyaml

# 可选（推理/评估/导出）
pip install onnx onnxruntime soundfile flask scipy matplotlib
```

---

## 二、训练

### 2.1 快速开始

```bash
# 使用配置文件（推荐）
python train/train_emotion_tcn.py --config train/config.yaml

# 临时覆盖参数（命令行优先级 > config.yaml > 硬编码默认值）
python train/train_emotion_tcn.py --config train/config.yaml --epochs 200 --lr 0.0005
```

### 2.2 配置文件说明（train/config.yaml）

```yaml
# ============ 路径 ============
wav_dir: "/path/to/wavs"                    # 音频目录
label_path: "/path/to/labels_new.jsonl"     # JSONL 标签
out_dir: "/path/to/output"                  # 输出目录（checkpoint 保存位置）

# ============ 训练参数 ============
epochs: 100
batch_size: 8
lr: 0.0001              # 学习率
weight_decay: 0.0001     # AdamW 权重衰减
seed: 42

# ============ 模型参数 ============
channels: 128            # TCN 通道数
layers: 6                # 因果卷积块数（感受野 = 3 * 2^layers - 1 帧）
dropout: 0.1

# ============ 边界检测头 ============
use_boundary_head: true

# ============ 损失函数 ============
w_type: 1.0              # 情感类型 loss 权重
w_lvl: 1.0               # 强度等级 loss 权重
w_bnd: 0.5               # 边界检测 loss 权重
label_smoothing: 0.1     # 标准 label smoothing（仅 CE loss 生效）
grad_clip: 1.0           # 梯度裁剪
pos_weight_cap: 10.0     # 边界 BCE pos_weight 上限

# ============ 亲缘损失（替代 type 的 CrossEntropy）============
use_affinity_loss: true           # 启用后 label_smoothing 对 type loss 不再生效
sibling_share: 0.08               # 兄弟类概率（如 angry ↔ angry_confused）
base_smooth: 0.02                 # 近邻类基础平滑系数

# ============ 验证集切分 ============
val_category: "不确定基调"         # 从该目录取验证集
n_val: 15                         # 验证集条数

# ============ 数据增广 ============
no_aug: false                     # true 禁用全部增广
aug_p_speed: 0.5                  # 变速概率
aug_p_noise: 0.5                  # 加噪概率
aug_snr_min: 15.0                 # 最小 SNR (dB)
aug_snr_max: 30.0                 # 最大 SNR (dB)
```

### 2.3 关键超参调优建议

| 场景 | 建议 |
|------|------|
| 过拟合（train↑ val↓） | 增大 `dropout`(0.2)、减小 `channels`(64)、增大增广概率 |
| 欠拟合（train/val 都低）| 增大 `channels`(256)、增加 `layers`(8)、减小 `dropout` |
| 稀有类别差 | 启用 `use_affinity_loss`，增大 `sibling_share`(0.10~0.15) |
| 边界检测不准 | 调整 `w_bnd`(0.3~1.0)、`pos_weight_cap`(5~30) |

### 2.4 输出文件

训练完成后 `out_dir/` 下产生：
```
out_dir/
└── best.pt              # 最优 checkpoint（按 acc_type + acc_lvl + f1_bnd 选取）
```

checkpoint 内包含完整模型配置，推理时无需手动指定模型参数。

---

## 三、推理 / 测试

### 3.1 单文件推理

```bash
python infer/infer_file.py \
    --wav test.wav \
    --ckpt outputs/best.pt \
    --out outputs/result.json
```

可选参数：
```
--device cpu|cuda
--switch_thr_on 0.78       # 边界检测激活阈值
--switch_thr_off 0.60      # 边界检测复位阈值
--switch_confirm_win 3     # 确认窗口（帧）
--switch_min_gap 5         # 最小间隔（帧）
```

输出 JSON 结构：
```json
{
  "wav": "test.wav",
  "fps": 30,
  "duration": 5.2,
  "frames": [
    {"frame_idx": 0, "type_id": 2, "level_id": 3, "boundary_p": 0.01, ...},
    ...
  ],
  "switch_frames": [45, 112],
  "switch_times": [1.5, 3.73],
  "cpp_emotion_sync": { "timeline": [...], "segments": [...] }
}
```

### 3.2 批量推理

```bash
python infer/batch_infer.py \
    --wav_dir ./test_wavs \
    --ckpt outputs/best.pt \
    --out_dir outputs/test_predictions
```

递归处理目录下所有 `.wav` 文件，每个文件输出一个 `.json`。

### 3.3 流式推理（模拟在线场景）

```bash
# 带文本融合的全流水线
python tools/scan_stream_all.py --tag my_run

# 文本+音频边界融合
python tools/stream_fuse_predictions_with_text.py \
    --pred_dir ./outputs/test_emotion_codes \
    --text_csv ./test/transcriptions.csv \
    --out_dir ./outputs/fused
```

### 3.4 评估

```bash
python tools/eval_text_fusion.py \
    --tags baseline my_run \
    --labels ./annotater/labels_new.jsonl \
    --out_dir ./outputs/stream_online
```

输出指标：
- `type_acc` — 情感类型分类准确率
- `level_mae` — 强度等级平均绝对误差
- `constrained%` — 文本约束生效帧比例
- 按情感类型拆分的准确率 + 混淆矩阵

### 3.5 ONNX 导出

```bash
python tools/export_onnx.py \
    --ckpt outputs/best.pt \
    --out outputs/emotion_tcn.onnx
```

导出后自动进行数值验证（PyTorch vs ONNX 输出对比）。

---

## 四、可视化 & 标注工具

### 4.1 Web 可视化服务器

```bash
# 设置环境变量
export WAV_DIR=./test_wavs
export LABEL_PATH=./annotater/labels_new.jsonl
export PRED_PATH=./outputs/test_predictions
export PORT=7861

python test_server.py
# 浏览器打开 http://localhost:7861
```

API 端点：
```
GET /api/files                  # 文件列表
GET /api/pred/<wav>             # 预测结果
GET /api/label/<wav>            # 标注数据
GET /audio/<wav>                # 音频流
```

### 4.2 情感模板渲染

```bash
python tools/render_eye_socket_levels.py \
    --data-dir ./annotater/data \
    --out-dir ./outputs/eye_socket_views
```

---

## 五、调试指南

### 5.1 训练 loss 异常排查

**loss 为 NaN：**
```bash
# 1. 减小学习率
--lr 0.00005

# 2. 增大梯度裁剪
--grad_clip 0.5

# 3. 检查音频是否有全零/极短文件
python -c "
from train.emotion_data import load_items, load_wav
items = load_items('annotater/labels_new.jsonl')
for it in items:
    wav = load_wav(f'annotater/wavs/{it[\"wav\"]}')
    dur = wav.size(1) / 16000
    if dur < 0.5 or wav.abs().max() < 1e-6:
        print(f'问题文件: {it[\"wav\"]}  时长={dur:.2f}s  max_amp={wav.abs().max():.6f}')
"
```

**type_acc 停在 ~20%（随机水平）：**
- 检查标签分布是否极度不均衡
- 启用 `use_affinity_loss` 或增大 `label_smoothing`
- 确认 `ALLOWED_TYPES` 与标注中的 `type` 字段匹配

### 5.2 验证数据加载

```python
from train.emotion_data import DataConfig, EmotionSeqDataset, load_items, collate, ALLOWED_TYPES

items = load_items("annotater/labels_new.jsonl")
print(f"总样本: {len(items)}")

cfg = DataConfig(wav_dir="annotater/wavs", label_path="annotater/labels_new.jsonl")
ds = EmotionSeqDataset(cfg, items, aug_config=None)

# 检查单个样本
sample = ds[0]
print(f"wav: {sample['wav']}")
print(f"mel shape: {sample['mel'].shape}")      # 期望 [T, 80]
print(f"y_type shape: {sample['y_type'].shape}") # 期望 [T]
print(f"y_lvl shape: {sample['y_lvl'].shape}")   # 期望 [T]
print(f"y_bnd sum: {sample['y_bnd'].sum()}")     # 边界帧数

# 检查 batch collate
from torch.utils.data import DataLoader
loader = DataLoader(ds, batch_size=4, collate_fn=collate)
batch = next(iter(loader))
print(f"batch mel: {batch['mel'].shape}")        # 期望 [B, maxT, 80]
print(f"batch mask: {batch['mask'].sum(dim=1)}") # 每条有效帧数
```

### 5.3 验证亲缘损失软标签

```python
from train.emotion_data import AffinityAwareLoss, ALLOWED_TYPES

loss_fn = AffinityAwareLoss(num_classes=10, sibling_share=0.08, base_smooth=0.02)

# 打印软标签矩阵
for i, name in enumerate(ALLOWED_TYPES):
    row = loss_fn.soft_targets[i]
    top3 = row.topk(3)
    print(f"{name:18s} → 主类={row[i]:.3f}  "
          f"top3: {', '.join(f'{ALLOWED_TYPES[j]}={v:.3f}' for v, j in zip(top3.values, top3.indices))}")
```

### 5.4 推理结果检查

```python
import json

with open("outputs/result.json") as f:
    pred = json.load(f)

print(f"时长: {pred['duration']:.2f}s  帧数: {len(pred['frames'])}")
print(f"切换点: {pred['switch_times']}")

# 统计情感分布
from collections import Counter
types = Counter(f['type_id'] for f in pred['frames'])
print(f"情感分布: {dict(types)}")
```

### 5.5 Checkpoint 内容检查

```python
import torch

ckpt = torch.load("outputs/best.pt", map_location="cpu")
print(f"训练 epoch: {ckpt['epoch']}")
print(f"模型配置: {ckpt['cfg']}")
print(f"边界正样本比例: {ckpt['pos_ratio']:.6f}")
print(f"参数量: {sum(p.numel() for p in ckpt['model'].values()):,}")

# 查看是否使用了亲缘损失
if ckpt['cfg'].get('use_affinity_loss'):
    print(f"亲缘损失: sibling_share={ckpt['cfg']['sibling_share']} "
          f"base_smooth={ckpt['cfg']['base_smooth']}")
```

### 5.6 常见问题

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| `RuntimeError: failed to load wav` | 音频格式不支持 / 路径错误 | 确认路径、尝试 `ffmpeg -i x.wav -ar 16000 -ac 1 out.wav` |
| `CUDA out of memory` | batch_size 过大 / 音频过长 | 减小 `batch_size`，或在数据中过滤超长音频 |
| val_acc 不提升 | 验证集太小 / 过拟合 | 增大 `n_val`，增大 `dropout`，启用增广 |
| 边界 F1 = 0 | `pos_weight_cap` 过低 / 边界太稀疏 | 增大 `pos_weight_cap`(20~50)，增大 `w_bnd` |
| ONNX 导出失败 | 动态 shape 不兼容 | 确认 `use_boundary_head` 与 checkpoint 一致 |

---

## 六、完整工作流示例

```bash
# 1. 训练
python train/train_emotion_tcn.py --config train/config.yaml

# 2. 批量推理
python infer/batch_infer.py \
    --wav_dir ./annotater/wavs \
    --ckpt ./output/best.pt \
    --out_dir ./outputs/test_emotion_codes

# 3. 评估
python tools/eval_text_fusion.py \
    --tags baseline \
    --labels ./annotater/labels_new.jsonl

# 4. 导出 ONNX（部署用）
python tools/export_onnx.py \
    --ckpt ./output/best.pt \
    --out ./outputs/emotion_tcn.onnx

# 5. 可视化
WAV_DIR=./annotater/wavs PRED_PATH=./outputs/test_emotion_codes python test_server.py
```
