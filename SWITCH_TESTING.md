# 情绪切换点识别测试手册

本文档用于测试“单条音频存在多个情绪切换点”的识别效果，数据基准使用：
- `annotater/labels_new.jsonl`

## 0. 实现原理（核心）

本项目的切换点识别不是“分类一个情绪标签”，而是“在时间轴上检测边界事件”。

### 0.1 训练目标定义

每条音频先被采样到 30fps 帧序列。对每一帧有两个主标签：
- `type_id[t]`：情绪类别
- `level_id[t]`：情绪强度等级

据此构造边界标签：
- `y_bnd[t] = 1` 当且仅当 `type_id[t] != type_id[t-1]` 或 `level_id[t] != level_id[t-1]`
- 否则 `y_bnd[t] = 0`

这意味着：一条音频可以有任意多个切换点，本质是多事件检测问题。

### 0.2 模型输出含义

模型在每帧输出：
- `type` logits
- `lvl` logits
- `bnd` logits（边界头）

推理时通过 `sigmoid(bnd_logits[t])` 得到 `boundary_p[t]`，表示“第 t 帧是切换点”的概率。

### 0.3 从概率到切换点（解码原理）

单看 `boundary_p[t] > 阈值` 会抖动，因此用了状态机解码（见 `infer/postprocess.py`）：

1. `thr_on`：进入候选边界状态  
2. 在候选窗口内持续记录局部最大值（峰值帧）  
3. 满足任一条件就确认该峰值为切换点：
   - 到达 `confirm_win` 确认窗
   - 概率回落到 `thr_off` 以下
4. `min_gap`：与上一个切换点之间至少间隔若干帧，抑制连跳

所以该解码器天然支持多切换点，输出 `switch_frames: List[int]`。

### 0.4 为什么要容差评估（tol）

标注点和预测点可能存在几帧偏差（尤其在真实语音中），因此评估采用“容差匹配”：
- 预测点与某个 GT 点距离 `<= tol`（帧）则记为 TP
- 未匹配预测是 FP，未匹配 GT 是 FN

据此计算：
- `Precision = TP/(TP+FP)`
- `Recall = TP/(TP+FN)`
- `F1 = 2PR/(P+R)`

`tol=5`（约 167ms）更贴近落地体验；`tol=3`（约 100ms）更严格。

### 0.5 参数的工程意义

- `thr_on`：越高越保守（FP 降，FN 可能升）
- `thr_off`：回落确认阈值，越高越早结束候选段
- `confirm_win`：越大越稳，但响应稍慢
- `min_gap`：越大越抗抖，但可能吞掉密集切换

你当前通过网格搜索得到的参数是：
- `thr_on=0.74`
- `thr_off=0.50`
- `confirm_win=2`
- `min_gap=7`

在当前数据上（40条）可达到较高的容差 F1。

## 1. 前置条件

- 已安装并可用的 Python：
  - `E:\Program Files (x86)\Anaconda\envs\s2e\python.exe`
- 已训练完成模型：
  - `outputs/ckpt_bnd_v1/best.pt`

## 2. 生成推理结果

批量推理（每个结果 JSON 内包含 `switch_frames` / `switch_times`）：

```powershell
& "E:\Program Files (x86)\Anaconda\envs\s2e\python.exe" infer/batch_infer.py --wav_dir ./annotater/wavs --ckpt ./outputs/ckpt_bnd_v1/best.pt --out_dir ./outputs/emotion_codes --device cpu
```

单文件推理（可显式指定解码参数）：

```powershell
& "E:\Program Files (x86)\Anaconda\envs\s2e\python.exe" infer/infer_file.py --wav ./annotater/wavs/utt_0001.wav --ckpt ./outputs/ckpt_bnd_v1/best.pt --out ./outputs/emotion_codes/utt_0001.json --device cpu --switch_thr_on 0.74 --switch_thr_off 0.50 --switch_confirm_win 2 --switch_min_gap 7
```

## 3. 量化评估

运行自动评估与调参（容差 ±5 帧）：

```powershell
& "E:\Program Files (x86)\Anaconda\envs\s2e\python.exe" tools/tune_switch_decode.py --pred_dir ./outputs/emotion_codes --label_path ./annotater/labels_new.jsonl --tol 5 --topk 10
```

建议再跑一遍更严格容差（±3 帧）：

```powershell
& "E:\Program Files (x86)\Anaconda\envs\s2e\python.exe" tools/tune_switch_decode.py --pred_dir ./outputs/emotion_codes --label_path ./annotater/labels_new.jsonl --tol 3 --topk 10
```

当前推荐参数（基于现有结果）：
- `thr_on=0.74`
- `thr_off=0.50`
- `confirm_win=2`
- `min_gap=7`

## 4. 可视化检查

单条样本绘图：

```powershell
& "E:\Program Files (x86)\Anaconda\envs\s2e\python.exe" tools/plot_switch_compare.py --pred_dir ./outputs/emotion_codes --label_path ./annotater/labels_new.jsonl --wav utt_0001.wav --tol 5
```

输出路径：
- `outputs/plots/utt_0001.svg`

批量查看最差样本：

```powershell
& "E:\Program Files (x86)\Anaconda\envs\s2e\python.exe" tools/plot_switch_compare_batch.py --pred_dir ./outputs/emotion_codes --label_path ./annotater/labels_new.jsonl --plot_dir ./outputs/plots --tol 5 --worst_k 8 --python "E:\Program Files (x86)\Anaconda\envs\s2e\python.exe"
```

图中颜色说明：
- 蓝线：`boundary_p`（边界概率）
- 绿线：GT 切换点（标注真值）
- 青线：预测 TP（命中）
- 红线：预测 FP（误报）
- 紫线：GT Miss / FN（漏检）
- 橙/黄虚线：阈值线

## 5. 如何判定效果好坏

- 看整体：`F1 / P / R`
- 看稳定性：是否出现大量紧邻双点抖动
- 看定位误差：`tol=5` 与 `tol=3` 差距是否过大

经验：
- 红线太多：提高 `thr_on` 或增大 `min_gap`
- 紫线太多：降低 `thr_on` 或减小 `min_gap`

## 6. 常见问题

- `FutureWarning: torch.load weights_only=False`
  - 警告，可先忽略，不影响当前流程。
- 无法安装 `matplotlib`
  - 可视化脚本已支持 SVG fallback，不依赖 matplotlib 也可出图。
