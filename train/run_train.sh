#!/bin/bash
# run_train.sh - 训练启动脚本
# 路径：/home/borrun/Speech2Emotion/train/run_train.sh

# 读取配置文件并运行训练
CONFIG_FILE="config.yaml"

# 解析 YAML 配置（使用 Python）
python3 << 'EOF'
import yaml
import subprocess
import sys

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

cmd = ["python", "train_emotion_tcn.py"]

# 必需参数
cmd.extend(["--wav_dir", str(config["wav_dir"])])
cmd.extend(["--label_path", str(config["label_path"])])
cmd.extend(["--out_dir", str(config["out_dir"])])

# 可选参数
optional_params = [
    "epochs", "batch_size", "lr", "weight_decay", "seed", "val_ratio",
    "channels", "layers", "dropout", "label_smoothing",
    "w_type", "w_lvl", "w_bnd", "grad_clip", "pos_weight_cap"
]

for param in optional_params:
    if param in config and config[param] is not None:
        cmd.extend([f"--{param}", str(config[param])])

# 布尔参数
if config.get("use_boundary_head", False):
    cmd.append("--use_boundary_head")
else:
    cmd.append("--no_boundary_head")

print("执行命令:")
print(" ".join(cmd))
print("-" * 50)

subprocess.run(cmd)
EOF