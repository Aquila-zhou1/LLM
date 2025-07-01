#!/bin/bash

# GPT预训练脚本 - 专门显示评估时的token解码内容
# 注释掉其他token展示，只在评估时显示详细的token解码

echo "=== GPT预训练开始（评估token解码版本）==="
echo "时间: $(date)"
echo "工作目录: $(pwd)"

# 检查conda环境
if [[ "$CONDA_DEFAULT_ENV" != "gpt-training" ]]; then
    echo "警告: 当前不在gpt-training环境中"
    echo "请运行: conda activate gpt-training"
    exit 1
fi

# 检查GPU
echo "=== GPU信息 ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
else
    echo "未检测到nvidia-smi，可能在CPU模式下运行"
fi
echo ""

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0  # 使用第一张GPU
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 训练参数
OUTPUT_DIR="./outputs/pretrain_eval_tokens_$(date +%Y%m%d_%H%M%S)"
BATCH_SIZE=4  # 使用较小的批量大小以便更频繁地看到评估
LEARNING_RATE=1e-4
NUM_EPOCHS=3  # 较少的轮数用于测试

echo "=== 训练配置 ==="
echo "输出目录: $OUTPUT_DIR"
echo "批量大小: $BATCH_SIZE"
echo "学习率: $LEARNING_RATE"
echo "训练轮数: $NUM_EPOCHS"
echo "评估间隔: 每200步（更频繁的评估）"
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 检查必要文件
echo "=== 检查必要文件 ==="
required_files=(
    "train/pretrain.py"
    "data/tinystories_loader.py"
    "model/gpt_model.py"
)

for file in "${required_files[@]}"; do
    if [[ -f "$file" ]]; then
        echo "✓ $file"
    else
        echo "✗ $file (缺失)"
        exit 1
    fi
done
echo ""

echo "=== 特别说明 ==="
echo "本次训练将在评估时显示详细的token解码内容，包括："
echo "- 原始完整文本"
echo "- 输入文本部分"
echo "- 目标文本部分"
echo "- 前10个token的预测vs真实对比"
echo "- 每个token的预测概率"
echo ""

# 开始训练
echo "=== 开始训练（专注评估token解码）==="

# 创建临时配置，设置更频繁的评估
python -c "
import sys
sys.path.append('.')
from train.pretrain import PretrainConfig, train_model

# 创建配置
config = PretrainConfig()
config.output_dir = '$OUTPUT_DIR'
config.batch_size = $BATCH_SIZE
config.learning_rate = $LEARNING_RATE
config.num_epochs = $NUM_EPOCHS
config.eval_interval = 200  # 每200步评估一次（更频繁）
config.eval_steps = 50      # 每次评估50步
config.log_interval = 25    # 每25步打印日志

print('开始训练，重点关注评估时的token解码内容...')
train_model(config, deepspeed_config=None)
"

# 检查训练结果
if [[ $? -eq 0 ]]; then
    echo ""
    echo "=== 训练完成 ==="
    echo "输出目录: $OUTPUT_DIR"
    echo ""
    echo "在训练过程中，您应该已经看到了详细的评估token解码内容，包括："
    echo "- 每20个评估步骤的token解码分析"
    echo "- 原始文本、输入文本、目标文本的对比"
    echo "- 前10个token的预测准确性分析"
    echo ""
    echo "检查点文件:"
    find "$OUTPUT_DIR" -name "*.bin" -o -name "*.json" | head -5
else
    echo ""
    echo "=== 训练失败 ==="
    echo "请检查错误信息并重试"
    exit 1
fi

echo "=== 脚本执行完成 ==="
echo "时间: $(date)"
