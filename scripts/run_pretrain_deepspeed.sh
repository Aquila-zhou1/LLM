#!/bin/bash

# GPT预训练脚本 - DeepSpeed版本
# 使用DeepSpeed进行分布式训练和内存优化

echo "=== GPT预训练开始 ==="
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
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
echo ""

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0  # 使用第一张GPU
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 训练参数
OUTPUT_DIR="./outputs/pretrain_$(date +%Y%m%d_%H%M%S)"
DEEPSPEED_CONFIG="./configs/ds_config_pretrain.json"
BATCH_SIZE=8
LEARNING_RATE=1e-4
NUM_EPOCHS=10

echo "=== 训练配置 ==="
echo "输出目录: $OUTPUT_DIR"
echo "DeepSpeed配置: $DEEPSPEED_CONFIG"
echo "批量大小: $BATCH_SIZE"
echo "学习率: $LEARNING_RATE"
echo "训练轮数: $NUM_EPOCHS"
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 检查必要文件
echo "=== 检查必要文件 ==="
required_files=(
    "train/pretrain.py"
    "data/tinystories_loader.py"
    "model/gpt_model.py"
    "$DEEPSPEED_CONFIG"
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

# 测试数据加载器
echo "=== 测试数据加载器 ==="
python -c "
import sys
sys.path.append('.')
from data.tinystories_loader import test_dataloader
test_dataloader()
"

if [[ $? -ne 0 ]]; then
    echo "数据加载器测试失败"
    exit 1
fi
echo "数据加载器测试通过"
echo ""

# 测试模型
echo "=== 测试模型 ==="
python -c "
import sys
sys.path.append('.')
from model.gpt_model import test_model
test_model()
"

if [[ $? -ne 0 ]]; then
    echo "模型测试失败"
    exit 1
fi
echo "模型测试通过"
echo ""

# 开始训练
echo "=== 开始DeepSpeed训练 ==="
echo "命令: deepspeed train/pretrain.py --deepspeed_config $DEEPSPEED_CONFIG --output_dir $OUTPUT_DIR --batch_size $BATCH_SIZE --learning_rate $LEARNING_RATE --num_epochs $NUM_EPOCHS"
echo ""

deepspeed train/pretrain.py \
    --deepspeed_config "$DEEPSPEED_CONFIG" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --num_epochs "$NUM_EPOCHS"

# 检查训练结果
if [[ $? -eq 0 ]]; then
    echo ""
    echo "=== 训练完成 ==="
    echo "输出目录: $OUTPUT_DIR"
    echo "检查点文件:"
    find "$OUTPUT_DIR" -name "*.bin" -o -name "*.json" | head -10
    echo ""
    echo "最佳模型:"
    if [[ -d "$OUTPUT_DIR/best_model" ]]; then
        ls -la "$OUTPUT_DIR/best_model/"
    else
        echo "未找到最佳模型目录"
    fi
else
    echo ""
    echo "=== 训练失败 ==="
    echo "请检查错误信息并重试"
    exit 1
fi

echo "=== 脚本执行完成 ==="
echo "时间: $(date)"
