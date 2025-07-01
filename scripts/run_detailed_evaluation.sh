#!/bin/bash

# 详细模型评估脚本
# 对训练好的模型进行全面评估，包括困惑度计算和文本生成评分

echo "=== 详细模型评估开始 ==="
echo "时间: $(date)"
echo "工作目录: $(pwd)"

# 检查参数
if [[ $# -lt 1 ]]; then
    echo "用法: $0 <模型检查点路径> [批量大小] [评估批次数] [生成样本数]"
    echo "示例: $0 ./outputs/pretrain_20241201_120000/best_model 4 50 5"
    exit 1
fi

CHECKPOINT_PATH="$1"
BATCH_SIZE="${2:-4}"
MAX_EVAL_BATCHES="${3:-50}"
NUM_SAMPLES="${4:-5}"

# 检查conda环境
if [[ "$CONDA_DEFAULT_ENV" != "gpt-training" ]]; then
    echo "警告: 当前不在gpt-training环境中"
    echo "请运行: conda activate gpt-training"
    exit 1
fi

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "=== 评估配置 ==="
echo "模型检查点: $CHECKPOINT_PATH"
echo "批量大小: $BATCH_SIZE"
echo "最大评估批次: $MAX_EVAL_BATCHES"
echo "生成样本数: $NUM_SAMPLES"
echo ""

# 检查模型文件
if [[ ! -d "$CHECKPOINT_PATH" ]] && [[ ! -f "$CHECKPOINT_PATH" ]]; then
    echo "错误: 模型检查点不存在: $CHECKPOINT_PATH"
    exit 1
fi

echo "=== 检查必要文件 ==="
required_files=(
    "scripts/evaluate_model.py"
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

# 检查GPU
echo "=== GPU信息 ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
else
    echo "未检测到nvidia-smi，将在CPU模式下运行"
fi
echo ""

echo "=== 开始详细评估 ==="
echo "这将包括:"
echo "1. 验证集困惑度计算（带详细样例分析）"
echo "2. 文本生成样本（多温度对比和质量评分）"
echo "3. 模型性能统计"
echo ""

python scripts/evaluate_model.py \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --batch_size "$BATCH_SIZE" \
    --max_eval_batches "$MAX_EVAL_BATCHES" \
    --generate_samples \
    --num_samples "$NUM_SAMPLES"

# 检查评估结果
if [[ $? -eq 0 ]]; then
    echo ""
    echo "=== 详细评估完成 ==="
    echo "✓ 评估成功完成"
    echo ""
    echo "评估内容包括:"
    echo "- 验证集困惑度（带样例分析）"
    echo "- 文本生成质量评分"
    echo "- 模型性能统计"
    echo ""
    echo "如需查看更多详细信息，请查看上方的输出日志。"
else
    echo ""
    echo "=== 评估失败 ==="
    echo "✗ 请检查模型文件和配置"
    exit 1
fi

echo "=== 脚本执行完成 ==="
echo "时间: $(date)"
