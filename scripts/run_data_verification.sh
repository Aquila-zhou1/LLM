#!/bin/bash

# 数据验证脚本
# 验证训练数据的token化是否正确

echo "=== 数据验证开始 ==="
echo "时间: $(date)"
echo "工作目录: $(pwd)"

# 检查conda环境
if [[ "$CONDA_DEFAULT_ENV" != "gpt-training" ]]; then
    echo "警告: 当前不在gpt-training环境中"
    echo "请运行: conda activate gpt-training"
    exit 1
fi

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "=== 检查必要文件 ==="
required_files=(
    "data/tinystories_loader.py"
    "scripts/verify_data.py"
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

echo "=== 开始数据验证 ==="
python scripts/verify_data.py

# 检查验证结果
if [[ $? -eq 0 ]]; then
    echo ""
    echo "=== 数据验证完成 ==="
    echo "✓ 所有验证通过，数据准备就绪"
    echo ""
    echo "接下来可以运行:"
    echo "  ./scripts/run_pretrain_simple.sh     # 简单训练"
    echo "  ./scripts/run_pretrain_deepspeed.sh  # DeepSpeed训练"
else
    echo ""
    echo "=== 数据验证失败 ==="
    echo "✗ 请检查数据处理流程"
    exit 1
fi

echo "=== 脚本执行完成 ==="
echo "时间: $(date)"
