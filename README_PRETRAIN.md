# 预训练完整指南

## 概述

本项目实现了完整的GPT预训练流程，包含详细的数据验证、训练监控和模型评估功能。

## 快速开始

### 1. 环境准备

```bash
conda env create -f scripts/env.yaml --yes
conda activate gpt-training
```

### 2. 数据验证

```bash
# 验证数据token化是否正确
./scripts/run_data_verification.sh
```

### 3. 组件测试

```bash
# 测试所有组件是否正常工作
python scripts/test_components.py
```

### 4. 开始训练

#### 方式1：DeepSpeed训练（推荐）

```bash
# 使用DeepSpeed进行内存优化训练
./scripts/run_pretrain_deepspeed.sh
```

#### 方式2：简单训练

```bash
# 标准PyTorch训练
./scripts/run_pretrain_simple.sh
```

### 5. 模型评估

```bash
# 详细评估训练好的模型
./scripts/run_detailed_evaluation.sh ./outputs/pretrain_XXXXXX/best_model
```

## 输出文件结构

```
outputs/
└── pretrain_20241201_120000/
    ├── best_model/
    │   ├── pytorch_model.bin    # 最佳模型权重
    │   └── config.json          # 模型配置
    ├── checkpoint-1000/         # 定期检查点
    ├── checkpoint-2000/
    └── epoch_1/                 # 轮次检查点
```

## 配置说明

### DeepSpeed配置 (`configs/ds_config_pretrain.json`)

- **ZeRO Stage 2**：优化器状态分割
- **CPU Offload**：将优化器状态卸载到CPU
- **FP16**：混合精度训练
- **批量大小**：32 (8 per GPU × 4 accumulation)

### 模型配置

- **层数**：6层Transformer
- **隐藏维度**：512
- **注意力头**：8个
- **上下文长度**：1024
- **词表大小**：50257 (GPT-2词表)

## 性能目标

- **困惑度目标**：< 40 (文档要求)
- **训练时间**：预计3-8小时（取决于数据量）
- **显存使用**：约6-8GB（RTX 3080）