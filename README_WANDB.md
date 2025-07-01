# Wandb集成使用指南

## 📊 概述

本项目已集成Wandb（Weights & Biases）实验跟踪功能，可以实时监控训练和评估的loss和困惑度（PPL）。

## 🚀 快速开始

### 1. 安装wandb
```bash
pip install wandb
```

### 2. 登录wandb
```bash
wandb login
```
首次使用需要在 https://wandb.ai 注册账号并获取API key。

### 3. 启用wandb训练
```bash
# DeepSpeed训练（默认启用wandb）
./scripts/run_pretrain_deepspeed.sh

# 简单训练（默认启用wandb）
./scripts/run_pretrain_simple.sh

# 手动指定wandb参数
python train/pretrain.py \
    --use_wandb \
    --wandb_project "my-gpt-project" \
    --wandb_run_name "experiment-1" \
    --output_dir ./outputs \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --num_epochs 10
```

## 📈 监控指标

### 训练指标 (train/)
- **train/loss**: 训练损失
- **train/perplexity**: 训练困惑度
- **train/learning_rate**: 学习率
- **train/epoch**: 当前轮次
- **train/step**: 全局步数
- **train/elapsed_time**: 每步用时

### 评估指标 (eval/)
- **eval/loss**: 验证集损失
- **eval/perplexity**: 验证集困惑度
- **eval/step**: 评估步数
- **eval/best_perplexity**: 历史最佳困惑度

### 轮次指标 (epoch/)
- **epoch/train_loss**: 轮次平均训练损失
- **epoch/train_perplexity**: 轮次平均训练困惑度
- **epoch/val_loss**: 轮次验证损失
- **epoch/val_perplexity**: 轮次验证困惑度
- **epoch/number**: 轮次编号
- **epoch/best_perplexity**: 当前最佳困惑度

### 最终指标 (final/)
- **final/best_perplexity**: 最终最佳困惑度
- **final/total_steps**: 总训练步数
- **final/total_epochs**: 总训练轮数

## 🎯 使用示例

### 基本使用
```bash
# 启用wandb的简单训练
./scripts/run_pretrain_simple.sh
```

### 自定义项目名称
修改脚本中的wandb参数：
```bash
# 在脚本中修改
WANDB_PROJECT="my-custom-project"
WANDB_RUN_NAME="experiment-$(date +%Y%m%d_%H%M%S)"
```

### 禁用wandb
```bash
# 修改脚本中的USE_WANDB=false
# 或者直接运行python命令不加--use_wandb参数
python train/pretrain.py --output_dir ./outputs --batch_size 8 --num_epochs 3
```

## 📊 Wandb界面功能

### 1. 实时图表
- **Loss曲线**: 训练和验证损失的实时变化
- **困惑度曲线**: 训练和验证困惑度的实时变化
- **学习率曲线**: 学习率调度的变化

### 2. 系统监控
- **GPU使用率**: 实时GPU利用率和显存使用
- **CPU使用率**: CPU和内存使用情况
- **网络I/O**: 数据加载性能

### 3. 超参数跟踪
- **模型配置**: 层数、隐藏维度、注意力头数等
- **训练配置**: 批量大小、学习率、优化器参数等
- **数据配置**: 序列长度、词表大小等

### 4. 实验比较
- **多实验对比**: 不同超参数设置的效果对比
- **最佳模型追踪**: 自动记录最佳困惑度和对应的模型

## 🔧 配置选项

### 命令行参数
```bash
--use_wandb              # 启用wandb跟踪
--wandb_project PROJECT  # wandb项目名称（默认：gpt-pretrain）
--wandb_entity ENTITY    # wandb团队名称（可选）
--wandb_run_name NAME    # 运行名称（可选，默认自动生成）
```

### 配置类参数
在 `PretrainConfig` 类中：
```python
self.use_wandb = True                    # 是否使用wandb
self.wandb_project = "gpt-pretrain"      # 项目名称
self.wandb_entity = None                 # 团队名称
self.wandb_run_name = None               # 运行名称
```

## 📝 最佳实践

### 1. 命名规范
```bash
# 推荐的运行名称格式
WANDB_RUN_NAME="model-${MODEL_SIZE}-lr-${LEARNING_RATE}-$(date +%Y%m%d_%H%M%S)"

# 例如
WANDB_RUN_NAME="model-6layer-lr-1e4-20241201_120000"
```

### 2. 项目组织
```bash
# 按实验类型组织项目
WANDB_PROJECT="gpt-pretrain-baseline"     # 基线实验
WANDB_PROJECT="gpt-pretrain-ablation"     # 消融实验
WANDB_PROJECT="gpt-pretrain-hyperopt"     # 超参数优化
```

### 3. 标签使用
在wandb界面中为实验添加标签：
- `baseline`: 基线模型
- `large-model`: 大模型实验
- `fast-training`: 快速训练测试
- `production`: 生产环境模型

## 🔍 故障排除

### 常见问题

1. **wandb未安装**
   ```bash
   pip install wandb
   ```

2. **未登录wandb**
   ```bash
   wandb login
   ```

3. **网络连接问题**
   ```bash
   # 设置代理（如果需要）
   export WANDB_BASE_URL="https://api.wandb.ai"
   ```

4. **禁用wandb**
   ```bash
   # 在脚本中设置
   USE_WANDB=false
   
   # 或者不使用--use_wandb参数
   ```

### 离线模式
如果网络不稳定，可以使用离线模式：
```bash
export WANDB_MODE=offline
```

稍后同步：
```bash
wandb sync wandb/offline-run-*
```

## 📊 示例输出

训练开始时会看到：
```
wandb初始化成功
wandb: Currently logged in as: your-username
wandb: Tracking run at https://wandb.ai/your-username/gpt-pretrain/runs/abc123
```

训练过程中的日志：
```
步骤    250 | 轮次  1/10 | 批次  125/500 | 损失 4.2345 | 困惑度  68.45 | 学习率 8.5e-05 | 用时 1.23s
--- 第 500 步评估 ---
评估完成 - 平均损失: 3.8934, 困惑度: 49.12
```

对应的wandb指标会实时更新到网页界面。

## 🎉 总结

通过wandb集成，您可以：
- 实时监控训练进度
- 比较不同实验的效果
- 追踪最佳模型性能
- 分析训练过程中的问题
- 与团队分享实验结果

现在开始您的wandb增强训练吧！
