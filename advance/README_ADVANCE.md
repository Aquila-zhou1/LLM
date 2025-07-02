# 高级任务完整指南

## 概述

本项目实现了预训练(Pretrain)、监督微调(SFT)、LoRA微调， 直接偏好强化学习(DPO)算法、训练推理(蒸馏)模型算法等全过程代码

## 快速开始

### 1. 环境准备

```bash
cd advance
pip install -r requirements.txt
```

### 2. **预训练**

注意：后续任务基于该预训练模型，故不可省略

```bash
cd trainer
python train_pretrain.py --use_wandb --val_ratio 0.1 --eval_interval 300
```

### 3. SFT训练

```bash
python train_full_sft.py --use_wandb --val_ratio 0.1 --eval_interval 300
```

### 4. LoRA微调

```bash
python train_lora.py --use_wandb --val_ratio 0.1 --eval_interval 300
```

### 5. DPO训练

```bash
python train_dpo.py --use_wandb --val_ratio 0.1 --eval_interval 300
```

### 6. 训练推理模型

```
python train_distill_reason.py
```


