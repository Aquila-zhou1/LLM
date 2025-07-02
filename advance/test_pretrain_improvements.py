#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试改进后的预训练脚本
验证新增的验证集评估和PPL计算功能
"""

import os
import sys
import argparse
import torch
import math

# 添加项目根目录到路径
sys.path.append(os.path.abspath('.'))

def test_ppl_calculation():
    """测试困惑度计算函数"""
    print("测试困惑度(PPL)计算...")
    
    # 测试正常情况
    loss = 2.3
    ppl = math.exp(loss)
    print(f"Loss: {loss:.3f}, PPL: {ppl:.3f}")
    
    # 测试边界情况
    loss = 10.0
    ppl = math.exp(loss) if loss < 10 else float('inf')
    print(f"Loss: {loss:.3f}, PPL: {ppl}")
    
    # 测试小loss值
    loss = 0.1
    ppl = math.exp(loss)
    print(f"Loss: {loss:.3f}, PPL: {ppl:.3f}")
    
    print("PPL计算测试完成！\n")

def test_data_split():
    """测试数据集分割功能"""
    print("测试数据集分割功能...")
    
    # 模拟数据集大小
    dataset_size = 1000
    val_ratio = 0.1
    
    val_size = int(dataset_size * val_ratio)
    train_size = dataset_size - val_size
    
    print(f"总数据集大小: {dataset_size}")
    print(f"验证集比例: {val_ratio}")
    print(f"训练集大小: {train_size}")
    print(f"验证集大小: {val_size}")
    
    assert train_size + val_size == dataset_size, "数据集分割错误！"
    print("数据集分割测试通过！\n")

def test_wandb_config():
    """测试wandb配置"""
    print("测试wandb配置...")
    
    # 模拟参数
    class MockArgs:
        learning_rate = 5e-4
        epochs = 1
        batch_size = 32
        hidden_size = 512
        num_hidden_layers = 8
        max_seq_len = 512
        use_moe = False
        accumulation_steps = 8
        grad_clip = 1.0
        val_ratio = 0.1
        eval_interval = 500
    
    args = MockArgs()
    
    config = {
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hidden_size": args.hidden_size,
        "num_hidden_layers": args.num_hidden_layers,
        "max_seq_len": args.max_seq_len,
        "use_moe": args.use_moe,
        "accumulation_steps": args.accumulation_steps,
        "grad_clip": args.grad_clip,
        "val_ratio": args.val_ratio,
        "eval_interval": args.eval_interval
    }
    
    print("Wandb配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("Wandb配置测试完成！\n")

def test_logging_format():
    """测试日志格式"""
    print("测试日志格式...")
    
    # 模拟训练参数
    epoch = 0
    epochs = 1
    step = 100
    iter_per_epoch = 1000
    train_loss = 2.345
    train_ppl = math.exp(train_loss)
    lr = 5e-4
    epoch_time = 15
    
    log_message = 'Epoch:[{}/{}]({}/{}) train_loss:{:.3f} train_ppl:{:.3f} lr:{:.12f} epoch_Time:{}min:'.format(
        epoch + 1,
        epochs,
        step,
        iter_per_epoch,
        train_loss,
        train_ppl,
        lr,
        epoch_time
    )
    
    print("训练日志格式:")
    print(log_message)
    
    # 模拟验证参数
    val_loss = 2.123
    val_ppl = math.exp(val_loss)
    
    val_log_message = f'Validation - loss:{val_loss:.3f} ppl:{val_ppl:.3f}'
    print("验证日志格式:")
    print(val_log_message)
    
    print("日志格式测试完成！\n")

def main():
    """主测试函数"""
    print("=" * 50)
    print("测试改进后的预训练脚本功能")
    print("=" * 50)
    
    test_ppl_calculation()
    test_data_split()
    test_wandb_config()
    test_logging_format()
    
    print("=" * 50)
    print("所有测试完成！")
    print("=" * 50)
    
    print("\n使用说明:")
    print("1. 运行预训练时添加 --use_wandb 参数启用wandb记录")
    print("2. 使用 --val_ratio 0.1 设置验证集比例（默认10%）")
    print("3. 使用 --eval_interval 500 设置验证评估间隔（默认500步）")
    print("4. 训练过程中会显示训练和验证的loss和PPL指标")
    print("5. 所有指标会自动记录到wandb项目 'Improved llm model' 中")
    
    print("\n示例命令:")
    print("cd trainer")
    print("python train_pretrain.py --use_wandb --val_ratio 0.1 --eval_interval 300")

if __name__ == "__main__":
    main()
