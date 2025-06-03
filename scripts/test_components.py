"""
组件测试脚本
测试所有预训练组件是否正常工作
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_data_loader():
    """测试数据加载器"""
    logger.info("=== 测试数据加载器 ===")
    try:
        from data.tinystories_loader import create_dataloaders
        
        # 创建小规模数据加载器
        train_loader, val_loader, tokenizer = create_dataloaders(
            batch_size=2,
            max_length=128,
            num_workers=0,  # 避免多进程问题
            train_subset_size=10,
            val_subset_size=5
        )
        
        # 测试一个批次
        batch = next(iter(train_loader))
        logger.info(f"✓ 数据加载器测试通过")
        logger.info(f"  批次形状: {batch['input_ids'].shape}")
        logger.info(f"  分词器词表大小: {len(tokenizer)}")
        
        return True
    except Exception as e:
        logger.error(f"✗ 数据加载器测试失败: {e}")
        return False

def test_model():
    """测试模型"""
    logger.info("=== 测试模型 ===")
    try:
        from model.gpt_model import GPTSmall
        
        # 创建小模型
        model = GPTSmall(
            vocab_size=50257,
            hidden_size=256,  # 较小的隐藏层用于测试
            num_layers=2,     # 较少的层数用于测试
            num_heads=4,
            max_seq_len=128
        )
        
        # 测试前向传播
        batch_size, seq_len = 2, 64
        input_ids = torch.randint(0, 50257, (batch_size, seq_len))
        
        with torch.no_grad():
            logits = model(input_ids)
        
        logger.info(f"✓ 模型测试通过")
        logger.info(f"  模型参数数量: {model.get_num_params():,}")
        logger.info(f"  输入形状: {input_ids.shape}")
        logger.info(f"  输出形状: {logits.shape}")
        
        return True
    except Exception as e:
        logger.error(f"✗ 模型测试失败: {e}")
        return False

def test_training_step():
    """测试训练步骤"""
    logger.info("=== 测试训练步骤 ===")
    try:
        from data.tinystories_loader import create_dataloaders
        from model.gpt_model import GPTSmall
        import torch.nn as nn
        
        # 创建数据和模型
        train_loader, _, tokenizer = create_dataloaders(
            batch_size=2,
            max_length=128,
            num_workers=0,
            train_subset_size=5
        )
        
        model = GPTSmall(
            vocab_size=50257,
            hidden_size=256,
            num_layers=2,
            num_heads=4,
            max_seq_len=128
        )
        
        # 创建优化器
        print("==创建优化器==")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        
        # 测试一个训练步骤
        print("==测试一个训练步骤==")
        batch = next(iter(train_loader))
        input_ids = batch['input_ids']
        
        # 构造输入和目标
        print("==构造输入和目标==")
        inputs = input_ids[:, :-1]
        targets = input_ids[:, 1:]
        
        # 前向传播
        print("==前向传播==")
        logits = model(inputs)
        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        
        # 反向传播
        print("==反向传播==")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        logger.info(f"✓ 训练步骤测试通过")
        logger.info(f"  损失值: {loss.item():.4f}")
        logger.info(f"  困惑度: {torch.exp(loss).item():.2f}")
        
        return True
    except Exception as e:
        logger.error(f"✗ 训练步骤测试失败: {e}")
        return False

def test_gpu():
    """测试GPU可用性"""
    logger.info("=== 测试GPU ===")
    try:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"✓ GPU可用")
            logger.info(f"  设备: {torch.cuda.get_device_name()}")
            logger.info(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            
            # 测试GPU计算
            x = torch.randn(100, 100).to(device)
            y = torch.mm(x, x.t())
            logger.info(f"  GPU计算测试通过")
            
            return True
        else:
            logger.warning("GPU不可用，将使用CPU")
            return False
    except Exception as e:
        logger.error(f"✗ GPU测试失败: {e}")
        return False

def test_deepspeed_config():
    """测试DeepSpeed配置"""
    logger.info("=== 测试DeepSpeed配置 ===")
    try:
        import json
        config_path = "configs/ds_config_pretrain.json"
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            logger.info(f"✓ DeepSpeed配置文件存在")
            logger.info(f"  批量大小: {config.get('train_batch_size', 'N/A')}")
            logger.info(f"  ZeRO阶段: {config.get('zero_optimization', {}).get('stage', 'N/A')}")
            logger.info(f"  FP16: {config.get('fp16', {}).get('enabled', False)}")
            
            return True
        else:
            logger.error(f"✗ DeepSpeed配置文件不存在: {config_path}")
            return False
    except Exception as e:
        logger.error(f"✗ DeepSpeed配置测试失败: {e}")
        return False

def main():
    """主函数"""
    logger.info("开始组件测试...")
    
    tests = [
        ("GPU", test_gpu),
        ("数据加载器", test_data_loader),
        ("模型", test_model),
        ("训练步骤", test_training_step),
        ("DeepSpeed配置", test_deepspeed_config)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"测试 {test_name} 时发生异常: {e}")
            results[test_name] = False
    
    # 输出测试结果摘要
    logger.info(f"\n{'='*50}")
    logger.info("=== 测试结果摘要 ===")
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败"
        logger.info(f"{test_name:15s}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        logger.info("\n🎉 所有测试通过！可以开始训练。")
    else:
        logger.info("\n⚠️  部分测试失败，请检查相关组件。")
    
    return all_passed

if __name__ == "__main__":
    main()
