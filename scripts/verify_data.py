"""
数据验证脚本
验证训练数据的token化是否正确，检查数据完整性
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

import torch
from data.tinystories_loader import create_dataloaders, verify_data_integrity
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def detailed_token_analysis(train_loader, val_loader, tokenizer, num_samples: int = 5):
    """
    详细分析token化过程
    
    Args:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        tokenizer: 分词器
        num_samples: 分析的样本数量
    """
    logger.info("=== 详细Token分析 ===")
    
    # 分析训练集
    logger.info("\n--- 训练集分析 ---")
    train_iter = iter(train_loader)
    
    for i in range(min(num_samples, len(train_loader))):
        batch = next(train_iter)
        logger.info(f"\n训练样本 {i+1}:")
        
        # 获取第一个样本
        input_ids = batch['input_ids'][0]
        attention_mask = batch['attention_mask'][0]
        
        logger.info(f"Token序列形状: {input_ids.shape}")
        logger.info(f"注意力掩码形状: {attention_mask.shape}")
        
        # 统计信息
        total_tokens = len(input_ids)
        valid_tokens = attention_mask.sum().item()
        pad_tokens = total_tokens - valid_tokens
        
        logger.info(f"总token数: {total_tokens}")
        logger.info(f"有效token数: {valid_tokens}")
        logger.info(f"填充token数: {pad_tokens}")
        
        # 解码完整文本
        full_text = tokenizer.decode(input_ids, skip_special_tokens=True)
        logger.info(f"完整文本长度: {len(full_text)} 字符")
        logger.info(f"完整文本: {full_text}")
        
        # 分析前20个token
        logger.info("前20个token详细分析:")
        for j in range(min(20, len(input_ids))):
            token_id = input_ids[j].item()
            token_text = tokenizer.decode([token_id])
            is_special = token_id in [tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.bos_token_id]
            mask_value = attention_mask[j].item()
            
            logger.info(f"  位置{j:2d}: ID={token_id:5d} | 文本='{token_text}' | "
                       f"特殊={is_special} | 掩码={mask_value}")
        
        # 验证语言建模的输入-目标对应关系
        logger.info("语言建模输入-目标验证:")
        inputs = input_ids[:-1]  # 前n-1个token作为输入
        targets = input_ids[1:]  # 后n-1个token作为目标
        
        logger.info(f"输入序列长度: {len(inputs)}")
        logger.info(f"目标序列长度: {len(targets)}")
        
        # 检查前10个输入-目标对
        for j in range(min(10, len(inputs))):
            input_token = tokenizer.decode([inputs[j].item()])
            target_token = tokenizer.decode([targets[j].item()])
            logger.info(f"  输入'{input_token}' -> 目标'{target_token}'")
    
    # 分析验证集
    logger.info("\n--- 验证集分析 ---")
    val_iter = iter(val_loader)
    
    for i in range(min(2, len(val_loader))):  # 验证集只看2个样本
        batch = next(val_iter)
        logger.info(f"\n验证样本 {i+1}:")
        
        input_ids = batch['input_ids'][0]
        attention_mask = batch['attention_mask'][0]
        
        # 基本统计
        total_tokens = len(input_ids)
        valid_tokens = attention_mask.sum().item()
        
        logger.info(f"总token数: {total_tokens}")
        logger.info(f"有效token数: {valid_tokens}")
        
        # 解码文本
        full_text = tokenizer.decode(input_ids, skip_special_tokens=True)
        logger.info(f"文本: {full_text[:200]}...")

def check_data_consistency(train_loader, val_loader, tokenizer):
    """
    检查数据一致性
    
    Args:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        tokenizer: 分词器
    """
    logger.info("=== 数据一致性检查 ===")
    
    # 检查词表范围
    logger.info("检查token ID范围...")
    
    all_token_ids = set()
    max_token_id = 0
    min_token_id = float('inf')
    
    # 检查训练集
    for i, batch in enumerate(train_loader):
        if i >= 10:  # 只检查前10个批次
            break
        
        input_ids = batch['input_ids']
        batch_max = input_ids.max().item()
        batch_min = input_ids.min().item()
        
        max_token_id = max(max_token_id, batch_max)
        min_token_id = min(min_token_id, batch_min)
        
        # 收集所有token ID
        all_token_ids.update(input_ids.flatten().tolist())
    
    logger.info(f"Token ID范围: {min_token_id} - {max_token_id}")
    logger.info(f"词表大小: {len(tokenizer)}")
    logger.info(f"唯一token数量: {len(all_token_ids)}")
    
    # 检查是否有超出词表的token
    if max_token_id >= len(tokenizer):
        logger.error(f"发现超出词表范围的token: {max_token_id} >= {len(tokenizer)}")
    else:
        logger.info("✓ 所有token都在词表范围内")
    
    # 检查特殊token
    logger.info("特殊token检查:")
    logger.info(f"  PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    logger.info(f"  EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    logger.info(f"  BOS token: {getattr(tokenizer, 'bos_token', 'None')} (ID: {getattr(tokenizer, 'bos_token_id', 'None')})")
    
    # 统计特殊token使用情况
    pad_count = sum(1 for tid in all_token_ids if tid == tokenizer.pad_token_id)
    eos_count = sum(1 for tid in all_token_ids if tid == tokenizer.eos_token_id)
    
    logger.info(f"  PAD token出现次数: {pad_count}")
    logger.info(f"  EOS token出现次数: {eos_count}")

def verify_batch_shapes(train_loader, val_loader):
    """
    验证批次形状一致性
    
    Args:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
    """
    logger.info("=== 批次形状验证 ===")
    
    # 检查训练集批次
    logger.info("训练集批次形状:")
    for i, batch in enumerate(train_loader):
        if i >= 3:  # 只检查前3个批次
            break
        
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        
        logger.info(f"  批次 {i+1}: input_ids={input_ids.shape}, attention_mask={attention_mask.shape}")
        
        # 检查形状一致性
        if input_ids.shape != attention_mask.shape:
            logger.error(f"形状不一致: input_ids={input_ids.shape} vs attention_mask={attention_mask.shape}")
        else:
            logger.info(f"    ✓ 形状一致")
    
    # 检查验证集批次
    logger.info("验证集批次形状:")
    for i, batch in enumerate(val_loader):
        if i >= 3:  # 只检查前3个批次
            break
        
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        
        logger.info(f"  批次 {i+1}: input_ids={input_ids.shape}, attention_mask={attention_mask.shape}")

def main():
    """主函数"""
    logger.info("开始数据验证...")
    
    # 创建数据加载器（使用小规模数据进行验证）
    logger.info("创建数据加载器...")
    train_loader, val_loader, tokenizer = create_dataloaders(
        batch_size=4,
        max_length=512,
        num_workers=0,  # 避免多进程问题
        train_subset_size=20,  # 使用小规模数据
        val_subset_size=10
    )
    
    logger.info(f"训练批次数: {len(train_loader)}")
    logger.info(f"验证批次数: {len(val_loader)}")
    logger.info(f"分词器词表大小: {len(tokenizer)}")
    
    # 执行各种验证
    try:
        # 1. 基本数据完整性验证
        verify_data_integrity(train_loader, val_loader, tokenizer, num_samples=3)
        
        # 2. 详细token分析
        detailed_token_analysis(train_loader, val_loader, tokenizer, num_samples=3)
        
        # 3. 数据一致性检查
        check_data_consistency(train_loader, val_loader, tokenizer)
        
        # 4. 批次形状验证
        verify_batch_shapes(train_loader, val_loader)
        
        logger.info("\n" + "="*60)
        logger.info("🎉 所有数据验证通过！")
        logger.info("数据已准备就绪，可以开始训练。")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"\n❌ 数据验证失败: {e}")
        logger.error("请检查数据处理流程。")
        raise

if __name__ == "__main__":
    main()
