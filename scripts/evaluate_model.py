"""
模型评估脚本
用于评估训练好的模型，计算困惑度和生成文本示例
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from transformers import GPT2Tokenizer

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from data.tinystories_loader import create_dataloaders
from model.gpt_model import GPTSmall

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> tuple[GPTSmall, dict]:
    """
    从检查点加载模型
    
    Args:
        checkpoint_path: 检查点路径
        device: 设备
        
    Returns:
        (模型, 配置)
    """
    logger.info(f"正在加载检查点: {checkpoint_path}")
    
    # 加载检查点
    if os.path.isdir(checkpoint_path):
        model_path = os.path.join(checkpoint_path, "pytorch_model.bin")
        config_path = os.path.join(checkpoint_path, "config.json")
    else:
        model_path = checkpoint_path
        config_path = None
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # 加载配置
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        # 使用默认配置
        config = {
            'vocab_size': 50257,
            'hidden_size': 512,
            'num_layers': 6,
            'num_heads': 8,
            'max_seq_len': 1024,
            'dropout': 0.1
        }
        logger.warning("未找到配置文件，使用默认配置")
    
    # 创建模型
    model = GPTSmall(
        vocab_size=config['vocab_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        max_seq_len=config['max_seq_len'],
        dropout=config['dropout']
    )
    
    # 加载权重
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    logger.info(f"模型加载完成，参数数量: {model.get_num_params():,}")
    
    return model, config

def evaluate_perplexity(model: GPTSmall, val_loader, device: torch.device, tokenizer) -> float:
    """
    计算模型在验证集上的困惑度
    
    Args:
        model: 模型
        val_loader: 验证数据加载器
        device: 设备
        tokenizer: 分词器
        
    Returns:
        困惑度
    """
    logger.info("正在计算困惑度...")
    
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # 构造输入和目标
            inputs = input_ids[:, :-1]
            targets = input_ids[:, 1:]
            
            # 前向传播
            logits = model(inputs, attention_mask[:, :-1])
            
            # 计算损失
            loss = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='sum')(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
            
            # 计算有效token数量
            valid_tokens = (targets != tokenizer.pad_token_id).sum().item()
            
            total_loss += loss.item()
            total_tokens += valid_tokens
            
            if (i + 1) % 50 == 0:
                logger.info(f"已处理 {i + 1} 个批次...")
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    logger.info(f"困惑度计算完成: {perplexity:.2f}")
    
    return perplexity

def generate_text_samples(model: GPTSmall, tokenizer: GPT2Tokenizer, device: torch.device, num_samples: int = 5):
    """
    生成文本样本
    
    Args:
        model: 模型
        tokenizer: 分词器
        device: 设备
        num_samples: 生成样本数量
    """
    logger.info(f"正在生成 {num_samples} 个文本样本...")
    
    model.eval()
    
    # 预定义的提示词
    prompts = [
        "Once upon a time",
        "The little girl",
        "In a magical forest",
        "The brave knight",
        "A beautiful princess"
    ]
    
    for i in range(min(num_samples, len(prompts))):
        prompt = prompts[i]
        logger.info(f"\n--- 样本 {i+1} ---")
        logger.info(f"提示词: '{prompt}'")
        
        # 编码提示词
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        
        # 生成文本
        with torch.no_grad():
            generated = model.generate(
                input_ids,
                max_new_tokens=100,
                temperature=0.8,
                top_k=50
            )
        
        # 解码生成的文本
        generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        logger.info(f"生成文本: {generated_text}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='模型评估')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--batch_size', type=int, default=8, help='批量大小')
    parser.add_argument('--max_eval_batches', type=int, default=100, help='最大评估批次数')
    parser.add_argument('--generate_samples', action='store_true', help='是否生成文本样本')
    parser.add_argument('--num_samples', type=int, default=5, help='生成样本数量')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 加载模型
    model, config = load_model_from_checkpoint(args.checkpoint_path, device)
    
    # 创建分词器
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 创建验证数据加载器
    logger.info("正在创建验证数据加载器...")
    _, val_loader, _ = create_dataloaders(
        batch_size=args.batch_size,
        max_length=config['max_seq_len'],
        num_workers=2
    )
    
    # 限制评估批次数
    if args.max_eval_batches > 0:
        val_data = []
        for i, batch in enumerate(val_loader):
            if i >= args.max_eval_batches:
                break
            val_data.append(batch)
        val_loader = val_data
        logger.info(f"限制评估批次数为: {len(val_loader)}")
    
    # 计算困惑度
    perplexity = evaluate_perplexity(model, val_loader, device, tokenizer)
    
    # 生成文本样本
    if args.generate_samples:
        generate_text_samples(model, tokenizer, device, args.num_samples)
    
    # 输出结果摘要
    logger.info("\n=== 评估结果摘要 ===")
    logger.info(f"模型检查点: {args.checkpoint_path}")
    logger.info(f"模型参数数量: {model.get_num_params():,}")
    logger.info(f"验证集困惑度: {perplexity:.2f}")
    
    if perplexity < 40:
        logger.info("✓ 困惑度达到目标要求 (< 40)")
    else:
        logger.info("✗ 困惑度未达到目标要求 (< 40)")

if __name__ == "__main__":
    main()
