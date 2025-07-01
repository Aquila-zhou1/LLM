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

def evaluate_perplexity_detailed(model: GPTSmall, val_loader, device: torch.device, tokenizer, show_examples: bool = True) -> float:
    """
    计算模型在验证集上的困惑度，并显示详细的评估样例

    Args:
        model: 模型
        val_loader: 验证数据加载器
        device: 设备
        tokenizer: 分词器
        show_examples: 是否显示评估样例

    Returns:
        困惑度
    """
    logger.info("正在计算困惑度...")

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    batch_losses = []

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
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )

            # 计算有效token数量
            valid_tokens = (targets != tokenizer.pad_token_id).sum().item()

            if valid_tokens > 0:
                batch_loss = loss.item() / valid_tokens
                batch_losses.append(batch_loss)
                total_loss += loss.item()
                total_tokens += valid_tokens

                # 显示详细评估样例
                if show_examples and i < 3:  # 显示前3个批次的详细信息
                    logger.info(f"\n=== 评估样例 {i+1} ===")

                    # 选择第一个样本进行详细分析
                    sample_input = inputs[0]
                    sample_target = targets[0]
                    sample_logits = logits[0]

                    # 解码原始文本
                    original_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
                    input_text = tokenizer.decode(sample_input, skip_special_tokens=True)
                    target_text = tokenizer.decode(sample_target, skip_special_tokens=True)

                    logger.info(f"原始文本: {original_text[:200]}...")
                    logger.info(f"输入文本: {input_text[:150]}...")
                    logger.info(f"目标文本: {target_text[:150]}...")

                    # 计算该样本的困惑度
                    sample_loss = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)(
                        sample_logits.reshape(-1, sample_logits.size(-1)),
                        sample_target.reshape(-1)
                    )
                    sample_ppl = torch.exp(sample_loss).item()
                    logger.info(f"样本损失: {sample_loss.item():.4f}")
                    logger.info(f"样本困惑度: {sample_ppl:.2f}")

                    # 分析前几个预测
                    logger.info("前10个token预测分析:")
                    for j in range(min(10, len(sample_target))):
                        if sample_target[j] != tokenizer.pad_token_id:
                            # 获取预测概率
                            token_logits = sample_logits[j]
                            token_probs = torch.softmax(token_logits, dim=-1)

                            # 真实token
                            true_token_id = sample_target[j].item()
                            true_token = tokenizer.decode([true_token_id])
                            true_prob = token_probs[true_token_id].item()

                            # 预测的最高概率token
                            pred_token_id = torch.argmax(token_probs).item()
                            pred_token = tokenizer.decode([pred_token_id])
                            pred_prob = token_probs[pred_token_id].item()

                            correct = "✓" if pred_token_id == true_token_id else "✗"

                            logger.info(f"  位置{j:2d}: 真实='{true_token}' (概率={true_prob:.3f}) | "
                                      f"预测='{pred_token}' (概率={pred_prob:.3f}) {correct}")

            if (i + 1) % 50 == 0:
                logger.info(f"已处理 {i + 1} 个批次...")

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    # 计算困惑度统计信息
    if batch_losses:
        import numpy as np
        batch_ppls = [math.exp(loss) for loss in batch_losses]
        logger.info(f"\n=== 困惑度统计 ===")
        logger.info(f"平均困惑度: {perplexity:.2f}")
        logger.info(f"困惑度标准差: {np.std(batch_ppls):.2f}")
        logger.info(f"困惑度范围: {min(batch_ppls):.2f} - {max(batch_ppls):.2f}")

    logger.info(f"困惑度计算完成: {perplexity:.2f}")

    return perplexity

def evaluate_perplexity(model: GPTSmall, val_loader, device: torch.device, tokenizer) -> float:
    """
    计算模型在验证集上的困惑度（简化版本）
    """
    return evaluate_perplexity_detailed(model, val_loader, device, tokenizer, show_examples=False)

def evaluate_perplexity_on_prompts(model: GPTSmall, prompts: list, device: torch.device, tokenizer) -> float:
    """
    计算模型在给定提示词上的困惑度
    
    Args:
        model: 模型
        prompts: 提示词列表
        device: 设备
        tokenizer: 分词器
        
    Returns:
        困惑度
    """
    logger.info("正在计算给定提示词的困惑度...")
    
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for prompt in prompts:
            # 编码提示词
            input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
            attention_mask = torch.ones(input_ids.shape, device=device)  # 假设没有padding，所有位置都是有效的
            
            # 构造输入和目标
            inputs = input_ids[:, :-1]
            targets = input_ids[:, 1:]
            
            # 前向传播
            logits = model(inputs, attention_mask[:, :-1])
            
            # 计算损失
            loss = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='sum')(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )
            
            # 计算有效token数量
            valid_tokens = (targets != tokenizer.pad_token_id).sum().item()
            
            total_loss += loss.item()
            total_tokens += valid_tokens
            
            logger.info(f"处理提示词: '{prompt}'")
            logger.info(f"损失：'{loss.item()}'")
            perplexity = torch.exp(loss)
            logger.info(f"困惑度：'{perplexity.item()}'")

    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    logger.info(f"困惑度计算完成: {perplexity:.2f}")
    
    return perplexity


def evaluate_text_quality(text: str, prompt: str) -> dict:
    """
    评估生成文本的质量

    Args:
        text: 生成的文本
        prompt: 原始提示词

    Returns:
        评估结果字典
    """
    # 基本统计
    total_length = len(text)
    prompt_length = len(prompt)
    generated_length = total_length - prompt_length

    # 词汇统计
    words = text.split()
    unique_words = len(set(words))
    word_diversity = unique_words / len(words) if words else 0

    # 重复检测
    sentences = text.split('.')
    repeated_phrases = 0
    for i, sentence in enumerate(sentences):
        for j, other_sentence in enumerate(sentences):
            if i != j and sentence.strip() and sentence.strip() in other_sentence:
                repeated_phrases += 1
                break

    # 连贯性评分（简单启发式）
    coherence_score = 0
    if "once upon a time" in text.lower():
        coherence_score += 1
    if any(word in text.lower() for word in ["the", "and", "was", "were", "is", "are"]):
        coherence_score += 1
    if len(sentences) > 1:
        coherence_score += 1

    # 语法完整性（简单检查）
    grammar_score = 0
    if text.endswith('.') or text.endswith('!') or text.endswith('?'):
        grammar_score += 1
    if text[0].isupper():
        grammar_score += 1

    return {
        'total_length': total_length,
        'generated_length': generated_length,
        'word_count': len(words),
        'unique_words': unique_words,
        'word_diversity': word_diversity,
        'repeated_phrases': repeated_phrases,
        'coherence_score': coherence_score,
        'grammar_score': grammar_score,
        'sentences': len(sentences)
    }

def generate_text_samples_with_scoring(model: GPTSmall, tokenizer: GPT2Tokenizer, device: torch.device, num_samples: int = 5):
    """
    生成文本样本并进行详细评分

    Args:
        model: 模型
        tokenizer: 分词器
        device: 设备
        num_samples: 生成样本数量
    """
    logger.info(f"正在生成 {num_samples} 个文本样本并评分...")

    model.eval()

    # 预定义的提示词
    prompts = [
        "Once upon a time",
        "The little girl",
        "In a magical forest",
        "The brave knight",
        "A beautiful princess",
        "Tom likes drinking juice",
        "I love you",
        "Why training a natural language model is important",
        "If I were a god"
    ]

    all_scores = []

    for i in range(min(num_samples, len(prompts))):
        prompt = prompts[i]
        logger.info(f"\n{'='*60}")
        logger.info(f"样本 {i+1}: '{prompt}'")
        logger.info(f"{'='*60}")

        # 编码提示词
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

        # 计算提示词的困惑度
        with torch.no_grad():
            prompt_logits = model(input_ids)
            if input_ids.size(1) > 1:
                prompt_targets = input_ids[:, 1:]
                prompt_inputs = input_ids[:, :-1]
                prompt_loss = nn.CrossEntropyLoss()(
                    prompt_logits[:, :-1, :].reshape(-1, prompt_logits.size(-1)),
                    prompt_targets.reshape(-1)
                )
                prompt_ppl = torch.exp(prompt_loss).item()
            else:
                prompt_ppl = float('inf')

        logger.info(f"提示词困惑度: {prompt_ppl:.2f}")

        # 生成多个版本进行比较
        temperatures = [0.7, 0.8, 1.0]
        best_text = ""
        best_score = -1

        for temp in temperatures:
            logger.info(f"\n--- 温度 {temp} ---")

            # 生成文本
            with torch.no_grad():
                generated = model.generate(
                    input_ids,
                    max_new_tokens=80,
                    temperature=temp,
                    top_k=50
                )

            # 解码生成的文本
            generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
            logger.info(f"生成文本: {generated_text}")

            # 评估文本质量
            quality_scores = evaluate_text_quality(generated_text, prompt)

            # 计算生成部分的困惑度
            if generated.size(1) > input_ids.size(1):
                generated_part = generated[:, input_ids.size(1):]
                with torch.no_grad():
                    gen_logits = model(generated[:, :-1])
                    gen_targets = generated[:, 1:]
                    gen_loss = nn.CrossEntropyLoss()(
                        gen_logits[:, input_ids.size(1)-1:, :].reshape(-1, gen_logits.size(-1)),
                        gen_targets[:, input_ids.size(1)-1:].reshape(-1)
                    )
                    gen_ppl = torch.exp(gen_loss).item()
            else:
                gen_ppl = float('inf')

            # 综合评分
            total_score = (
                quality_scores['coherence_score'] * 2 +
                quality_scores['grammar_score'] * 2 +
                quality_scores['word_diversity'] * 3 +
                max(0, 3 - quality_scores['repeated_phrases']) +
                max(0, 5 - gen_ppl/10)  # 困惑度越低越好
            )

            logger.info(f"质量评分:")
            logger.info(f"  生成长度: {quality_scores['generated_length']} 字符")
            logger.info(f"  词汇数量: {quality_scores['word_count']}")
            logger.info(f"  词汇多样性: {quality_scores['word_diversity']:.3f}")
            logger.info(f"  连贯性评分: {quality_scores['coherence_score']}/3")
            logger.info(f"  语法评分: {quality_scores['grammar_score']}/2")
            logger.info(f"  重复短语: {quality_scores['repeated_phrases']}")
            logger.info(f"  生成困惑度: {gen_ppl:.2f}")
            logger.info(f"  综合评分: {total_score:.2f}")

            if total_score > best_score:
                best_score = total_score
                best_text = generated_text
                best_temp = temp

        logger.info(f"\n🏆 最佳生成 (温度={best_temp}, 评分={best_score:.2f}):")
        logger.info(f"'{best_text}'")

        all_scores.append({
            'prompt': prompt,
            'best_text': best_text,
            'best_score': best_score,
            'best_temperature': best_temp
        })

    # 输出总体统计
    logger.info(f"\n{'='*60}")
    logger.info("总体评估统计:")
    logger.info(f"{'='*60}")

    avg_score = sum(s['best_score'] for s in all_scores) / len(all_scores)
    logger.info(f"平均评分: {avg_score:.2f}")

    best_sample = max(all_scores, key=lambda x: x['best_score'])
    worst_sample = min(all_scores, key=lambda x: x['best_score'])

    logger.info(f"最佳样本: '{best_sample['prompt']}' (评分: {best_sample['best_score']:.2f})")
    logger.info(f"最差样本: '{worst_sample['prompt']}' (评分: {worst_sample['best_score']:.2f})")

def generate_text_samples(model: GPTSmall, tokenizer: GPT2Tokenizer, device: torch.device, num_samples: int = 5):
    """
    生成文本样本（简化版本）
    """
    generate_text_samples_with_scoring(model, tokenizer, device, num_samples)

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

    # 计算验证集困惑度（详细版本）
    logger.info("\n" + "="*60)
    logger.info("开始详细评估验证集困惑度")
    logger.info("="*60)
    perplexity = evaluate_perplexity_detailed(model, val_loader, device, tokenizer, show_examples=True)

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
