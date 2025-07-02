"""
GPT模型预训练脚本
实现完整的预训练流程，包括训练循环、验证、模型保存等
"""

import os
import sys
import time
import math
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import deepspeed
from transformers import get_linear_schedule_with_warmup

# 导入wandb用于实验跟踪
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("警告: wandb未安装，将跳过实验跟踪功能")

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from data.tinystories_loader import create_dataloaders
from model.gpt_model import GPTSmall

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PretrainConfig:
    """预训练配置类"""
    
    def __init__(self):
        # 模型配置
        self.vocab_size = 50257 # 尝试过3000，则会有runtimeerror
        self.hidden_size = 1024
        self.num_layers = 12
        self.num_heads = 8
        self.max_seq_len = 1024
        self.dropout = 0.1
        
        # 训练配置
        self.batch_size = 8
        self.learning_rate = 1e-4
        self.weight_decay = 1e-2
        self.num_epochs = 10
        self.warmup_steps = 1000
        self.gradient_accumulation_steps = 4
        self.max_grad_norm = 1.0
        
        # 评估配置
        self.eval_interval = 500  # 每500步评估一次
        self.eval_steps = 100     # 评估时使用的步数
        self.save_interval = 1000 # 每1000步保存一次
        
        # 数据配置
        self.num_workers = 4 # debugs
        # self.num_workers = 4
        self.cache_dir = "./cache"
        
        # 输出配置
        self.output_dir = "./outputs"
        self.log_interval = 50    # 每50步打印一次日志

        # wandb配置
        self.use_wandb = True
        self.wandb_project = "gpt-pretrain"
        self.wandb_entity = None  # 如果有团队名称可以设置
        self.wandb_run_name = None  # 运行名称，如果为None会自动生成

def calculate_perplexity(loss: float) -> float:
    """计算困惑度"""
    return math.exp(loss)

def evaluate_model(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    tokenizer,
    max_eval_steps: int = 100
) -> tuple[float, float]:
    """
    评估模型

    Args:
        model: 模型
        val_loader: 验证数据加载器
        device: 设备
        tokenizer: 分词器
        max_eval_steps: 最大评估步数

    Returns:
        (平均损失, 困惑度)
    """
    model.eval()
    total_loss = 0.0
    num_steps = 0

    logger.info(f"开始评估模型（最多{max_eval_steps}步）...")

    with torch.no_grad():
        for step, batch in enumerate(val_loader):
            if step >= max_eval_steps:
                break

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # 构造输入和目标
            inputs = input_ids[:, :-1]
            targets = input_ids[:, 1:]

            # 前向传播
            logits = model(inputs, attention_mask[:, :-1])

            # 计算损失
            loss = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )

            total_loss += loss.item()
            num_steps += 1

            if (step + 1) % 20 == 0:
                logger.info(f"  评估步骤 {step + 1}/{max_eval_steps}, 当前损失: {loss.item():.4f}")

                # 显示第一个样本的token解码内容
                sample_input_ids = input_ids[0]
                sample_inputs = inputs[0]
                sample_targets = targets[0]
                sample_logits = logits[0]

                logger.info(f"  === 评估样本token解码内容 ===")

                # 解码完整原始文本
                original_text = tokenizer.decode(sample_input_ids, skip_special_tokens=True)
                logger.info(f"  原始完整文本: {original_text}")

                # 解码输入部分
                input_text = tokenizer.decode(sample_inputs, skip_special_tokens=True)
                logger.info(f"  输入文本: {input_text}")

                # 解码目标部分
                target_text = tokenizer.decode(sample_targets, skip_special_tokens=True)
                logger.info(f"  目标文本: {target_text}")

                # 显示前10个token的预测情况
                logger.info(f"  前10个token预测分析:")
                for i in range(min(10, len(sample_targets))):
                    if sample_targets[i] != tokenizer.pad_token_id:
                        # 真实token
                        true_token_id = sample_targets[i].item()
                        true_token = tokenizer.decode([true_token_id])

                        # 预测token（最高概率）
                        pred_token_id = torch.argmax(sample_logits[i]).item()
                        pred_token = tokenizer.decode([pred_token_id])

                        # 预测概率
                        token_probs = torch.softmax(sample_logits[i], dim=-1)
                        true_prob = token_probs[true_token_id].item()
                        pred_prob = token_probs[pred_token_id].item()

                        correct = "✓" if pred_token_id == true_token_id else "✗"

                        logger.info(f"    位置{i:2d}: 真实='{true_token}' (概率={true_prob:.3f}) | "
                                  f"预测='{pred_token}' (概率={pred_prob:.3f}) {correct}")

                logger.info(f"  ================================")

    avg_loss = total_loss / num_steps
    perplexity = calculate_perplexity(avg_loss)

    logger.info(f"评估完成 - 平均损失: {avg_loss:.4f}, 困惑度: {perplexity:.2f}")

    return avg_loss, perplexity

def save_checkpoint(
    model: nn.Module,
    optimizer,
    scheduler,
    step: int,
    loss: float,
    config: PretrainConfig,
    output_dir: str
):
    """保存检查点"""
    checkpoint_dir = Path(output_dir) / f"checkpoint-{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存模型状态
    if hasattr(model, 'module'):  # DeepSpeed包装的模型
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
    
    checkpoint = {
        'step': step,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict() if hasattr(optimizer, 'state_dict') else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'config': config.__dict__
    }
    
    torch.save(checkpoint, checkpoint_dir / "pytorch_model.bin")
    
    # 保存配置
    with open(checkpoint_dir / "config.json", 'w') as f:
        json.dump(config.__dict__, f, indent=2)
    
    logger.info(f"检查点已保存到: {checkpoint_dir}")

def verify_training_data(train_loader, tokenizer, num_batches: int = 2):
    """
    验证训练数据的token化是否正确

    Args:
        train_loader: 训练数据加载器
        tokenizer: 分词器
        num_batches: 验证的批次数量
    """
    logger.info("=== 验证训练数据token化 ===")

    train_iter = iter(train_loader)
    for batch_idx in range(min(num_batches, len(train_loader))):
        batch = next(train_iter)
        logger.info(f"\n--- 批次 {batch_idx + 1} ---")

        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        logger.info(f"批次形状: {input_ids.shape}")

        # 检查第一个样本
        sample_input = input_ids[0]
        sample_mask = attention_mask[0]

        logger.info(f"有效token数量: {sample_mask.sum().item()}")

        # 构造训练用的输入和目标
        inputs = sample_input[:-1]  # 前n-1个token作为输入
        targets = sample_input[1:]  # 后n-1个token作为目标

        logger.info(f"输入序列长度: {len(inputs)}")
        logger.info(f"目标序列长度: {len(targets)}")

        # 解码输入和目标
        input_text = tokenizer.decode(inputs, skip_special_tokens=True)
        target_text = tokenizer.decode(targets, skip_special_tokens=True)

        logger.info(f"输入文本: {input_text[:150]}...")
        logger.info(f"目标文本: {target_text[:150]}...")

        # 检查输入和目标的对应关系
        logger.info("验证输入-目标对应关系:")
        for i in range(min(10, len(inputs))):
            input_token = tokenizer.decode([inputs[i].item()])
            target_token = tokenizer.decode([targets[i].item()])
            logger.info(f"  位置{i}: '{input_token}' -> '{target_token}'")

        # 检查pad token
        pad_count = (sample_input == tokenizer.pad_token_id).sum().item()
        logger.info(f"Pad token数量: {pad_count}")

        if batch_idx == 0:  # 只详细显示第一个批次
            logger.info(f"完整原始文本: {tokenizer.decode(sample_input, skip_special_tokens=True)}")

def train_model(config: PretrainConfig, deepspeed_config: dict = None):
    """
    训练模型主函数

    Args:
        config: 训练配置
        deepspeed_config: DeepSpeed配置
    """
    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)

    # 初始化wandb
    wandb_run = None
    if config.use_wandb and WANDB_AVAILABLE:
        try:
            wandb_run = wandb.init(
                entity=config.wandb_entity,
                project=config.wandb_project,
                name=config.wandb_run_name,
                config={
                    # 模型配置
                    "vocab_size": config.vocab_size,
                    "hidden_size": config.hidden_size,
                    "num_layers": config.num_layers,
                    "num_heads": config.num_heads,
                    "max_seq_len": config.max_seq_len,
                    "dropout": config.dropout,

                    # 训练配置
                    "batch_size": config.batch_size,
                    "learning_rate": config.learning_rate,
                    "weight_decay": config.weight_decay,
                    "num_epochs": config.num_epochs,
                    "warmup_steps": config.warmup_steps,
                    "gradient_accumulation_steps": config.gradient_accumulation_steps,
                    "max_grad_norm": config.max_grad_norm,

                    # 其他配置
                    "eval_interval": config.eval_interval,
                    "eval_steps": config.eval_steps,
                    "save_interval": config.save_interval,
                    "use_deepspeed": deepspeed_config is not None,
                }
            )
            logger.info("wandb初始化成功")
        except Exception as e:
            logger.warning(f"wandb初始化失败: {e}")
            wandb_run = None
    elif config.use_wandb and not WANDB_AVAILABLE:
        logger.warning("wandb未安装，跳过实验跟踪")

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    # 创建数据加载器
    logger.info("正在创建数据加载器...")
    train_loader, val_loader, tokenizer = create_dataloaders(
        batch_size=config.batch_size,
        max_length=config.max_seq_len,
        num_workers=config.num_workers,
        cache_dir=config.cache_dir
    )

    # 验证训练数据（注释掉以减少输出）
    # verify_training_data(train_loader, tokenizer, num_batches=2)
    
    # 创建模型
    logger.info("正在创建模型...")
    model = GPTSmall(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout
    )
    
    # 计算总训练步数
    total_steps = len(train_loader) * config.num_epochs
    logger.info(f"总训练步数: {total_steps}")
    
    # 初始化DeepSpeed
    if deepspeed_config:
        logger.info("正在初始化DeepSpeed...")
        model_engine, optimizer, _, scheduler = deepspeed.initialize(
            model=model,
            config=deepspeed_config
        )
        device = model_engine.device
    else:
        # 标准PyTorch训练
        model = model.to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=total_steps
        )
        model_engine = model
    
    # 训练循环
    logger.info("开始训练...")
    global_step = 0
    best_perplexity = float('inf')
    
    for epoch in range(config.num_epochs):
        logger.info(f"\n=== 开始第 {epoch + 1}/{config.num_epochs} 轮训练 ===")
        
        model_engine.train()
        epoch_loss = 0.0
        
        for step, batch in enumerate(train_loader):
            start_time = time.time()
            
            # 数据移到设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # 构造输入和目标（语言建模任务）
            inputs = input_ids[:, :-1]      # 前n-1个token作为输入
            targets = input_ids[:, 1:]      # 后n-1个token作为目标
            input_mask = attention_mask[:, :-1]
            
            # 前向传播
            logits = model_engine(inputs, input_mask)
            
            # 计算损失
            loss = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )
            
            # 反向传播
            if deepspeed_config:
                model_engine.backward(loss)
                model_engine.step()
            else:
                loss = loss / config.gradient_accumulation_steps
                loss.backward()
                
                if (step + 1) % config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
            
            # 记录损失
            epoch_loss += loss.item()
            global_step += 1
            
            # 打印训练日志
            if global_step % config.log_interval == 0:
                elapsed_time = time.time() - start_time
                current_lr = scheduler.get_last_lr()[0] if scheduler else config.learning_rate
                perplexity = calculate_perplexity(loss.item())

                logger.info(
                    f"步骤 {global_step:6d} | "
                    f"轮次 {epoch+1:2d}/{config.num_epochs} | "
                    f"批次 {step+1:4d}/{len(train_loader)} | "
                    f"损失 {loss.item():.4f} | "
                    f"困惑度 {perplexity:6.2f} | "
                    f"学习率 {current_lr:.2e} | "
                    f"用时 {elapsed_time:.2f}s"
                )

                # 记录到wandb
                if wandb_run is not None:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/perplexity": perplexity,
                        "train/learning_rate": current_lr,
                        "train/epoch": epoch + 1,
                        "train/step": global_step,
                        "train/elapsed_time": elapsed_time
                    }, step=global_step)
            
            # 定期评估
            if global_step % config.eval_interval == 0:
                logger.info(f"\n--- 第 {global_step} 步评估 ---")
                val_loss, val_perplexity = evaluate_model(
                    model_engine, val_loader, device, tokenizer, config.eval_steps
                )

                # 记录评估结果到wandb
                if wandb_run is not None:
                    wandb.log({
                        "eval/loss": val_loss,
                        "eval/perplexity": val_perplexity,
                        "eval/step": global_step,
                        "eval/best_perplexity": min(best_perplexity, val_perplexity)
                    }, step=global_step)

                # 保存最佳模型
                if val_perplexity < best_perplexity:
                    best_perplexity = val_perplexity
                    logger.info(f"发现更好的模型！困惑度: {val_perplexity:.2f}")
                    save_checkpoint(
                        model_engine, optimizer, scheduler,
                        global_step, val_loss, config,
                        os.path.join(config.output_dir, "best_model")
                    )

                model_engine.train()
            
            # 定期保存检查点
            if global_step % config.save_interval == 0:
                save_checkpoint(
                    model_engine, optimizer, scheduler,
                    global_step, loss.item(), config, config.output_dir
                )
        
        # 轮次结束评估
        logger.info(f"\n=== 第 {epoch + 1} 轮训练完成 ===")
        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_epoch_perplexity = calculate_perplexity(avg_epoch_loss)
        logger.info(f"平均训练损失: {avg_epoch_loss:.4f}")
        logger.info(f"平均训练困惑度: {avg_epoch_perplexity:.2f}")

        # 轮次结束时进行完整评估
        val_loss, val_perplexity = evaluate_model(
            model_engine, val_loader, device, tokenizer, len(val_loader)
        )

        logger.info(f"验证损失: {val_loss:.4f}, 验证困惑度: {val_perplexity:.2f}")

        # 记录轮次结果到wandb
        if wandb_run is not None:
            wandb.log({
                "epoch/train_loss": avg_epoch_loss,
                "epoch/train_perplexity": avg_epoch_perplexity,
                "epoch/val_loss": val_loss,
                "epoch/val_perplexity": val_perplexity,
                "epoch/number": epoch + 1,
                "epoch/best_perplexity": best_perplexity
            }, step=global_step)

        # 保存轮次检查点
        save_checkpoint(
            model_engine, optimizer, scheduler,
            global_step, val_loss, config,
            os.path.join(config.output_dir, f"epoch_{epoch+1}")
        )
    
    logger.info(f"\n训练完成！最佳困惑度: {best_perplexity:.2f}")

    # 记录最终结果到wandb
    if wandb_run is not None:
        wandb.log({
            "final/best_perplexity": best_perplexity,
            "final/total_steps": global_step,
            "final/total_epochs": config.num_epochs
        })

        # 结束wandb运行
        wandb.finish()
        logger.info("wandb运行已结束")

def main():
    """主函数"""
    print("==主函数==")
    parser = argparse.ArgumentParser(description='GPT预训练')
    parser.add_argument('--deepspeed_config', type=str, help='DeepSpeed配置文件路径')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='输出目录')
    parser.add_argument('--batch_size', type=int, default=8, help='批量大小')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--num_epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank passed from DeepSpeed')

    # wandb相关参数
    parser.add_argument('--use_wandb', action='store_true', help='是否使用wandb跟踪实验')
    parser.add_argument('--wandb_project', type=str, default='gpt-pretrain', help='wandb项目名称')
    parser.add_argument('--wandb_entity', type=str, default=None, help='wandb团队名称')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='wandb运行名称')

    args = parser.parse_args()

    # 创建配置
    print("==创建配置==")
    config = PretrainConfig()
    config.output_dir = args.output_dir
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.num_epochs = args.num_epochs

    # 设置wandb配置
    config.use_wandb = args.use_wandb
    config.wandb_project = args.wandb_project
    config.wandb_entity = args.wandb_entity
    config.wandb_run_name = args.wandb_run_name
    
    # 加载DeepSpeed配置
    print("==加载DeepSpeed配置==")
    deepspeed_config = None
    if args.deepspeed_config and os.path.exists(args.deepspeed_config):
        with open(args.deepspeed_config, 'r') as f:
            deepspeed_config = json.load(f)
        logger.info(f"加载DeepSpeed配置: {args.deepspeed_config}")
    
    # 开始训练
    print("==开始训练==")
    train_model(config, deepspeed_config)

if __name__ == "__main__":
    main()
