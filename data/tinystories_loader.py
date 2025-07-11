"""
TinyStories数据集加载器
用于加载和预处理TinyStories数据集，构建PyTorch Dataset和DataLoader
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from datasets import load_dataset
import logging
from typing import Dict, List, Optional

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TinyStoriesDataset(Dataset):
    """TinyStories数据集类"""
    
    def __init__(
        self, 
        tokenizer: GPT2Tokenizer,
        max_length: int = 1024,
        split: str = "train",
        cache_dir: Optional[str] = None
    ):
        """
        初始化数据集
        
        Args:
            tokenizer: GPT2分词器
            max_length: 最大序列长度
            split: 数据集分割 ("train" 或 "validation")
            cache_dir: 缓存目录
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        
        logger.info(f"正在加载TinyStories数据集 - {split}分割...")
        
        # 加载数据集
        try:
            self.dataset = load_dataset(
                "roneneldan/TinyStories", 
                split=split,
                cache_dir=cache_dir
            )
            logger.info(f"成功加载{len(self.dataset)}条{split}数据")
        except Exception as e:
            logger.error(f"加载数据集失败: {e}")
            raise
        
        # 设置分词器的pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个数据样本
        
        Args:
            idx: 数据索引
            
        Returns:
            包含input_ids和attention_mask的字典
        """
        # 获取文本
        text = self.dataset[idx]['text']
        
        # 分词并编码
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)
        
        # 添加防御性检查：如果是空样本，抛出异常
        if input_ids.numel() == 0 or attention_mask.numel() == 0:
            return None # 返回 None，后续在 collate_fn 里过滤掉

    
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

def safe_collate_fn(batch: List[Optional[Dict[str, torch.Tensor]]]) -> Dict[str, torch.Tensor]:
    # 去掉空样本（None）
    batch = [x for x in batch if x is not None]
    
    if len(batch) == 0:
        raise ValueError("整批数据都是无效样本（None），请检查数据或调小 batch size。")

    return {
        'input_ids': torch.stack([x['input_ids'] for x in batch]),
        'attention_mask': torch.stack([x['attention_mask'] for x in batch])
    }


def create_dataloaders(
    batch_size: int = 8,    # debug  元数据为8
    max_length: int = 1024,
    num_workers: int = 4,
    cache_dir: Optional[str] = None,
    train_subset_size: Optional[int] = None, # 10240,
    val_subset_size: Optional[int] = None # 10240   # debug
) -> tuple[DataLoader, DataLoader, GPT2Tokenizer]:
    """
    创建训练和验证数据加载器
    
    Args:
        batch_size: 批量大小
        max_length: 最大序列长度
        num_workers: 数据加载工作进程数
        cache_dir: 缓存目录
        train_subset_size: 训练集子集大小（用于快速测试）
        val_subset_size: 验证集子集大小（用于快速测试）
        
    Returns:
        (train_loader, val_loader, tokenizer)
    """
    logger.info("正在初始化GPT2分词器...")
    
    # 初始化分词器
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # 设置特殊token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"分词器词表大小: {len(tokenizer)}")
    
    # 创建数据集
    logger.info("正在创建训练数据集...")
    train_dataset = TinyStoriesDataset(
        tokenizer=tokenizer,
        max_length=max_length,
        split="train",
        cache_dir=cache_dir
    )
    
    logger.info("正在创建验证数据集...")
    val_dataset = TinyStoriesDataset(
        tokenizer=tokenizer,
        max_length=max_length,
        split="validation",
        cache_dir=cache_dir
    )
    
    # 如果指定了子集大小，则创建子集
    if train_subset_size is not None:
        train_indices = list(range(min(train_subset_size, len(train_dataset))))
        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        logger.info(f"使用训练子集，大小: {len(train_dataset)}")
    
    if val_subset_size is not None:
        val_indices = list(range(min(val_subset_size, len(val_dataset))))
        val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
        logger.info(f"使用验证子集，大小: {len(val_dataset)}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=safe_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=safe_collate_fn
    )
    
    logger.info(f"训练数据加载器: {len(train_loader)}个批次")
    logger.info(f"验证数据加载器: {len(val_loader)}个批次")
    
    return train_loader, val_loader, tokenizer

def verify_data_integrity(
    train_loader,
    val_loader,
    tokenizer,
    num_samples: int = 3
):
    """
    验证数据完整性，检查token化和解码是否正确

    Args:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        tokenizer: 分词器
        num_samples: 验证的样本数量
    """
    logger.info("=== 开始数据完整性验证 ===")

    # 验证训练集
    logger.info("验证训练集数据...")
    train_iter = iter(train_loader)
    for i in range(min(num_samples, len(train_loader))):
        batch = next(train_iter)
        logger.info(f"\n--- 训练样本 {i+1} ---")

        # 获取第一个样本
        input_ids = batch['input_ids'][0]
        attention_mask = batch['attention_mask'][0]

        logger.info(f"Token序列长度: {len(input_ids)}")
        logger.info(f"有效token数量: {attention_mask.sum().item()}")

        # 解码完整文本
        full_text = tokenizer.decode(input_ids, skip_special_tokens=True)
        logger.info(f"解码文本长度: {len(full_text)} 字符")
        logger.info(f"解码文本预览: {full_text[:200]}...")

        # 检查特殊token
        special_tokens = []
        for token_id in input_ids[:20]:  # 检查前20个token
            if token_id.item() in [tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.bos_token_id]:
                token_name = tokenizer.decode([token_id.item()])
                special_tokens.append(f"{token_id.item()}({token_name})")

        if special_tokens:
            logger.info(f"前20个token中的特殊token: {special_tokens}")

        # 验证token范围
        min_token = input_ids.min().item()
        max_token = input_ids.max().item()
        logger.info(f"Token ID范围: {min_token} - {max_token}")

        if max_token >= tokenizer.vocab_size:
            logger.warning(f"发现超出词表范围的token: {max_token} >= {tokenizer.vocab_size}")

    # 验证验证集
    logger.info("\n验证验证集数据...")
    val_iter = iter(val_loader)
    for i in range(min(num_samples, len(val_loader))):
        batch = next(val_iter)
        logger.info(f"\n--- 验证样本 {i+1} ---")

        input_ids = batch['input_ids'][0]
        attention_mask = batch['attention_mask'][0]

        logger.info(f"Token序列长度: {len(input_ids)}")
        logger.info(f"有效token数量: {attention_mask.sum().item()}")

        # 解码文本
        full_text = tokenizer.decode(input_ids, skip_special_tokens=True)
        logger.info(f"解码文本预览: {full_text[:200]}...")

    logger.info("\n=== 数据完整性验证完成 ===")

def test_dataloader():
    """测试数据加载器功能"""
    logger.info("开始测试数据加载器...")

    # 创建小规模数据加载器进行测试
    train_loader, val_loader, tokenizer = create_dataloaders(
        batch_size=2,
        max_length=512,
        train_subset_size=10,
        val_subset_size=5
    )

    # 测试训练数据加载器
    logger.info("测试训练数据加载器...")
    for i, batch in enumerate(train_loader):
        logger.info(f"批次 {i+1}:")
        logger.info(f"  input_ids shape: {batch['input_ids'].shape}")
        logger.info(f"  attention_mask shape: {batch['attention_mask'].shape}")

        # 解码第一个样本查看内容
        sample_text = tokenizer.decode(batch['input_ids'][0], skip_special_tokens=True)
        logger.info(f"  样本文本预览: {sample_text[:100]}...")

        if i >= 1:  # 只测试前2个批次
            break

    # 进行数据完整性验证（注释掉以减少输出）
    # verify_data_integrity(train_loader, val_loader, tokenizer, num_samples=2)

    logger.info("数据加载器测试完成！")

if __name__ == "__main__":
    test_dataloader()
