"""
GPT模型定义
实现类GPT的Transformer模型，包含多头注意力、前馈网络等核心组件
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        """
        初始化多头注意力
        
        Args:
            hidden_size: 隐藏层维度
            num_heads: 注意力头数
            dropout: dropout概率
        """
        super().__init__()
        assert hidden_size % num_heads == 0
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # 线性变换层：将输入投影到Q、K、V
        self.c_attn = nn.Linear(hidden_size, 3 * hidden_size)
        # 输出投影层
        self.c_proj = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, hidden_size]
            mask: 注意力掩码 [batch_size, seq_len, seq_len]
            
        Returns:
            输出张量 [batch_size, seq_len, hidden_size]
        """
        B, T, C = x.size()  # batch_size, seq_len, hidden_size
        
        # 计算Q、K、V
        qkv = self.c_attn(x)  # [B, T, 3*C]
        q, k, v = qkv.split(self.hidden_size, dim=2)  # 每个都是 [B, T, C]
        
        # 重塑为多头形式
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, T, head_dim]
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, T, head_dim]
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, T, head_dim]
        
        # 计算注意力分数
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))  # [B, num_heads, T, T]
        
        # 应用因果掩码（下三角矩阵）
        causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        att = att.masked_fill(~causal_mask, float('-inf'))
        
        # 应用额外掩码（如padding掩码）
        if mask is not None:
            att = att.masked_fill(~mask.unsqueeze(1).unsqueeze(1), float('-inf'))
        
        # Softmax归一化
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        
        # 应用注意力权重到值
        y = att @ v  # [B, num_heads, T, head_dim]
        
        # 重新组合多头输出
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # [B, T, C]
        
        # 输出投影
        y = self.c_proj(y)
        
        return y

class FeedForward(nn.Module):
    """前馈网络"""
    
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        """
        初始化前馈网络
        
        Args:
            hidden_size: 隐藏层维度
            dropout: dropout概率
        """
        super().__init__()
        # 通常前馈网络的中间层维度是隐藏层的4倍
        self.c_fc = nn.Linear(hidden_size, 4 * hidden_size)
        self.c_proj = nn.Linear(4 * hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, hidden_size]
            
        Returns:
            输出张量 [batch_size, seq_len, hidden_size]
        """
        x = self.c_fc(x)
        x = F.gelu(x)  # 使用GELU激活函数
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    """Transformer块"""
    
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        """
        初始化Transformer块
        
        Args:
            hidden_size: 隐藏层维度
            num_heads: 注意力头数
            dropout: dropout概率
        """
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.attn = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.ln_2 = nn.LayerNorm(hidden_size)
        self.mlp = FeedForward(hidden_size, dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, hidden_size]
            mask: 注意力掩码
            
        Returns:
            输出张量 [batch_size, seq_len, hidden_size]
        """
        # 注意力子层（带残差连接）
        x = x + self.attn(self.ln_1(x), mask)
        # 前馈子层（带残差连接）
        x = x + self.mlp(self.ln_2(x))
        return x

class GPTSmall(nn.Module):
    """小型GPT模型"""
    
    def __init__(
        self,
        vocab_size: int = 50257,
        hidden_size: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        max_seq_len: int = 1024,
        dropout: float = 0.1
    ):
        """
        初始化GPT模型
        
        Args:
            vocab_size: 词表大小
            hidden_size: 隐藏层维度
            num_layers: Transformer层数
            num_heads: 注意力头数
            max_seq_len: 最大序列长度
            dropout: dropout概率
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        
        # 词嵌入层
        self.tok_emb = nn.Embedding(vocab_size, hidden_size)
        # 位置嵌入层
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, hidden_size))
        
        # Transformer层
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # 最终层归一化
        self.ln_f = nn.LayerNorm(hidden_size)
        
        # 语言模型头（输出层）
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # 权重共享：词嵌入和输出层共享权重
        self.lm_head.weight = self.tok_emb.weight
        
        # 初始化权重
        self.apply(self._init_weights)
        
        logger.info(f"初始化GPT模型: {self.get_num_params():,} 参数")
        
    def _init_weights(self, module):
        """初始化模型权重"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def get_num_params(self) -> int:
        """获取模型参数数量"""
        return sum(p.numel() for p in self.parameters())
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            input_ids: 输入token ids [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            
        Returns:
            logits: 输出logits [batch_size, seq_len, vocab_size]
        """
        B, T = input_ids.shape
        assert T <= self.max_seq_len, f"序列长度 {T} 超过最大长度 {self.max_seq_len}"
        
        # 词嵌入 + 位置嵌入
        tok_emb = self.tok_emb(input_ids)  # [B, T, hidden_size]
        pos_emb = self.pos_emb[:, :T, :]   # [1, T, hidden_size]
        x = tok_emb + pos_emb              # [B, T, hidden_size]
        
        # 通过所有Transformer层
        for block in self.blocks:
            x = block(x, attention_mask)
        
        # 最终层归一化
        x = self.ln_f(x)
        
        # 语言模型头
        logits = self.lm_head(x)  # [B, T, vocab_size]
        
        return logits
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        """
        生成文本
        
        Args:
            input_ids: 输入token ids [batch_size, seq_len]
            max_new_tokens: 最大生成token数
            temperature: 温度参数
            top_k: top-k采样
            
        Returns:
            生成的token ids [batch_size, seq_len + max_new_tokens]
        """
        self.eval()
        
        for _ in range(max_new_tokens):
            # 如果序列太长，截取最后max_seq_len个token
            input_ids_cond = input_ids if input_ids.size(1) <= self.max_seq_len else input_ids[:, -self.max_seq_len:]
            
            # 前向传播
            logits = self(input_ids_cond)
            
            # 取最后一个位置的logits
            logits = logits[:, -1, :] / temperature
            
            # top-k采样
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # 采样
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # 拼接到输入序列
            input_ids = torch.cat((input_ids, next_token), dim=1)
        
        return input_ids

def test_model():
    """测试模型功能"""
    logger.info("开始测试GPT模型...")
    
    # 创建模型
    model = GPTSmall(
        vocab_size=50257,
        hidden_size=512,
        num_layers=6,
        num_heads=8,
        max_seq_len=1024
    )
    
    # 测试前向传播
    batch_size, seq_len = 2, 100
    input_ids = torch.randint(0, 50257, (batch_size, seq_len))
    
    logger.info(f"输入形状: {input_ids.shape}")
    
    with torch.no_grad():
        logits = model(input_ids)
        logger.info(f"输出logits形状: {logits.shape}")
        
        # 测试生成
        generated = model.generate(input_ids[:1, :10], max_new_tokens=20)
        logger.info(f"生成序列形状: {generated.shape}")
    
    logger.info("模型测试完成！")

if __name__ == "__main__":
    test_model()
