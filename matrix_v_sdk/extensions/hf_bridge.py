import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .torch_bridge import MatrixVLinear

class MatrixVAttention(nn.Module):
    """
    A Transformer Attention layer accelerated by Matrix-V.
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Q, K, V projections using Matrix-V acceleration
        self.q_proj = MatrixVLinear(embed_dim, embed_dim)
        self.k_proj = MatrixVLinear(embed_dim, embed_dim)
        self.v_proj = MatrixVLinear(embed_dim, embed_dim)
        
        self.out_proj = MatrixVLinear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # 1. Projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Split into heads
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 2. Scaled Dot-Product Attention
        # Note: torch.matmul is used here for head-wise operations, 
        # but the projections themselves are O(N^1.5) accelerated.
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        attn_output = torch.matmul(attn_weights, v)
        
        # Recombine heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        # 3. Output projection (Accelerated)
        return self.out_proj(attn_output)

class MatrixVTransformerBlock(nn.Module):
    """
    A full Transformer block (Attention + MLP) accelerated by Matrix-V.
    """
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()
        self.attention = MatrixVAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.ffn = nn.Sequential(
            MatrixVLinear(embed_dim, ff_dim),
            nn.ReLU(),
            MatrixVLinear(ff_dim, embed_dim)
        )

    def forward(self, x):
        # Attention + Rediual
        x = x + self.attention(self.norm1(x))
        # FFN + Residual
        x = x + self.ffn(self.norm2(x))
        return x

