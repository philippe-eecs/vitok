import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    if freqs_cis.shape == (x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim-2 else 1 for i, d in enumerate(x.shape)]
    elif freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim-3 else 1 for i, d in enumerate(x.shape)]
        
    return freqs_cis.view(*shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor
    ):
        xq_r, xq_i = xq.float().reshape(*xq.shape[:-1], -1, 2).unbind(-1)
        xk_r, xk_i = xk.float().reshape(*xk.shape[:-1], -1, 2).unbind(-1)

        # reshape freqs_cos and freqs_sin for broadcasting
        freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
        freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

        # apply rotation using real numbers
        xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
        xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
        xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
        xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

        # flatten last two dimensions
        xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
        xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

        return xq_out.type_as(xq), xk_out.type_as(xk)

class CrossAttention(nn.Module):
    def __init__(
        self,
        encoder_dim,
        decoder_dim,
        num_heads=8,
        qkv_bias=False,
        qk_norm: bool = False,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = decoder_dim // num_heads
        self.q = nn.Linear(decoder_dim, decoder_dim, bias=qkv_bias)
        self.kv = nn.Linear(encoder_dim, decoder_dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(decoder_dim, decoder_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.q_norm = nn.LayerNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(head_dim) if qk_norm else nn.Identity()

    def forward(self, x, y, attn_mask=None):
        """
        query from decoder (x), key and value from encoder (y)
        """
        B, N, C = x.shape
        Ny = y.shape[1]
        q = (
            self.q(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        kv = (
            self.kv(y)
            .reshape(B, Ny, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
                attn_mask=attn_mask,
            )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_norm: bool = False,
                 block_mask=None,
                 attn_drop=0.,
                 proj_drop=0., #Precomputed frequencies for the cosine positional encoding
                 ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.q_norm = nn.LayerNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(head_dim) if qk_norm else nn.Identity()
        self.block_mask = block_mask

    def forward(self, x, attn_mask=None, freqs_cos=None, freqs_sin=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k) # make torchscript happy (cannot use tensor as tuple)
        if freqs_cos is not None:
            q, k = apply_rotary_emb(q, k, freqs_cos, freqs_sin)
        
        if self.block_mask is not None:
            x = F.scaled_dot_product_attention( #TODO: Implement FlexAttention with Block Casual Attention for Videos
                    q, k, v,
                    dropout_p=self.attn_drop.p if self.training else 0.,
                    attn_mask=self.block_mask,
                )
        else:
            x = F.scaled_dot_product_attention( #TODO: Implement FlexAttention with Block Casual Attention for Videos
                    q, k, v,
                    dropout_p=self.attn_drop.p if self.training else 0.,
                    attn_mask=attn_mask,
                )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
