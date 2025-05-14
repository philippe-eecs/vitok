import torch
import torch.nn as nn
from .attention import Attention, CrossAttention
from torch.utils.checkpoint import checkpoint

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class T5_MLP(nn.Module):
    def __init__(self,
                in_features,
                hidden_features=None,
                out_features=None,
                activation=nn.GELU,
                drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(in_features, hidden_features)
        self.fc3 = nn.Linear(hidden_features, out_features)
        self.act = activation()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.act(self.fc1(x)) * self.fc2(x)
        x = self.fc3(x)
        x = self.drop(x)
        return x

class MLP(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 activation=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = activation()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_norm=None,
                 t5_mlp=False, 
                 drop=0., 
                 attn_drop=0.,
                 norm_layer=nn.LayerNorm,
                 block_mask=None,
                 adaln=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            block_mask=block_mask,
            attn_drop=attn_drop,
            proj_drop=drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if t5_mlp:
            self.mlp = T5_MLP(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                drop=drop)
        else:
            self.mlp = MLP(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                drop=drop)
        self.adaln = adaln
        if adaln:
            self.linear_modulation = nn.Linear(dim, dim * 6)
    def forward(self, x, cond=None, attn_mask=None, freqs_cos=None, freqs_sin=None):
        if self.adaln:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.linear_modulation(cond).chunk(6, dim=-1)
            x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), attn_mask=attn_mask, freqs_cos=freqs_cos, freqs_sin=freqs_sin)
            x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        else:
            assert cond is None, 'Passing in condition to non-adaln block, use adaln=True'
            x = x + self.attn(self.norm1(x), attn_mask=attn_mask, freqs_cos=freqs_cos, freqs_sin=freqs_sin)
            x = x + self.mlp(self.norm2(x))
        return x

class Transformer(nn.Module):
    """ Bi-directional Transformer
    """
    def __init__(self,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 checkpoint=False, #TODO: Bugged with Compile + DDP, either use FSDP or fix bug
                 qkv_bias=False,
                 qk_norm=False,
                 t5_mlp=False,
                 block_mask=None,
                 norm_layer=nn.LayerNorm,
                 final_ln=False,
                 adaln=False): #TODO: Check if this is best conditioning for DiT
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                norm_layer=norm_layer,
                block_mask=block_mask,
                t5_mlp=t5_mlp,
                adaln=adaln) for _ in range(depth)
        ])

        if final_ln:
            self.norm = nn.LayerNorm(embed_dim)
        self.final_ln = final_ln
        self.checkpoint = checkpoint

    def forward(self, x, cond=None, attn_mask=None, freqs=None):
        if freqs is None:
            freqs_cos = [None] * len(self.blocks)
            freqs_sin = [None] * len(self.blocks)
        else:
            freqs_cos, freqs_sin = freqs
            if freqs_cos.shape[0] != len(self.blocks):
                freqs_cos = [freqs_cos] * len(self.blocks)
                freqs_sin = [freqs_sin] * len(self.blocks)
        for i, blk in enumerate(self.blocks):
            if self.checkpoint:
                x = checkpoint(blk, x, cond=cond, attn_mask=attn_mask, freqs_cos=freqs_cos[i], freqs_sin=freqs_sin[i], use_reentrant=False)
            else:
                x = blk(x, attn_mask=attn_mask, cond=cond, freqs_cos=freqs_cos[i], freqs_sin=freqs_sin[i])
        if self.final_ln:
            x = self.norm(x)
        return x