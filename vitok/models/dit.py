import torch
import torch.nn as nn
import math
from vitok.models.vit_modules import Transformer, compute_freqs_cis
from functools import partial
import numpy as np

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, output_width):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, output_width)
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(hidden_size, 2 * hidden_size)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class DiT(nn.Module):
    """ 
    Diffusion Transformer just in embedding space, no need to patchify or unpatchify.
    """
    def __init__(
        self,
        code_width=16,
        width=1024,
        depth=24,
        num_heads=16,
        checkpoint=False,
        class_dropout_prob=0.1,
        rope=10000.0,
        num_classes=1000,
        num_tokens=256,
        learn_sigma=True,
    ):
        super().__init__()

        self.encoder = Transformer(
            embed_dim=width,
            depth=depth,
            num_heads=num_heads,
            checkpoint=checkpoint,
            norm_layer=nn.LayerNorm,
            mlp_ratio=2.67,
            t5_mlp=True,
            adaln=True,
            final_ln=True)

        self.num_tokens = num_tokens
        self.num_classes = num_classes
        
        self.posemb = nn.Parameter(torch.randn(1, self.num_tokens, width) * math.sqrt(1.0 / self.num_tokens))
        self.width = width
        self.depth = depth
        self.num_heads = num_heads
        self.code_width = code_width

        self.learn_sigma = learn_sigma

        self.t_embedder = TimestepEmbedder(width)
        self.y_embedder = LabelEmbedder(num_classes, width, class_dropout_prob)
        self.silu = nn.SiLU()
        
        self.input_width_to_width = nn.Linear(code_width, width)
        if self.learn_sigma:
            self.final_layer = FinalLayer(width, code_width * 2)
        else:
            self.final_layer = FinalLayer(width, code_width)

        self.rope = rope
        if self.rope:
            self.compute_cis = partial(compute_freqs_cis, theta=self.rope, dim=width // num_heads)
        self.initialize_weights()
    
    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.0)
            nn.init.constant_(m.weight, 1.0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.encoder.blocks:
            nn.init.constant_(block.linear_modulation.weight, 0)
            nn.init.constant_(block.linear_modulation.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, y, train=False):
        x = self.input_width_to_width(x)
        t = self.t_embedder(t)
        y = self.y_embedder(y, train)
        c = self.silu(t + y)
        x = x + self.posemb
        if self.rope:
            length = torch.Tensor(np.arange(x.shape[1])).to(x.device)
            freqs = self.compute_cis(length)
        else:
            freqs = None
        x = self.encoder(x, cond=c, freqs=freqs)
        x = self.final_layer(x, c)
        return x

    def forward_cfg(self, x, t, y, cfg_scale):
        x = torch.cat([x, x], dim=0)
        t = torch.cat([t, t], dim=0)
        y_null = torch.ones_like(y) * self.num_classes
        y = torch.cat([y, y_null], dim=0)
        out = self.forward(x, t, y)
        cond_out, uncond_out = torch.split(out, x.shape[0] // 2, dim=0)
        out = uncond_out + cfg_scale * (cond_out - uncond_out)
        return out

def Model(**kw):  # pylint: disable=invalid-name
  """Factory function, because linen really don't like what I'm doing!"""
  return DiT(**kw)
