# Copyright 2024 Big Vision Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NaFlex ViT = NaViT + FlexiViT.

Based on:
* FlexiViT: https://arxiv.org/abs/2212.08013
* NaViT: https://arxiv.org/abs/2307.06304
"""

import math
import re
from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np



from big_vision.models import vit
import big_vision.models.proj.image_text.utils as it_utils
from typing import Optional
from big_vision.models.proj.image_text.utils import sequence_parallel_attention
from jax.sharding import PartitionSpec as P

try:
    # Kvax splash kernel (JAX‑only).  If missing, we'll raise at runtime.
    from kvax.splash_attention import splash_attention
except ImportError:  # pragma: no cover
    print("Using TPU not GPU, splash attention not available")
    splash_attention = None  # lazy guard later

def _decode_posemb(posemb):
  if (m := re.fullmatch(r"learn_2d(\(\d+\))", posemb)):
    grid_size = int(m.groups()[0][1:-1])
    return "learn_2d", grid_size
  return posemb, None

def _ring_block_mask(seq_len: int, radius: int, num_heads: int):
    """Returns (num_heads, B, B) boolean mask for Splash.
    Block size is fixed at 32.  radius is in *tokens* (not blocks)."""
    blk = 32
    blocks = math.ceil(seq_len / blk)
    r_blocks = math.ceil(radius / blk)
    idx = jnp.arange(blocks)
    mask2d = (jnp.abs(idx[:, None] - idx[None, :]) <= r_blocks)
    mask3d = jnp.broadcast_to(mask2d, (num_heads, blocks, blocks))
    return mask3d

class FlashGQAAttention(nn.Module):
    embed_dim: int
    num_q_heads: int
    kv_groups: int = 1           # Q heads per KV head
    head_dim: int = 64
    dropout: float = 0.0
    local_ws: tuple[int, int] | None = None  # token window (L,R)
    use_splash: bool = False
    use_seq_parallel: bool = False
    dtype: str = "bfloat16"

    @nn.compact
    def __call__(self, x, mask=None, *, deterministic: bool = True):
        B, T, _ = x.shape
        num_kv_heads = max(1, self.num_q_heads // self.kv_groups)

        # Project Q/K/V
        q = nn.DenseGeneral(self.num_q_heads * self.head_dim, axis=-1,
                            dtype=self.dtype, name="q_proj")(x)
        k = nn.DenseGeneral(num_kv_heads * self.head_dim, axis=-1,
                            dtype=self.dtype, name="k_proj")(x)
        v = nn.DenseGeneral(num_kv_heads * self.head_dim, axis=-1,
                            dtype=self.dtype, name="v_proj")(x)

        def split(t, h):
            return t.reshape(B, T, h, self.head_dim)
        q, k, v = map(split, (q, k, v), (self.num_q_heads, num_kv_heads, num_kv_heads))

        # ------------------------------------------------------------------
        # Attention kernel selection
        # ------------------------------------------------------------------
        if self.use_splash:
            if splash_attention is None:
                raise ImportError("kvax.splash_attention not found. Install Kvax or set use_splash=False")
            # Sliding window → ring mask radius
            if self.local_ws is None:
                raise ValueError("Splash requires local_ws (sliding window) to define block mask")
            left, right = self.local_ws
            radius = max(left, right)
            blk_mask = _ring_block_mask(T, radius, self.num_q_heads)  # share mask across all Q heads
            attn_fn = partial(splash_attention, block_layout=blk_mask)
        else:
            attn_fn = partial(jax.nn.dot_product_attention,
                              local_window_size=self.local_ws,
                              mask=mask)

        # Wrap with sequence parallel if requested
        if self.use_seq_parallel: #TODO: Need to set this up
            attn_fn = sequence_parallel_attention(attn_fn)

        y = attn_fn(q, k, v)  # (B, T, Qh, Hd)
        y = y.reshape(B, T, -1)
        y = nn.Dense(self.embed_dim, dtype=self.dtype, name="out_proj")(y)
        y = nn.Dropout(self.dropout)(y, deterministic)
        return y


def _pos_emb_resize(pos_emb, shapes, coords, l):
  """Resizes the positional embeddings to match the input image size.
  
  Args:
    pos_emb: Positional embeddings.
    shapes: Image shapes (usually `coords.max(axis=1) + 1`).
    coords: Patch coordinates.
    l: Maximum number of patches per side. Necesary in order to have a static
      return shape.

  Setting l to 64 is a heuristic. Ideally, we would use
  `l = tokens.shape[1]` here, but that requires too much memory,
  especially for high-resolution inputs. Using a lower value
  effectively limits the maximum resolution to `l x patch_size`.
  Resolutions above that will lead to NaNs in the positional
  embeddings and NaN model outputs.
  Note: this value can be adjusted post-hoc without retraining.

  Returns:
    Postional embeddings for every patch.
  """

  def resize_fn(shape, coords):
    emb = jax.image.scale_and_translate(
        pos_emb,
        shape=(l, l, pos_emb.shape[-1]),
        spatial_dims=(0, 1),
        scale=shape / jnp.asarray(pos_emb.shape[:2]),
        translation=jnp.asarray([0, 0]),
        method="lanczos3", antialias=True)
    gather_dim = jax.lax.GatherDimensionNumbers(
        offset_dims=(1,),
        collapsed_slice_dims=(0, 1),
        start_index_map=(0, 1, 2)
    )
    return jax.lax.gather(
        emb,
        jnp.pad(coords, [[0, 0], [0, 1]]),
        gather_dim,
        [1, 1, emb.shape[-1]],
        mode="fill")
  return it_utils.batch_shmap(
      jax.vmap(resize_fn, in_axes=(0, 0), out_axes=0),
      shapes, coords)

class T5_MLP(nn.Module):
  """SwiGLU T5-style MLP block (Noam Shazeer style, no biases, 3 matmuls)."""
  mlp_dim: Optional[int] = None  # Defaults to 2.3x input dim
  dropout: float = 0.0
  dtype_mm: str = "bfloat16"

  @nn.compact
  def __call__(self, x, deterministic=True):
    """Applies SwiGLU T5 MLP block (no biases, 3 matmuls)."""
    d = x.shape[-1]
    hidden_dim = self.mlp_dim or int(2.3 * d)

    # No bias, 3 matmuls: gate, up, and out
    gate = nn.Dense(hidden_dim, use_bias=False, dtype=self.dtype_mm,
                    kernel_init=nn.initializers.xavier_uniform())(x)
    up = nn.Dense(hidden_dim, use_bias=False, dtype=self.dtype_mm,
                  kernel_init=nn.initializers.xavier_uniform())(x)
    
    dense_out_layer = nn.Dense(d, use_bias=False, dtype=self.dtype_mm, # Store the layer instance
                               kernel_init=nn.initializers.xavier_uniform())
    
    # SwiGLU activation: swish(gate) * up
    x_activated = nn.silu(gate) * up
    x_activated = nn.Dropout(rate=self.dropout)(x_activated, deterministic)
    
    # Get the output tensor from the final dense layer
    output_tensor = dense_out_layer(x_activated)
    
    # Apply sharding constraint to the output tensor
    output_tensor = nn.with_logical_constraint(output_tensor, ("act_batch", "act_len", "act_emb"))
    
    return output_tensor

class Encoder1DBlock(nn.Module):
    mlp_dim: int | None = None
    num_q_heads: int = 12
    kv_groups: int = 1
    dropout: float = 0.0
    dtype_mm: str = "bfloat16"
    head_dim: int = 64
    local_ws: tuple[int, int] | None = None
    use_splash: bool = False
    use_seq_parallel: bool = False
    use_t5_mlp: bool = False

    @nn.compact
    def __call__(self, x, mask=None, deterministic=True):
        x = nn.with_logical_constraint(x, ("act_batch", "act_len", "act_emb"))
        if mask is not None:
            mask = mask[..., None, :, :]
        y = nn.LayerNorm(dtype=self.dtype_mm)(x)
        y = FlashGQAAttention(embed_dim=x.shape[-1],
                               num_q_heads=self.num_q_heads,
                               kv_groups=self.kv_groups,
                               head_dim=self.head_dim,
                               dropout=self.dropout,
                               local_ws=self.local_ws,
                               use_splash=self.use_splash,
                               use_seq_parallel=self.use_seq_parallel,
                               dtype=self.dtype_mm)(y, mask=mask,
                                                   deterministic=deterministic)
        y = nn.with_logical_constraint(y, ("act_batch", "act_len", "act_emb"))
        x = x + y
        y = nn.LayerNorm(dtype=self.dtype_mm)(x)
        if self.use_t5_mlp:
            y = T5_MLP(mlp_dim=self.mlp_dim, dropout=self.dropout,
                       dtype_mm=self.dtype_mm)(y, deterministic)
        else:
            y = vit.MlpBlock(mlp_dim=self.mlp_dim, dropout=self.dropout,
                         dtype_mm=self.dtype_mm)(y, deterministic)
        y = nn.with_logical_constraint(y, ("act_batch", "act_len", "act_emb"))
        y = nn.Dropout(self.dropout)(y, deterministic)
        x = x + y
        x = nn.with_logical_constraint(x, ("act_batch", "act_len", "act_emb"))
        return x, {}


class Encoder(nn.Module):
  """Transformer Model Encoder for sequence to sequence translation."""
  depth: int
  mlp_dim: int | None = None  # Defaults to 4x input dim
  num_heads: int = 12
  dropout: float = 0.0
  scan: bool = False
  remat_policy: str = "nothing_saveable"
  dtype_mm: str = "bfloat16"
  local_ws: tuple[int, int] | None = None
  use_splash: bool = False
  use_seq_parallel: bool = False
  use_t5_mlp: bool = False
  kv_groups: int = 1
  head_dim: int = 64

  @nn.compact
  def __call__(self, x, mask=None, deterministic=True):
    out = {}

    if self.scan:
      block = nn.remat(
          Encoder1DBlock,
          prevent_cse=False,
          static_argnums=(3,),  # 0=self, 3=deterministic
          policy=getattr(jax.checkpoint_policies, self.remat_policy, None),
          )
      x, scan_out = nn.scan(
          block,
          variable_axes={"params": 0},
          split_rngs={"params": True, "dropout": True},
          in_axes=nn.broadcast,
          length=self.depth)(
              name="encoderblock",
              dtype_mm=self.dtype_mm,
              mlp_dim=self.mlp_dim,
              num_q_heads=self.num_heads,
              kv_groups=self.kv_groups,
              head_dim=self.head_dim,
              dropout=self.dropout,
              local_ws=self.local_ws,
              use_splash=self.use_splash,
              use_seq_parallel=self.use_seq_parallel,
              use_t5_mlp=self.use_t5_mlp
          )(x, mask, deterministic)
      for lyr in range(self.depth):
        out[f"block{lyr:02d}"] = jax.tree.map(lambda o, l=lyr: o[l], scan_out)
    else:
      # Input Encoder
      for lyr in range(self.depth):
        block_cur = Encoder1DBlock(
            name=f"encoderblock_{lyr}",
            dtype_mm=self.dtype_mm,
            mlp_dim=self.mlp_dim, num_q_heads=self.num_heads,
            kv_groups=self.kv_groups,
            head_dim=self.head_dim,
            dropout=self.dropout,
            local_ws=self.local_ws,
            use_splash=self.use_splash,
            use_seq_parallel=self.use_seq_parallel,
            use_t5_mlp=self.use_t5_mlp
        )
        x, out[f"block{lyr:02d}"] = block_cur(x, mask, deterministic)
      out["pre_ln"] = x  # Alias for last block, but without the number in it.

    return nn.LayerNorm(name="encoder_norm")(x), out


class MAPHead(nn.Module):
  """Multihead Attention Pooling."""
  mlp_dim: int | None = None  # Defaults to 4x input dim
  num_heads: int = 12

  @nn.compact
  def __call__(self, x, mask=None):
    n, l, d = x.shape  # pylint: disable=unused-variable
    probe = self.param("probe", nn.initializers.xavier_uniform(),
                       (1, 1, d), x.dtype)
    probe = jnp.tile(probe, [n, 1, 1])

    if mask is not None:
      mask = mask[..., None, None, :]  # Add query and head dims.

    x = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        kernel_init=nn.initializers.xavier_uniform())(probe, x, mask=mask)

    y = nn.LayerNorm()(x)
    x = x + vit.MlpBlock(mlp_dim=self.mlp_dim)(y)
    return x[:, 0]


class _Model(nn.Module):
  """ViT model."""

  num_classes: int | None = None
  width: int = 768
  depth: int = 12
  mlp_dim: int | None = None  # Defaults to 4x input dim
  num_heads: int = 12
  rep_size: int | bool = False
  pool_type: str = "gap"  # Can also be "map" or "tok"
  head_zeroinit: bool = True
  scan: bool = False
  # or "dots_with_no_batch_dims_saveable" for more speed (memory costly)
  remat_policy: str = "nothing_saveable"
  dtype_mm: str = "bfloat16"

  posemb: str = "learn_2d(128)"
  nposemb: int | None = None  # Needs to be overwritten

  patchln_pre: bool = False
  patchln_post: bool = False

  # New options for GQA/T5/Local/SeqParallel
  local_ws: tuple[int, int] | None = None
  use_splash: bool = False
  use_seq_parallel: bool = False
  use_t5_mlp: bool = False
  kv_groups: int = 1
  head_dim: int = 64

  @nn.compact
  def __call__(self, image, *, train=False):
    out = {}
    patches, ptype, yabs, xabs = image
    patches = jnp.asarray(patches, self.dtype_mm)  # BN(hw3) of float32

    if self.patchln_pre:
      patches = nn.LayerNorm(name="patchln_pre")(patches)

    # Embed the patches.
    tokens = out["stem"] = nn.Dense(
        self.width, name="embedding", dtype=self.dtype_mm)(patches)

    if self.patchln_post:
      tokens = nn.LayerNorm(name="patchln_post")(tokens)

    x = tokens
    posemb, posemb_grid_size = _decode_posemb(self.posemb)
    if posemb == "learn_2d":
      posembs = self.param(
          "pos_embedding",
          nn.initializers.normal(stddev=1/np.sqrt(self.width)),
          (self.nposemb, self.nposemb, self.width), self.dtype_mm)

      coords = jnp.stack([yabs, xabs], axis=-1)
      shapes = coords.max(axis=1) + 1
      # See comment in `_pos_emb_resize` for details.
      x += _pos_emb_resize(posembs, shapes, coords, posemb_grid_size)
    elif posemb == "rope":
      pass #TODO: Implement Rope 1D
    else:
      raise ValueError(f"Unknown posemb: '{self.posemb}'")

    out["with_posemb"] = x

    n, l, c = x.shape # Original sequence length before any CLS token logic
    # num_cls_tokens = 0 # Removed CLS logic
    # if self.pool_type == "tok":
    #   num_cls_tokens = 4 # Assuming 4 CLS tokens from `cls = self.param("cls", ..., (1, 4, c), ...)`
    #   cls_params = self.param("cls", nn.initializers.zeros, (1, num_cls_tokens, c), x.dtype)
    #   x = jnp.concatenate([jnp.tile(cls_params, [n, 1, 1]), x], axis=1)
    #   # Update ptype for the new CLS tokens. Assuming ptype was (batch, old_L).
    #   # cls_ptype should mark these as valid, non-padding tokens.
    #   # If ptype uses 1 for valid patches, CLS should also be 1.
    #   cls_ptypes = jnp.ones((n, num_cls_tokens), dtype=ptype.dtype)
    #   ptype = jnp.concatenate([cls_ptypes, ptype], axis=1)
    
    # current_seq_len = x.shape[1] # Sequence length is now just l

    # 1. Basic padding mask based on original ptype
    # ptype == 1 means valid token (patch), 0 means padding.
    padding_mask_1d = (ptype == 1) # ptype here is the original, without CLS modifications
    # Create a 2D mask: a query can attend to a key if both are valid (not padding)
    sa_mask = jnp.logical_and(padding_mask_1d[..., :, None], padding_mask_1d[..., None, :])
    # This sa_mask is (batch, seq_len, seq_len). 
    # jax.nn.dot_product_attention will use this AND local_window_size if both are provided.

    # The complex custom mask logic for CLS global + patch local is removed.
    # We now rely purely on `local_ws` passed to the attention module for windowing.

    x, out["encoder"] = Encoder(
        depth=self.depth,
        mlp_dim=self.mlp_dim,
        num_heads=self.num_heads,
        scan=self.scan,
        remat_policy=self.remat_policy,
        dtype_mm=self.dtype_mm,
        local_ws=self.local_ws, # Restore passing self.local_ws
        use_splash=self.use_splash, # Kernel choice
        use_seq_parallel=self.use_seq_parallel, # Should be False
        use_t5_mlp=self.use_t5_mlp,
        kv_groups=self.kv_groups,
        head_dim=self.head_dim,
        name="Transformer")(
            x, mask=sa_mask, deterministic=not train) # Pass the simple padding mask
    encoded = x
    # Ignore the padding tokens when pooling:
    pool_mask = (ptype == 1)  # 1 == patch (not pad)
    if self.pool_type == "map":
      maphead = MAPHead(num_heads=self.num_heads, mlp_dim=self.mlp_dim)
      x = maphead(x, mask=pool_mask)
    elif self.pool_type == "gap":
      pool_mask = pool_mask[..., None]
      x = jnp.sum(x * pool_mask, axis=1) / jnp.sum(pool_mask, axis=1)
    elif self.pool_type == "max":
      # Tested in (internal link)
      pool_mask = pool_mask[..., None]
      ignore = jnp.where(pool_mask, 0, jnp.finfo(x.dtype).min)
      x = jnp.max(pool_mask * x + ignore, axis=1)
    elif self.pool_type == "tok":
      x, encoded = x[:, :1], encoded[:, 1:]
      #average CLS tokens
      x = jnp.mean(x, axis=1)
      #ptype = ptype[:, 0]
    elif self.pool_type == "none":
      pass
    else:
      raise ValueError(f"Unknown pool type: '{self.pool_type}'")
    out["head_input"] = x

    if self.rep_size:
      rep_size = self.width if self.rep_size is True else self.rep_size
      hid = nn.Dense(rep_size, name="pre_logits")
      x = nn.tanh(hid(x))

    out["pre_logits"] = x
    out["encoded"] = encoded
    if self.num_classes:
      kw = {"kernel_init": nn.initializers.zeros} if self.head_zeroinit else {}
      head = nn.Dense(self.num_classes, name="head", **kw)
      x = out["logits"] = head(x)

    return x, out


def Model(num_classes=None, *, variant=None, **kw):  # pylint: disable=invalid-name
  """Factory function, because linen really don't like what I'm doing!"""
  return _Model(num_classes, **{**vit.decode_variant(variant), **kw})

load = vit.load
