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

"""NaFlex SigLIP-VAE model.

Combines NaFlex (NaViT + FlexiViT) with SigLIP-VAE to handle random resolutions and aspect ratios.
"""

import abc
import re
from typing import Any, Optional, Tuple, Union

import importlib
import flax.linen as nn
import jax
import jax.numpy as jnp
import einops
from big_vision.models.proj.givt.vae import Model as VAE
from big_vision.models.vit import MlpBlock
from big_vision import utils
import flax
from big_vision.models import common

def patches_to_image(image,
                     max_x, max_y,
                     patch_size=16,
                     fill_value=0.0):
    patches, ptype, yidx, xidx = image
    B, N, token_dim = patches.shape

    # 1) Instead of boolean indexing, use a where operation
    padding_token = jnp.full((token_dim,), fill_value, dtype=patches.dtype)
    # Create a mask and use where instead of .at[].set()
    mask = (ptype == 0)[:, :, None]  # Add channel dimension
    patches = jnp.where(mask, padding_token, patches)
    
    # 2) Create the empty grid
    token_grid = jnp.full((B, max_y, max_x, token_dim),
                          fill_value, dtype=patches.dtype)
    
    # 3) Make a batch index of shape [B, N]
    bidx = jnp.arange(B)[:, None]  # shape [B, 1]
    bidx = jnp.broadcast_to(bidx, (B, N))  # shape [B, N]

    # 4) Scatter using (batch, y, x) indexing
    token_grid = token_grid.at[bidx, yidx, xidx].set(patches)

    # 5) Optionally override (0,0) with tokens[:, 0]
    token_grid = token_grid.at[jnp.arange(B), 0, 0].set(patches[:, 0])
    
    # 6) Reshape + rearrange
    token_grid = token_grid.reshape(B, max_y * max_x, patch_size, patch_size, 3)
    images = einops.rearrange(
        token_grid,
        'B (gh gw) ph pw c -> B (gh ph) (gw pw) c',
        gh=max_y, gw=max_x, ph=patch_size, pw=patch_size
    )
    return images

def reparametrize(
      mu: jax.Array,
      logvar: jax.Array,
      rng: jax.Array,
  ) -> jax.Array:
    std = jnp.exp(0.5 * logvar)
    eps = jax.random.normal(rng, shape=std.shape, dtype=std.dtype)
    return mu + std * eps


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
    x = x + MlpBlock(mlp_dim=self.mlp_dim)(y)
    return x[:, 0]


ConfigDict = Any

class Model(VAE):
    """NaFlex SigLIP VAE with dual encoders (image + text)."""
    # Image encoder config
    image: Optional[ConfigDict] = None
    decoder_image: Optional[ConfigDict] = None
    image_model: str = "proj.image_text.naflex_vit"
    
    # Text encoder config
    text: Optional[ConfigDict] = None
    text_model: str = "proj.image_text.text_transformer"
    
    # Shared VAE config
    out_dims: Union[int, Tuple[int, int]] = 128
    channel_dim: int = 16
    temperature_init: float = 1.0
    freeze_encoder: bool = False
    bias_init: Optional[float] = None
    # Use this flag to decide whether to use layernorm in the bottleneck.
    layernorm_code: bool = False  
    deterministic_ae: bool = False
    patch_size: int = 16
    
    def setup(self):
        # Set up the text encoder if a config is provided.
        if self.text is not None:
            text_module = importlib.import_module(f"big_vision.models.{self.text_model}")
            self.text_encoder = text_module.Model(
                **{"num_classes": self.out_dims[1], **(self.text or {})},
                name="txt"
            )
        else:
            self.text_encoder = None

        # Set up the image encoder and the autoencoder components if the config is provided.
        if self.image is not None:
            image_module = importlib.import_module(f"big_vision.models.{self.image_model}")
            self.image_encoder = image_module.Model(
                **{"num_classes": self.out_dims[0], **(self.image or {})},
                name="img"
            )
            # Initialize the decoder submodule.
            decoder_image = importlib.import_module(f"big_vision.models.{self.image_model}")
            self.decoder = decoder_image.Model(
                **{"num_classes": self.out_dims[0], **(self.decoder_image or {})},
                name="decoder_image"
            )
            # Bottleneck down: produce either [channel_dim] (with normalization) or 2 channels for (mu, logvar)
            if self.layernorm_code:
                self.bottleneck_down = nn.Dense(self.channel_dim, name="Dense_0")
            else:
                self.bottleneck_down = nn.Dense(self.channel_dim * 2, name="Dense_0")
            # Bottleneck up: project latent code to decoder width.
            self.bottleneck_up = nn.Dense(self.decoder_image['width'], name="Dense_1")
            # Final projection to RGB values
            self.final_proj = nn.Dense(self.patch_size * self.patch_size * 3, name="final_proj") #TODO: Worth checking if final_conv 
        else:
            self.image_encoder = None

        # Initialize parameters for temperature and bias if provided.
        if self.temperature_init is not None:
            self.t = self.param(
                "t",
                lambda key, shape, dtype: jnp.log(self.temperature_init) * jnp.ones(shape, dtype),
                (1,), jnp.float32)
        else:
            self.t = None

        if self.bias_init is not None:
            self.b = self.param(
                "b",
                lambda key, shape, dtype: self.bias_init * jnp.ones(shape, dtype),
                (1,), jnp.float32)
        else:
            self.b = None

        # Instead of overwriting the layernorm_code boolean flag, store the LayerNorm module separately.
        if self.layernorm_code:
            self.layernorm_module = nn.LayerNorm(name='LayerNorm_0')

    def encode(self, image=None, text=None, resample=False, train=False):
        """Encodes the image input into latent variables for VAE reconstruction."""
        out = {}
        # Process text input if available.
        if text is not None:
            ztxt, out_txt = self.text_encoder(text)
            for k, v in out_txt.items():
                out[f"txt/{k}"] = v
            # Freeze gradients for text as in the original code.
            ztxt = jax.lax.stop_gradient(ztxt)
            out_txt = jax.tree_map(jax.lax.stop_gradient, out_txt)
            ztxt = ztxt / (jnp.linalg.norm(ztxt, axis=1, keepdims=True) + 1e-8)
        else:
            ztxt = None

        if image is not None:
            zimg, out_img = self.image_encoder(image)
                
            if self.freeze_encoder:
                zimg = jax.lax.stop_gradient(zimg)
                out_img = jax.tree_map(jax.lax.stop_gradient, out_img)
            zimg = zimg / (jnp.linalg.norm(zimg, axis=1, keepdims=True) + 1e-8)

            x_2d = out_img["encoded"]
            encoded = self.bottleneck_down(x_2d)
            if self.layernorm_code:
                # Apply the LayerNorm module stored in self.layernorm_module.
                mu = self.layernorm_module(encoded)
                logvar = jnp.zeros_like(mu)
            else:
                mu, logvar = jnp.split(encoded, 2, axis=-1)
            if self.deterministic_ae:
                z = mu
            else:
                if not train and resample:
                    z = reparametrize(mu, logvar, self.make_rng("vae"))
                else:
                    z = mu
        else:
            zimg = None
            mu = None
            logvar = None
            z = None
        
        return z, {"zimg": zimg, "ztxt": ztxt, "t": self.t, "b": self.b, "mu": mu, "logvar": logvar}

    def decode(self, latent, orig_image_shape=None, train=False):
        """Decodes a latent vector back to an image.
        
        For NaFlex, we directly use the patches, ptype, yabs, xabs structure.
        
        Args:
            latent: The latent vector to decode.
            orig_image_shape: For NaFlex, this should be a tuple of (patches, ptype, yabs, xabs).
            train: Whether we're in training mode.
            
        Returns:
            Decoded image patches in the same format as the input.
        """
        if orig_image_shape is None:
            raise ValueError("orig_image_shape must be provided for NaFlex decoding")
            

        ptype, yabs, xabs = orig_image_shape
        patches = latent
        image_input = (patches, ptype, yabs, xabs)  
        _, out = self.decoder(image_input, train=train)
        x = out["encoded"]
        decoded_patches = self.final_proj(x)
        decoded_patches = jnp.tanh(decoded_patches) #map to [-1, 1]
        return decoded_patches
    
    def __call__(self, image=None, text=None, train=False):
        z, out = self.encode(image, text, train)
        if image is not None:
            decoded_patches = self.decode(z, orig_image_shape=image[1:], train=train)
            return (decoded_patches, image[1], image[2], image[3]), out
        else:
            return None, out

def load(init_params, init_files, model_cfg, img_load_kw={}, txt_load_kw={}):  # pylint: disable=dangerous-default-value
  """Loads both towers, `init_files` is now a dict with `img` and `txt` keys."""
  if isinstance(init_files, str):
    init_files = VANITY_NAMES.get(init_files)

  if isinstance(init_files, str):
    # A shortcut for a single file checkpoint of a two_towers model.
    init_files = {k: f"{init_files}:{k}" for k in ("img", "t", "b")}
  else:
    init_files = {**init_files}  # Shallow copy because we'll pop stuff off.

  if not init_params:  # Convenience to skip checks in colab.
    init_params = {"img": None, "txt": None}
  restored_params = {**init_params}

  img_init = init_files.pop("image", init_files.pop("img", None))
  if img_init:
    restored_params["img"] = importlib.import_module(
        f"big_vision.models.{model_cfg.get('image_model', 'vit')}"
    ).load(init_params["img"], img_init, model_cfg.image, **img_load_kw)

  txt_init = init_files.pop("text", init_files.pop("txt", None))
  if txt_init:
    restored_params["txt"] = importlib.import_module(
        f"big_vision.models.{model_cfg.get('text_model', 'proj.image_text.text_transformer')}"  # pylint: disable=line-too-long
    ).load(init_params["txt"], txt_init, model_cfg.text, **txt_load_kw)

  t_init = init_files.pop("temperature", init_files.pop("t", None))
  if t_init:
    restored_params["t"] = utils.load_params(t_init)

  b_init = init_files.pop("bias", init_files.pop("b", None))
  if b_init:
    restored_params["b"] = utils.load_params(b_init)

  assert not init_files, (
      f"There's something unused left in `config.model_init`. You probably got "
      f"a typo. Here it is: {init_files}")

  return restored_params

def simple_load(init_params, init_files, dont_load=None, load_ema=False):
    """Loads both towers, `init_files` is now a dict with `img` and `txt` keys."""
    params = flax.core.unfreeze(utils.load_params(init_files, load_ema=load_ema))
    if init_params is not None:
        # Use empty list as default if dont_load is None
        if dont_load is None:
            dont_load = []
        # If dont_load is a dict (config), extract the dont_load field or use empty list
        elif isinstance(dont_load, dict):
            dont_load = dont_load.get("dont_load", [])
        params = common.merge_params(params, init_params, dont_load)
    return params

VANITY_NAMES = {
    # pylint: disable=line-too-long
    # SigLIP image encoder checkpoints from https://arxiv.org/abs/2303.15343
    "SigLIP B/16 224": "gs://big_vision/siglip/webli_en_b16_224_63724782.npz",
    "SigLIP B/16 256": "gs://big_vision/siglip/webli_en_b16_256_60500360.npz",
    "SigLIP B/16 384": "gs://big_vision/siglip/webli_en_b16_384_68578854.npz",
    "SigLIP B/16 512": "gs://big_vision/siglip/webli_en_b16_512_68580893.npz",
    "SigLIP L/16 256": "gs://big_vision/siglip/webli_en_l16_256_60552751.npz",
    "SigLIP L/16 384": "gs://big_vision/siglip/webli_en_l16_384_63634585.npz",
    "SigLIP So400m/14 224": "gs://big_vision/siglip/webli_en_so400m_224_57633886.npz",
    "SigLIP So400m/14 384": "gs://big_vision/siglip/webli_en_so400m_384_58765454.npz",
    "SigLIP B/16-i18n 256": "gs://big_vision/siglip/webli_i18n_b16_256_66117334.npz",

    # SigLIP 2 image and text encoder checkpoints from https://arxiv.org/abs/2502.14786
    "SigLIP2 B/16 224": "gs://big_vision/siglip2/siglip2_b16_224.npz",
    "SigLIP2 B/16 256": "gs://big_vision/siglip2/siglip2_b16_256.npz",
    "SigLIP2 B/16 384": "gs://big_vision/siglip2/siglip2_b16_384.npz",
    "SigLIP2 B/16 512": "gs://big_vision/siglip2/siglip2_b16_512.npz",
    "SigLIP2 B/32 256": "gs://big_vision/siglip2/siglip2_b32_256.npz",
    "SigLIP2 L/16 256": "gs://big_vision/siglip2/siglip2_l16_256.npz",
    "SigLIP2 L/16 384": "gs://big_vision/siglip2/siglip2_l16_384.npz",
    "SigLIP2 L/16 512": "gs://big_vision/siglip2/siglip2_l16_512.npz",
    "SigLIP2 So400m/14 224": "gs://big_vision/siglip2/siglip2_so400m14_224.npz",
    "SigLIP2 So400m/14 384": "gs://big_vision/siglip2/siglip2_so400m14_384.npz",
    "SigLIP2 So400m/16 256": "gs://big_vision/siglip2/siglip2_so400m16_256.npz",
    "SigLIP2 So400m/16 384": "gs://big_vision/siglip2/siglip2_so400m16_384.npz",
    "SigLIP2 So400m/16 512": "gs://big_vision/siglip2/siglip2_so400m16_512.npz",
    "SigLIP2 g-opt/16 256": "gs://big_vision/siglip2/siglip2_g-opt16_256.npz",
    "SigLIP2 g-opt/16 384": "gs://big_vision/siglip2/siglip2_g-opt16_384.npz",
    # SigLIP 2 NaFlex image and text encoder checkpoints.
    # These need `image_model="proj.image_text.naflex_vit"` for the image encoder
    # and a non-standard preprocessing, see configs/proj/image_text/README_siglip2.md.
    "SigLIP2 B/16 NaFlex": "gs://big_vision/siglip2/siglip2_b16_naflex.npz",
    "SigLIP2 So400m/16 NaFlex": "gs://big_vision/siglip2/siglip2_so400m16_naflex.npz",
    # pylint: enable=line-too-long
}