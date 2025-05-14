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

# pylint: disable=line-too-long
r"""Train VAE for ViTok-VAE with frozen SigLip-VAE image encoder and learned bottleneck.
"""

import big_vision.configs.common as bvcc
import numpy as np
from ml_collections import ConfigDict
import jax
#Batch size * grad_accum = gives total batch size

def get_config(arg=None):
  """The base configuration."""
  arg = bvcc.parse_arg(
      arg, variant='S_B/24x64', max_tokens=1600, batch_size=64, grad_accum=2, window_size=128, #Only use splash if you are GPU not TPU
      lpips=1.0, discriminator=0.02, beta=1e-3, encoder_lr_scale=0.25, fsdp=False, num_tiles=1, peak_lr=1e-4, warmup_steps=0.05,
      steps=100000, init_file="", distributed=True, finetune=False, runlocal=False)

  # --- New options for mesh, splash/local window, sequence parallelism ---
  # Example: 4096 tokens (64x64 grid), 8 GPUs (2 data, 4 seq)
  # You can override these via arg or by editing here
  #mesh = [('data', jax.device_count())] # Default to all devices on data axis
  use_splash = False # Requires GPUs
  use_seq_parallel = False #TODO: Broken right now
  local_ws = (arg.window_size, arg.window_size)

  config = ConfigDict()
  config.pp_modules = ['ops_general', 'ops_image', 'proj.image_text.ops_naflex']
  n_patches = arg.max_tokens
  config.ema_decay = 0.9995
  variant = decode_variant(arg.variant)
  patch_size = variant['patch_size']
  channel_dim = variant['channel_dim']
  config.patch_size = patch_size
  config.channel_dim = channel_dim
  #Limit image size
  posemb_grid_size = (2048 // patch_size) + 1 #max resolution possible for any side
  config.posemb_grid_size = posemb_grid_size
  config.grad_accum_steps = arg.grad_accum
  config.distributed = arg.distributed
  config.variant = arg.variant

  # --- Mesh and sharding rules ---
  #config.mesh = mesh
  config.sharding_rules = [
      ("batch", "data"),
      ("act_batch", "data"),
      ("data", "data"),
      ("seq", None),       # Replicated
      ("act_len", None),   # Replicated
      ("embed", None),
      ("act_emb", None),
      ("mlp", None),
      ("heads", None),
  ]

  config.input = {}
  config.input.data = {}
  def get_pp_train(seq_length, arg):
    resize_to_sequence = f'|resize_to_sequence({patch_size}, {seq_length}, max_image_size={posemb_grid_size * patch_size}, outkey="image")'
    return (
      f'decode'
      f'|value_range(-1, 1)'
      f'{resize_to_sequence}'
      f'|patchify({patch_size}, key="image")'
      f'|flatten(["image"])'
      f'|pad_to_shape(key="image/patches", shape=({n_patches}, None))'
      f'|pad_to_shape(key="image/type", shape=({n_patches},))'
      f'|pad_to_shape(key="image/yidx", shape=({n_patches},))'
      f'|pad_to_shape(key="image/xidx", shape=({n_patches},))'
      f'|tuplify(["image/patches", "image/type", "image/yidx", "image/xidx"], "image")'
      f'|keep("image")'
    )

  def add_sequence_length_datasets(sequence_lengths):
    for i, seq_len in enumerate(sequence_lengths):
      suffix = "" if i == 0 and seq_len == n_patches else f"_{seq_len}"  
      config.input[f"laion_{suffix}"] = dict(
          pp=get_pp_train(seq_len, arg),
          shuffle_buffer=500_000 if not arg.runlocal else 1000,
          data=dict(
              name='webdataset:laion', 
              tfrecord_paths=["gs://vidtok-data/data/laion400m/tfrecord_dedup", "gs://vidtok-data/laion2b-dedup-tfrecord"],
              mem_buffer_size=1024 * 1024 * 16,
              preshuffle=True,
              filter_highest_part=True,
          ),
      )
      config.input.data[f"laion_{suffix}"] = 1.0

  min_token = int(256 / patch_size) ** 2
  sequence_lengths = [min_token]
  while sequence_lengths[-1] * 2 <= arg.max_tokens:
    sequence_lengths.append(sequence_lengths[-1] * 2)
  add_sequence_length_datasets(sequence_lengths)

  config.input.batch_size = arg.batch_size if not arg.runlocal else 32
  config.input.prefetch = 8
  config.total_steps = arg.steps * arg.grad_accum
  config.init_types = ['float32', 'int32']
  config.log_training_steps = 100
  config.ckpt_steps = 5000
  config.keep_ckpt_steps = None
  config.model_name = 'proj.vitok.naflex_vit_vae'
  config.model_load = {}

  # --- Model config with new options ---
  config.model = ConfigDict()
  config.model.image_model = 'proj.vitok.gqa_naflex_vit'
  config.model.out_dims = (None, 768)
  config.model.channel_dim = channel_dim
  config.model.patch_size = patch_size
  config.model.image = ConfigDict()
  config.model.image.width = variant['enc_width']
  config.model.image.depth = variant['enc_depth']
  
  config.model.image.num_heads = 12 if variant['enc_width'] == 768 else 16
  config.model.image.kv_groups = 2  # Try 2 for more efficient attention (was 1)
  config.model.image.head_dim = config.model.image.width // config.model.image.num_heads
  # If you want to set head_dim, do so in the encoder/attention submodule configs, not here.
  config.model.image.local_ws = local_ws
  config.model.image.use_splash = use_splash
  config.model.image.use_seq_parallel = use_seq_parallel # Will be False
  config.model.image.posemb = f'learn_2d({posemb_grid_size})'
  config.model.image.nposemb = posemb_grid_size
  config.model.image.dtype_mm = "bfloat16"
  config.model.image.use_t5_mlp = True

  config.model.decoder_image = ConfigDict()
  config.model.decoder_image.width = variant['dec_width']
  config.model.decoder_image.depth = variant['dec_depth']
  config.model.decoder_image.num_heads = 12 if variant['dec_width'] == 768 else 16
  config.model.decoder_image.kv_groups = 2  # Try 2 for more efficient attention (was 1)
  config.model.decoder_image.head_dim = config.model.decoder_image.width // config.model.decoder_image.num_heads

  config.model.decoder_image.local_ws = local_ws
  config.model.decoder_image.use_splash = use_splash
  config.model.decoder_image.use_seq_parallel = use_seq_parallel # Will be False
  config.model.decoder_image.posemb = f'learn_2d({posemb_grid_size})'
  config.model.decoder_image.nposemb = posemb_grid_size
  config.model.decoder_image.dtype_mm = "bfloat16"
  config.model.decoder_image.use_t5_mlp = True

  #config.model.patch_size = patch_size

  # --- Rest of config as before ---
  config.optax_name = 'scale_by_adam'
  config.optax = dict(b2=0.95)
  config.grad_clip_norm = 1.0
  if arg.fsdp:
    config.sharding_strategy = [('.*', 'fsdp(axis="data")')]

  config.model_init_ckpt = arg.init_file
  config.beta = arg.beta
  config.lpips = arg.lpips
  config.crop_size = 256
  config.image_dim_lpips = (1, 256, 256, 3)
  config.num_tiles = arg.num_tiles
  config.discriminator = arg.discriminator
  config.start_gen_loss = 5000
  config.gen_loss_warmup_steps = 5000

  if config.discriminator:
    config.discriminator_optax_name = 'scale_by_adam'
    config.discriminator_optax = dict(b2=0.95)
    config.discriminator_lr = (arg.peak_lr / 5)
    config.discriminator_wd = 1e-4
    config.discriminator_schedule = [
      (".*", dict(decay_type='cosine', warmup_steps=arg.warmup_steps)),
    ]

  config.lr = arg.peak_lr
  config.wd = 1e-4
  config.schedule = [
    (".*", dict(decay_type='cosine', warmup_steps=arg.warmup_steps)),
  ]
  config.lr_mults = [
    ("img/.*", arg.encoder_lr_scale),
    ("Dense_0", arg.encoder_lr_scale),
    (".*", 1.0),
  ]

  if arg.runlocal:
    config.input.batch_size = 32
    config.log_training_steps = 5
    config.model_init = ""
  return config

def vit_variant(string):
  if string == 'S':
    return {'depth': 6, 'width': 768}
  elif string == 'B':
    return {'depth': 12, 'width': 768}
  elif string == 'L':
    return {'depth': 24, 'width': 1024}
  elif string == 'So400m':
    return {'depth': 27, 'width': 1152}
  elif string == 'H':
    return {'depth': 32, 'width': 1280}
  else:
    raise ValueError(f"Invalid variant: {string}")

def decode_variant(variant):
  #Split variant into encoder size, decoder size, patch_size, and channel_dim
  model_size, latent_size = variant.split('/')
  encoder_size, decoder_size = model_size.split('_')
  patch_size, channel_dim = latent_size.split('x')
  patch_size = int(patch_size)
  channel_dim = int(channel_dim)

  config = {}

  encoder_variant = vit_variant(encoder_size)
  encoder_variant = {'enc_' + k: v for k, v in encoder_variant.items()}
  decoder_variant = vit_variant(decoder_size)
  decoder_variant = {'dec_' + k: v for k, v in decoder_variant.items()}

  config.update(encoder_variant)
  config.update(decoder_variant)
  config['patch_size'] = patch_size
  config['channel_dim'] = channel_dim
  
  return config