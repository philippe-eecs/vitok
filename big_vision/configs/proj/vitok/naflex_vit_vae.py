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

#Batch size * grad_accum = gives total batch size

def get_config(arg=None):
  """The base configuration."""
  arg = bvcc.parse_arg(
      arg, variant='S_B/20x64', max_tokens=576, batch_size=64, grad_accum=4,
      lpips=2.5, discriminator=0.05, beta=1e-3, encoder_lr_scale=0.25, fsdp=True, num_tiles=2, peak_lr=1e-4, warmup_steps=0.05,
      steps=100000, init_file="", distributed=True, finetune=False, runlocal=False)

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
              tfrecord_paths=["gs://vidtok-data/data/laion400m/tfrecord_dedup",],  #"gs://vidtok-data/laion2b-dedup-tfrecord"],
              mem_buffer_size=1024 * 1024 * 16,
              preshuffle=True,
              filter_highest_part=True,
          ),
      )
      #bias larger res so training is more stable and efficient on token density
      config.input.data[f"laion_{suffix}"] = 1.0 * seq_len / arg.max_tokens

  #given arg.max_length, place all sequences from 256 doubling until reaching arg.max_length
  #min tokens is closest to 256p
  min_token = int(256 / patch_size) ** 2
  print(f"Min token: {min_token}")
  sequence_lengths = [min_token]
  while sequence_lengths[-1] * 2 <= arg.max_tokens:
    sequence_lengths.append(sequence_lengths[-1] * 2)
  print(f"Sequence lengths: {sequence_lengths}")
  add_sequence_length_datasets(sequence_lengths)
  
  config.input.batch_size = arg.batch_size if not arg.runlocal else 32
  config.input.prefetch = 8 #small batch size so prefetch aggressively
  config.total_steps = arg.steps * arg.grad_accum
  config.init_types = ['float32', 'int32']
  config.log_training_steps = 100
  config.ckpt_steps = 5000
  config.keep_ckpt_steps = None
  config.model_name = 'proj.vitok.naflex_vit_vae'
  config.model_load = {}

  config.model = ConfigDict()
  config.model.patch_size = patch_size
  config.model.channel_dim = channel_dim
  config.model.image_model = 'proj.image_text.naflex_vit'
  config.model.out_dims = (None, 768)
  config.model.freeze_encoder = (arg.encoder_lr_scale == 0.0)  # Set based on LR scale
  config.model.image = ConfigDict({
      'depth': variant['enc_depth'],
      'width': variant['enc_width'], 
      'pool_type': 'tok',
      'posemb': f'learn_2d({posemb_grid_size})',
      'nposemb': 16,
      'num_heads': 12 if variant['enc_width'] == 768 else 16,
      'scan': True,
  })

  # Decoder config - only include parameters that Encoder expects
  config.model.decoder_image = ConfigDict({
      'depth': variant['dec_depth'],
      'width': variant['dec_width'], 
      'pool_type': 'tok',
      'scan': True,
      'posemb': f'learn_2d({posemb_grid_size})',
      'nposemb': 16,
      'num_heads': 12 if variant['dec_width'] == 768 else 16,
  })

  
  config.optax_name = 'scale_by_adam'
  config.optax = dict(b2=0.95)
  config.grad_clip_norm = 1.0
  if arg.fsdp:
    config.sharding_strategy = [('.*', 'fsdp(axis="data")')]
    config.sharding_rules = [('act_batch', ('data',))]
  
  config.model_init_ckpt = arg.init_file

  #KL beta term
  config.beta = arg.beta

  # Perceputal Loss Settings
  config.lpips = arg.lpips
  config.crop_size = 256 #always use 256 for memory savings
  config.image_dim_lpips = (1, 256, 256, 3)
  config.num_tiles = arg.num_tiles
  config.discriminator = arg.discriminator
  config.start_gen_loss = 5000
  config.gen_loss_warmup_steps = 5000
  
  # If using StyleGAN discriminator, add its config
  if config.discriminator:
    config.discriminator_model = ConfigDict()
    config.discriminator_model.input_size = 256
    config.discriminator_model.channel_multiplier = 1
    config.discriminator_model.blur_resample = True
  
  # Discriminator optimizer config
  #if finetune, use constant LR

  if config.discriminator:
    config.discriminator_optax_name = 'scale_by_adam'
    config.discriminator_optax = dict(b2=0.95)
    config.discriminator_lr = (arg.peak_lr / 5) #Lower discriminator LR helps stability a lot
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


  ''' #Slow at high resolutions and requires loading lots of images, LPIPS loss gives decent estimate of quality
  pp_img_val = (
    f'decode'
    f'|value_range(-1, 1)'
    f'|resize_to_sequence({patch_size}, {n_patches}, outkey="image")'
    f'|patchify({patch_size}, key="image")'
    f'|flatten(["image"])'
    f'|pad_to_shape(key="image/patches", shape=({n_patches}, None))'
    f'|pad_to_shape(key="image/type", shape=({n_patches},))'
    f'|pad_to_shape(key="image/yidx", shape=({n_patches},))'
    f'|pad_to_shape(key="image/xidx", shape=({n_patches},))'
    f'|tuplify(["image/patches", "image/type", "image/yidx", "image/xidx"], "image")'
    f'|keep("image", "id", "labels")'
    )

  config.grad_clip_norm = 0.5
  config.seed = 0
  config.evals = {}
  
  config.evals.reconstruction = dict(
        type='proj.vitok.reconstruction', #TODO: Setup aspect ratio for NaFlex
        data=dict(name='imagenet2012', split='validation'),
        pp_fn=pp_img_val,
        pred='recon',
        compute_siglip=True,
        log_steps=10000,  # Very fast O(seconds) so it's fine to run it often.
    )
  '''

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