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
import ml_collections as mlc
import numpy as np
from big_vision.configs.proj.image_text import common
from ml_collections import ConfigDict

def get_config(arg=None):
  """The base configuration."""
  arg = bvcc.parse_arg(
      arg, max_tokens=256, max_ar=2.0, patch_size=16, channel_dim=64, 
      batch_size=128, enc_depth=6, dec_depth=12, enc_width=768, dec_width=768, lpips=3.0, encoder_lr_scale=0.5, 
      siglip=False, siglip_feedback=0.0, siglip_distill=0.0, siglip_feature_weight=0.0, fsdp=False, num_tiles=1,
      discriminator=True, discriminator_type='style', discriminator_weight=0.05, steps=1000000, crop_size=256,
      start_gen_loss=10000, gen_loss_warmup_steps=20000, init_file="", distributed=False,
      finetune=False, runlocal=False, grad_accum=2)

  config = ConfigDict()
  config.lpips = arg.lpips
  config.siglip_feature_weight = arg.siglip_feature_weight
  config.num_tiles = arg.num_tiles
  config.input = {}
  config.pp_modules = ['ops_general', 'ops_image', 'ops_text', 'proj.image_text.ops_naflex', 'proj.paligemma.ops']
  patch_size = arg.patch_size
  n_patches = arg.max_tokens
  config.ema_decay = 0.9999
  config.input.data = {}
  grid_size = int(np.sqrt(n_patches))
  max_grid_size = int(np.sqrt(n_patches) * arg.max_ar) #Max grid size to be decoded in either x or y direction, need to fix do to static shape nature of Jax
  config.max_grid_size = max_grid_size
  config.image_dim_lpips = (1, arg.crop_size, arg.crop_size, 3)
  config.crop_size = arg.crop_size
  config.grad_accum_steps = arg.grad_accum
  config.finetune = arg.finetune
  config.distributed = arg.distributed

  # Define a tokenizer that directly processes 'texts' to 'labels'
  # SigLIP2 Tokenizer with NaFlex
  tokenizer = lambda inkey, outkey: f'lower(key="{inkey}")|tok(length={64}, model="gemma", bos="no", eos="sticky", inkey="{inkey}", outkey="{outkey}")'
  
  def get_pp_train(seq_length, arg):
    resize_to_sequence = f'|resize_to_sequence({patch_size}, {seq_length}, outkey="image", min_aspect_ratio={1/arg.max_ar}, max_aspect_ratio={arg.max_ar})'

    return (
      f'decode'
      f'|value_range(-1, 1)'
      f'{resize_to_sequence}'
      f'|{tokenizer("labels", "labels")}'
      f'|patchify({patch_size}, key="image")'
      f'|flatten(["image"])'
      f'|pad_to_shape(key="image/patches", shape=({n_patches}, None))'
      f'|pad_to_shape(key="image/type", shape=({n_patches},))'
      f'|pad_to_shape(key="image/yidx", shape=({n_patches},))'
      f'|pad_to_shape(key="image/xidx", shape=({n_patches},))'
      f'|tuplify(["image/patches", "image/type", "image/yidx", "image/xidx"], "image")'
      f'|keep("image", "labels")'
    )
  
  def add_sequence_length_datasets(sequence_lengths):
    for i, seq_len in enumerate(sequence_lengths):
      suffix = "" if i == 0 and seq_len == n_patches else f"_{seq_len}"  
      config.input[f"laion_400m{suffix}"] = dict(
          pp=get_pp_train(seq_len, arg),
          shuffle_buffer=500_000 if not arg.runlocal else 1000,
          data=dict(
              name='webdataset:laion', 
              tfrecord_paths=["gs://vidtok-data/data/laion400m/tfrecord_dedup", "gs://vidtok-data/laion2b-dedup-tfrecord"],
              mem_buffer_size=1024 * 1024 * 4,
              preshuffle=True,
              filter_highest_part=True,
          ),
      )
      config.input.data[f"laion_400m{suffix}"] = 1.0 #actually about 100m

  #given arg.max_length, place all sequences from 256 doubling until reaching arg.max_length
  sequence_lengths = [256]
  while sequence_lengths[-1] < arg.max_tokens:
    sequence_lengths.append(sequence_lengths[-1] * 2)
  add_sequence_length_datasets(sequence_lengths)

  config.input.batch_size = arg.batch_size if not arg.runlocal else 32
  config.input.prefetch = 4
  config.total_steps = arg.steps
  config.init_shapes = [(1, grid_size * patch_size, grid_size * patch_size, 3), (1, grid_size * patch_size,)]
  config.init_types = ['float32', 'int32']
  config.init_input = config.input
  config.log_training_steps = 100
  config.ckpt_steps = 1000
  config.keep_ckpt_steps = None
  config.model_name = 'proj.vitok.naflex_vit_vae'
  config.model_load = {}

  config.model = ConfigDict()
  config.model.channel_dim = arg.channel_dim
  config.model.image_model = 'proj.image_text.naflex_vit'
  config.model.out_dims = (None, 768)
  config.model.temperature_init = 10.0
  config.model.bias_init = -2.71
  config.model.freeze_encoder = (arg.encoder_lr_scale == 0.0)  # Set based on LR scale
  config.model.image = ConfigDict({
      'depth': arg.enc_depth,
      'width': arg.enc_width, 
      'pool_type': 'tok',
      'nposemb': 16,
      'num_heads': 12 if arg.enc_width == 768 else 16,
      'scan': True,
  })

  # Decoder config - only include parameters that Encoder expects
  config.model.decoder_image = ConfigDict({
      'depth': arg.dec_depth,
      'width': arg.dec_width, 
      'pool_type': 'tok',
      'scan': True,
      'nposemb': 16,
      'num_heads': 12 if arg.dec_width == 768 else 16,
  })

  config.siglip_feedback = arg.siglip_feedback
  config.siglip_distill = arg.siglip_distill
  config.siglip = arg.siglip
  if arg.siglip:
    config.siglip_model_name = 'proj.image_text.two_towers'
    config.siglip_model_init = 'SigLIP2 B/16 NaFlex'

    config.siglip_model = ConfigDict()
    config.siglip_model.image_model = 'proj.image_text.naflex_vit'
    config.siglip_model.out_dim = (None, 768)
    config.siglip_model.temperature_init = 10.0
    config.siglip_model.bias_init = -2.71
    config.siglip_model.image = ConfigDict({
        'variant': 'B',
        'pool_type': 'map',
        'posemb': 'learn_2d',
        'nposemb': 16, #Square root of num patches
        'scan': True,
        #'num_classes': None,  # Explicitly set num_classes to None to avoid head parameters
    })

    config.siglip_model.text_model = 'proj.image_text.text_transformer'
    config.siglip_model.text = ConfigDict({
        'variant': 'B',
        'scan': True,
        'vocab_size': 256_000,
        #'num_classes': None,
    })

  config.beta = 5e-3 #Should sweep over
  config.contrastive_weight = 0.0 #Need to sweep
  config.optax_name = 'scale_by_adam'
  config.optax = dict(b2=0.95)
  config.grad_clip_norm = 1.0
  if arg.fsdp:
    config.sharding_strategy = [('.*', 'fsdp(axis="data")')]
    config.sharding_rules = [('act_batch', ('data',))]
  
  config.model_init_ckpt = arg.init_file

  # Discriminator settings
  config.discriminator = arg.discriminator
  config.discriminator_type = arg.discriminator_type
  config.discriminator_weight = arg.discriminator_weight
  config.start_gen_loss = arg.start_gen_loss
  config.gen_loss_warmup_steps = arg.gen_loss_warmup_steps
  
  # If using StyleGAN discriminator, add its config
  if config.discriminator and config.discriminator_type == 'style':
    config.discriminator_model = ConfigDict()
    config.discriminator_model.input_size = config.max_grid_size * patch_size
    config.discriminator_model.channel_multiplier = 1
    config.discriminator_model.blur_resample = True
  
  # Discriminator optimizer config
  #if finetune, use constant LR

  if config.discriminator:
    config.discriminator_optax_name = 'scale_by_adam'
    config.discriminator_optax = dict(b2=0.95)
    config.discriminator_lr = 3e-5
    config.discriminator_wd = 1e-4
    if not arg.finetune:
      config.discriminator_schedule = [
        (".*", dict(decay_type='cosine', warmup_steps=0.05)),
      ]
    else:
      config.discriminator_schedule = [
        (".*", dict(decay_type='cosine', warmup_steps=0.01)), # Use empty dict for constant schedule
      ]

  config.lr = 3e-4
  config.wd = 1e-4

  if arg.encoder_lr_scale > 0:
    if not arg.finetune:
      config.schedule = [
        (".*", dict(decay_type='cosine', warmup_steps=0.05)),
      ]
    else:
      config.schedule = [
        (".*", dict(decay_type='cosine', warmup_steps=0.01)), # Use empty dict for constant schedule
      ]
    config.lr_mults = [
      ("img/.*", arg.encoder_lr_scale * 0.5 if arg.finetune else arg.encoder_lr_scale),
      ("Dense_0", arg.encoder_lr_scale * 0.5 if arg.finetune else arg.encoder_lr_scale),
      (".*", 1.0),
    ]
  else:
    config.lr = 5e-5 #start off lower
    config.schedule = [
      ("img/.*", None),
      ("Dense_0", None),
      ("Dense_1", None), # Freeze bottleneck up layer too
      (".*", dict(decay_type='cosine', warmup_steps=0.05)), #warmup quickly
    ]
    config.lr_mults = [
      (".*", 1.0),
    ]

  pp_img_val = (
    f'decode'
    f'|value_range(-1, 1)'
    f'|resize_to_sequence({patch_size}, {n_patches}, outkey="image", min_aspect_ratio={1/arg.max_ar}, max_aspect_ratio={arg.max_ar})'
    f'|patchify({patch_size}, key="image")'
    f'|flatten(["image"])'
    f'|pad_to_shape(key="image/patches", shape=({n_patches}, None))'
    f'|pad_to_shape(key="image/type", shape=({n_patches},))'
    f'|pad_to_shape(key="image/yidx", shape=({n_patches},))'
    f'|pad_to_shape(key="image/xidx", shape=({n_patches},))'
    f'|tuplify(["image/patches", "image/type", "image/yidx", "image/xidx"], "image")'
    f'|keep("image", "id", "labels")'
    )

  config.grad_clip_norm = 1.0
  config.seed = 0
  config.evals = {}
  '''
  config.evals.reconstruction = dict(
        type='proj.vitok.reconstruction', #TODO: Setup aspect ratio for NaFlex
        data=dict(name='imagenet2012', split='validation'),
        pp_fn=pp_img_val,
        pred='recon',
        compute_siglip=True,
        log_steps=10000,  # Very fast O(seconds) so it's fine to run it often.
    )
  '''

  pp_img_val = (
    f'decode'
    f'|value_range(-1, 1)'
    f'|resize_to_sequence({patch_size}, {n_patches}, outkey="image", min_aspect_ratio={1/arg.max_ar}, max_aspect_ratio={arg.max_ar})'
    f'|patchify({patch_size}, key="image")'
    f'|flatten(["image"])'
    f'|pad_to_shape(key="image/patches", shape=({n_patches}, None))'
    f'|pad_to_shape(key="image/type", shape=({n_patches},))'
    f'|pad_to_shape(key="image/yidx", shape=({n_patches},))'
    f'|pad_to_shape(key="image/xidx", shape=({n_patches},))'
    f'|tuplify(["image/patches", "image/type", "image/yidx", "image/xidx"], "image")'
    f'|keep("image", "id", "labels")'
    )
    #Remove decode from pp_img_val
  #pp_img_val = pp_img_val.replace('decode|', '')
  #if arg.siglip:
    #config.evals.retrieval_coco_supervised = common.get_coco(
    #    pp_img=pp_img_val,
    #    pp_txt=f'{tokenizer("texts", "labels")}',
    #    log_steps=1000,
    #    pred="supervised_predict",
    #)

  if arg.runlocal:
    config.input.batch_size = 32
    config.log_training_steps = 5
    config.model_init = ""

    if arg.siglip:
      config.siglip_model_init = ""
  return config