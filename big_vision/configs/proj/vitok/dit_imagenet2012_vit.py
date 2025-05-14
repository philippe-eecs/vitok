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
r"""Train Generative Infinite Vocabulary Transformer (GIVT) on ImageNet.

Example launch command (local; see main README for launching on TPU servers):

  python -m big_vision.trainers.proj.givt.generative \
    --config big_vision/configs/proj/givt/givt_imagenet2012.py \
    --workdir gs://$GS_BUCKET_NAME/big_vision/`date '+%m-%d_%H%M'`

Add the suffix `:key1=value1,key2=value2,...` to the config path in the launch 
command to modify the the config with the arguments below. For example:
`--config big_vision/configs/proj/givt/givt_imagenet_2012.py:model_size=large`
"""

import big_vision.configs.common as bvcc
import ml_collections


RES = 256
PATCH_SIZE = 16

def get_config(arg=None):
  """A config for training a simple VAE on imagenet2012."""
  arg = bvcc.parse_arg(arg, res=RES, patch_size=PATCH_SIZE, runlocal=False, singlehost=False)
  config = ml_collections.ConfigDict()

  config.input = {}
  ### Using Imagenette here to ensure this config is runnable without manual
  ### download of ImageNet. This is only meant for testing and will overfit
  ### immediately. Please download ImageNet to reproduce the paper results.
  # config.input.data = dict(name='imagenet2012', split='train[4096:]')
  config.input.data = dict(name='imagenet2012', split='train[4096:]')

  config.input.batch_size = 1024 if not arg.runlocal else 8
  config.input.shuffle_buffer_size = 25_000 if not arg.runlocal else 10

  config.total_epochs = 500

  config.input.pp = (
      f'decode_jpeg_and_inception_crop({arg.res},'
      f'area_min=80, area_max=100, ratio_min=1.0, ratio_max=1.0,'
      f'method="bicubic", antialias=True)'
      f'|flip_lr'
      f'|value_range(-1, 1, key="image")'
      f'|copy("label", "labels")'
      f'|keep("image", "labels")')

  pp_eval = (
      f'decode'
      f'|resize_small({arg.res}, inkey="image", outkey="image",'
      f'method="bicubic", antialias=True)'
      f'|central_crop({arg.res})'
      f'|value_range(-1, 1, key="image")'
      f'|copy("label", "labels")'
      f'|keep("image", "labels")')

  config.log_training_steps = 50
  config.ckpt_steps = 1000
  config.keep_ckpt_steps = None

  # Used for eval sweep.
  config.eval_only = False

  # VAE section
  config.vae = {}
  config.vae.model = ml_collections.ConfigDict()
  config.vae.model.code_len = 256
  config.vae.model_name = 'proj.givt.vit'
  config.vae.model.input_size = (arg.res, arg.res)
  config.vae.model.patch_size = (arg.patch_size, arg.patch_size)
  config.vae.model.codeword_dim = 16
  config.vae.model.width = 768
  config.vae.model.enc_depth = 6
  config.vae.model.dec_depth = 12
  config.vae.model.mlp_dim = 3072
  config.vae.model.num_heads = 12
  config.vae.model.code_dropout = 'none'
  config.vae.model.bottleneck_resize = False
  config.vae.model.scan = True
  config.vae.model.remat_policy = 'nothing_saveable'
  config.vae.model_init = 'gs://us-central2-b/big_vision/givt/vitok_SB_1e-4_beta_bs_512/checkpoint.bv'

  # GIVT section
  config.model_name = 'dit'
  config.model = ml_collections.ConfigDict()
  config.model.seq_len = 256
  config.model.out_dim = 16
  config.model.num_classes = 1000
  config.model.code_len = 256
  config.model.width = 1024
  config.model.depth = 24
  config.model.mlp_dim = 4096
  config.model.num_heads = 16
  config.model.scan = True
  config.model.remat_policy = 'nothing_saveable'
  config.model.dtype_mm = 'float32'
  config.model.adaln = True
  config.model.cfg_dropout_rate = 0.1
  config.model.num_cls = 1
  #config.grad_clip_norm = 1.0

  # FSDP training by default
  #config.sharding_strategy = [('.*', 'fsdp(axis="data")')]
  #config.sharding_rules = [('act_batch', ('data',))]

  # Standard schedule
  config.lr = 0.0001
  config.wd = 0.0001
  config.schedule = dict(decay_type='cosine', warmup_percent=0.1)

  ### Evaluation section
  config.evals = {}

  config.evals.save_pred_sampling = dict(
      type='proj.givt.save_predictions',
      pp_fn=pp_eval,
      log_steps=10_000,
      pred='sample',
      batch_size=1024,
      data=dict(name=config.input.data.name, split='validation[:1024]'),
      outfile='inference_sampled.npz',
  )

  config.seed = 0

  config.ckpt_timeout = 30

  if arg.runlocal:
    config.input.batch_size = 4
    config.input.shuffle_buffer_size = 10
    config.log_training_steps = 5
    config.model.num_decoder_layers = 2

    config.evals.val.data.split = 'validation[:16]'
    config.evals.val.log_steps = 20

  return config
