# Copyright 2023 Big Vision Authors.
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

"""Evaluator for the classfication task."""
# pylint: disable=consider-using-from-import

import functools

import big_vision.datasets.core as ds_core
import big_vision.input_pipeline as input_pipeline
import big_vision.pp.builder as pp_builder
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
import jax.lax as lax
import PIL
from .fid import InceptionV3, compute_frechet_distance, compute_inception_score
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from .image_quality import psnr, ssim
from .cmmd import compute_cmmd
API = 'jit'

# To avoid re-compiling the function for every new instance of the same
# evaluator on a different dataset!
@functools.cache
def get_eval_fn(predict_fn, inception_fn=None):
  """Produces eval function, also applies pmap."""
  @jax.jit
  def _patch_predict_fn(train_state, batch, rng):
    x, out = predict_fn(train_state, batch, rng)
    img = ((jnp.clip(out['image'], -1, 1) + 1) * 127.5).astype(jnp.uint8)
    ref_img = ((jnp.clip(out['ref_image'], -1, 1) + 1) * 127.5).astype(jnp.uint8)

    ssim_score = ssim(ref_img, img, L=255.0)  # Correctly compute SSIM.
    psnr_score = psnr(ref_img, img)
    outs = {'psnr': psnr_score, 'ssim': ssim_score,
            'img': img, 'ref_img': ref_img}
    
    if inception_fn:
      acts, softmax_outputs = inception_fn(out['recon_crops'])
      ref_acts, ref_softmax_outputs = inception_fn(out['true_crops'])
      outs['acts'] = acts
      outs['softmax_outputs'] = softmax_outputs
      outs['ref_acts'] = ref_acts
      outs['ref_softmax_outputs'] = ref_softmax_outputs

    if 'siglip_feature_loss' in out:
      outs['siglip_feature_loss'] = out['siglip_feature_loss']
      outs['siglip_true_zimg'] = out['siglip_true_zimg']
      outs['siglip_recon_zimg'] = out['siglip_recon_zimg']
    
    if 'lpips_loss' in out:
      outs['lpips_loss'] = out['lpips_loss']
    
    return outs
  return _patch_predict_fn

class Evaluator:
  """Classification evaluator."""

  def __init__(self, predict_fn, data, pp_fn, batch_size,
               cache_final=True, cache_raw=False, prefetch=1,
               resize=True, compute_fid=False, compute_siglip=False,
               label_key='labels', *, devices):
    data = ds_core.get(**data)
    pp_fn = pp_builder.get_preprocess_fn(pp_fn)
    self.ds, self.steps = input_pipeline.make_for_inference(
        data.get_tfdata(ordered=True), pp_fn, batch_size,
        num_ex_per_process=data.num_examples_per_process(),
        cache_final=cache_final, cache_raw=cache_raw)
    self.data_iter = input_pipeline.start_global(self.ds, devices, prefetch)
    self.num_examples = 64
    self.resize = resize

    if compute_fid:
      inception_model = InceptionV3() #TODO: SigLIP features for FID?
      with jax.transfer_guard("allow"):
        params_dict = inception_model.init(jax.random.PRNGKey(0), jnp.ones((1, 299, 299, 3)))
      def inception_forward(x):
        preds = jax.lax.stop_gradient(inception_model.apply(params_dict, x, train=False))
        return preds['acts'].squeeze(axis=1).squeeze(axis=1), jax.nn.softmax(preds['logits'], axis=1)
      self.eval_fn = get_eval_fn(predict_fn, inception_forward)
    else:
      self.eval_fn = get_eval_fn(predict_fn)
    
    self.label_key = label_key
    self.compute_fid = compute_fid
    self.compute_siglip = compute_siglip

    mesh = jax.sharding.Mesh(devices, ("devices",))
    self._all_gather_p = jax.jit(
        lambda x: x, out_shardings=NamedSharding(mesh, P()))
            
  def run(self, train_state):
    """Computes all metrics."""
    total_ssim, total_psnr, total_lpips, nseen = 0, 0, 0, 0
    if self.compute_fid:
      acts, softmax_outputs, ref_acts = [], [], []
    if self.compute_siglip:
      siglip_true_zimg, siglip_recon_zimg, siglip_feature_loss = [], [], 0
    rng = jax.random.PRNGKey(0)
    for _, batch in tqdm(zip(range(self.steps), self.data_iter)):
      #If multihost, we need to do allgather
      with jax.transfer_guard("allow"):
        outs = self._all_gather_p(self.eval_fn(train_state, batch, rng))
      rng, _ = jax.random.split(rng)
      outs = jax.device_get(outs)
      total_ssim += outs['ssim']
      total_psnr += outs['psnr']
      if 'lpips_loss' in outs:
        total_lpips += outs['lpips_loss']
      nseen += 1
      
      if self.compute_fid:
        acts.append(outs['acts'])
        softmax_outputs.append(outs['softmax_outputs'])
        ref_acts.append(outs['ref_acts'])
      if self.compute_siglip:
        if 'siglip_true_zimg' in outs:
          siglip_true_zimg.append(outs['siglip_true_zimg'])
          siglip_recon_zimg.append(outs['siglip_recon_zimg'])
          siglip_feature_loss += outs['siglip_feature_loss']
    if self.compute_fid:
      acts = np.concatenate(acts, axis=0)
      softmax_outputs = np.concatenate(softmax_outputs, axis=0)
      ref_acts = np.concatenate(ref_acts, axis=0)
      mu = np.mean(acts, axis=0)
      sigma = np.cov(acts, rowvar=False)
      ref_mu = np.mean(ref_acts, axis=0)
      ref_sigma = np.cov(ref_acts, rowvar=False)

      fid_score = compute_frechet_distance(ref_mu, mu, ref_sigma, sigma)
      inception_score = compute_inception_score(softmax_outputs)
    if 'siglip_true_zimg' in outs:
      siglip_true_zimg = np.concatenate(siglip_true_zimg, axis=0)
      siglip_recon_zimg = np.concatenate(siglip_recon_zimg, axis=0)
      cmmd_score = compute_cmmd(siglip_true_zimg, siglip_recon_zimg)
      siglip_feature_loss /= nseen
    avg_ssim = total_ssim / nseen
    avg_psnr = total_psnr / nseen
    avg_lpips = total_lpips / nseen
    img_array = np.array(outs['img'][:self.num_examples])
    ref_img_array = np.array(outs['ref_img'][:self.num_examples])

    grid_size = 8
    h, w = img_array.shape[1:3]
    grid = np.ones((h * grid_size, w * grid_size, 3), dtype=np.uint8) * 255
    ref_grid = np.ones((h * grid_size, w * grid_size, 3), dtype=np.uint8) * 255

    for idx, img in enumerate(img_array):
      i = idx // grid_size
      j = idx % grid_size
      grid[i*h:(i+1)*h, j*w:(j+1)*w] = img
      ref_grid[i*h:(i+1)*h, j*w:(j+1)*w] = ref_img_array[idx]
    
    grid = PIL.Image.fromarray(grid)
    ref_grid = PIL.Image.fromarray(ref_grid)

    yield ('visuals/img', grid)
    yield ('visuals/ref_img', ref_grid)
    if self.compute_fid:
      yield ('fid_score', fid_score)
      yield ('inception_score', inception_score)
    if self.compute_siglip:
      yield ('cmmd_score', cmmd_score)
      yield ('siglip_feature_loss', siglip_feature_loss)
    yield ('ssim', avg_ssim)
    yield ('psnr', avg_psnr)
    if 'lpips_loss' in outs:
      yield ('lpips', avg_lpips)