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

"""Train loop for training a VAE or beta-VAE with a Gaussian encoder."""
# pylint: disable=consider-using-from-import
import functools
import importlib
import math
import multiprocessing.pool
import os

from absl import app
from absl import flags
from absl import logging
from big_vision import input_pipeline
import big_vision.evaluators.common as eval_common
import big_vision.optax as bv_optax
import big_vision.sharding as bv_sharding
#import big_vision.trainers.proj.givt.utils as trainer_utils
#from big_vision.trainers.proj.uvim import panoptic_task
import big_vision.utils as u
from clu import parameter_overview
import flax.linen as nn
import jax
from jax.experimental import mesh_utils
from jax.experimental import multihost_utils
from jax.experimental.array_serialization import serialization as array_serial
import jax.numpy as jnp
from ml_collections import config_flags
import numpy as np
import optax
import tensorflow as tf
from tensorflow.io import gfile
from big_vision.evaluators.proj.vitok.lpips import LPIPS, load as lpips_load
from big_vision.models.proj.vitok.naflex_vit_vae import patches_to_image
from ml_collections import ConfigDict
from big_vision.models.proj.vitok.stylegan_discriminator import Discriminator, hinge_d_loss
# pylint: disable=logging-fstring-interpolation

from tqdm import tqdm

partial = functools.partial

config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=True)

flags.DEFINE_string("workdir", default=None, help="Work unit directory.")
flags.DEFINE_boolean("cleanup", default=False,
                     help="Delete workdir (only) after successful completion.")

# Adds jax flags to the program.
jax.config.parse_flags_with_absl()
# Transfer guard will fail the program whenever that data between a host and
# a device is transferred implicitly. This often catches subtle bugs that
# cause slowdowns and memory fragmentation. Explicit transfers are done
# with jax.device_put and jax.device_get.
#jax.config.update("jax_transfer_guard", "disallow")
# Fixes design flaw in jax.random that may cause unnecessary d2d comms.
jax.config.update("jax_threefry_partitionable", True)


NamedSharding = jax.sharding.NamedSharding
P = jax.sharding.PartitionSpec


def main(argv):
  del argv
  config = flags.FLAGS.config
  workdir = flags.FLAGS.workdir

  if config.get("distributed", False):
    jax.distributed.initialize()

  # Make sure TF does not touch GPUs.
  tf.config.set_visible_devices([], "GPU")
  logging.info(
      f"\u001b[33mHello from process {jax.process_index()} holding "
      f"{jax.local_device_count()}/{jax.device_count()} devices and "
      f"writing to workdir {workdir}.\u001b[0m")

  save_ckpt_path = None
  if workdir:  # Always create if requested, even if we may not write into it.
    gfile.makedirs(workdir)
    save_ckpt_path = os.path.join(workdir, "checkpoint.bv")

  # The pool is used to perform misc operations such as logging in async way.
  pool = multiprocessing.pool.ThreadPool()

  # Here we register preprocessing ops from modules listed on `pp_modules`.
  for m in config.get("pp_modules",
                      ["ops_general", "ops_image", "proj.uvim.pp_ops",
                       "proj.givt.pp_ops"]):
    importlib.import_module(f"big_vision.pp.{m}")

  # Setup up logging and experiment manager.
  xid, wid = -1, -1
  fillin = lambda s: s
  def info(s, *a):
    logging.info("\u001b[33mNOTE\u001b[0m: " + s, *a)
  def write_note(note):
    if jax.process_index() == 0:
      info("%s", note)

  mw = u.BigVisionMetricWriter(xid, wid, workdir, config)

  # Allow for things like timings as early as possible!
  u.chrono.inform(measure=mw.measure, write_note=write_note)

################################################################################
#                                                                              #
#                                Set up Mesh                                   #
#                                                                              #
################################################################################

  # We rely on jax mesh_utils to organize devices, such that communication
  # speed is the fastest for the last dimension, second fastest for the
  # penultimate dimension, etc.
  config_mesh = config.get("mesh", [("data", jax.device_count())])

  # Sharding rules with default
  sharding_rules = config.get("sharding_rules", [("act_batch", "data")])

  mesh_axes, mesh_size = tuple(zip(*config_mesh))

  # Because jax.utils do not support `-1` shape size.
  mesh_size = np.array(jax.devices()).reshape(mesh_size).shape

  device_mesh = mesh_utils.create_device_mesh(mesh_size)

  # Consistent device order is important to ensure correctness of various train
  # loop components, such as input pipeline, update step, evaluators. The
  # order presribed by the `devices_flat` variable should be used throughout
  # the program.
  devices_flat = device_mesh.flatten()

################################################################################
#                                                                              #
#                                Input Pipeline                                #
#                                                                              #
################################################################################

  write_note("Initializing train dataset...")
  batch_size = config.input.batch_size
  if batch_size % jax.device_count() != 0:
    raise ValueError(f"Batch size ({batch_size}) must "
                     f"be divisible by device number ({jax.device_count()})")
  info("Global batch size %d on %d hosts results in %d local batch size. With "
       "%d dev per host (%d dev total), that's a %d per-device batch size.",
       batch_size, jax.process_count(), batch_size // jax.process_count(),
       jax.local_device_count(), jax.device_count(),
       batch_size // jax.device_count())

  train_ds, ntrain_img = input_pipeline.training(config.input)

  total_steps = u.steps("total", config, ntrain_img, batch_size)
  def get_steps(name, default=ValueError, cfg=config):
    return u.steps(name, cfg, ntrain_img, batch_size, total_steps, default)

  u.chrono.inform(total_steps=total_steps, global_bs=batch_size,
                  steps_per_epoch=ntrain_img / batch_size)

  info("Running for %d steps, that means %f epochs",
       total_steps, total_steps * batch_size / ntrain_img)

  # Start input pipeline as early as possible.
  n_prefetch = config.get("prefetch_to_device", 1)
  train_iter = input_pipeline.start_global(train_ds, devices_flat, n_prefetch)

################################################################################
#                                                                              #
#                           Create Model & Optimizer                           #
#                                                                              #
################################################################################

  write_note("Creating model...")
  model_mod = importlib.import_module(f"big_vision.models.{config.model_name}")
  model = model_mod.Model(**config.get("model", {}))

  def init(rng):
    batch = jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype.as_numpy_dtype), train_ds.element_spec)
    params = model.init(rng, batch["image"], text=None)["params"]
    return params

  # This seed makes the Jax part of things (like model init) deterministic.
  # However, full training still won't be deterministic, for example due to the
  # tf.data pipeline not being deterministic even if we would set TF seed.
  rng = jax.random.PRNGKey(u.put_cpu(config.get("seed", 0)))

  write_note("Inferring parameter shapes...")
  rng, rng_init = jax.random.split(rng)
  params_shape = jax.eval_shape(init, rng_init)

  write_note("Inferring optimizer state shapes...")
  # Create the base optimizer and schedules.
  inner_tx, sched_fns = bv_optax.make(config, nn.unbox(params_shape), sched_kw=dict(
      total_steps=total_steps // config.get("grad_accum_steps", 1), batch_size=batch_size, data_size=ntrain_img))

  # Get gradient accumulation steps.
  grad_accum_steps = config.get("grad_accum_steps", 1)
  if grad_accum_steps < 1:
    raise ValueError("grad_accum_steps must be >= 1")

  # Wrap the optimizer with MultiSteps if gradient accumulation is enabled.
  if grad_accum_steps > 1:
    tx = optax.MultiSteps(inner_tx, every_k_schedule=grad_accum_steps)
    write_note(f"Using gradient accumulation with {grad_accum_steps} steps.")
  else:
    tx = inner_tx  # Use the base optimizer directly if no accumulation.

  opt_shape = jax.eval_shape(tx.init, params_shape)
  # We jit this, such that the arrays are created on the CPU, not device[0].
  sched_fns_cpu = [u.jit_cpu()(sched_fn) for sched_fn in sched_fns]

  if jax.process_index() == 0:
    num_params = sum(np.prod(p.shape) for p in jax.tree.leaves(params_shape))
    mw.measure("num_params", num_params)

################################################################################
#                                                                              #
#                               Shard & Transfer                               #
#                                                                              #
################################################################################

  write_note("Creating device mesh...")
  mesh = jax.sharding.Mesh(device_mesh, mesh_axes)
  repl_sharding = jax.sharding.NamedSharding(mesh, P())

  write_note("Inferring shardings...")
  train_state_shape = {"params": params_shape, "opt": opt_shape, "ema_params": params_shape}

  strategy = config.get("sharding_strategy", [(".*", "replicate")])
  with nn.logical_axis_rules(sharding_rules):
    train_state_sharding = bv_sharding.infer_sharding(
        train_state_shape, strategy=strategy, mesh=mesh)

  write_note("Transferring train_state to devices...")
  # RNG is always replicated
  rng_init = u.reshard(rng_init, repl_sharding)

  # Parameters and the optimizer are now global (distributed) jax arrays.
  params = jax.jit(init, out_shardings=train_state_sharding["params"])(rng_init)
  opt = jax.jit(tx.init, out_shardings=train_state_sharding["opt"])(params)
  ema_params = jax.jit(lambda x: x, out_shardings=train_state_sharding["params"])(params)
  rng, rng_loop = jax.random.split(rng, 2)
  rng_loop = u.reshard(rng_loop, repl_sharding)
  del rng  # not used anymore, so delete it.

  # At this point we have everything we need to form a train state. It contains
  # all the parameters that are passed and updated by the main training step.
  # From here on, we have no need for Flax AxisMetadata (such as partitioning).
  train_state = nn.unbox({"params": params, "opt": opt, "ema_params": ema_params})
  del params, opt, ema_params  # Delete to avoid memory leak or accidental reuse.

  write_note("Logging parameter overview...")
  parameter_overview.log_parameter_overview(
      train_state["params"], msg="Init params",
      include_stats="global", jax_logging_process=0)

  # Computing ELBO or beta-VAE loss for Gaussian encoder.
  def extract_crops_vectorized(images, y_starts, x_starts, crop_size):
    """
    Extract patches from a batch of images using jax.lax.dynamic_slice.

    Args:
        images: Array of shape (batch_size, height, width, channels)
        y_starts: Array of shape (total_positions, 1), y-coordinates of patch top-left corners
        x_starts: Array of shape (total_positions, 1), x-coordinates of patch top-left corners
        crop_size: int, size of the square patches to extract

    Returns:
        patches: Array of shape (total_positions, crop_size, crop_size, channels)
        
    Note:
        This function handles the case where the number of crop positions (y_starts/x_starts)
        is a multiple of the batch size. In this case, it will repeat each image the appropriate
        number of times to match the positions.
    """
    # Check if dimensions need adjustment
    if y_starts.shape[0] % images.shape[0] == 0 and y_starts.shape[0] != images.shape[0]:
        # Calculate number of repeats needed
        repeats = y_starts.shape[0] // images.shape[0]
        # Repeat each image to match the number of crop positions
        images = jnp.repeat(images, repeats, axis=0)
    
    def extract_single_patch(image, y_start, x_start):
        # Extract one patch from one image using dynamic_slice
        return jax.lax.dynamic_slice(
            image,
            (y_start[0], x_start[0], 0),
            (crop_size, crop_size, image.shape[2])
        )

    # Vectorize over batch dimension using jax.vmap
    extract_patches = jax.vmap(extract_single_patch, in_axes=(0, 0, 0))
    return extract_patches(images, y_starts, x_starts)

  def extra_crops(image, rng_crop,
                  patch_size=16, num_tiles=1, crop_size=224):
    """
    Generate crop positions for evenly spaced tiles across the valid coordinate range.
    
    Args:
        image: Tuple of (patches, ptype, yabs, xabs)
        rng_crop: JAX PRNG key for random crop selection
        patch_size: Size of each patch in pixels
        num_tiles: Number of tiles along each dimension
        crop_size: Size of each crop in pixels
        
    Returns:
        (y_starts_flat, x_starts_flat): Flattened arrays of crop positions.
        The returned arrays have shape (batch_size * num_tiles^2, 1),
        where each image in the batch gets num_tiles^2 crop positions.
    """
    # Unpack the image tuples
    patches, ptype, yabs, xabs = image
    
    # Get batch size from patches
    batch_size = patches.shape[0]
    
    # Get maximum valid starting position for crops
    y_max = (jnp.max(yabs, axis=1, keepdims=True)) * patch_size - crop_size
    x_max = (jnp.max(xabs, axis=1, keepdims=True)) * patch_size - crop_size
    #make sure y_max and x_max are not negative
    y_max = jnp.maximum(y_max, 0)
    x_max = jnp.maximum(x_max, 0)

    # Split RNG key for independent x and y sampling
    rng_y, rng_x = jax.random.split(rng_crop)

    # Generate num_tiles independent random relative positions for y and x for each image
    # Shape: (batch_size, num_tiles)
    y_rel_positions = jax.random.uniform(rng_y, shape=(batch_size, num_tiles))
    x_rel_positions = jax.random.uniform(rng_x, shape=(batch_size, num_tiles))
    
    # Apply the relative positions to each batch item's max values, round down and cast to int
    y_starts = jnp.floor(y_max * y_rel_positions).astype(jnp.int32)  # Shape: (batch_size, num_tiles)
    x_starts = jnp.floor(x_max * x_rel_positions).astype(jnp.int32)  # Shape: (batch_size, num_tiles)
    
    # Flatten for extraction and convert to integers
    y_starts_flat = y_starts.reshape(-1, 1).astype(jnp.int32)  # Shape: (batch_size * num_tiles, 1)
    x_starts_flat = x_starts.reshape(-1, 1).astype(jnp.int32)  # Shape: (batch_size * num_tiles, 1)
    
    return y_starts_flat, x_starts_flat

  def compute_feature_similarity(true_out, recon_out, feature_type="sa", similarity="dot_product"):
      """Compute similarity loss between specific feature types from original and reconstructed images.
      
      Args:
          true_out: Dictionary containing outputs from true image in SigLIP model
          recon_out: Dictionary containing outputs from reconstructed image in SigLIP model
          feature_type: String indicating which feature to use ("sa", "+sa", "+mlp")
          similarity: String indicating similarity metric - "dot_product" or "kl"
          
      Returns:
          Scalar loss based on feature differences
      """
      loss = 0.0
      num_blocks = 0
      
      # Extract encoder outputs
      true_encoder = true_out.get("img/encoder", true_out.get("encoder", {}))
      recon_encoder = recon_out.get("img/encoder", recon_out.get("encoder", {}))
      
      # Process all blocks for the specific feature type
      for key, block in true_encoder.items():
          if not key.startswith("block") or key not in recon_encoder:
              continue
              
          # Only process the specified feature type
          if feature_type not in block or feature_type not in recon_encoder[key]:
              continue
              
          true_feat = block[feature_type]
          recon_feat = recon_encoder[key][feature_type]
          
          # Skip if shapes don't match
          if true_feat.shape != recon_feat.shape:
              continue
          
          # Compute similarity based on selected metric
          if similarity == "kl":
              # Normalize with softmax along feature dimension
              true_max = jnp.max(true_feat, axis=-1, keepdims=True)
              recon_max = jnp.max(recon_feat, axis=-1, keepdims=True)
              
              true_exp = jnp.exp(true_feat - true_max)
              recon_exp = jnp.exp(recon_feat - recon_max)
              
              true_sum = jnp.sum(true_exp, axis=-1, keepdims=True)
              recon_sum = jnp.sum(recon_exp, axis=-1, keepdims=True)
              
              true_norm = true_exp / (true_sum + 1e-8)
              recon_norm = recon_exp / (recon_sum + 1e-8)
              
              # KL divergence
              kl = true_norm * (jnp.log(true_norm + 1e-8) - jnp.log(recon_norm + 1e-8))
              feat_loss = jnp.mean(kl)
          else:  # Default to dot product
              # Calculate dot product similarity
              dot_prod = jnp.sum(true_feat * recon_feat, axis=-1)
              true_mag = jnp.sqrt(jnp.sum(true_feat**2, axis=-1))
              recon_mag = jnp.sqrt(jnp.sum(recon_feat**2, axis=-1))
              
              # Normalized dot product (cosine similarity)
              feat_loss = 1.0 - jnp.mean(dot_prod / (true_mag * recon_mag + 1e-8))
          
          loss += feat_loss
          num_blocks += 1
      
      # Return average loss or 0 if no features found
      return loss / (num_blocks + 1e-8) if num_blocks > 0 else 0.0 # Average the loss over blocks

  # Computing ELBO or beta-VAE loss for Gaussian encoder.
  def vae_loss_fn(image, recon_image, mu, logvar, beta=1.0,
                keep_batch_dim=False, loss_type="l1"):
    # Unpack the image tuples
    patches, ptype, yabs, xabs = image
    decoded_patches, _, _, _ = recon_image
    
    # Reconstruction loss per patch
    padding_mask = (ptype == 1).astype(patches.dtype) # Shape (N, num_patches)
    
    # Calculate element-wise loss (depends on loss_type)
    # Shape: (N, num_patches, C) assuming patches have channel dimension C
    element_loss = jnp.abs(patches - decoded_patches) + 5 * jnp.square(patches - decoded_patches) #5 is a measure of scale of loss based on experiments (0.01 - 0.05 for L1 and 0.001 - 0.005 for L2)

    weighted_loss_elements = element_loss * padding_mask[:, :, None] # Shape (N, num_patches, C)

    # Sum weighted loss over patches dimension (axis=1) -> Shape (N, C)
    loss_rec_summed_over_patches = jnp.sum(weighted_loss_elements, axis=1)

    # KL divergence per patch element -> Shape (N, num_patches, C)
    loss_kl_elements = -0.5 * (1 + logvar - mu**2 - jnp.exp(logvar))
    
    # Apply padding mask to KL elements
    masked_kl_elements = loss_kl_elements * padding_mask[:, :, None] # Shape (N, num_patches, C)
    
    # Sum KL loss over patches dimension (axis=1) -> Shape (N, C)
    loss_kl_summed_over_patches = jnp.sum(masked_kl_elements, axis=1)

    # Correct mean normalization by counting non-padded patches
    num_valid_patches = jnp.sum(padding_mask, axis=1, keepdims=True) # Shape (N, 1)
    
    # Normalize summed losses by number of valid patches (broadcasts correctly)
    loss_rec = loss_rec_summed_over_patches / (num_valid_patches + 1e-8) # Shape (N, C)
    loss_kl = loss_kl_summed_over_patches / (num_valid_patches + 1e-8) # Shape (N, C)

    if not keep_batch_dim:
        # Average over batch (N) and feature (C) dimensions
        loss_rec = jnp.mean(loss_rec)
        loss_kl = jnp.mean(loss_kl)

    loss = loss_rec + beta * loss_kl
    return loss, {"loss_rec": loss_rec, "loss_kl": loss_kl}
  
  def contrastive_loss_fn(zimg, ztxt, t, b): #Requires larger batch sizes...
    """Contrastive loss between image and text embeddings."""
    logits = jnp.dot(zimg, ztxt.T)
    logits = logits * t + b
    eye = jnp.eye(zimg.shape[0])
    m1_diag1 = -jnp.ones_like(logits) + 2 * eye
    loglik = jax.nn.log_sigmoid(m1_diag1 * logits)
    nll = -jnp.sum(loglik, axis=-1)
    l = jnp.mean(nll)
    return l
  
  if config.get("lpips", 0.0) > 0.0:
    write_note("Initializing LPIPS loss...")
    # Create a dummy image from the dataset element spec.
    lpips_model = LPIPS()
    def lpips_init(rng):
      dummy_image = jnp.zeros(config.image_dim_lpips)
      lpips_vars = lpips_model.init(rng, dummy_image, dummy_image)
      return lpips_vars["params"]

    # Infer shape and sharding for LPIPS params
    lpips_params_shape = jax.eval_shape(lpips_init, rng_loop)
    with nn.logical_axis_rules(sharding_rules):
      lpips_params_sharding = bv_sharding.infer_sharding(
          lpips_params_shape, strategy=strategy, mesh=mesh)

    # Initialize with inferred sharding
    lpips_params = jax.jit(lpips_init, out_shardings=lpips_params_sharding)(rng_loop)
    lpips_params = lpips_load(lpips_params)
  else:
    lpips_model = None
    lpips_params = None
  
  if config.get("siglip", False):
    siglip_model_mod = importlib.import_module(f"big_vision.models.{config.siglip_model_name}")
    siglip_model = siglip_model_mod.Model(**config.get("siglip_model", {}))

    def siglip_init(rng):
      batch = jax.tree.map(
          lambda x: jnp.zeros(x.shape, x.dtype.as_numpy_dtype),
          train_ds.element_spec)
      siglip_params =  siglip_model.init(rng, batch["image"], text=batch.get("labels", None))["params"]
      return siglip_params

    #load pre-trained siglip model
    siglip_params_eval_shape = jax.eval_shape(siglip_init, rng_loop)

    with nn.logical_axis_rules(sharding_rules):
        siglip_params_sharding = bv_sharding.infer_sharding(
            siglip_params_eval_shape, strategy=strategy, mesh=mesh)

    siglip_params = jax.jit(siglip_init)(rng_loop)
    if config.siglip_model_init:
      siglip_params = siglip_model_mod.load(
          siglip_params, config.siglip_model_init, config.get("siglip_model"))

      # shard model with FSDP
      siglip_params = u.reshard(siglip_params, siglip_params_sharding)

    # Remove the assertion and add informative logging
    logging.info("SigLIP feedback model loaded successfully")

  ################################################################################
  #                                                                              #
  #                            Discriminator Setup                               #
  #                                                                              #
  ################################################################################

  # Initialize discriminator if enabled in config
  discriminator_state = None
  if config.get("discriminator", 0.0):
    write_note("Creating discriminator...")
    discriminator = Discriminator(input_size=config.image_dim_lpips[1])
      
    def disc_init(rng):
      # Create a dummy image of the right shape
      dummy_image = jnp.zeros(config.image_dim_lpips) #Use same shape as LPIPS model, since we will take crops
      params = discriminator.init(rng, dummy_image)["params"]
      return params
      
    # Initialize discriminator params
    _, rng_disc = jax.random.split(rng_loop)
    discriminator_params_shape = jax.eval_shape(disc_init, rng_disc)
    
    # Create discriminator optimizer
    # Create a proper ConfigDict object that has a get() method
    
    disc_config = ConfigDict()
    disc_config.optax_name = config.get("discriminator_optax_name", "scale_by_adam")
    disc_config.optax = config.get("discriminator_optax", dict(b2=0.95))
    disc_config.lr = config.get("discriminator_lr", 2e-5)
    disc_config.wd = config.get("discriminator_wd", 1e-4)
    disc_config.schedule = config.get("discriminator_schedule", [(".*", dict(decay_type='cosine', warmup_steps=0.05))])
    
    disc_tx, disc_sched_fns = bv_optax.make(
        disc_config,
        nn.unbox(discriminator_params_shape),
        sched_kw=dict(total_steps=total_steps // config.get("grad_accum_steps", 1), batch_size=batch_size, data_size=ntrain_img)
    )

    # Wrap the discriminator optimizer with MultiSteps if gradient accumulation is enabled.
    if grad_accum_steps > 1:
      disc_tx = optax.MultiSteps(disc_tx, every_k_schedule=grad_accum_steps)
      write_note(f"Using gradient accumulation for discriminator with {grad_accum_steps} steps.")

    disc_opt_shape = jax.eval_shape(disc_tx.init, discriminator_params_shape)
    
    # Get schedule functions for CPU
    disc_sched_fns_cpu = [u.jit_cpu()(sched_fn) for sched_fn in disc_sched_fns]
    
    # Infer sharding for discriminator state
    disc_state_shape = {"params": discriminator_params_shape, "opt": disc_opt_shape}
    with nn.logical_axis_rules(sharding_rules):
      disc_state_sharding = bv_sharding.infer_sharding(
          disc_state_shape, strategy=strategy, mesh=mesh)
          
    # Initialize discriminator params
    discriminator_params = jax.jit(disc_init, out_shardings=disc_state_sharding["params"])(rng_disc)
    disc_opt = jax.jit(disc_tx.init, out_shardings=disc_state_sharding["opt"])(discriminator_params)
    
    # Create discriminator state
    discriminator_state = nn.unbox({"params": discriminator_params, "opt": disc_opt})
    del discriminator_params, disc_opt  # Delete to avoid memory leak
    
    write_note("Logging discriminator parameter overview...")
    parameter_overview.log_parameter_overview(
        discriminator_state["params"], msg="Init discriminator params",
        include_stats="global", jax_logging_process=0)
  else:
    disc_sched_fns_cpu = None

################################################################################
#                                                                              #
#                                 Update Step                                  #
#                                                                              #
################################################################################

  @functools.partial(
      jax.jit,
      donate_argnums=(0, 1),
      out_shardings=(train_state_sharding, 
                     disc_state_sharding if config.get("discriminator", 0.0) else repl_sharding, 
                     repl_sharding))
  def update_fn(train_state, discriminator_state, rng, batch):
    """Update step for both the generator and discriminator."""
    # Extract params and opt from train_state
    params, opt, ema_params = train_state["params"], train_state["opt"], train_state["ema_params"]
    
    step_count = bv_optax.get_count(opt, jittable=True)
    rng = jax.random.fold_in(rng, step_count)

    # Get device-specific loss rng.
    rng_vae, rng_model, rng_crop, rng_disc = jax.random.split(rng, 4)

    def loss_fn(params):
        recon_image, out = model.apply(
            {"params": params},
            batch["image"],
            text=None,
            train=True,
            rngs={"dropout": rng_model, "vae": rng_vae})
        mu = out["mu"]
        logvar = out["logvar"]
        zimg = out["zimg"]
        t = out["t"]
        b = out["b"]

        y_starts_flat, x_starts_flat = extra_crops(batch["image"], rng_crop, num_tiles=config.num_tiles)
        _2D_image = patches_to_image(batch["image"], config.posemb_grid_size, config.posemb_grid_size, patch_size=config.patch_size)
        recon_2D_image = patches_to_image(recon_image, config.posemb_grid_size, config.posemb_grid_size, patch_size=config.patch_size)
        
        # Now extract_crops_vectorized will handle the dimension mismatch internally
        true_crops = extract_crops_vectorized(_2D_image, y_starts_flat, x_starts_flat, crop_size=config.crop_size)
        recon_crops = extract_crops_vectorized(recon_2D_image, y_starts_flat, x_starts_flat, crop_size=config.crop_size)

        # Fixed VAE loss function call
        loss, aux_loss = vae_loss_fn(
            batch["image"], recon_image, mu, logvar, config.get("beta", 1.0),
        )
        
        # Include contrastive loss if configured
        if config.get("contrastive_weight", 0) > 0:
            contrastive_loss = contrastive_loss_fn(out["zimg"], out["ztxt"], out["t"], out["b"])
            loss += config.get("contrastive_weight") * contrastive_loss
            aux_loss["sigmoid_loss"] = contrastive_loss

        # Fixed LPIPS loss
        if config.get("lpips", 0.0) > 0.0:
            loss_lpips = lpips_model.apply({"params": lpips_params}, recon_crops, true_crops).mean()
            loss += config.get("lpips") * loss_lpips
            aux_loss["lpips_loss"] = loss_lpips
        
        # SigLIP section 
        recon_siglip_out = None # Initialize to None
        true_siglip_out = None  # Initialize to None
        if config.get("siglip", False):
            # Run SigLIP on true image - assumes output dict contains necessary features
            true_siglip_zimg, _, true_siglip_out = siglip_model.apply(
                {"params": siglip_params}, batch["image"], text=None
            )

            if config.get("siglip_feedback", 0.0) > 0.0:
                # Run SigLIP on reconstructed image
                recon_siglip_zimg, _, recon_siglip_out = siglip_model.apply(
                    {"params": siglip_params}, recon_image, text=None
                )
                aux_loss["siglip_feedback_loss"] = jnp.sum((recon_siglip_zimg - true_siglip_zimg)**2)
                loss += config.get("siglip_feedback") * aux_loss["siglip_feedback_loss"]
            
            # Add the feature similarity calculation here:
            if config.get("siglip_feature_weight", 0.0) > 0.0:
                # Get chosen feature type from config
                feature_type = config.get("siglip_feature_type", "sa")
                similarity = config.get("siglip_feature_similarity", "dot_product")
                
                # We need recon_siglip_out, compute if not already done for feedback loss
                if recon_siglip_out is None:
                    _, _, recon_siglip_out = siglip_model.apply(
                        {"params": siglip_params}, recon_image, text=None
                    )
                
                # Pass the full output dictionaries
                feature_loss = compute_feature_similarity(
                    true_siglip_out, 
                    recon_siglip_out,
                    feature_type=feature_type,
                    similarity=similarity
                )
                aux_loss["siglip_feature_loss"] = feature_loss
                
                # Apply optional warmup
                warmup_steps = config.get("siglip_feature_warmup_steps", 0)
                if warmup_steps > 0:
                    feature_weight = config.get("siglip_feature_weight") * jnp.minimum(1.0, step_count / warmup_steps)
                    aux_loss["siglip_feature_weight"] = feature_weight
                    loss += feature_weight * feature_loss
                else:
                    loss += config.get("siglip_feature_weight") * feature_loss

            if config.get("siglip_distill", 0.0) > 0.0:
                # Supervised contrastive loss on zimg
                # Ensure true_siglip_zimg is available (already computed above)
                logits = jnp.dot(zimg, true_siglip_zimg.T) 
                logits = logits * t + b # Use t, b from the main VAE model's output
                eye = jnp.eye(zimg.shape[0])
                m1_diag1 = -jnp.ones_like(logits) + 2 * eye
                loglik = jax.nn.log_sigmoid(m1_diag1 * logits)
                nll = -jnp.sum(loglik, axis=-1)
                aux_loss["siglip_distill_loss"] = jnp.mean(nll)
                loss += config.get("siglip_distill") * aux_loss["siglip_distill_loss"]
        
        # Generator loss from discriminator if enabled
        has_discriminator = config.get("discriminator", 0.0) and discriminator_state is not None
        if has_discriminator:
          # Apply discriminator loss but scale with warmup
          disc_fake_logits = discriminator.apply({"params": discriminator_state["params"]}, recon_crops)
          g_loss = -jnp.mean(disc_fake_logits)
          
          # Apply warmup to generator loss based on step_count
          warmup_steps = config.get("gen_loss_warmup_steps", 10000)
          start_step = config.get("start_gen_loss", 0)
          aux_weight = jnp.maximum(0.0, jnp.minimum(1.0, (step_count - start_step) / warmup_steps))
          
          # Store metrics regardless of whether we're in warmup
          aux_loss["generator_loss"] = g_loss
          aux_loss["generator_loss_weight"] = aux_weight
          
          # Apply weighted loss
          loss += config.get("discriminator", 0.0) * g_loss * aux_weight
    
        aux_loss["loss"] = loss
        return loss, (aux_loss, true_crops, recon_crops)
  
    (loss, (measurements, true_crops, recon_crops)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        params)
    updates, opt = tx.update(grads, opt, params)
    params = optax.apply_updates(params, updates)
    measurements["training_loss"] = loss
    gs = jax.tree.leaves(bv_optax.replace_frozen(config.schedule, grads, 0.))
    measurements["l2_grads"] = jnp.sqrt(sum([jnp.vdot(g, g) for g in gs]))
    ps = jax.tree.leaves(params)
    measurements["l2_params"] = jnp.sqrt(sum([jnp.vdot(p, p) for p in ps]))
    us = jax.tree.leaves(updates)
    measurements["l2_updates"] = jnp.sqrt(sum([jnp.vdot(u, u) for u in us]))

    ema_params = optax.incremental_update(
        ema_params, params, config.get("ema_decay", 0.9999))
    
    # Now update the discriminator if it exists
    has_discriminator = config.get("discriminator", 0.0) and discriminator_state is not None
    if has_discriminator:
      disc_params, disc_opt = discriminator_state["params"], discriminator_state["opt"]
      
      def disc_loss_fn(disc_params):
        # Get discriminator outputs for real and fake
        logits_real = discriminator.apply({"params": disc_params}, true_crops)
        logits_fake = discriminator.apply({"params": disc_params}, recon_crops)
        
        # Calculate discriminator loss (hinge loss)
        disc_loss = hinge_d_loss(logits_real, logits_fake)
        
        return disc_loss, {
            "disc_loss": disc_loss,
            "logits_real_mean": jnp.mean(logits_real),
            "logits_fake_mean": jnp.mean(logits_fake),
        }
      
      (disc_loss, disc_measurements), disc_grads = jax.value_and_grad(disc_loss_fn, has_aux=True)(disc_params)
      disc_updates, disc_opt = disc_tx.update(disc_grads, disc_opt, disc_params)
      disc_params = optax.apply_updates(disc_params, disc_updates)

      disc_measurements["disc_loss"] = disc_loss
      gs = jax.tree.leaves(bv_optax.replace_frozen(config.schedule, disc_grads, 0.))
      disc_measurements["l2_disc_grads"] = jnp.sqrt(sum([jnp.vdot(g, g) for g in gs]))
      ps = jax.tree.leaves(disc_params)
      disc_measurements["l2_disc_params"] = jnp.sqrt(sum([jnp.vdot(p, p) for p in ps]))
      us = jax.tree.leaves(disc_updates)
      disc_measurements["l2_disc_updates"] = jnp.sqrt(sum([jnp.vdot(u, u) for u in us]))
      
      # Update discriminator state
      discriminator_state = {"params": disc_params, "opt": disc_opt}
      
      # Merge measurements
      measurements.update(disc_measurements)

    return {"params": params, "opt": opt, "ema_params": ema_params}, discriminator_state, measurements

################################################################################
#                                                                              #
#                               Load Checkpoint                                #
#                                                                              #
################################################################################

  # Decide how to initialize training. The order is important.
  # 1. Always resumes from the existing checkpoint, e.g. resumes a finetune job.
  # 2. Resume from a previous checkpoint, e.g. start a cooldown training job.
  # 3. Initialize model from something, e,g, start a fine-tuning job.
  # 4. Train from scratch.
  resume_ckpt_path = None
  if save_ckpt_path and gfile.exists(f"{save_ckpt_path}-LAST"):
    resume_ckpt_path = save_ckpt_path
  elif config.get("resume"):
    resume_ckpt_path = fillin(config.resume)

  ckpt_mngr = None
  if save_ckpt_path or resume_ckpt_path:
    ckpt_mngr = array_serial.GlobalAsyncCheckpointManager()

  if resume_ckpt_path:
    write_note(f"Resuming training from checkpoint {resume_ckpt_path}...")
    jax.tree.map(lambda x: x.delete(), train_state)
    del train_state
    shardings = {
        **train_state_sharding,
        "chrono": jax.tree.map(lambda _: repl_sharding,
                               u.chrono.save()),
    }
    loaded = u.load_checkpoint_ts(
        resume_ckpt_path, tree=shardings, shardings=shardings)
    train_state = {key: loaded[key] for key in train_state_sharding.keys()}

    u.chrono.load(jax.device_get(loaded["chrono"]))
    del loaded
  elif config.get("model_init_ckpt", ""):
    write_note(f"Initialize model from {config.model_init_ckpt}...")
    train_state["params"] = model_mod.simple_load(
        train_state["params"], config.model_init_ckpt)
    train_state["ema_params"] = model_mod.simple_load(
        train_state["ema_params"], config.model_init_ckpt, load_ema=True)
    # Ensure params are properly sharded after loading
    train_state["params"] = u.reshard(
        train_state["params"], train_state_sharding["params"])
    train_state["ema_params"] = u.reshard(
        train_state["ema_params"], train_state_sharding["ema_params"])
    
    parameter_overview.log_parameter_overview(
        train_state["params"], msg="restored params",
        include_stats="global", jax_logging_process=0)

  elif config.get("model_init"):
    write_note(f"Initialize model from {config.model_init}...")
    train_state["params"] = model_mod.load(
        train_state["params"], config.model_init, config.get("model"),
        **config.get("model_load", {}))

    # load has the freedom to return params not correctly sharded. Think of for
    # example ViT resampling position embedings on CPU as numpy arrays.
    train_state["params"] = u.reshard(
        train_state["params"], train_state_sharding["params"])

    parameter_overview.log_parameter_overview(
        train_state["params"], msg="restored params",
        include_stats="global", jax_logging_process=0)


################################################################################
#                                                                              #
#                                 Setup Evals                                  #
#                                                                              #
################################################################################

  # We do not jit/pmap this function, because it is passed to evaluator that
  # does it later. We output as many intermediate tensors as possible for
  # maximal flexibility. Later `jit` will prune out things that are not needed.
  def predict_fn(train_state, batch):
    _, out = model.apply(
        {"params": train_state["ema_params"]},
        batch.get("image"), text=batch.get("labels", None))
    zimg = out["zimg"]
    ztxt = out["ztxt"]
    return zimg, ztxt, {}
  def recon_fn(train_state, batch, rng):
    x, out = model.apply(
        {"params": train_state["ema_params"]},
        batch.get("image"), None)
    image = patches_to_image(x, config.posemb_grid_size, config.posemb_grid_size)
    ref_image = patches_to_image(batch["image"], config.posemb_grid_size, config.posemb_grid_size)
    total_loss, aux_loss = vae_loss_fn(batch["image"], x, out["mu"], out["logvar"], config.get("beta", 1.0))
    
    # Split the incoming RNG
    rng_lpips, rng_fid = jax.random.split(rng)

    # Initialize lpips_loss to handle case where it's not computed
    lpips_loss = 0.0

    #Compute LPIPS loss with more tiles
    if config.get("lpips", 0.0) > 0.0:
      # Use the split key for LPIPS crops
      y_starts_flat, x_starts_flat = extra_crops(batch["image"], rng_lpips, num_tiles=2, crop_size=224)
      recon_crops = extract_crops_vectorized(image, y_starts_flat, x_starts_flat, crop_size=224)
      true_crops = extract_crops_vectorized(ref_image, y_starts_flat, x_starts_flat, crop_size=224)
      lpips_loss = lpips_model.apply({"params": lpips_params}, recon_crops, true_crops).mean()
    # fid crops should be 299x299
    # Use the other split key for FID crops
    y_starts_flat, x_starts_flat = extra_crops(batch["image"], rng_fid, num_tiles=2, crop_size=299)
    true_crops = extract_crops_vectorized(ref_image, y_starts_flat, x_starts_flat, crop_size=299)
    recon_crops = extract_crops_vectorized(image, y_starts_flat, x_starts_flat, crop_size=299)

    out = {"kl_loss": aux_loss["loss_kl"], "recon_loss": aux_loss["loss_rec"], 
           "total_loss": total_loss, "ref_image": ref_image, "image": image, "lpips_loss": lpips_loss, # Now always defined
           "true_crops": true_crops, "recon_crops": recon_crops}

    if config.get("siglip", False):
      true_siglip_zimg, _, true_siglip_out = siglip_model.apply(
          {"params": siglip_params}, batch["image"], text=None
      )
      recon_siglip_zimg, _, recon_siglip_out = siglip_model.apply(
          {"params": siglip_params}, x, text=None
      )
      siglip_feature_loss = compute_feature_similarity(
        true_siglip_out, 
        recon_siglip_out,
        feature_type=config.get("siglip_feature_type", "sa"),
        similarity=config.get("siglip_feature_similarity", "dot_product")
      )
      out["siglip_feature_loss"] = siglip_feature_loss
      out["siglip_true_zimg"] = true_siglip_zimg
      out["siglip_recon_zimg"] = recon_siglip_zimg
    return x, out
  
  def siglip_predict_fn(train_state, batch):
    zimg, ztxt, out = siglip_model.apply(
        {"params": siglip_params},
        batch.get("image"), text=batch.get("labels", None)
    )
    return zimg, ztxt, {}
  
  def supervised_predict_fn(train_state, batch):
    if batch.get("labels", None) is None:
      _, out = model.apply(
          {"params": train_state["params"]},
          batch['image'], text=None
      )
      zimg = out["zimg"]
      ztxt = None
    else:
      zimg, ztxt, out = siglip_model.apply(
          {"params": siglip_params},
          None, text=batch['labels']
      )
    return zimg, ztxt, {}
  # Only initialize evaluators when they are first needed.
  # TODO: add evaluators for SigLIP to test for performance decay (or not) after fine tuning encoder
  @functools.lru_cache(maxsize=None) 
  def evaluators():
    return eval_common.from_config(
        config, {
            "predict": predict_fn,
            "siglip_predict": siglip_predict_fn,
            "supervised_predict": supervised_predict_fn,
            "recon": recon_fn,
            },
        lambda s: write_note(f"Init evaluator: {s}…\n{u.chrono.note}"),
        lambda key, cfg: get_steps(key, default=None, cfg=cfg),
        devices_flat,
    )

  # At this point we need to know the current step to see whether to run evals.
  write_note("Inferring the first step number...")
  first_step_device = bv_optax.get_count(train_state["opt"], jittable=True)
  first_step = int(jax.device_get(first_step_device))
  u.chrono.inform(first_step=first_step)

  # Note that training can be pre-empted during the final evaluation (i.e.
  # just after the final checkpoint has been written to disc), in which case we
  # want to run the evals.
  for (name, evaluator, log_steps, prefix) in evaluators():
    with u.chrono.log_timing(f"z/secs/eval/{name}"):
      with mesh, nn.logical_axis_rules(sharding_rules):
        for key, value in evaluator.run(train_state):
          mw.measure(f"{prefix}{key}", jax.device_get(value))

################################################################################
#                                                                              #
#                                  Train Loop                                  #
#                                                                              #
################################################################################

  prof = None  # Keeps track of start/stop of profiler state.

  write_note("Starting training loop, compiling the first step...")
  step = first_step + 1
  pbar = tqdm(total=total_steps - first_step)
  while step <= total_steps:
    try:
      batch = next(train_iter)
    except Exception as e:
      logging.warning(f"Skipping batch at step {step} due to error: {e}")
      
      #assert False, "Skipping batch at step {step} due to error: {e}"
      #continue  # Skip this batch and try the next one

    mw.step_start(step)

    with jax.profiler.StepTraceAnnotation("train_step", step_num=step):
      with u.chrono.log_timing("z/secs/update0", noop=step > first_step + 1):
        with mesh, nn.logical_axis_rules(sharding_rules):
          train_state, discriminator_state, measurements = update_fn(train_state, discriminator_state, rng_loop, batch)
            
    # On the first host, let's always profile a handful of early steps.
    if jax.process_index() == 0:
      prof = u.startstop_prof(prof, step, first_step, get_steps("log_training"))

    # Report training progress
    if (u.itstime(step, get_steps("log_training"), total_steps, host=0)
        or u.chrono.warmup and jax.process_index() == 0):
      for i, sched_fn_cpu in enumerate(sched_fns_cpu):
        mw.measure(f"global_schedule{i if i else ''}",
                   sched_fn_cpu(u.put_cpu(step - 1)))
      
      # Report discriminator scheduler if it exists
      if disc_sched_fns_cpu and discriminator_state is not None:
        for i, sched_fn_cpu in enumerate(disc_sched_fns_cpu):
          mw.measure(f"disc_schedule{i if i else ''}",
                     sched_fn_cpu(u.put_cpu(step - 1)))
                     
      measurements = jax.device_get(measurements)
      for name, value in measurements.items():
        mw.measure(name, value)
      u.chrono.tick(step)
      if not np.isfinite(measurements["training_loss"]):
        write_note(f"Training loss became nan or inf at step {step}. Resuming training from checkpoint {resume_ckpt_path}...")
        write_note(f"Resuming training from checkpoint {resume_ckpt_path}...")
        jax.tree.map(lambda x: x.delete(), train_state)
        del train_state
        
        # Include discriminator state shardings if it exists
        all_shardings = {
            **train_state_sharding,
            "chrono": jax.tree.map(lambda _: repl_sharding, u.chrono.save()),
        }
        
        if config.get("discriminator", 0.0):
          all_shardings["discriminator"] = disc_state_sharding
        
        # When finetuning, we don't load the optimizer state from the checkpoint.
        if config.get("finetune", False):
          if "opt" in all_shardings:
            del all_shardings["opt"]
        
        loaded = u.load_checkpoint_ts(
            resume_ckpt_path, tree=all_shardings, shardings=all_shardings)
        
        train_state = {key: loaded[key] for key in train_state_sharding.keys()}
        # Load discriminator state if it exists in the checkpoint
        if config.get("discriminator", 0.0) and "discriminator" in loaded:
          discriminator_state = loaded["discriminator"]
        #step = int(jax.device_get(u.get_count(train_state["opt"], jittable=True)))
        del loaded

    # Checkpoint saving
    keep_ckpt_steps = get_steps("keep_ckpt", None) or total_steps
    if save_ckpt_path and (
        (keep := u.itstime(step, keep_ckpt_steps, total_steps, first=False))
        or u.itstime(step, get_steps("ckpt", None), total_steps, first=True)
    ):
      u.chrono.pause(wait_for=train_state)

      # Copy because we add extra stuff to the checkpoint.
      ckpt = {**train_state}
      
      # Add discriminator state to checkpoint if it exists
      if discriminator_state is not None:
        ckpt["discriminator"] = discriminator_state

      # To save chrono state correctly and safely in a multihost setup, we
      # broadcast the state to all hosts and convert it to a global array.
      with jax.transfer_guard("allow"):
        chrono_ckpt = multihost_utils.broadcast_one_to_all(u.chrono.save())
      chrono_shardings = jax.tree.map(lambda _: repl_sharding, chrono_ckpt)
      ckpt = ckpt | {"chrono": u.reshard(chrono_ckpt, chrono_shardings)}

      u.save_checkpoint_ts(ckpt_mngr, ckpt, save_ckpt_path, step, keep)
      u.chrono.resume()

    for (name, evaluator, log_steps, prefix) in evaluators():
      if u.itstime(step, log_steps, total_steps, first=False, last=True):
        u.chrono.pause(wait_for=train_state)
        u.chrono.tick(step)  # Record things like epoch number, core hours etc.
        write_note(f"{name} evaluation...\n{u.chrono.note}")
        with u.chrono.log_timing(f"z/secs/eval/{name}"):
          with mesh, nn.logical_axis_rules(sharding_rules):
            for key, value in evaluator.run(train_state):
              mw.measure(f"{prefix}{key}", jax.device_get(value))
        u.chrono.resume()
    mw.step_end()
    step += 1
    pbar.update(1)
  pbar.close()

  # Always give a chance to stop the profiler, no matter how things ended.
  if jax.process_index() == 0 and prof is not None:
    u.startstop_prof(prof)

  # Last note needs to happen before the pool's closed =)
  write_note(f"Done!\n{u.chrono.note}")

  pool.close()
  pool.join()
  mw.close()

  if ckpt_mngr:
    ckpt_mngr.wait_until_finished()

  # Make sure all hosts stay up until the end of main.
  u.sync()

  u.maybe_cleanup_workdir(workdir, flags.FLAGS.cleanup, info)


if __name__ == "__main__":
  app.run(main)