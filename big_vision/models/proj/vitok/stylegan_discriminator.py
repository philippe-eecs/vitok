import math
from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import lax

import flax.linen as nn


# -------------------
# Utility functions
# -------------------

def hinge_d_loss(logits_real: jnp.ndarray, logits_fake: jnp.ndarray) -> jnp.ndarray:
    """
    Hinge loss for the discriminator (JAX version).
    """
    loss_real = jnp.mean(nn.relu(1.0 - logits_real))
    loss_fake = jnp.mean(nn.relu(1.0 + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

def compute_lecam_loss(
    logits_real_mean: jnp.ndarray,
    logits_fake_mean: jnp.ndarray,
    ema_logits_real_mean: jnp.ndarray,
    ema_logits_fake_mean: jnp.ndarray
) -> jnp.ndarray:
    """
    Computes the LeCam loss for given average real and fake logits (JAX version).
    """
    term1 = jnp.mean((nn.relu(logits_real_mean - ema_logits_fake_mean)) ** 2)
    term2 = jnp.mean((nn.relu(ema_logits_real_mean - logits_fake_mean)) ** 2)
    return term1 + term2


# -------------------
# ActNorm in Flax
# -------------------
class ActNorm(nn.Module):
    """
    Flax version of ActNorm. To emulate data-dependent initialization, we store
    an `initialized` flag in the Module's batch/state collection. Once initialized,
    the loc and scale remain learned parameters.
    """
    num_features: int
    logdet: bool = False
    affine: bool = True
    allow_reverse_init: bool = False  # Unused here, but included for completeness.

    @nn.compact
    def __call__(self, 
                 x: jnp.ndarray, 
                 reverse: bool = False,
                 train: bool = True):
        """
        If reverse=True, performs the reverse operation: (x / scale) - loc.
        Otherwise, does scale * (x + loc).
        """
        # Handle the case of (N, C) → (N, C, 1, 1)
        # so that we stay consistent with PyTorch's layout for 2D data.
        squeeze = False
        if x.ndim == 2:
            x = x[:, :, None, None]
            squeeze = True

        # Create or retrieve the parameters.
        # By default, we initialize them as zeros and ones, but we can update them
        # after the first forward pass if we want data-dependent init.
        loc = self.param("loc", 
                         nn.initializers.zeros, 
                         (1, x.shape[1], 1, 1))
        scale = self.param("scale", 
                           nn.initializers.ones, 
                           (1, x.shape[1], 1, 1))
        
        # Keep track of an "initialized" flag as batch/state variable.
        initialized = self.variable("batch_stats", "initialized", 
                                    lambda: jnp.array(0, dtype=jnp.uint8))

        # Data-dependent initialization logic:
        def _initialize_actnorm(x, loc, scale):
            # Compute mean & std over (N, H, W).
            # (N, C, H, W) => flatten over N,H,W
            flatten = jnp.reshape(
                jnp.transpose(x, (1, 0, 2, 3)), 
                (x.shape[1], -1)
            )
            mean = jnp.mean(flatten, axis=1, keepdims=True)  # shape (C, 1)
            std = jnp.std(flatten, axis=1, keepdims=True)    # shape (C, 1)

            # Expand back to (1, C, 1, 1)
            mean = mean[:, None, None]
            std = std[:, None, None]

            # Return new "loc" and "scale"
            new_loc = -mean
            new_scale = 1.0 / (std + 1e-6)
            return new_loc, new_scale

        # If training and not yet initialized, do data-dependent init.
        # Then mark as initialized=1
        if train and (initialized.value == 0):
            new_loc, new_scale = _initialize_actnorm(x, loc, scale)
            # Because loc/scale are parameters, we cannot simply do "loc = new_loc".
            # In Flax, one typical approach is: re-initialize them via a custom init function
            # or treat them as variables, not params. For demonstration, we do the variable approach:
            loc_var = self.variable("batch_stats", "loc_var", lambda: loc)
            scale_var = self.variable("batch_stats", "scale_var", lambda: scale)
            loc_var.value = new_loc
            scale_var.value = new_scale
            initialized.value = jnp.array(1, dtype=jnp.uint8)

            # Now use the updated variables for forward pass:
            loc_actual = loc_var.value
            scale_actual = scale_var.value
        else:
            # If we have previously stored them in batch_stats, use that.
            # Otherwise, just the param. (One might unify these logic paths.)
            loc_var = self.variable("batch_stats", "loc_var", lambda: loc)
            scale_var = self.variable("batch_stats", "scale_var", lambda: scale)
            loc_actual = loc_var.value
            scale_actual = scale_var.value

        if reverse:
            # Reverse pass: (x / scale) - loc
            out = (x / (scale_actual + 1e-8)) - loc_actual
        else:
            # Forward pass: scale * (x + loc)
            out = scale_actual * (x + loc_actual)

        # Unsqueeze to original shape if needed.
        if squeeze:
            out = out[:, :, 0, 0]

        # Optionally compute log-determinant if self.logdet
        if self.logdet:
            # sum of log|scale| across channels, multiplied by H*W
            _, _, height, width = x.shape
            log_abs = jnp.log(jnp.abs(scale_actual + 1e-8))
            # sum along channels
            logdet_val = height * width * jnp.sum(log_abs)
            # expand for batch dimension
            logdet_out = logdet_val * jnp.ones(x.shape[0])
            return out, logdet_out

        return out


# -------------------
# BlurPool2D in Flax
# -------------------
class BlurPool2D(nn.Module):
    filter_size: int = 4
    stride: int = 2

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.filter_size == 3:
            filt = jnp.array([1., 2., 1.], dtype=jnp.float32)
            pad = 1
        elif self.filter_size == 4:
            filt = jnp.array([1., 3., 3., 1.], dtype=jnp.float32)
            pad = 1
        elif self.filter_size == 5:
            filt = jnp.array([1., 4., 6., 4., 1.], dtype=jnp.float32)
            pad = 2
        elif self.filter_size == 6:
            filt = jnp.array([1., 5., 10., 10., 5., 1.], dtype=jnp.float32)
            pad = 2
        elif self.filter_size == 7:
            filt = jnp.array([1., 6., 15., 20., 15., 6., 1.], dtype=jnp.float32)
            pad = 3
        else:
            raise ValueError("Only filter_size of 3, 4, 5, 6 or 7 is supported.")

        filt_2d = jnp.outer(filt, filt)
        filt_2d = filt_2d / jnp.sum(filt_2d)
        # shape = (1, 1, H, W) to do depthwise
        filt_2d = filt_2d[None, None, :, :]

        # repeat filter for each channel
        channels = x.shape[1]
        depthwise_filter = jnp.tile(filt_2d, [channels, 1, 1, 1])

        # Use lax.conv_general_dilated for depthwise filtering
        # dims: N, C, H, W
        out = lax.conv_general_dilated(
            lhs=x,
            rhs=depthwise_filter,
            window_strides=(self.stride, self.stride),
            padding=[(pad, pad), (pad, pad)],
            # Fix: Use proper dimension numbers format for JAX
            dimension_numbers=('NCHW', 'OIHW', 'NCHW'),
            feature_group_count=channels  # depthwise
        )
        return out


# -------------------
# StyleGANResBlock in Flax
# -------------------
class StyleGANResBlock(nn.Module):
    input_dim: int
    output_dim: int
    blur_resample: bool = True
    act_fn: Any = nn.leaky_relu  # We can pass in jax.nn.leaky_relu or define our own.

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Use Xavier uniform initialization as in the PyTorch version
        xavier_init = nn.initializers.xavier_uniform()
        
        # conv1
        x_res = x
        conv1 = nn.Conv(
            features=self.output_dim, 
            kernel_size=(3, 3), 
            padding=((1,1),(1,1)),
            kernel_init=xavier_init
        )
        x = conv1(x)
        x = self.act_fn(x)

        # downsample
        if self.blur_resample:
            x = BlurPool2D(filter_size=4, stride=2)(x)
            x_res = BlurPool2D(filter_size=4, stride=2)(x_res)
        else:
            # average pool 2d
            x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding='VALID')
            x_res = nn.avg_pool(x_res, window_shape=(2, 2), strides=(2, 2), padding='VALID')

        # residual path
        res_conv = nn.Conv(
            features=self.output_dim, 
            kernel_size=(1, 1), 
            use_bias=False,
            kernel_init=xavier_init
        )
        x_res = res_conv(x_res)

        # conv2
        conv2 = nn.Conv(
            features=self.output_dim, 
            kernel_size=(3, 3), 
            padding=((1,1),(1,1)),
            kernel_init=xavier_init
        )
        x = conv2(x)
        x = self.act_fn(x)

        # sum + scale
        return (x_res + x) / math.sqrt(2.0)


# -------------------
# Discriminator in Flax
# -------------------
class Discriminator(nn.Module):
    """
    Flax version of StyleGAN-like Discriminator.
    """
    input_size: int = 256
    channel_multiplier: int = 1
    blur_resample: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Map input_size -> filter dictionary
        # (Following your PyTorch version)
        filters_map = {
            4:   512,
            8:   512,
            16:  512,
            32:  512,
            64:  256 * self.channel_multiplier,
            128: 128 * self.channel_multiplier,
            224: 64  * self.channel_multiplier,
            256: 64  * self.channel_multiplier,
            384: 32  * self.channel_multiplier,
            512: 32  * self.channel_multiplier,
            1024:16  * self.channel_multiplier
        }

        # Use Xavier uniform initialization for weights as in PyTorch version
        xavier_init = nn.initializers.xavier_uniform()
        
        # Create a leaky_relu function with fixed negative_slope=0.2
        activation_fn = lambda x: nn.leaky_relu(x, negative_slope=0.2)
        
        # conv1
        cfirst = nn.Conv(
            features=filters_map[self.input_size], 
            kernel_size=(3,3), 
            padding=((1,1),(1,1)),
            kernel_init=xavier_init
        )
        x = cfirst(x)
        x = activation_fn(x)

        # build a stack of ResBlocks
        log_size = int(math.log2(self.input_size))
        in_ft = filters_map[self.input_size]

        for i in range(log_size, 2, -1):
            out_ft = filters_map[2 ** (i - 1)]
            # define and remat each block in a sub-scope
            block = StyleGANResBlock(
                input_dim=in_ft,
                output_dim=out_ft,
                blur_resample=self.blur_resample,
                act_fn=activation_fn
            )
            # Call the rematerialized block
            x = block(x)
            in_ft = out_ft

        # conv_last
        conv_last = nn.Conv(
            features=filters_map[4], 
            kernel_size=(3,3), 
            padding=((1,1),(1,1)),
            kernel_init=xavier_init
        )
        x = conv_last(x)
        x = activation_fn(x)

        # flatten
        x = x.reshape((x.shape[0], -1))

        # fc1
        # In PyTorch this assumes 4x4 spatial dimensions after downsampling
        fc1 = nn.Dense(
            features=filters_map[4],
            kernel_init=xavier_init
        )
        x = fc1(x)
        x = activation_fn(x)

        # fc2
        fc2 = nn.Dense(
            features=1,
            kernel_init=xavier_init
        )
        x = fc2(x)
        return x
