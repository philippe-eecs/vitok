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

"""Utility functions."""

import jax
from jax.experimental import shard_map
from jax.interpreters import pxla
import flax.linen as nn

from jax.sharding import PartitionSpec as PS  # logical-axis helper
from jax.core import Tracer # For robust type checking of JAX tracers

P = PS


def batch_shmap(fn, *args, **kwargs):
  """Shard map to map along the data dimension w/o triggering communication."""

  mesh = pxla.thread_resources.env.physical_mesh
  if not mesh.empty:
    devices_flat = mesh.devices.flatten()
    mesh_flat = jax.sharding.Mesh(devices_flat, ("data",))
    fn = shard_map.shard_map(
        fn,
        mesh=mesh_flat,
        in_specs=P("data"), out_specs=P("data"), check_rep=True)
  return fn(*args, **kwargs)


def subsample_batch(x, subsample: int):
  """Shard map to subsample the data dimension w/o triggering communication."""
  fn = lambda x: jax.tree.map(lambda xx: xx[::subsample], x)
  return batch_shmap(fn, x) if subsample > 1 else x


def sequence_parallel_attention(attn_core):
    """Wrap an attention kernel so it works with a sequence-sharded mesh.

    * Q stays local (already sharded on 'seq').
    * K and V are all-gathered across mesh axis **'seq'** so every shard
      sees the full context (or full sliding window).
    * The output keeps the original ('data','seq',…) layout.
    * Logical-axis assertions catch silent resharding bugs.
    """
    _axes = ('act_batch', 'act_len', 'head', 'dim')   # (B, T, H, Dh)

    def wrapped(q, k, v, *args, **kwargs):
        q_orig_type = type(q) # For debugging print
        # Apply logical constraints first, q, k, v will be tracers if in abstract context.
        q = nn.with_logical_constraint(q, _axes)
        k = nn.with_logical_constraint(k, _axes)
        v = nn.with_logical_constraint(v, _axes)

        # Check if q is a JAX tracer (like ShapedArray or DynamicJaxprTracer)
        # which indicates we are in an abstract evaluation context like eval_shape or jit tracing.
        # We use `q` after with_logical_constraint as that's the object whose properties we care about here.
        is_abstract_evaluation = isinstance(q, Tracer)

        if is_abstract_evaluation:
            if jax.process_index() == 0:
                print(f"DEBUG sequence_parallel_attention: Bypassing all_gather for abstract q_orig_type: {q_orig_type}, q type after constraint: {type(q)}", flush=True)
            k_full = k
            v_full = v
        else:
            if jax.process_index() == 0:
                print(f"DEBUG sequence_parallel_attention: Performing all_gather for concrete q_orig_type: {q_orig_type}, q type after constraint: {type(q)}", flush=True)
            k_full = jax.lax.all_gather(k, 'seq', axis=1, tiled=True)
            v_full = jax.lax.all_gather(v, 'seq', axis=1, tiled=True)

        # keep logical labels after the gather (or bypass)
        k_full = nn.with_logical_constraint(k_full, _axes)
        v_full = nn.with_logical_constraint(v_full, _axes)

        # ── attention ─────────────────────────────────────────────────────────
        out = attn_core(q, k_full, v_full, *args, **kwargs)

        # output stays sharded on ('data','seq',…)
        out = nn.with_logical_constraint(out, _axes)
        return out

    return wrapped
