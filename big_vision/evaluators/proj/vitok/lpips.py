import jax
import h5py
import jax.numpy as jnp
import flax.linen as nn
from .vgg import VGG
from huggingface_hub import hf_hub_download

class LPIPS(nn.Module):    
    def setup(self):
        # Use VGG without rematerialization
        self.vgg = VGG(
            output='activations',
            pretrained='imagenet',
            architecture='vgg16',
            include_head=False,
            #pooling='max_pool'  # Added pooling parameter
        )
        self.lins = [nn.Conv(1, (1,1), strides=None, padding=0, use_bias=False, name=name) for name in ['lins_0', 'lins_1', 'lins_2', 'lins_3', 'lins_4']]
        self.feature_names = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3']
        
    def __call__(self, x, t):
        # Note: Input images should be 224x224 or will be processed by VGG accordingly
        x = self.vgg((x + 1) / 2) # Based on ImageNet Mean and Std
        t = self.vgg((t + 1) / 2)
            
        feats_x, feats_t, diffs = {}, {}, {}
        for i, f in enumerate(self.feature_names):
            feats_x[i] = normalize_tensor(x[f])
            feats_t[i] = normalize_tensor(t[f])
            diffs[i] = (feats_x[i] - feats_t[i]) ** 2

        res = [spatial_average(self.lins[i](diffs[i]), keepdims=True) for i in range(len(self.feature_names))]
        val = sum(res)
        return val

def recursive_load(h5_obj):
    # If it's a dataset, return its data as a NumPy array.
    if isinstance(h5_obj, h5py.Dataset):
        return h5_obj[()]
    # If it's a group, iterate over its items.
    elif isinstance(h5_obj, h5py.Group):
        result = {}
        for key, item in h5_obj.items():
            result[key] = recursive_load(item)
        return result
    else:
        # Fallback: return the object as is.
        return h5_obj

def rename_weights(tree):
    if isinstance(tree, dict):
        # Replace key "weight" with "kernel"
        return {("kernel" if key == "weight" else key): rename_weights(value)
                for key, value in tree.items()}
    elif isinstance(tree, list):
        return [rename_weights(item) for item in tree]
    else:
        return tree

def load(params):  # pylint: disable=invalid-name because we had to CamelCase above.
  """Load init from checkpoint, both old model and this one. +Hi-res posemb."""
  vgg_weights = hf_hub_download(repo_id="pcuenq/lpips-jax", filename="vgg16_weights.h5")
  lpips_lin = hf_hub_download(repo_id="pcuenq/lpips-jax", filename="lpips_lin.h5")
  with jax.transfer_guard("allow"):
    vgg_weights = recursive_load(h5py.File(vgg_weights))
    vgg_weights = rename_weights(vgg_weights)
    vgg_weights = jax.tree.map(jnp.array, vgg_weights)

    lpips_lin = recursive_load(h5py.File(lpips_lin))
    lpips_lin = jax.tree.map(jnp.array, lpips_lin)
  params['vgg'] = vgg_weights
  for i, name in enumerate(['lins_0', 'lins_1', 'lins_2', 'lins_3', 'lins_4']):
    params[name]['kernel'] = lpips_lin[f'lin{i}']
  return params

def normalize_tensor(x, eps=1e-10):
    norm_factor = jnp.sqrt(jnp.sum(x**2, axis=-1, keepdims=True))
    return x / (norm_factor + eps)

def spatial_average(x, keepdims=True):
    return jnp.mean(x, axis=[1, 2], keepdims=keepdims)