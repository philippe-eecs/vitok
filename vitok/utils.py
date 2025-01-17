import os
import random

import numpy as np
import torch
import torch.distributed as dist
import wandb
from collections import OrderedDict
from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd
import math

class NumpyDataset(Dataset):
    def __init__(self, root: str):
        super().__init__()
        # Assuming all your .npz files are in the root directory
        self.file_paths = [os.path.join(root, f) for f in os.listdir(root) if f.endswith('.npz') and f != "stats.npz"]
        # Optionally, you can handle labels here if they are not part of the .npz files
    def __len__(self):
        return len(self.file_paths)
    def __getitem__(self, index: int):
        path = self.file_paths[index]
        try:
            npz = np.load(path)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            rand_idx = np.random.randint(0, self.__len__())
            return self.__getitem__(rand_idx)
        z, y = npz["latents"], npz["labels"]
        z = torch.from_numpy(z).float()
        y = torch.from_numpy(y).long()
        return z, y

def adjust_learning_rate(
    optimizer,
    step,
    warmup_steps,
    total_steps,
    learning_rate,
    schedule="const",
    min_lr=0,
):
    """Constant after warmup"""
    if step < warmup_steps:
        lr = max(learning_rate * step / warmup_steps, min_lr)
    elif schedule == "const":
        lr = max(learning_rate, min_lr)
    elif schedule == "cos":
        lr = min_lr + (learning_rate - min_lr) * 0.5 * (
            1.0
            + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps))
        )
    else:
        raise NotImplementedError

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return torch.Tensor([lr]).to('cuda')

def run_eval(model, evaluators, step, max_visuals=5, fps=12, csv_path=''):
    fid = None
    avg_main_metric = 0
    if get_rank() == 0 and csv_path:
        results_df = pd.DataFrame(columns=['Prefix', 'Key', 'Value'])
    for (name, evaluator, _, prefix) in evaluators:
        eval_results = evaluator.run(model)
        if get_rank() == 0:
            for key, value in eval_results.items():
                print(f"Logging {prefix}{key} at step {step}")
                if "wandbvideo" in key:
                    for i in range(min(len(value), max_visuals)):
                        vid = wandb.Video(value[i], fps=fps)
                        wandb.log({f"{prefix}{key}/{i}": vid}, step=step)
                else:
                    wandb.log({f"{prefix}{key}": value}, step=step)
                    if csv_path:
                        new_row = pd.DataFrame({'Prefix': [prefix], 'Key': [key], 'Value': [value]})
                        results_df = pd.concat([results_df, new_row], ignore_index=True)
        if "main_metric" in eval_results.keys() and eval_results["main_metric"] is not None:
            avg_main_metric += eval_results["main_metric"]
        dist.barrier()
    if get_rank() == 0 and csv_path:
        results_df.to_csv(csv_path, index=False)
    return avg_main_metric

def patchify(x, patch_size=2): #Hardcoded for 32x32x4 images to 256 x 16
    bsz = x.shape[0]
    x = x.reshape(shape=(bsz, 4, 16, 2, 16, 2))
    x = torch.einsum("nchpwq->nhwpqc", x)
    x = x.reshape(shape=(bsz, 256, 16))
    return x

def unpatchify(x):
    bsz = x.shape[0]
    x = x.reshape(shape=(bsz, 16, 16, 2, 2, 4))
    x = torch.einsum("nhwpqc->nchpwq", x)
    x = x.reshape(shape=(bsz, 4, 32, 32))
    return x

def postprocess_video(videos: torch.Tensor):
    assert videos.ndim == 5
    assert videos.dtype == torch.float32
    videos = videos.permute(0, 2, 1, 3, 4) # B, C, F, H, W -> B, F, C, H, W
    videos = videos * 0.5 + 0.5
    videos = torch.clamp(videos * 255, 0, 255).to(torch.uint8)
    return videos

def reshape_video_to_img_batch(x: torch.Tensor):
    assert x.ndim == 5
    x_reshape = x.permute(0, 2, 1, 3, 4) # B, C, F, H, W -> B, F, C, H, W
    bs, frames, ch, h, w = x_reshape.shape
    x_reshape = x_reshape.contiguous().view(bs * frames, ch, h, w)
    return x_reshape

def save_checkpoint(train_state, workdir, filename="checkpoint_CURRENT"):
    save_path = os.path.join(workdir, filename)
    assert 'model' in train_state.keys() or 'ema' in train_state.keys()
    if is_main_process():
        state_dict = {}
        for key, value in train_state.items():
            if hasattr(value, 'state_dict'):
                try:
                    state_dict[key] = value.module.state_dict()
                except AttributeError:
                    state_dict[key] = value.state_dict()
            else:
                state_dict[key] = value
        torch.save(state_dict, save_path)
    print(f"Checkpoint saved to {save_path}")
    dist.barrier()

def load_checkpoint(train_state, workdir, filename="checkpoint_CURRENT", device=torch.device('cuda'), strict=True):
    assert 'model' in train_state.keys() or 'ema' in train_state.keys()
    load_path = os.path.join(workdir, filename)
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"No checkpoint found at {load_path}")
    checkpoint = torch.load(load_path, map_location=device)
    for key, value in train_state.items():
        if key in checkpoint:
            if hasattr(value, 'load_state_dict') and isinstance(checkpoint[key], dict):
                if not strict:
                    print(f"Loading model state dict from {load_path} with strict=False")
                    loaded_state_dict = checkpoint[key]
                    current_state_dict = value.state_dict()
                    for k in list(loaded_state_dict.keys()):
                        if k not in current_state_dict or current_state_dict[k].shape != loaded_state_dict[k].shape:
                            del loaded_state_dict[k]
                    value.load_state_dict(loaded_state_dict, strict=False)
                else:
                    value.load_state_dict(checkpoint[key])
            else:
                train_state[key] = checkpoint[key]
        else:
            print(f"Key {key} not found in checkpoint")
            raise KeyError(f"Key {key} not found in checkpoint")
    
    print(f"Checkpoint loaded from {load_path}")

def collect_and_save_latents(predict_fn, model, dataloader, path, stats_only=False, epochs=2):
    idx = 0
    mean = 0
    std = 0
    count = 0
    latents = []
    labels = []
    #Compute the number of samples each process will save
    samples_per_proc = (len(dataloader) * epochs * dataloader.batch_size + 1)
    print(f"Collecting {samples_per_proc} samples per process")
    for epoch in range(epochs):
        seed = torch.randint(0, 2**32, (1,)).item()
        dataloader.sampler.set_epoch(seed)
        data_iter = iter(dataloader)
        for x, y in tqdm(data_iter, disable=get_rank() != 0):
            x = x.to('cuda')
            y = y.to('cuda')
            z = predict_fn(model, x)
            if not stats_only:
                latents.append(z.cpu())
                labels.append(y.cpu())
            mean += gather(z.mean())
            std += gather(z.std())
            count += 1
            # Save an individual npz file for each epoch
            if count % 100 == 0 and not stats_only:
                # Make sure idx is unique across all processes
                z = torch.cat(latents, dim=0).numpy()
                y = torch.cat(labels, dim=0).numpy()
                for z_, y_ in zip(z, y):
                    save_idx = idx + get_rank() * samples_per_proc
                    save_path = f"{path}/cached_sample_{save_idx}.npz"
                    if not os.path.exists(save_path):
                        np.savez(save_path, latents=z_, labels=y_)
                    idx += 1
                del z, y, latents, labels
                latents = []
                labels = []
            dist.barrier()
    return mean / count, std / count

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def gather(data, group=dist.group.WORLD, mode='mean'):
    def gather_tensor(tensor):
        tensor = tensor  # Ensure that the tensor is contiguous
        if not is_dist_avail_and_initialized():
            return tensor
        world_size = dist.get_world_size(group=group)
        if world_size == 1:
            return tensor.cpu().numpy()
        if tensor.numel() > 1:
            gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
            dist.all_gather(gathered_tensors, tensor, group=group)
            concatenated = torch.cat(gathered_tensors, dim=0).cpu().numpy()
            return concatenated
        elif tensor.numel() == 1:  # Want to average a scalar over all ranks
            gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
            dist.all_gather(gathered_tensors, tensor, group=group)
            if mode == 'mean':
                value = torch.stack(gathered_tensors).mean().item()
            elif mode == 'sum':
                value = torch.stack(gathered_tensors).sum().item()
            else:
                raise ValueError("mode must be either 'mean' or 'sum'")
            return value
        else:
            raise TypeError("Unsupported data type. Only tensors are supported.")
    if isinstance(data, (tuple, list)):
        return [gather(tensor) for tensor in data]
    elif isinstance(data, dict):
        return {key: gather(tensor) for key, tensor in data.items()}
    elif isinstance(data, torch.Tensor):
        assert data.device.type == "cuda", "Each tensor must be on cuda"
        return gather_tensor(data.detach().contiguous())
    else:
        raise TypeError("Unsupported data type. Only tensors, tuples, lists and dictionaries are supported.")

def is_main_process():
    return get_rank() == 0

@torch.no_grad()
def update_ema(ema_model, model, decay=0.999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def collect_latents(predict_fn, model, dataloader, epochs=1):
    encodings = []
    labels = []
    seed = torch.randint(0, 2**32, (1,)).item()
    dataloader.sampler.set_epoch(seed)
    data_iter = iter(dataloader)
    for x, y in tqdm(data_iter, disable=get_rank() != 0):
        x = x.to('cuda')
        y = y.to('cuda')
        z = predict_fn(model, x)
        encodings.append(gather(z))
        labels.append(gather(y))
        dist.barrier()
    encodings = np.concatenate(encodings, axis=0)
    labels = np.concatenate(labels, axis=0)
    return encodings, labels

def init_distributed_mode(local):
    if not local:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        gpu = 0

    torch.cuda.set_device(gpu)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    print(f"Initialized process group: rank {rank}, world size {world_size}, GPU {gpu}")
    torch.cuda.empty_cache()
    dist.barrier()
    setup_print(rank == 0)
    return rank, world_size, gpu

def setup_print(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


import ctypes
import fnmatch
import importlib
import inspect
import numpy as np
import os
import shutil
import sys
import types
import io
import pickle
import re
import requests
import html
import hashlib
import glob
import tempfile
import urllib
import urllib.request
import uuid

from distutils.util import strtobool
from typing import Any, List, Tuple, Union, Dict

def open_url(url: str, num_attempts: int = 10, verbose: bool = True, return_filename: bool = False) -> Any:
    """Download the given URL and return a binary-mode file object to access the data."""
    assert num_attempts >= 1

    # Doesn't look like an URL scheme so interpret it as a local filename.
    if not re.match('^[a-z]+://', url):
        return url if return_filename else open(url, "rb")

    if url.startswith('file://'):
        filename = urllib.parse.urlparse(url).path
        if re.match(r'^/[a-zA-Z]:', filename):
            filename = filename[1:]
        return filename if return_filename else open(filename, "rb")

    url_md5 = hashlib.md5(url.encode("utf-8")).hexdigest()

    # Download.
    url_name = None
    url_data = None
    with requests.Session() as session:
        if verbose:
            print("Downloading %s ..." % url, end="", flush=True)
        for attempts_left in reversed(range(num_attempts)):
            try:
                with session.get(url) as res:
                    res.raise_for_status()
                    if len(res.content) == 0:
                        raise IOError("No data received")

                    if len(res.content) < 8192:
                        content_str = res.content.decode("utf-8")
                        if "download_warning" in res.headers.get("Set-Cookie", ""):
                            links = [html.unescape(link) for link in content_str.split('"') if "export=download" in link]
                            if len(links) == 1:
                                url = requests.compat.urljoin(url, links[0])
                                raise IOError("Google Drive virus checker nag")
                        if "Google Drive - Quota exceeded" in content_str:
                            raise IOError("Google Drive download quota exceeded -- please try again later")

                    match = re.search(r'filename="([^"]*)"', res.headers.get("Content-Disposition", ""))
                    url_name = match[1] if match else url
                    url_data = res.content
                    if verbose:
                        print(" done")
                    break
            except KeyboardInterrupt:
                raise
            except:
                if not attempts_left:
                    if verbose:
                        print(" failed")
                    raise
                if verbose:
                    print(".", end="", flush=True)

    # Return data as file object.
    assert not return_filename
    return io.BytesIO(url_data)