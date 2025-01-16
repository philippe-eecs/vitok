from vitok.datasets.videodataset import VideoDataset
from vitok.datasets.webdataset import prepare_webdataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from vitok.utils import get_rank, get_world_size
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import os

class NoLabelImage(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = [os.path.join(root, img) for img in os.listdir(root)]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Assign all images a dummy label, e.g., 0
        label = 0
        return image, label

def replicate_padding(image, num_frames=2):
    """
    Replicate the image across the temporal dimension to simulate a video.
    """
    # image shape: [C, H, W]
    # unsqueeze to [C, 1, H, W] and repeat along the temporal dimension
    return image.unsqueeze(1).repeat(1, num_frames, 1, 1)

def get_transform(train=True, image_size=256, num_frames=1, scale=(0.8, 1.0), normalization='-1,1'):
    """
    Returns the appropriate transformation depending on whether it's for training or evaluation.
    """
    start_transforms = []
    if train:
        # Transformations for training
        start_transforms += [
            transforms.RandomResizedCrop(image_size, scale=scale),
            transforms.RandomHorizontalFlip()]
    else:
        # Transformations for evaluation
        start_transforms += [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size)]
    
    start_transforms += [transforms.ToTensor()]
    if normalization == '-1,1':
        start_transforms += [transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)]
    elif normalization == '0,1':
        pass
    elif normalization == 'imagenet':
        start_transforms += [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    else:
        raise ValueError(f"Normalization {normalization} not supported")
        
    return transforms.Compose(start_transforms + [transforms.Lambda(lambda x: replicate_padding(x, num_frames=num_frames))])

def compute_rank_cutoffs(batch_size, configs, frame_ratios):
    world_size = get_world_size()
    num_frames = np.array([config.get("weight_num_frames", 1) for config in configs])
    example_ratios = np.array(frame_ratios) / num_frames
    example_ratios /= sum(example_ratios)
    num_examples = example_ratios * batch_size
    total_num_frames = sum(num_examples * num_frames)
    frames_per_rank = total_num_frames / world_size
    examples_per_rank = frames_per_rank / num_frames
    num_ranks_per_dataset = num_examples / examples_per_rank
    examples_per_rank = np.round(examples_per_rank)
    rank_cutoffs = np.cumsum(num_ranks_per_dataset)
    return rank_cutoffs, examples_per_rank

def create_dataloader(seed, batch_size, config, num_replicas):
    transform = get_transform(train=config['train'], 
                              image_size=config['img_size'], 
                              num_frames=config.get('num_frames', 1), 
                              scale=config.get('scale', (0.8, 1.0)))
    if config['type'] == "video":
        dataset = VideoDataset(
            data_dir=config['root'],
            reference_csv=config.get('reference_csv', ''),
            img_size=config['img_size'],
            new_length=config['num_frames'],
            new_step=config['sampling_rate'],
            train=config['train'],
            data_format=config['data_format'],
        )
        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_replicas, seed=seed, rank=get_rank() % num_replicas, shuffle=True)
        dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=config['num_workers'],
        pin_memory=config['train'],
        drop_last=True,
        persistent_workers=config['train'])

    elif config['type'] == "webdataset":
        dataloader = prepare_webdataset(
            batch_size=batch_size,
            num_workers=config['num_workers'],
            bucket_paths=config['bucket_paths'],
            transform=transform,
            cache_dir=config.get('cache_dir', None))
    elif config['type'] == "jpeg":
        if config.get('no_label', False):
            dataset = NoLabelImage(root=config['root'], transform=transform)
        else:
            dataset = ImageFolder(root=config['root'], transform=transform)
        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_replicas, seed=seed, rank=get_rank() % num_replicas, shuffle=True)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=config['num_workers'],
            pin_memory=config['train'],
            drop_last=True,
            persistent_workers=config['train'])
    else:
        raise ValueError(f"Dataset type {config['type']} not supported")
    
    return dataloader

def create_dataloader_distributed(seed, batch_size, configs, frame_ratios=None):
    assert isinstance(configs, list), "Configs must be a list"
    if len(configs) == 0:
        raise ValueError("Configs must be a non-empty list")
    elif len(configs) == 1:
        return create_dataloader(seed, batch_size // get_world_size(), configs[0], num_replicas=get_world_size())
    else:
        if frame_ratios is None:
            frame_ratios = [1 / len(configs) for _ in range(len(configs))]
        assert len(configs) == len(frame_ratios), "Number of configs must match number of frames"
        assert sum(frame_ratios) == 1, "Frame ratios must sum to 1"
        rank_cutoffs, examples_per_rank = compute_rank_cutoffs(batch_size, configs, frame_ratios)
        print(f"Rank cutoffs: {rank_cutoffs}, Examples per rank: {examples_per_rank}")
        rank = get_rank()
        for i, cutoff in enumerate(rank_cutoffs):
            num_replicas = rank_cutoffs[i] if i == 0 else rank_cutoffs[i] - rank_cutoffs[i - 1]
            if rank < cutoff:
                print(f"Rank {rank} using config {i} with {num_replicas} replicas and {examples_per_rank[i]} examples")
                return create_dataloader(seed, int(examples_per_rank[i]), configs[i], num_replicas=int(num_replicas))
