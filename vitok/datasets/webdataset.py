import os
import glob
import webdataset as wds
import torch
import random

def nodesplitter(src, group=None):
    if torch.distributed.is_initialized():
        if group is None:
            group = torch.distributed.group.WORLD
        rank = torch.distributed.get_rank(group=group)
        size = torch.distributed.get_world_size(group=group)
        print(f"nodesplitter: rank={rank} size={size}")
        count = 0
        for i, item in enumerate(src):
            if i % size == rank:
                yield item
                count += 1
        print(f"nodesplitter: rank={rank} size={size} count={count} DONE")
    else:
        yield from src

def prepare_webdataset(batch_size, num_workers, bucket_paths, transform, cache_dir=None):
    #Prepares JPG webdataset for video compression by fake padding the images to simulate a video to meet minimum tube length requirements.
    tar_files = []
    for bucket_path in bucket_paths:
        for folder in os.listdir(bucket_path):
            if folder.endswith(".csv"):
                folder_path = os.path.join(bucket_path, folder)
                tar_files.extend(glob.glob(os.path.join(folder_path, "*.tar")))
    
    #Split into train, val - make sure val has at least 1 tar
    def make_sample(sample):
        return transform(sample["jpg"]), 0

    #randomly shuffle the tar files with fixed seed (so its the same train val split every time)
    random_int = random.randint(0, 100000)
    local_random = random.Random(random_int)
    local_random.shuffle(tar_files)
    
    raw_jpg_dataset = wds.WebDataset(tar_files, 
                                     cache_dir=cache_dir,
                                     shardshuffle=True,
                                     resampled=True,
                                     handler=wds.ignore_and_continue,
                                     nodesplitter=wds.split_by_node)
    trainset = raw_jpg_dataset.shuffle(2000).decode("pil").map(make_sample).batched(64)
    trainloader = wds.WebLoader(trainset, batch_size=None, num_workers=num_workers)
    trainloader = trainloader.unbatched().shuffle(2000).batched(batch_size)
    return trainloader