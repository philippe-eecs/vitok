### ViTok Code Repo!
Star/watch the repo! Weights and more updates coming soon!

### Launching 256p Image Compression Experiments

```
MASTER_PORT=01234 torchrun --nproc_per_node=8 -m vitok.run_ae --args vitok/configs/image_compression.py:variant=S-B/1x16,wandb=1 --workdir checkpoints/ViTok_S-B_16_256p
```

### Launching 512p Image Compression Experiments

```
MASTER_PORT=01234 torchrun --nproc_per_node=8 -m vitok.run_ae --args vitok/configs/image_compression.py:variant=S-B/1x16,wandb=1,size=512 --workdir checkpoints/ViTok_S-B_16_512p
```

### Launching i1k Generation experiments

```
MASTER_PORT=01234 torchrun --nproc_per_node=8 -m vitok.run_dit --args vitok/configs/i1k_generation.py:variant=L/256,compression_variant=S-B/1x16,wandb=1,name=dit_L --workdir checkpoints/ViTok_S-B_16_256p
```