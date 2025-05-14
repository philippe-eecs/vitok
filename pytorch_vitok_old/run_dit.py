import os
import random
import importlib
from absl import app
from absl import flags
from ml_collections import config_flags
import numpy as np
import torch
from tqdm import trange
from copy import deepcopy
import vitok.utils as utils
import vitok.evaluators.common as eval_common
from vitok.datasets import create_dataloader_distributed
import importlib
import os
import wandb
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
import torch.distributed as dist
from torch.amp import GradScaler, autocast
from vitok.diffusion import create_diffusion
from diffusers.models import AutoencoderKL

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

flags.DEFINE_string("main", default="vit_ae", help="What train main to use")
flags.DEFINE_string("key", default=None, help="Wandb key")
flags.DEFINE_integer("seed", default=42, help="Random seed")
config_flags.DEFINE_config_file(
    "args", None, "Training configuration.", lock_config=False)
flags.DEFINE_string("workdir", default=None, help="Work unit directory.")

import warnings
# Ignore specific deprecation warnings
warnings.filterwarnings("ignore", message="The parameter 'pretrained' is deprecated")
warnings.filterwarnings("ignore", message="Arguments other than a weight enum or `None` for 'weights' are deprecated")
# Ignore all user warnings
warnings.filterwarnings("ignore", category=UserWarning)
# Ignore all warnings from a specific module
warnings.filterwarnings("ignore", module="torchvision.models._utils")

def main(argv):
    del argv

    args = flags.FLAGS.args
    workdir = flags.FLAGS.workdir
    args.output_dir = workdir
    args.seed = flags.FLAGS.seed
    
    rank, world_size, gpu = utils.init_distributed_mode(args.get("local", True))
    args.rank = rank
    args.world_size = world_size
    args.gpu = gpu

    device = torch.device('cuda')
    seed = args.seed + utils.get_rank() #Starts dataloader in different locations
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if args.get("use_cnn", False):
        compression_model = AutoencoderKL.from_pretrained(
            f"stabilityai/sd-vae-ft-ema"
        )
    else:
        model_mod = importlib.import_module(f"video_compression.models.{args.compression_model_name}")
        compression_model = model_mod.Model(**args.get("compression_model", {}))
        try:
            utils.load_checkpoint({"ema": compression_model}, args.output_dir, filename=args.comp_checkpoint_load)
        except KeyError:
            print("No EMA found, fall backing to model")
            utils.load_checkpoint({"model": compression_model}, args.output_dir, filename=args.comp_checkpoint_load)

    model_mod = importlib.import_module(f"video_compression.models.{args.model_name}")
    model = model_mod.Model(**args.get("model", {}))

    if not args.eval_only:
        dataloader = create_dataloader_distributed(args.seed, args.batch_size, args.train_configs, args.get("frame_ratios", None))

    if args.rank == 0:
        wandb.init(project=args.get("wandb_project", None),
                    id=args.get("wandb_id", None),
                    mode=args.get("wandb_mode", "offline"),
                    group=args.output_dir + "/" + args.checkpoint_name, #Group all runs in the same output directory
                    config=args, dir=workdir)
    
    total_steps = int(args.steps)
    utils.requires_grad(compression_model, False)
    model.to(device)
    compression_model.to(device)
    if args.get("ema_decay", None):
        ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
        utils.requires_grad(ema, False)
    n_parameters = sum(p.numel() for p in model.parameters()
                       if p.requires_grad)
 
    print('number of params: {} M'.format(n_parameters / 1e6))
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % args.batch_size)
    print("Number of training steps = %d" % args.steps) 

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=args.betas)
    needs_scaler = args.precision == "f16" or args.precision == "bf16"
    amp_dtype = torch.float16 if args.precision == "f16" else torch.bfloat16
    scaler = GradScaler(enabled=needs_scaler)
    diffusion = create_diffusion(timestep_respacing="") 

    @torch.no_grad()
    def encode(compression_model, x):
        with autocast(dtype=amp_dtype, enabled=needs_scaler, device_type="cuda"):
            if args.get("use_cnn", False):
                preds = compression_model.encode(x.squeeze()).latent_dist.sample()
                preds = utils.patchify(preds) * 0.18215
            else:
                preds, _ = compression_model.encode(x)
                preds = preds.sample()
        return preds.to(torch.float32)

    torch.cuda.empty_cache()
    dist.barrier()
    model.train()

    if args.schedule == "cosine":
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=args.steps,
            cycle_mult=1.0,
            max_lr=args.lr,
            min_lr=args.get("min_lr", 1e-5),
            warmup_steps=int(0.1 * args.steps),
        )
    elif args.schedule == "linear":
        scheduler = 1
    else:
        scheduler = None

    train_state = {"model": model, "optimizer": optimizer, "step": 1, "scaler": scaler, "scheduler": scheduler}
    if args.get("ema_decay", None):
        train_state['ema'] = ema

    if args.checkpoint_name and os.path.exists(os.path.join(args.output_dir, args.checkpoint_name)): #load weights and optimizer state for continuing training
        print(f"Loading model at {os.path.join(args.output_dir, args.checkpoint_name)}")
        utils.load_checkpoint(train_state, args.output_dir, filename=args.checkpoint_name)
    elif args.checkpoint_load and os.path.exists(os.path.join(args.output_dir, args.checkpoint_load)): #
        print(f"Loading model at {os.path.join(args.output_dir, args.checkpoint_load)}")
        utils.load_checkpoint({"model": model}, args.output_dir, filename=args.checkpoint_load, strict=False)
        if args.get("ema_decay", None):
            utils.load_checkpoint({"ema": ema}, args.output_dir, filename=args.checkpoint_load, strict=False)
    elif not os.path.exists(args.output_dir):
        if args.rank == 0:
            os.makedirs(args.output_dir, exist_ok=True)
    
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.gpu], find_unused_parameters=False)
    
    if args.compile: #Compile after you load weights, otherwise load_checkpoint might bug
        torch._dynamo.config.compiled_autograd = True
        model = torch.compile(model)
        compression_model = torch.compile(compression_model)
    
    @torch.no_grad()
    def sample(models, batch_size, cfg_scale=1.5, sampling_steps=250):
        compression_model, model = models
        initial_noise = torch.randn(batch_size, args.num_tokens, args.code_width, device=device)
        ys = torch.randint(0, args.num_classes, (batch_size,), device=device)
        model_kwargs = dict(y=ys, cfg_scale=cfg_scale)
        diffusion_sampler = create_diffusion(timestep_respacing=f"{sampling_steps}")
        if args.get("ema_decay", None):
            def forward_cfg_autocast(*args, **kwargs):
                with autocast(dtype=amp_dtype, enabled=needs_scaler, device_type="cuda"):
                    out = model.forward_cfg(*args, **kwargs)
                return out.to(torch.float32) #Convert to float32 for inference diffusion process
        else:
            def forward_cfg_autocast(*args, **kwargs):
                with autocast(dtype=amp_dtype, enabled=needs_scaler, device_type="cuda"):
                    out = model.module.forward_cfg(*args, **kwargs)
                return out.to(torch.float32) #Convert to float32 for inference diffusion process
        samples = diffusion_sampler.ddim_sample_loop(forward_cfg_autocast, initial_noise.shape, initial_noise, clip_denoised=False, model_kwargs=model_kwargs, device=device)
        with autocast(dtype=amp_dtype, enabled=needs_scaler, device_type="cuda"):
            if args.get("use_cnn", False):
                samples = utils.unpatchify(samples) / 0.18215
                samples = compression_model.decode(samples).sample.unsqueeze(2)
            else:
                samples = compression_model.decode(samples, args.grid_size)
        return samples.to(torch.float32)
    
    @torch.no_grad()
    def decode(compression_model, z):
        with autocast(dtype=amp_dtype, enabled=needs_scaler, device_type="cuda"):
            if args.get("use_cnn", False):
                z = utils.unpatchify(z) / 0.18215
                x = compression_model.decode(z).sample.unsqueeze(2)
            else:
                x = compression_model.decode(z, args.grid_size)
        return x.to(torch.float32)
    
    eval_fns = {'sample': sample}

    def evaluators():
        return eval_common.from_config(
            args, eval_fns
    )
    evalulators = evaluators()

    if args.eval_only:
        if args.get("ema_decay", None):
            test_model = ema
        else:
            test_model = model
        test_model.eval()
        utils.run_eval((compression_model, test_model), evalulators, train_state['step'], max_visuals=args.max_visual,
                       fps=args.get("fps", 12), csv_path=os.path.join(args.output_dir, args.checkpoint_name + "_eval.csv"))
        if args.rank == 0:
            wandb.finish()
        dist.destroy_process_group()
        return

    model.train()
    compression_model.eval()
    if args.get("ema_decay", None):
        ema.eval()
    disable_tqdm = args.rank != 0
    data_iter = iter(dataloader)
    utils.save_checkpoint(train_state, args.output_dir, filename=args.checkpoint_name)
    skip_eval = True
    for step in trange(train_state['step'], total_steps + 1, disable=disable_tqdm):
        if args.schedule == "linear":        
            lr = utils.adjust_learning_rate(
                    optimizer,
                    step,
                    int(0.1 * total_steps),
                    total_steps,
                    args.lr,
                )
        else:
            lr = scheduler.get_lr()[0]

        lr = torch.Tensor([lr]).to(device)
        metrics = {}
        train_state['step'] = step
        metrics['step'] = torch.Tensor([step]).to(device)
        try:
            x, y = next(data_iter)
        except Exception as e:
            print(e)
            if hasattr(dataloader, 'sampler'):
                dataloader.sampler.set_epoch(step)
            data_iter = iter(dataloader)
            x, y = next(data_iter)
        dist.barrier()
        with autocast(dtype=amp_dtype, enabled=needs_scaler, device_type="cuda"):
            with torch.no_grad():
                x = x.to(device)
                y = y.to(device)
                x = encode(compression_model, x)
                if not args.get("use_cnn", False):
                    z = x
                    z_trunc = z[:, :model.module.num_tokens]
                else:
                    z = x
                    z_trunc = z
            t = torch.randint(0, diffusion.num_timesteps, (z_trunc.shape[0],), device=device)
            model_kwargs = dict(y=y, train=True)
            loss_dict = diffusion.training_losses(model, z_trunc, t, model_kwargs=model_kwargs)
            loss = loss_dict["loss"].mean()
            mse = loss_dict["mse"].mean()
            vb = loss_dict["vb"].mean()
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        if args.schedule == "cosine":
            scheduler.step()
        if args.get("ema_decay", None):
            utils.update_ema(ema, model.module)
        
        if step % args.log_freq == 0 or step == 1:
            metrics = {"lr": lr, "loss": loss, 'mean_z': z_trunc.mean(), 
                       'std_z': z_trunc.std(), "mse": mse, "vb": vb, **metrics}
            metrics = utils.gather(metrics)
            if args.rank == 0:
                wandb.log(metrics, step=step)
        
        if step % args.eval_freq == 0 and not skip_eval:
            if args.get("ema_decay", None):
                test_model = ema
            else:
                test_model = model
            test_model.eval()
            utils.run_eval((compression_model, test_model), evalulators, step, max_visuals=args.max_visuals, 
                           fps=args.get("fps", 12), csv_path=os.path.join(args.output_dir, args.checkpoint_name + f"_eval_{step}.csv"))
            model.train()
        else:
            skip_eval = False
        
        if step % args.sample_freq == 0 or step == 1:
            model.eval()
            count = 64 // utils.get_world_size()
            if args.get("use_cnn", False):
                z_trunc = z_trunc.detach()[:count]
                xt = loss_dict["x_t"].detach()[:count]
                xstart = loss_dict["pred_xstart"].detach()[:count]
            else:
                z_trunc = z_trunc.detach()[:count]
                xt = loss_dict["x_t"].detach()[:count]
                xstart = loss_dict["pred_xstart"].detach()[:count]

            decode_xt = decode(compression_model, xt)
            decode_truth = decode(compression_model, z_trunc)
            decode_xstart = decode(compression_model, xstart)
            if args.get("ema_decay", None):
                test_model = ema
            else:
                test_model = model

            test_model.eval()
            samples = sample((compression_model, test_model), count, cfg_scale=3.0, sampling_steps=250)
            
            samples = utils.postprocess_video(samples)
            decode_xt = utils.postprocess_video(decode_xt)
            decode_truth = utils.postprocess_video(decode_truth)
            decode_xstart = utils.postprocess_video(decode_xstart)

            samples = utils.gather(samples)
            decode_xt = utils.gather(decode_xt)
            decode_truth = utils.gather(decode_truth)
            decode_xstart = utils.gather(decode_xstart)

            if args.rank == 0:
                wandb.log({"eval/samples" : wandb.Video(samples)})
                for i, (x, y, z) in enumerate(zip(decode_truth, decode_xstart, decode_xt)):
                    if i < args.max_visuals:
                        video_tensor = np.stack([x, y, z], axis=0)
                        wandb.log({f"eval/decode_truth_{i}": wandb.Video(video_tensor)})
            model.train()
        
        if args.output_dir and step % args.save_ckpt_freq == 0:
            print(f"Saving model at step {step}")
            utils.save_checkpoint(train_state, args.output_dir, filename=args.checkpoint_name + f"_{step}")
        dist.barrier()
        torch.cuda.synchronize()
    
    if args.rank == 0:
        wandb.finish()
    dist.destroy_process_group()

if __name__ == "__main__":
    app.run(main)