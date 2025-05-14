import os
import random
import importlib
import numpy as np
import torch
import torch.distributed as dist
import wandb
import warnings
import vitok.utils as utils
import vitok.evaluators.common as eval_common
from vitok.models.perceptual_networks import Discriminator, LPIPS, hinge_d_loss, compute_lecam_loss, NLayerDiscriminator, NLayerDiscriminator3D
from absl import app
from absl import flags
from ml_collections import config_flags
from tqdm import trange
from copy import deepcopy
from vitok.datasets import create_dataloader_distributed
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from torch.amp import GradScaler, autocast

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

    if args.get('ignore_warning', False):
        warnings.filterwarnings("ignore")

    device = torch.device('cuda')
    seed = args.seed + utils.get_rank() #Starts dataloader in different locations
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    model_mod = importlib.import_module(f"video_compression.models.{args.model_name}")
    model = model_mod.Model(**args.get("model", {}))
    
    if not args.eval_only:
        dataloader = create_dataloader_distributed(args.seed, args.batch_size, args.train_configs, args.get("frame_ratios", None))

    if args.rank == 0:
        wandb.init(project=args.get("wandb_project", None),
                    id=args.get("wandb_id", None),
                    mode=args.get("wandb_mode", "offline"),
                    group=args.output_dir + "/" + args.checkpoint_name, #Group all runs in same wandb incase of multiple runs
                    config=args, dir=workdir)
    
    total_steps = int(args.steps)
    model.to(device)
    if args.get("ema_decay", None):
        print("Using EMA")
        ema = deepcopy(model).to(device)
        utils.requires_grad(ema, False)
    n_parameters = sum(p.numel() for p in model.parameters()
                       if p.requires_grad)
 
    print('number of params: {} M'.format(n_parameters / 1e6))
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % args.batch_size)
    print("Number of training steps = %d" % args.steps)

    if args.finetune_decoder:
        model.finetune_decoder()
    #filter params to only include those that require grad
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay, betas=args.betas)
    #TODO: Try MuP? SOAP? New optimizers!
    needs_scaler = args.precision == "f16" or args.precision == "bf16"
    amp_dtype = torch.float16 if args.precision == "f16" else torch.bfloat16
    scaler = GradScaler(enabled=needs_scaler)

    l2 = torch.nn.MSELoss(reduction="mean")
    l1 = torch.nn.SmoothL1Loss(beta=0.05, reduction="mean")  # Huber loss

    def create_encode_decode(token_count):
        @torch.no_grad()
        def encode_decode(model, video):
            with autocast(dtype=amp_dtype, enabled=needs_scaler, device_type="cuda"):
                x, _ = model(video, sample_posterior=False, token_count=token_count)
            return x.to(torch.float32)
        return encode_decode
    
    eval_fns = {f"encode_compress_{token_count}": create_encode_decode(token_count) for token_count in args.get("eval_token_counts", [16, 32, 64, 128, 192, 256, 512, 1024])}

    def evaluators():
        return eval_common.from_config(
            args, eval_fns
    )
    evalulators = evaluators()
    if args.schedule == "cosine":
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=args.steps,
            cycle_mult=1.0,
            max_lr=args.lr,
            min_lr=1e-5,
            warmup_steps=int(0.03 * args.steps),
        )
    elif args.schedule == "linear":
        scheduler = 1
    else:
        scheduler = None

    train_state = {"model": model, "optimizer": optimizer, "step": 1, "scaler": scaler, "scheduler": scheduler, 'best_metric': None}
    if args.get("ema_decay", None):
        train_state['ema'] = ema
    if args.lpips:
        lpips = LPIPS().to(device).eval()
    
    if args.discriminator:
        if args.discriminator_type=='patch':
            discriminator = NLayerDiscriminator().to(device)
        elif args.discriminator_type=='style':
            discriminator = Discriminator(input_size=args.size).to(device)
        elif args.discriminator_type=='3d_patch':
            discriminator = NLayerDiscriminator3D().to(device)
        else:
            raise NotImplementedError
        discriminator_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=args.discrim_lr, weight_decay=args.weight_decay, betas=args.betas)
        train_state = {**train_state, "discriminator": discriminator, "discriminator_optimizer": discriminator_optimizer}
    
    if os.path.exists(os.path.join(args.output_dir, args.checkpoint_name)): #load weights and optimizer state for continuing training
        print(f"Loading model at {os.path.join(args.output_dir, args.checkpoint_name)}")
        utils.load_checkpoint(train_state, args.output_dir, filename=args.checkpoint_name)
    elif os.path.exists(os.path.join(args.output_dir, args.checkpoint_load)) and args.checkpoint_load: #initialize weights from a different model
        print(f"Loading model at {os.path.join(args.output_dir, args.checkpoint_load)}")
        utils.load_checkpoint({"model": model}, args.output_dir, filename=args.checkpoint_load, strict=False)
        if args.get("ema_decay", None):
            try:
                utils.load_checkpoint({"ema": ema}, args.output_dir, filename=args.checkpoint_load, strict=False)
            except KeyError:
                print("No EMA weights found, copying model weights to EMA")
                ema = deepcopy(model).to(device)
                utils.requires_grad(ema, False)
                train_state['ema'] = ema
    elif not os.path.exists(args.output_dir):
        if args.rank == 0:
            os.makedirs(args.output_dir, exist_ok=True)
    
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.gpu], find_unused_parameters=False)

    if args.discriminator:
        discriminator = torch.nn.parallel.DistributedDataParallel( #Should be small enough to fit on a single GPU
            discriminator, device_ids=[args.gpu], broadcast_buffers=False, find_unused_parameters=False)
        train_state['discriminator'] = discriminator
    
    if args.compile: #Compile after you load weights, otherwise load_checkpoint might bug
        torch._dynamo.config.compiled_autograd = True
        model = torch.compile(model)
        #if args.discriminator:
        #    discriminator = torch.compile(discriminator) #Causes bugs 
        if args.lpips:
            lpips = torch.compile(lpips)
        
        
    if args.eval_only:
        if args.get("ema_decay", None):
            test_model = ema
        else:
            test_model = model
        test_model.eval()
        utils.run_eval(test_model, evalulators, train_state['step'], max_visuals=args.max_visuals, 
        fps=args.get("fps", 12), csv_path=os.path.join(args.output_dir, args.checkpoint_name + "_eval.csv"))
        if args.rank == 0:
            wandb.finish()
        dist.destroy_process_group()
        return

    skip_eval = True
    model.train()
    disable_tqdm = args.rank != 0
    data_iter = iter(dataloader)
    for step in trange(train_state['step'], total_steps + 1, disable=disable_tqdm):
        if args.schedule == "linear":        
            lr = utils.adjust_learning_rate(
                    optimizer,
                    step,
                    int(0.025 * total_steps),
                    total_steps,
                    args.lr,
                )
        else:
            lr = scheduler.get_lr()[0]

        lr = torch.Tensor([lr]).to(device)
        metrics = {}
        train_state['step'] = step

        try:
            inputs, labels = next(data_iter)
        except Exception as e:
            print(e)
            if hasattr(dataloader, 'sampler'):
                dataloader.sampler.set_epoch(step)
            data_iter = iter(dataloader)
            inputs, labels = next(data_iter)
        dist.barrier()

        with autocast(dtype=amp_dtype, enabled=needs_scaler, device_type="cuda"):
            vid_inputs = inputs.to(device)
            vid_preds, posterior = model(inputs)
        
            inputs = utils.reshape_video_to_img_batch(vid_inputs)
            preds = utils.reshape_video_to_img_batch(vid_preds)
    
            l2_loss = l2(preds, inputs)
            l1_loss = l1(preds, inputs)
            metrics['l2_loss'] = l2_loss
            metrics['l1_loss'] = l1_loss

            loss = torch.Tensor([0]).to(device)

            if args.l2_weight:
                loss = l2_loss * args.l2_weight
            
            if args.l1_weight:
                loss = l1_loss * args.l1_weight

            if args.variational:
                kl_loss = posterior.kl().mean()
                metrics['kl_loss'] = kl_loss
                loss += kl_loss * args.kl_weight

            if args.lpips:
                lpips_loss = lpips(preds, inputs).mean()
                metrics['lpips_loss'] = lpips_loss
                loss += lpips_loss * args.lpips
            
            if args.discriminator and step > args.start_gen_loss:
                utils.requires_grad(discriminator.module, False)
                if args.discriminator_type=='3d_patch':
                    g_loss = -discriminator(vid_preds).mean()
                else:
                    g_loss = -discriminator(preds).mean()
                metrics['generator_loss'] = g_loss
                aux_weight = min(1.0, ((step - args.start_gen_loss) / args.gen_loss_warmup_steps))
                loss += args.discriminator * g_loss * aux_weight
                metrics['generator_loss_aux_weight'] = torch.Tensor([aux_weight]).to(device)
                utils.requires_grad(discriminator.module, True)
        
        dist.barrier()

        inputs = inputs.to(torch.float32) #Recast from autocast
        vid_inputs = vid_inputs.to(torch.float32)
        preds = preds.detach().to(torch.float32)
        vid_preds = vid_preds.detach().to(torch.float32)

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        if args.schedule == "cosine":
            scheduler.step()
        if args.get("ema_decay", None):
            utils.update_ema(ema, model.module, decay=args.ema_decay)

        if args.discriminator and step > args.start_discriminator_loss:
            disc_lr = utils.adjust_learning_rate(
                    discriminator_optimizer,
                    step,
                    int(0.025 * (total_steps - args.start_discriminator_loss)),
                    (total_steps - args.start_discriminator_loss),
                    args.discrim_lr,
                )
            discriminator_optimizer.zero_grad() #Make sure prior gradients don't affect the discriminator
            if args.discriminator_type=='3d_patch':
                logits_real = discriminator(vid_inputs)
                logits_fake = discriminator(vid_preds)
            else:
                logits_real = discriminator(inputs)
                logits_fake = discriminator(preds)
                        
            disc_loss = hinge_d_loss(logits_real, logits_fake).mean()
            disc_loss.backward()
            discriminator_optimizer.step()
            metrics = {**metrics, 'logits_real': logits_real, 'logits_fake': logits_fake,
                                  'disc_loss': disc_loss, 'disc_lr': disc_lr}

        if step % args.log_freq == 0 or step == 0:
            mse = (preds - inputs).pow(2).mean()
            mae = (preds - inputs).abs().mean()
            metrics = {"lr": lr, "mse": mse,  "mae": mae, "loss": loss, **metrics}
            metrics = utils.gather(metrics)
            if args.rank == 0:
                wandb.log(metrics, step=step)
        
        if step % args.eval_freq == 0 and not skip_eval:
            if args.get("ema_decay", None):
                test_model = ema
            else:
                test_model = model
            test_model.eval()
            metric = utils.run_eval(test_model, evalulators, step, max_visuals=args.max_visuals, 
                                    fps=args.get("fps", 12), csv_path=os.path.join(args.output_dir, args.checkpoint_name + f"_eval_{step}.csv"))
            dist.barrier()
            if metric is not None:
                if train_state['best_metric'] is None or metric < train_state['best_metric']:
                    train_state['best_metric'] = metric
                    if args.output_dir:
                        print(f"Saving best checkpoint at {step}")
                        utils.save_checkpoint(train_state, args.output_dir, filename=args.checkpoint_name + "_best")
            model.train()

        dist.barrier()
        if step % args.save_ckpt_freq == 0:
            utils.save_checkpoint(train_state, args.output_dir, filename=args.checkpoint_name)
            print(f"Saving model at step {step}")

        dist.barrier()
        torch.cuda.synchronize()
        skip_eval = False
    
    if args.rank == 0:
        wandb.finish()
    dist.destroy_process_group()

if __name__ == "__main__":
    app.run(main)
