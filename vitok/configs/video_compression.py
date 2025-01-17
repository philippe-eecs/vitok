# pylint: disable=line-too-long
import vitok.configs.common as cc
import ml_collections as mlc

def get_config(arg=None):
    """Config for training."""
    arg = cc.parse_arg(arg, variant="S-B/4x16", batch_size=256, size=256, steps=1e5,
                        lpips=0.1, discriminator=0.0, compile=False, eval=False,
                        name="l2_lpips", load="", model="ae", block_casual=True,
                        lr=3e-4, disc_lr=1e-5, variational=True, finetune_decoder=False,
                        min_code_length=256, max_code_length=1024, code_width=16, precision="bf16",
                        disc_type='style', schedule="cosine", ema=0.0, num_frames=24, fps=12,
                        wandb_project="Vid_Cmp", wandb=0, local=False)
                        

    config = mlc.ConfigDict()
    config.wandb_project = arg.wandb_project
    config.wandb_id = None
    config.local = arg.local
    config.size = arg.size
    config.lpips = arg.lpips
    config.steps = arg.steps
    config.finetune_decoder = arg.finetune_decoder

    if arg.ema > 0:
        config.ema_decay = arg.ema

    config.checkpoint_name = arg.name
    config.checkpoint_load = arg.load
    config.schedule = arg.schedule

    config.save_ckpt_freq = int(5000 * (256 / arg.batch_size))
    config.log_freq = int(250 * (256 / arg.batch_size))
    config.eval_freq = int(25000 * (256 / arg.batch_size))

    config.start_discriminator_loss = int(0 * (256 / arg.batch_size))
    config.start_gen_loss = int(2500 * (256 / arg.batch_size))
    config.gen_loss_warmup_steps = int(25000 * (256 / arg.batch_size))
    config.discriminator = arg.discriminator
    config.discrim_lr = arg.disc_lr

    config.kl_weight = 3e-4 * (4096 / (((arg.max_code_length + arg.min_code_length) / 2) * arg.code_width))
    config.compile = arg.compile
    config.wandb_mode = "offline" if arg.wandb == 0 else "online"
    config.eval_only = arg.eval
    config.precision = arg.precision
    config.loss_fn = "l2" #Smooth l1 also works well
    config.batch_size = arg.batch_size
    config.discriminator_type = arg.disc_type

    variant_dic = decode_variant(arg.variant)

    config.video_train_data_1 = dict()
    config.video_train_data_1.type = "video"
    config.video_train_data_1.data_format = "shutterstock"
    config.video_train_data_1.root = "/fsx-muvigen"
    config.video_train_data_1.reference_csv = "/data/home/philippehansen/shutterstock_csvs/train_shutterstock_paths_only.csv"
    config.video_train_data_1.num_frames = arg.num_frames
    config.video_train_data_1.weight_num_frames = arg.num_frames * 2
    config.video_train_data_1.sampling_rate = int(24 / arg.fps)
    config.video_train_data_1.train = True
    config.video_train_data_1.num_workers = 4
    config.video_train_data_1.img_size = arg.size

    config.video_train_data_2 = dict()
    config.video_train_data_2.type = "video"
    config.video_train_data_2.data_format = "kinetics"
    config.video_train_data_2.root = "/fsx-project/philippehansen/kinetics-dataset/k700-2020/train"
    config.video_train_data_2.reference_csv = "/fsx-project/philippehansen/kinetics-dataset/k700-2020/annotations/train_with_integer_labels.csv"
    config.video_train_data_2.num_frames = arg.num_frames
    config.video_train_data_2.weight_num_frames = arg.num_frames * 2
    config.video_train_data_2.sampling_rate = 2
    config.video_train_data_2.train = True
    config.video_train_data_2.num_workers = 4
    config.video_train_data_2.img_size = arg.size
    config.video_train_data_2.distributed_sampler = True
    config.video_train_data_2.kinetics_path = True

    if arg.wandb == 0:
        config.train_configs = [config.video_train_data_1]
    else:
        config.train_configs = [config.video_train_data_1, config.video_train_data_2]
        config.frame_ratios = [7/8, 1/8]

    config.video_val_data_1 = dict()
    config.video_val_data_1.type = "video"
    config.video_val_data_1.data_format = "kinetics"
    config.video_val_data_1.root = "/fsx-project/philippehansen/kinetics-dataset/k700-2020/val"
    config.video_val_data_1.reference_csv = "/fsx-project/philippehansen/kinetics-dataset/k700-2020/annotations/val_with_integer_labels.csv"
    config.video_val_data_1.num_frames = arg.num_frames
    config.video_val_data_1.sampling_rate = 2
    config.video_val_data_1.train = False
    config.video_val_data_1.num_workers = 1
    config.video_val_data_1.img_size = arg.size
    config.video_val_data_1.distributed_sampler = True
    config.video_val_data_1.kinetics_path = True

    config.video_val_data_2 = dict()
    config.video_val_data_2.type = "video"
    config.video_val_data_2.data_format = "ucf101"
    config.video_val_data_2.root = "/fsx-project/philippehansen/ucf101/UCF-101"
    config.video_val_data_2.num_frames = arg.num_frames
    config.video_val_data_2.weight_num_frames = arg.num_frames
    config.video_val_data_2.sampling_rate = int(25 / arg.fps)
    config.video_val_data_2.train = False
    config.video_val_data_2.num_workers = 1
    config.video_val_data_2.img_size = arg.size

    config.video_val_data_3 = dict()
    config.video_val_data_3.type = "video"
    config.video_val_data_3.data_format = "shutterstock"
    config.video_val_data_3.root = "/fsx-muvigen"
    config.video_val_data_3.reference_csv = "/data/home/philippehansen/shutterstock_csvs/train_shutterstock_paths_only.csv"
    config.video_val_data_3.num_frames = arg.num_frames
    config.video_val_data_3.weight_num_frames = arg.num_frames * 2
    config.video_val_data_3.sampling_rate = int(24 / arg.fps)
    config.video_val_data_3.train = False
    config.video_val_data_3.num_workers = 1
    config.video_val_data_3.img_size = arg.size

    config.model_name = arg.model
    config.model = dict(
        **variant_dic,
        img_size=arg.size,
        num_frames=arg.num_frames,
        variational=arg.variational,
        code_width=arg.code_width,
        max_code_length=arg.max_code_length,
        rope_theta=10000.0,
        rope_style='1d_axial',
        posemb='sincos3d',
        simple=True,
        block_casual=arg.block_casual,
        checkpoint=False if arg.compile else True,
    )

    config.opt = 'adamw'
    config.weight_decay = 1e-4
    config.betas = (0.9, 0.95)
    config.lr = arg.lr * (arg.batch_size / 256)
    config.clip_grad = 1.0
    
    def get_kinetics_eval(token_count):
        return dict(
            type='reconstruction',
            config=config.video_val_data_1,
            compute_fvd=True,
            log_steps=10000,
            prefix=f'reconstruction/kinetics/val/{token_count}/',
            pred=f'encode_compress_{token_count}',
            val_size=50000,
        )
    
    def get_ucf101_eval(token_count):
        return dict(
            type='reconstruction',
            config=config.video_val_data_2,
            compute_fvd=True,
            log_steps=10000,
            prefix=f'reconstruction/ucf101/train/{token_count}/',
            pred=f'encode_compress_{token_count}',
            val_size=50000,
        )
    
    def get_shutterstock_eval(token_count):
        return dict(
            type='reconstruction',
            config=config.video_val_data_3,
            compute_fvd=True,
            log_steps=10000,
            prefix=f'reconstruction/shutterstock/train/{token_count}/',
            pred=f'encode_compress_{token_count}',
            val_size=256,
        )

    token_lengths = [256, 512, 1024, 2048, 4096, 8192, 16384]
    config.eval_token_counts = token_lengths
    config.evals = {}
    config.evals = {f"{idx}": get_kinetics_eval(token) for idx, token in enumerate(token_lengths) if token <= arg.max_code_length and token >= arg.min_code_length}
    config.evals.update({f"{idx + len(config.evals)}": get_ucf101_eval(token) for idx, token in enumerate(token_lengths) if token <= arg.max_code_length and token >= arg.min_code_length})
    config.evals.update({f"{idx + len(config.evals)}": get_shutterstock_eval(token) for idx, token in enumerate(token_lengths) if token <= arg.max_code_length and token >= arg.min_code_length})
    config.max_visuals = 4
    return config

def decode_variant(variant):
    """Converts a string "B/4x4 into (tubelet, patch) tuple"""
    if variant is None:
        return {}

    v, rest = variant.split("/")
    enc_v, dec_v = v.split("-")
    tubelet, patch = map(int, rest.split("x"))
    widths = {"S": 768, "B": 768, "L": 1152, "H": 1296}
    depths = {"S": 6, "B": 12, "L": 24, "H": 32}
    heads = {"S": 12, "B": 12, "L": 16, "H": 16}

    return {
        "width": widths[dec_v],
        "encoder_depth": depths[enc_v],
        "decoder_depth": depths[dec_v],
        "num_heads": heads[dec_v],
        "tubelet_size": tubelet,
        "patch_size": patch,
    }