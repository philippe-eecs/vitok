# pylint: disable=line-too-long
import vitok.configs.common as cc
import ml_collections as mlc

def get_config(arg=None):
    """Config for training."""
    arg = cc.parse_arg(arg, variant="S-B/1x16", batch_size=1024, size=256, steps=1e5,
                        lpips=1.0, discriminator=0.0, compile=False, eval=False,
                        name="l2_lpips", load="", model="ae", simple=True, ema=0.0,
                        lr=1e-4, variational=True, finetune_decoder=False, rope=10000.0,
                        posemb='nope', rope_style='1d_axial', max_code_length=256, min_code_length=256, code_width=16, 
                        precision="bf16", disc_type='style', disc_lr=2e-5, schedule="cosine", i1k_only=False,
                        wandb_project="ViTok_Image", wandb=0, local=False)

    config = mlc.ConfigDict()
    config.wandb_project = arg.wandb_project
    config.wandb_id = None
    config.local = arg.local
    config.lpips = arg.lpips
    config.l2_weight = 1.0
    config.l1_weight = 0.1
    config.steps = arg.steps
    config.size = arg.size
    config.ignore_warning = True
    config.finetune_decoder = arg.finetune_decoder
    config.schedule = arg.schedule
    config.variational = arg.variational

    if arg.ema > 0:
        config.ema_decay = arg.ema

    config.checkpoint_name = arg.name
    config.checkpoint_load = arg.load

    config.save_ckpt_freq = int(10000 * (256 / arg.batch_size))
    config.log_freq = int(250 * (256 / arg.batch_size))
    if arg.finetune_decoder:
        config.eval_freq = int(10000 * (256 / arg.batch_size))
    else:
        config.eval_freq = int(50000 * (256 / arg.batch_size))

    config.start_discriminator_loss = int(0 * (256 / arg.batch_size))
    config.start_gen_loss = int(5000 * (256 / arg.batch_size))
    config.gen_loss_warmup_steps = int(50000 * (256 / arg.batch_size))
    config.discriminator = arg.discriminator
    config.discrim_lr = arg.disc_lr * (arg.batch_size / 256)
    config.kl_weight = 1e-3
    config.compile = arg.compile
    config.wandb_mode = "offline" if arg.wandb == 0 else "online"
    config.eval_only = arg.eval
    config.precision = arg.precision
    config.batch_size = arg.batch_size
    config.discriminator_type = arg.disc_type

    variant_dic = decode_variant(arg.variant)

    config.ss_train_data = dict()
    config.ss_train_data.type = "webdataset"
    config.ss_train_data.bucket_paths = ["/fsx-shutterstock-image/dataset/first_cleaned/ss-photo-bucket-2/webdataset_512", "/fsx-shutterstock-image/dataset/second_batch/ss-photo-bucket-2/webdataset_512"]
    config.ss_train_data.num_frames = variant_dic["tubelet_size"]
    config.ss_train_data.train = True
    config.ss_train_data.num_workers = 4
    config.ss_train_data.img_size = arg.size
    config.ss_train_data.cache_dir = "/fsx-project/philippehansen/SS_cache"

    config.imagenet_train_data = dict()
    config.imagenet_train_data.type = "jpeg"
    config.imagenet_train_data.root = "/fsx-project/philippehansen/imagenet/train"
    config.imagenet_train_data.num_frames = variant_dic["tubelet_size"]
    config.imagenet_train_data.train = True
    config.imagenet_train_data.num_workers = 4
    config.imagenet_train_data.img_size = arg.size

    if arg.i1k_only:
        config.train_configs = [config.imagenet_train_data]
        config.frame_ratios = [1.0]
    else:
        config.train_configs = [config.imagenet_train_data, config.ss_train_data]
        config.frame_ratios = [1/8, 7/8]

    config.image_val_data = dict()
    config.image_val_data.type = "jpeg"
    config.image_val_data.root = '/fsx-project/philippehansen/imagenet/val'
    config.image_val_data.num_frames = variant_dic["tubelet_size"]
    config.image_val_data.num_workers = 0
    config.image_val_data.train = False
    config.image_val_data.img_size = arg.size

    config.image_val_data_1 = dict()
    config.image_val_data_1.type = "jpeg"
    config.image_val_data_1.root = '/fsx-project/philippehansen/coco/images/val2017'
    config.image_val_data_1.num_frames = variant_dic["tubelet_size"]
    config.image_val_data_1.no_label = True
    config.image_val_data_1.num_workers = 0
    config.image_val_data_1.train = False
    config.image_val_data_1.img_size = arg.size
    
    if arg.max_code_length > arg.min_code_length:
        train_lengths = [2**i for i in range(int(arg.min_code_length.bit_length()) - 1, int(arg.max_code_length.bit_length()))]
        eval_lengths = train_lengths
    else:
        train_lengths = None
        eval_lengths = [arg.max_code_length]
    
    print(f"Train lengths: {train_lengths}")
    print(f"Eval lengths: {eval_lengths}")

    config.model_name = arg.model
    config.model = dict(
    **variant_dic,
    img_size=arg.size,
    num_frames=variant_dic["tubelet_size"],
    variational=arg.variational,
    code_width=arg.code_width,
    code_length=arg.max_code_length,
    rope_theta=arg.rope,
    rope_style=arg.rope_style,
    lengths=train_lengths,
    posemb=arg.posemb,
    simple=arg.simple,
    checkpoint=False if arg.compile else True,
    )

    config.opt = 'adamw'
    config.weight_decay = 1e-4
    config.betas = (0.9, 0.95)
    config.lr = arg.lr * (arg.batch_size / 256)
    config.clip_grad = 1.0

    def get_imagenet_eval(token_count):
        return dict(
            type='reconstruction',
            config=config.image_val_data,
            prefix=f'reconstruction/i1k/val/{token_count}/',
            log_steps=10000,
            pred=f'encode_compress_{token_count}',
            val_size=50000,
        )
    def get_coco_eval(token_count):
        return dict(
            type='reconstruction',
            config=config.image_val_data_1,
            prefix=f'reconstruction/coco/val/{token_count}/',
            log_steps=10000,
            pred=f'encode_compress_{token_count}',
            val_size=50000,
        )
    image_token_lengths = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    config.eval_token_counts = image_token_lengths
    config.evals = {f"{idx}": get_imagenet_eval(token) for idx, token in enumerate(eval_lengths)}
    config.evals.update({f"{idx + len(config.evals)}": get_coco_eval(token) for idx, token in enumerate(eval_lengths)})
    if arg.eval:
        config.max_visuals = 64
    else:
        config.max_visuals = 4
    return config

def decode_variant(variant):
    """Converts a string "B/4x4 into (tubelet, patch) tuple"""
    if variant is None:
        return {}

    v, rest = variant.split("/")
    enc_v, dec_v = v.split("-")
    tubelet, patch = map(int, rest.split("x"))
    widths = {"S": 768, "B": 768, "L": 1152, "H": 1312}
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
