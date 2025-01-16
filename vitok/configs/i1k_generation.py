# pylint: disable=line-too-long
import vitok.configs.common as cc
import ml_collections as mlc

def get_config(arg=None):
    """Config for training."""
    arg = cc.parse_arg(arg, variant="XL/256", compression_variant="S-B/1x16", 
                        batch_size=1024, size=256, steps=2e6, predict_xstart=False,
                        compile=False, eval=False, use_cnn=False, 
                        code_width=16, code_length=256, ema=0.9999, lr=1e-4,
                        precision="bf16", variational=True, cache=False,
                        wandb_project="ViTok_i1k_Generation", wandb=0, local=False, schedule="cosine",
                        name="dit_XL_256", load="", comp_load="discrim_base")

    config = mlc.ConfigDict()
    config.wandb_project = arg.wandb_project
    config.wandb_id = None
    config.local = arg.local
    config.steps = int(arg.steps)
    
    if arg.ema > 0:
        config.ema_decay = arg.ema
    else:
        config.ema_decay = None
    
    config.schedule = arg.schedule
    config.min_lr = 3e-5

    config.num_classes = 1000
    if arg.use_cnn and arg.cache:
        config.cache_path = f'/fsx-project/philippehansen/imagenet/cache/sd_cnn_r'
    elif arg.cache:  
        config.cache_path = f'/fsx-project/philippehansen/imagenet/cache/{arg.compression_variant}/{arg.load}'
    else:
        config.cache_path = None

    config.predict_xstart = arg.predict_xstart
    config.use_cnn = arg.use_cnn

    config.checkpoint_name = arg.name
    config.checkpoint_load = arg.load
    config.comp_checkpoint_load = arg.comp_load

    config.save_ckpt_freq = int(50000 * (256 / arg.batch_size))
    config.log_freq = int(100 * (256 / arg.batch_size))
    config.sample_freq = int(50000 * (256 / arg.batch_size))
    config.eval_freq = int(500000 * (256 / arg.batch_size))

    config.compile = arg.compile
    config.wandb_mode = "offline" if arg.wandb == 0 else "online"
    config.eval_only = arg.eval
    config.precision = arg.precision
    config.batch_size = arg.batch_size

    variant_dic = decode_variant(arg.variant)
    compression_variant_dic = decode_compression_variant(arg.compression_variant)

    config.train_data = dict()
    config.train_data.type = "jpeg"
    config.train_data.root = '/fsx-project/philippehansen/imagenet/train'
    config.train_data.num_frames = compression_variant_dic["tubelet_size"]
    config.train_data.train = True
    config.train_data.num_workers = 4
    config.train_data.img_size = arg.size

    config.train_configs = [config.train_data]
    config.val_data = dict()
    config.val_data.type = "jpeg"
    config.val_data.root = '/fsx-project/philippehansen/imagenet/val'
    config.val_data.num_frames = compression_variant_dic["tubelet_size"]
    config.val_data.num_workers = 1
    config.val_data.train = True
    config.val_data.img_size = arg.size

    config.code_width = arg.code_width
    config.num_tokens = variant_dic["num_tokens"]
    
    if not arg.use_cnn:
        config.compression_model_name = 'ae' #generic vit based auto-encoder
        config.compression_model = dict(
            **compression_variant_dic,
            img_size=arg.size,
            num_frames=compression_variant_dic["tubelet_size"],
            variational=arg.variational,
            code_width=arg.code_width,
            code_length=arg.code_length,
            rope_theta=10000.0,
            rope_style='1d_axial',
            posemb='nope',
            simple=True,
            checkpoint=False if arg.compile else True, #Compile bugged with checkpointing + DDP
        )
        config.num_tokens = variant_dic["num_tokens"]
        config.code_width = arg.code_width
    else:
        variant_dic["num_tokens"] = 256
        config.code_width = 16
    
    config.grid_size = (compression_variant_dic["tubelet_size"], arg.size // compression_variant_dic["patch_size"], arg.size // compression_variant_dic["patch_size"])
    config.model_name = 'dit'
    config.model = dict(
        **variant_dic,
        code_width=config.code_width,
        checkpoint=False if arg.compile else True,
    )

    config.opt = 'adamw'
    config.weight_decay = 0.0001
    config.betas = (0.9, 0.95)
    config.lr = arg.lr * arg.batch_size / 256
    config.clip_grad = 1.0

    config.evals = dict()
    if arg.eval:
        config.max_visual = 16
        config.evals['generation_1_4'] = dict(
                                        type='generation',
                                        config=config.val_data,
                                        pred='sample',
                                        log_steps=1000,
                                        val_size=50000,
                                        predict_kwargs=dict(cfg_scale=1.4, sampling_steps=250),
                                    )        
    else:
        config.max_visual = 9                           
    return config

def decode_variant(variant):
    """Converts a string "B/128 into dictionary"""
    if variant is None:
        return {}

    v, num_tokens = variant.split("/")
    num_tokens = int(num_tokens)

    return {
        "width": {"B": 768, "L": 1024, "XL": 1152}[v],
        "depth": {"B": 12, "L": 24, "XL": 28}[v],
        "num_heads": {"B": 12, "L": 16, "XL": 16}[v],
        "num_tokens": num_tokens,
    }

def decode_compression_variant(variant):
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