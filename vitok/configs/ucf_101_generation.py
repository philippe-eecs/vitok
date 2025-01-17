# pylint: disable=line-too-long
import vitok.configs.common as cc
import ml_collections as mlc

def get_config(arg=None):
    """Config for training."""
    arg = cc.parse_arg(arg, variant="B/256", compression_variant="B/2x8x16", 
                        batch_size=256, size=128, steps=2e5, predict_xstart=False,
                        schedule="linear", compile=False, eval=False,
                        enc_dep=0, dec_dep=0, num_frames=16, fps=8,
                        variational=True, normalize=True, casual_code=False, cache=False,
                        wandb_project="UCF_101_Generation",  wandb=0, local=False,
                        code_min=4096, code_max=4096,
                        name="dit_B_128", load="")

    config = mlc.ConfigDict()
    config.wandb_project = arg.wandb_project
    config.wandb_id = None
    config.local = arg.local
    config.steps = int(arg.steps * (256 / arg.batch_size))
    config.ema_decay = 0.9995
    config.num_classes = 101
    config.normalize_latents = arg.normalize
    if arg.cache:
        config.cache_path = f'/fsx-project/philippehansen/imagenet/cache/sd_cnn'  
        config.cache_path = f'/fsx-project/philippehansen/ucf101/cache/{arg.compression_variant}/{arg.load}'
    else:
        config.cache_path = None

    config.predict_xstart = arg.predict_xstart
    config.schedule = arg.schedule

    config.checkpoint_name = arg.name
    config.checkpoint_load = arg.load

    config.save_ckpt_freq = int(10000 * (256 / arg.batch_size))
    config.log_freq = int(100 * (256 / arg.batch_size))
    config.sample_freq = int(5000 * (256 / arg.batch_size))
    config.eval_freq = int(10000 * (256 / arg.batch_size))

    config.compile = arg.compile
    config.wandb_mode = "offline" if arg.wandb == 0 else "online"
    config.eval_only = arg.eval
    config.precision = "f32"
    config.batch_size = arg.batch_size

    variant_dic = decode_variant(arg.variant)
    compression_variant_dic = decode_compression_variant(arg.compression_variant)

    if arg.enc_dep:
        compression_variant_dic["encoder_depth"] = arg.enc_dep
    if arg.dec_dep:
        compression_variant_dic["decoder_depth"] = arg.dec_dep

    config.train_data = dict()
    config.train_data = dict()
    config.train_data.type = "video"
    config.train_data.data_format = "ucf101"
    config.train_data.root = "/fsx-project/philippehansen/ucf101/UCF-101"
    config.train_data.num_frames = arg.num_frames
    config.train_data.weight_num_frames = arg.num_frames
    config.train_data.sampling_rate = int(25 / arg.fps)
    config.train_data.train = True
    config.train_data.num_workers = 8
    config.train_data.img_size = arg.size

    config.train_configs = [config.train_data]
    config.val_data = dict()
    #val data is the same
    config.val_data.type = "video"
    config.val_data.data_format = "ucf101"
    config.val_data.root = "/fsx-project/philippehansen/ucf101/UCF-101"
    config.val_data.num_frames = arg.num_frames
    config.val_data.weight_num_frames = arg.num_frames
    config.val_data.sampling_rate = int(25 / arg.fps)
    config.val_data.train = False
    config.val_data.num_workers = 2
    config.val_data.img_size = arg.size
    
    #num_patches = (arg.size // compression_variant_dic["patch_size"]) ** 2
    num_tokens = (arg.size // compression_variant_dic["patch_size"]) ** 2 * arg.num_frames // compression_variant_dic["tubelet_size"]
    code_length = arg.code_max // compression_variant_dic["code_width"]
    min_ratio = arg.code_min / (compression_variant_dic["code_width"] * code_length)
    max_ratio = 1.0
        
    print(f"Code length: {code_length}")
    
    config.compression_model_name = 'self_ae' #generic vit based auto-encoder
    config.compression_model = dict(
        **compression_variant_dic,
        img_size=arg.size,
        num_frames=arg.num_frames,
        variational=arg.variational,
        code_length=code_length,
        code_mask_ratio=(min_ratio, max_ratio),
        checkpoint=False if arg.compile else True, #Compile bugged with checkpointing + DDP
    )
    config.num_tokens = variant_dic["num_tokens"]
    config.code_width = compression_variant_dic["code_width"]
    
    config.grid_size = (arg.num_frames // compression_variant_dic["tubelet_size"], 
                        arg.size // compression_variant_dic["patch_size"], 
                        arg.size // compression_variant_dic["patch_size"])

    config.model_name = 'dit'
    config.model = dict(
        **variant_dic,
        code_width=config.code_width,
        checkpoint=False if arg.compile else True,
        num_classes=config.num_classes,
    )

    config.opt = 'adamw'
    config.weight_decay = 0.0
    config.betas = (0.9, 0.99)
    config.lr = 15e-5 * arg.batch_size / 256
    config.clip_grad = 1.0

    config.evals = dict()
    config.max_visuals = 16
    config.evals['generation_1_5'] = dict(
                                    type='generation',
                                    config=config.val_data,
                                    pred='sample',
                                    log_steps=1000,
                                    val_size=50000,
                                    predict_kwargs=dict(cfg_scale=1.5, sampling_steps=250),
                                )
    config.evals['generation_3_0'] = dict(
                                    type='generation',
                                    config=config.val_data,
                                    pred='sample',
                                    log_steps=1000,
                                    val_size=50000,
                                    predict_kwargs=dict(cfg_scale=3.0, sampling_steps=250),
                                )                         
    return config

def decode_variant(variant):
    """Converts a string "B/128 into dictionary"""
    if variant is None:
        return {}

    v, num_tokens = variant.split("/")
    num_tokens = int(num_tokens)

    return {
        "width": {"B": 768, "L": 1024, "H": 1280}[v],
        "depth": {"B": 12, "L": 24, "H": 32}[v],
        "num_heads": {"B": 12, "L": 16, "H": 16}[v],
        "num_tokens": num_tokens,
    }

def decode_compression_variant(variant):
    """Converts a string "B/4x4 into (tubelet, patch) tuple"""
    if variant is None:
        return {}

    v, rest = variant.split("/")
    tubelet, patch, code_width = map(int, rest.split("x"))

    return {
        "width": {"B": 768, "L": 1024, "H": 1280}[v],
        "encoder_depth": {"B": 6, "L": 12, "H": 16}[v],
        "decoder_depth": {"B": 12, "L": 24, "H": 32}[v],
        "num_heads": {"B": 12, "L": 16, "H": 16}[v],
        "tubelet_size": tubelet,
        "patch_size": patch,
        "code_width": code_width,
    }