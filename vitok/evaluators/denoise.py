import torch
import numpy as np
from vitok.datasets import create_dataloader_distributed
import torch.distributed as dist
import vitok.utils as utils
from vitok.evaluators.metrics import MetricCalculator, calculate_fid
from tqdm import tqdm

class Evaluator:
    def __init__(self, predict_fn, batch_size, config, val_size=50000):
        self.data_loader = create_dataloader_distributed(0, # Fixed seed
                                                         batch_size,
                                                         configs=[config])                                                
        self.predict_fn = predict_fn
        self.calls = val_size // batch_size
        self.metrics = MetricCalculator(keys=list(range(config['num_frames'])), metrics=('fid', 'ssim', 'psnr', 'l1', 'l2'))
        self.val_size = val_size
        self.batch_size = batch_size
        self.num_frames = config['num_frames']
        self.total_calls = val_size // batch_size

    def run(self, models):
        print("Starting run")
        self.metrics.move_model_to_device('cuda')
    
        for _, (batched_inputs_cpu, _) in tqdm(zip(range(self.total_calls), self.data_loader)):
            reference_inputs = batched_inputs_cpu.to('cuda')
            x_0_pred = self.predict_fn(models, reference_inputs)

            for frame_idx in range(self.num_frames):
                self.metrics.update(reference_inputs[:, frame_idx], x_0_pred[:, frame_idx], frame_idx)

        self.metrics.move_model_to_device('cpu')
            
        dist.barrier()
        returns = utils.gather_over_ranks(self.metrics.prepare_for_gather())

        fid_real_activations = returns['fid_real_activations']
        fid_fake_activations = returns['fid_fake_activations']
        ssim_stats = returns['ssim_stats']
        psnr_stats = returns['psnr_stats']

        fids = []
        ssims = []
        psnrs = []

        for frame in range(self.num_frames):
            fid = calculate_fid(fid_real_activations[frame], fid_fake_activations[frame])
            ssim = np.mean(ssim_stats[frame])
            psnr = np.mean(psnr_stats[frame])
            fids.append(fid)
            ssims.append(ssim)
            psnrs.append(psnr)
    
        fids = np.array(fids)
        ssims = np.array(ssims)
        psnrs = np.array(psnrs)
        dist.barrier()
        print("Resetting")
        self.metrics.reset()

        return {'mse': utils.gather_over_ranks(total_mse / nseen), 
                'mae': utils.gather_over_ranks(total_l1 / nseen),
                'fid': fids.mean(),
                'ssim': ssims.mean(),
                'psnr': psnrs.mean(), 
                'wandbvideo': utils.gather_over_ranks((reference_inputs, decoded_inputs))}