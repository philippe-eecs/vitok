import torch
import numpy as np
from vitok.datasets import create_dataloader_distributed
import torch.distributed as dist
import vitok.utils as utils
from vitok.evaluators.metrics import MetricCalculator, calculate_fid
from tqdm import tqdm

class Evaluator:
    def __init__(self, predict_fn, batch_size, config, val_size=50000, compute_fvd=False, predict_kwargs={}):
        self.data_loader = create_dataloader_distributed(0, # Fixed seed
                                                         batch_size,
                                                         configs=[config])                                                
        self.predict_fn = predict_fn
        self.calls = val_size // batch_size
        self.metrics = MetricCalculator(keys=list(range(config['num_frames'])), metrics=('fid', 'is'), pool='mean')
        if compute_fvd:
            self.fvd_metrics = MetricCalculator(keys=['video'], metrics=('fvd',), pool='mean')
        self.compute_fvd = compute_fvd
        self.val_size = val_size
        self.batch_size = batch_size
        self.num_frames = config['num_frames']
        self.total_calls = val_size // batch_size
        self.predict_kwargs = predict_kwargs

    def run(self, models):
        self.metrics.move_model_to_device('cuda')
        if self.compute_fvd:
            self.fvd_metrics.move_model_to_device('cuda')
        disable_tqdm = utils.get_rank() != 0
        dist.barrier()
        for _, (batched_inputs_cpu, _) in tqdm(zip(range(self.total_calls), self.data_loader), disable=disable_tqdm):
            reference_inputs = batched_inputs_cpu.to('cuda')
            samples = self.predict_fn(models, self.batch_size // dist.get_world_size(), **self.predict_kwargs)
            dist.barrier()
            original_inputs = utils.postprocess_video(reference_inputs) #For FID reference
            samples = utils.postprocess_video(samples)
            dist.barrier()
            for frame_idx in range(self.num_frames):
                self.metrics.update(original_inputs[:, frame_idx], samples[:, frame_idx], str(frame_idx))
            dist.barrier()
            if self.compute_fvd:
                self.fvd_metrics.update(reference_inputs, decoded_inputs, 'video')

        
        self.metrics.move_model_to_device('cpu')
        stats = self.metrics.gather()
        self.metrics.reset()
        if self.compute_fvd:
            self.fvd_metrics.move_model_to_device('cpu')
            fvd_stats = self.fvd_metrics.gather()
            self.fvd_metrics.reset()
            stats = {**stats, **fvd_stats}
        dist.barrier()
        return {**stats,
                'wandbvideo_samples': utils.gather(samples)}