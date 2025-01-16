import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import inception_v3, Inception_V3_Weights
import numpy as np
from scipy import linalg
import vitok.utils as utils
import torch.distributed as dist
import requests
import os
from torchmetrics.functional import (
    peak_signal_noise_ratio as PSNR,
    structural_similarity_index_measure as SSIM,)   

class MetricCalculator:
    def __init__(self, keys, metrics=('fvd', 'fid', 'ssim', 'psnr'), pool=None):
        if 'fid' in metrics or 'is' in metrics:
            self.inception = InceptionV3().to('cpu') #Keep on CPU to avoid OOM
            self.inception.eval()
        if 'fvd' in metrics:
            detector_url = 'https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1'
            local_file_path = 'i3d_torchscript.pt'
            #Download URL on rank 0
            if dist.get_rank() == 0 and not os.path.exists(local_file_path):
                with utils.open_url(detector_url, verbose=False) as f:
                    #f is io.BytesIO(url_data)
                    with open(local_file_path, 'wb') as out:
                        out.write(f.read())
            dist.barrier()
            with open(local_file_path, 'rb') as f:
                self.detector = torch.jit.load(f).eval()
            self.detector_kwargs = dict(rescale=False, resize=False, return_features=True)
        self.keys = [str(key) for key in keys]
        self.metrics = metrics
        self.pool = pool
        self.reset()
            
    def reset(self):
        self.fid_fake_activations = {key: [] for key in self.keys}
        self.fid_real_activations = {key: [] for key in self.keys}
        self.fvd_fake_activations = {key: [] for key in self.keys}
        self.fvd_real_activations = {key: [] for key in self.keys}
        self.inception_score = {key: [] for key in self.keys}
        self.ssim_stats = {key: [] for key in self.keys}
        self.psnr_stats = {key: [] for key in self.keys}
        self.mse = {key: [] for key in self.keys}
        self.mae = {key: [] for key in self.keys}
    
    def move_model_to_device(self, device):
        if 'fid' in self.metrics:
            self.inception = self.inception.to(device)
        if 'fvd' in self.metrics:
            self.detector = self.detector.to(device)
    
    @torch.no_grad()
    def update(self, real, generated, key):
        assert len(real) == len(generated), "Input shapes should be the same"
        assert real.dtype == torch.uint8 and generated.dtype == torch.uint8, "Input types should be uint8"
        real = real.float()
        generated = generated.float()

        if 'is' in self.metrics or 'fid' in self.metrics:
            assert len(real.shape) == 4 and len(generated.shape) == 4, "Input shapes should be [B, C, H, W]"
            real_activations, _ = self.inception(2 * (real / 255) - 1)
            fake_activations, fake_probs = self.inception(2 * (generated/ 255) - 1)
        
        if 'fvd' in self.metrics:
            assert len(real.shape) == 5 and len(generated.shape) == 5, "Input shapes should be [B, T, C, H, W]"
            real = F.interpolate(real.permute(0, 2, 1, 3, 4), size=(real.shape[1], 224, 224), mode='trilinear', align_corners=False)
            generated = F.interpolate(generated.permute(0, 2, 1, 3, 4), size=(generated.shape[1], 224, 224), mode='trilinear', align_corners=False)
            real = (2 * (real / 255) - 1)
            generated = (2 * (generated / 255) - 1)
            fvd_feats_real = self.detector(real, **self.detector_kwargs)
            fvd_feats_fake = self.detector(generated, **self.detector_kwargs)
            self.fvd_real_activations[key].append(fvd_feats_real)
            self.fvd_fake_activations[key].append(fvd_feats_fake)

        if 'mse' in self.metrics:
            self.mse[key].append(((real - generated) ** 2).mean().unsqueeze(0))
        
        if 'mae' in self.metrics:
            self.mae[key].append((real - generated).abs().mean().unsqueeze(0))

        if 'fid' in self.metrics:
            self.fid_real_activations[key].append(real_activations)
            self.fid_fake_activations[key].append(fake_activations)
        
        if 'is' in self.metrics:
            self.inception_score[key].append(fake_probs)

        if 'ssim' in self.metrics:
            ssims = SSIM(preds=generated, target=real, reduction="none")
            for ssim in ssims:
                self.ssim_stats[key].append(ssim.unsqueeze(0))
        
        if 'psnr' in self.metrics:
            self.psnr_stats[key].append(PSNR(preds=generated, target=real).unsqueeze(0))

    def gather(self):
        stats = {key : {} for key in self.keys}
        for key in self.keys:
            if 'mse' in self.metrics:
                stats[key]['mse'] = utils.gather(torch.cat(self.mse[key], dim=0)).mean()
            
            if 'mae' in self.metrics:
                stats[key]['mae'] = utils.gather(torch.cat(self.mae[key], dim=0)).mean()
            
            if 'fid' in self.metrics:
                fid_real_activations = utils.gather(torch.cat(self.fid_real_activations[key], dim=0))
                fid_fake_activations = utils.gather(torch.cat(self.fid_fake_activations[key], dim=0))
                stats[key]['fid'] = calculate_fid(fid_real_activations, fid_fake_activations)
            
            if 'fvd' in self.metrics:
                fvd_real_activations = utils.gather(torch.cat(self.fvd_real_activations[key], dim=0))
                fvd_fake_activations = utils.gather(torch.cat(self.fvd_fake_activations[key], dim=0))
                stats[key]['fvd'] = calculate_fid(fvd_fake_activations, fvd_real_activations)
            
            if 'is' in self.metrics:
                inception_score = utils.gather(torch.cat(self.inception_score[key], dim=0))
                stats[key]['is'] = compute_inception_score(inception_score)

            if 'ssim' in self.metrics:
                stats[key]['ssim'] = utils.gather(torch.cat(self.ssim_stats[key], dim=0))
                if not isinstance(stats[key]['ssim'], float):
                    stats[key]['ssim'] = stats[key]['ssim'].mean() 

            if 'psnr' in self.metrics:
                stats[key]['psnr'] = utils.gather(torch.cat(self.psnr_stats[key], dim=0))
                if not isinstance(stats[key]['psnr'], float):
                    stats[key]['psnr'] = stats[key]['psnr'].mean()
            dist.barrier()
        
        if self.pool is not None:
            final_stats = {metric: 0.0 for metric in self.metrics}
            for metric in self.metrics:
                for key in self.keys:
                    final_stats[metric] += stats[key][metric]
                final_stats[metric] /= len(self.keys)
            stats = final_stats

        return stats

def compute_inception_score(softmax_outputs):
    p_yx = softmax_outputs
    p_y = np.mean(p_yx, axis=0)
    kl_div = p_yx * (np.log(p_yx) - np.log(p_y))
    kl_mean = np.mean(np.sum(kl_div, axis=1))
    return np.exp(kl_mean)

def calculate_fid(real_activations, fake_activations):
    mu1, sigma1 = calculate_activation_statistics(real_activations)
    mu2, sigma2 = calculate_activation_statistics(fake_activations)
    fid_value = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid_value

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

def calculate_activation_statistics(act):
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

class InceptionV3(nn.Module):
    def __init__(self):
        super(InceptionV3, self).__init__()
        model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        for param in model.parameters():
            param.requires_grad = False
        self.Conv2d_1a_3x3 = model.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = model.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = model.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = model.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = model.Conv2d_4a_3x3
        self.Mixed_5b = model.Mixed_5b
        self.Mixed_5c = model.Mixed_5c
        self.Mixed_5d = model.Mixed_5d
        self.Mixed_6a = model.Mixed_6a
        self.Mixed_6b = model.Mixed_6b
        self.Mixed_6c = model.Mixed_6c
        self.Mixed_6d = model.Mixed_6d
        self.Mixed_6e = model.Mixed_6e
        self.Mixed_7a = model.Mixed_7a
        self.Mixed_7b = model.Mixed_7b
        self.Mixed_7c = model.Mixed_7c
        self.fc = model.fc

    def forward(self, x):
        x = nn.Upsample(size=(299, 299), mode="bilinear", align_corners=True)(x)
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        x = F.avg_pool2d(x, kernel_size=8)
        acts = x.view(x.size(0), -1)
        acts = F.dropout(acts, training=False)
        logits = self.fc(acts)
        return acts, F.softmax(logits, dim=1)