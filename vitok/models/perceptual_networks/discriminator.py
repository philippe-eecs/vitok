import functools
import math
from typing import Tuple


import torch
import torch.nn as nn
import torch.nn.functional as F

class ActNorm(nn.Module):
    def __init__(
        self, num_features, logdet=False, affine=True, allow_reverse_init=False
    ):
        assert affine
        super().__init__()
        self.logdet = logdet
        self.loc = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.allow_reverse_init = allow_reverse_init

        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input, reverse=False):
        if reverse:
            return self.reverse(input)
        if len(input.shape) == 2:
            input = input[:, :, None, None]
            squeeze = True
        else:
            squeeze = False

        _, _, height, width = input.shape

        if self.training and self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        h = self.scale * (input + self.loc)

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)

        if self.logdet:
            log_abs = torch.log(torch.abs(self.scale))
            logdet = height * width * torch.sum(log_abs)
            logdet = logdet * torch.ones(input.shape[0]).to(input)
            return h, logdet

        return h

def hinge_d_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    """Hinge loss for discrminator.

    This function is borrowed from
    https://github.com/CompVis/taming-transformers/blob/master/taming/modules/losses/vqperceptual.py#L20
    """
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

def compute_lecam_loss(
    logits_real_mean: torch.Tensor,
    logits_fake_mean: torch.Tensor,
    ema_logits_real_mean: torch.Tensor,
    ema_logits_fake_mean: torch.Tensor
) -> torch.Tensor:
    """Computes the LeCam loss for the given average real and fake logits.

    Args:
        logits_real_mean -> torch.Tensor: The average real logits.
        logits_fake_mean -> torch.Tensor: The average fake logits.
        ema_logits_real_mean -> torch.Tensor: The EMA of the average real logits.
        ema_logits_fake_mean -> torch.Tensor: The EMA of the average fake logits.

    Returns:
        lecam_loss -> torch.Tensor: The LeCam loss.
    """
    lecam_loss = torch.mean(torch.pow(F.relu(logits_real_mean - ema_logits_fake_mean), 2))
    lecam_loss += torch.mean(torch.pow(F.relu(ema_logits_real_mean - logits_fake_mean), 2))
    return lecam_loss

class BlurPool2D(nn.Module):
    def __init__(self, filter_size=4, stride=2):
        super(BlurPool2D, self).__init__()
        self.filter_size = filter_size
        self.stride = stride

        if self.filter_size == 3:
            filter = [1.0, 2.0, 1.0]
            self.padding = 1
        elif self.filter_size == 4:
            filter = [1.0, 3.0, 3.0, 1.0]
            self.padding = 1
        elif self.filter_size == 5:
            filter = [1.0, 4.0, 6.0, 4.0, 1.0]
            self.padding = 2
        elif self.filter_size == 6:
            filter = [1.0, 5.0, 10.0, 10.0, 5.0, 1.0]
            self.padding = 2
        elif self.filter_size == 7:
            filter = [1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0]
            self.padding = 3
        else:
            raise ValueError("Only filter_size of 3, 4, 5, 6 or 7 is supported.")

        filter = torch.tensor(filter, dtype=torch.float32)
        filter = filter[:, None] * filter[None, :]
        filter /= torch.sum(filter)
        filter = filter.view(1, 1, filter.shape[0], filter.shape[1])
        self.register_buffer("filter", filter)

    def forward(self, x):
        channel_num = x.shape[1]
        depthwise_filter = self.filter.repeat((channel_num, 1, 1, 1))

        return F.conv2d(
            x,
            depthwise_filter,
            stride=self.stride,
            padding=self.padding,
            groups=channel_num,
        )

class StyleGANResBlock(nn.Module):
    def __init__(self, input_dim, output_dim, activation_fn, blur_resample):
        super(StyleGANResBlock, self).__init__()
        self.activation_fn = activation_fn
        self.blur_resample = blur_resample

        self.conv1 = nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1)
        if self.blur_resample:
            self.blur_pool = BlurPool2D(filter_size=4)
        else:
            self.downsample = nn.AvgPool2d(kernel_size=2)

        self.res_conv = nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1)

        self.sqrt2 = math.sqrt(2)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.activation_fn(x)
        if self.blur_resample:
            x = self.blur_pool(x)
            residual = self.blur_pool(residual)
        else:
            x = self.downsample(x)
            residual = self.downsample(residual)
        residual = self.res_conv(residual)
        x = self.conv2(x)
        x = self.activation_fn(x)
        out = (residual + x) / self.sqrt2
        return out


class Discriminator(nn.Module):
    def __init__(
        self,
        input_size=256,
        filters=64,
        channel_multiplier=1,
        blur_resample=True,
    ):
        super(Discriminator, self).__init__()
        self.activation_fn = nn.LeakyReLU(0.2, True)
        self.channel_multiplier = channel_multiplier

        filters = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * self.channel_multiplier,
            128: 128 * self.channel_multiplier,
            256: 64 * self.channel_multiplier,
            512: 32 * self.channel_multiplier,
            1024: 16 * self.channel_multiplier,
        }

        self.conv1 = nn.Conv2d(3, filters[input_size], kernel_size=3, padding=1)
        self.res_blocks = nn.ModuleList()
        in_ft = filters[input_size]
        log_size = int(math.log2(input_size))
        for i in range(log_size, 2, -1):
            out_ft = filters[2 ** (i - 1)]
            self.res_blocks.append(
                StyleGANResBlock(in_ft, out_ft, self.activation_fn, blur_resample)
            )
            in_ft = out_ft
        self.conv_last = nn.Conv2d(in_ft, filters[4], kernel_size=3, padding=1)
        self.fc1 = nn.Linear(
            filters[4] * 4**2,
            filters[4],
        )
        self.fc2 = nn.Linear(filters[4], 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation_fn(x)
        for res_block in self.res_blocks:
            x = res_block(x)
        x = self.conv_last(x)
        x = self.activation_fn(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x)
        return x

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=True):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if not use_actnorm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = ActNorm
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)

class NLayerDiscriminator3D(nn.Module):
    """
    NLayerDiscriminator from Taming but with 3D convs and batch norms
    """

    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False):
        super(NLayerDiscriminator3D, self).__init__()
        if not use_actnorm:
            norm_layer = nn.BatchNorm3d
        else:
            raise NotImplementedError("Not implemented.")
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func != nn.BatchNorm3d
        else:
            use_bias = norm_layer != nn.BatchNorm3d

        kw = 3
        padw = 1
        sequence = [
            nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv3d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=(kw, kw, kw),
                    stride=(2 if n == 1 else 1, 2, 2),
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv3d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=(kw, kw, kw),
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        return self.main(input)