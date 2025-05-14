import torch
import torch.nn as nn
import torch.nn.functional as F

class TubeletEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768,
                 tubelet_size=2,
                 conv=True,
                 flatten=True):
        super().__init__()
        self.conv = conv
        if conv:
            self.proj = nn.Conv3d(
                in_channels=in_chans,
                out_channels=embed_dim,
                kernel_size=(tubelet_size, patch_size, patch_size),
                stride=(tubelet_size, patch_size, patch_size))
        else:
            self.proj = nn.Linear(in_chans * patch_size**2, embed_dim)
        self.flatten = flatten
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size

    def forward(self, x):
        if self.conv:
            x = self.proj(x)
        else:
            tb, p = self.tubelet_size, self.patch_size
            assert x.shape[3] == x.shape[4] and x.shape[3] % p == 0
            assert x.shape[2] % tb == 0
            ts, hs, ws = x.shape[2] // tb, x.shape[3] // p, x.shape[4] // p
            x = x.reshape(shape=(x.shape[0], x.shape[1], ts, tb, hs, p, ws, p))
            x = torch.einsum('nctbhpwq->nthwcpq', x)
            x = x.reshape(shape=(x.shape[0], ts * hs * ws, tb * p**2 * x.shape[1]))
            x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        return x

class TubeletDecode(nn.Module):
    def __init__(
        self,
        patch_size=16,
        tublet_size=8,
        out_channels=3,
        in_embed_dim=768,
        conv=True,
    ):
        super().__init__()
        self.tublet_size = tublet_size
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.conv = conv
        self.in_embed_dim = in_embed_dim
        if self.conv:
            self.proj = nn.ConvTranspose3d(
                in_channels=in_embed_dim,
                out_channels=out_channels,
                kernel_size=(tublet_size, patch_size, patch_size),
                stride=(tublet_size, patch_size, patch_size),
            )
        else:
            self.proj = nn.Linear(
                in_features=in_embed_dim,
                out_features=patch_size * patch_size * tublet_size * out_channels,
            )

    def forward(self, x, grid_size):
        c = self.out_channels
        tb, p, p = self.tublet_size, self.patch_size, self.patch_size
        t, h, w = grid_size
        if self.conv:
            x = x.reshape(shape=(x.shape[0], t, h, w, self.in_embed_dim))
            x = x.permute(0, 4, 1, 2, 3) # b, c, t, h, w
            return self.proj(x)
        else:
            x = self.proj(x)
            x = x.reshape(shape=(x.shape[0], t, h, w, tb, p, p, c))
            x = torch.einsum("bthwpqrc->bctphqwr", x)
            return  x.reshape(shape=(x.shape[0], c, t * tb, h * p, w * p))