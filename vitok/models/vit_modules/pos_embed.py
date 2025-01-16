import torch
import numpy as np

def posemb_sincos_1d(length, width, temperature=10_000., dtype=np.float32):
    """Generates 1D positional embeddings using sine-cosine method."""
    assert width % 2 == 0, "Width must be a multiple of 2 for sincos posemb"
    pos = np.arange(length)
    omega = np.arange(width // 2) / (width // 2 - 1)
    omega = 1. / (temperature ** omega)
    pos_emb = np.einsum("n,d->nd", pos, omega)
    pe = np.concatenate([np.sin(pos_emb), np.cos(pos_emb)], axis=1)
    return np.asarray(pe, dtype=dtype)

def posemb_sincos_2d(h, w, width, temperature=10_000., dtype=np.float32):
    """Generates 2D positional embeddings using sine-cosine method."""
    y, x = np.mgrid[:h, :w]
    assert width % 4 == 0, "Width must be mult of 4 for sincos posemb"
    omega = np.arange(width // 4) / (width // 4 - 1)
    omega = 1. / (temperature**omega)
    y = np.einsum("m,d->md", y.flatten(), omega)
    x = np.einsum("m,d->md", x.flatten(), omega)
    pe = np.concatenate([np.sin(x), np.cos(x), np.sin(y), np.cos(y)], axis=1)
    return np.asarray(pe, dtype)

def posemb_sincos_3d(t, h, w, width, temperature=10_000., dtype=np.float32): #TODO: You should check if this is correct?
    """Generates 3D positional embeddings using sine-cosine method."""
    t, y, x = np.mgrid[:t, :h, :w]
    omega = np.arange(width // 6) / (width // 6 - 1)
    omega = 1. / (temperature**omega)
    t = np.einsum("m,d->md", t.flatten(), omega)
    y = np.einsum("m,d->md", y.flatten(), omega)
    x = np.einsum("m,d->md", x.flatten(), omega)
    pe = np.concatenate([np.sin(t), np.cos(t), np.sin(x), np.cos(x), np.sin(y), np.cos(y)], axis=1)
    return np.asarray(pe, dtype)

def init_random_2d_freqs(dim: int, num_heads: int, theta: float = 10.0, rotate: bool = True):
    freqs_x = []
    freqs_y = []
    mag = 1 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    for i in range(num_heads):
        angles = torch.rand(1) * 2 * torch.pi if rotate else torch.zeros(1)        
        fx = torch.cat([mag * torch.cos(angles), mag * torch.cos(torch.pi/2 + angles)], dim=-1)
        fy = torch.cat([mag * torch.sin(angles), mag * torch.sin(torch.pi/2 + angles)], dim=-1)
        freqs_x.append(fx)
        freqs_y.append(fy)
    freqs_x = torch.stack(freqs_x, dim=0)
    freqs_y = torch.stack(freqs_y, dim=0)
    freqs = torch.stack([freqs_x, freqs_y], dim=0)
    return freqs

def init_t_xy(end_x: int, end_y: int):
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode='floor').float()
    return t_x, t_y

def compute_freqs_cis(t: int, dim: int = 768, theta: float = 10000.0):
    with torch.amp.autocast(enabled=False, device_type="cuda"):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        freqs = torch.outer(t, freqs.to(t.device)).float()
        freqs_cos = torch.cos(freqs)
        freqs_sin = torch.sin(freqs) 
    return freqs_cos, freqs_sin

def compute_axial_cis(t_x: int, t_y: int, t_z: int = None, dim: int = 768, theta: float = 100.0):
    with torch.amp.autocast(enabled=False, device_type="cuda"):
        if not t_z:
            assert dim % 4 == 0 and dim % 2 == 0, 'dim must be divisible by 4 and 2 for axial 2D'
            freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
            freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
        else:
            assert dim % 6 == 0 and dim % 3 == 0, 'dim must be divisible by 6 and 3 for axial 3D'
            freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 6)[: (dim // 6)].float() / dim))
            freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 6)[: (dim // 6)].float() / dim))
            freqs_z = 1.0 / (theta ** (torch.arange(0, dim, 6)[: (dim // 6)].float() / dim))

        freqs_x = torch.outer(t_x, freqs_x.to(t_x.device))
        freqs_y = torch.outer(t_y, freqs_y.to(t_y.device))
        freqs_x_cos = torch.cos(freqs_x)
        freqs_x_sin = torch.sin(freqs_x)
        freqs_y_cos = torch.cos(freqs_y)
        freqs_y_sin = torch.sin(freqs_y)
        if t_z:
            freqs_z = torch.outer(t_z, freqs_z.to(t_z.device))
            freqs_z_cos = torch.cos(freqs_z)
            freqs_z_sin = torch.sin(freqs_z)
            freqs_cos = torch.cat([freqs_x_cos, freqs_y_cos, freqs_z_cos], dim=-1)
            freqs_sin = torch.cat([freqs_x_sin, freqs_y_sin, freqs_z_sin], dim=-1)
        else:
            freqs_cos = torch.cat([freqs_x_cos, freqs_y_cos], dim=-1)
            freqs_sin = torch.cat([freqs_x_sin, freqs_y_sin], dim=-1)
    return freqs_cos, freqs_sin

def compute_mixed_cis(freqs: torch.Tensor, t_x: torch.Tensor, t_y: torch.Tensor, num_heads: int):
    N = t_x.shape[0]
    depth = freqs.shape[1]
    with torch.amp.autocast(enabled=False, device_type="cuda"):
        freqs_x = (t_x.unsqueeze(-1) @ freqs[0].unsqueeze(-2)).view(depth, N, num_heads, -1).permute(0, 2, 1, 3)
        freqs_y = (t_y.unsqueeze(-1) @ freqs[1].unsqueeze(-2)).view(depth, N, num_heads, -1).permute(0, 2, 1, 3)
        freqs = freqs_x + freqs_y
        freqs_cos = torch.cos(freqs)
        freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin