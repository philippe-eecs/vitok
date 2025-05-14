import torch
import torch.nn as nn
from vitok.models.vit_modules import Transformer, TubeletEmbed, TubeletDecode, posemb_sincos_1d, posemb_sincos_2d, posemb_sincos_3d, init_random_2d_freqs, init_t_xy, compute_mixed_cis, compute_axial_cis, compute_freqs_cis
from vitok.models.perceptual_networks.distributions import DiagonalGaussianDistribution #TODO: Ablate on turning this off
import numpy as np
from functools import partial
from torch.nn.attention.flex_attention import create_block_mask

def create_block_casual_mask(num_t, num_h, num_w, device='cuda'):
    sequence_length = num_t * num_h * num_w
    block_size = num_h * num_w
    blocks = torch.ones(sequence_length // block_size, block_size, block_size)
    block_diag_enable_mask = torch.block_diag(*blocks)
    causal_enable_mask = torch.ones(sequence_length, sequence_length).tril_(0)
    disable_mask = ((block_diag_enable_mask + causal_enable_mask) < 0.5).to(torch.bool)
    return disable_mask.to(device)

class AE(nn.Module):
    """ 
    Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(
        self,
        channels=3,
        img_size=256,
        num_frames=1,
        patch_size=16,
        tubelet_size=1,
        width=768,
        encoder_depth=6,
        num_heads=12,
        decoder_depth=12,
        code_length=256,
        code_width=16,
        checkpoint=False,
        variational=True,
        block_casual=False,
        lengths=None,
        rope_theta=10000.0,
        rope_style='1d_axial',
        simple=False, #Directly converts patch to code
        posemb='nope',
        norm_layer='layer',
    ):
        super().__init__()
        self.tubelet_size = tubelet_size
        self.patch_size = patch_size
        self.max_num_input_tokens = (img_size // patch_size) ** 2 * (num_frames // tubelet_size)
        self.grid_size = (num_frames // tubelet_size, img_size // patch_size, img_size // patch_size)
        self.num_h = self.num_w = img_size // patch_size
        self.num_t = num_frames // tubelet_size
        self.rope_theta = rope_theta
        self.encoder_depth = encoder_depth
        self.decoder_depth = decoder_depth
        self.block_casual = block_casual

        self.tubelet_embed = TubeletEmbed(
            patch_size=patch_size,
            in_chans=channels,
            embed_dim=width,
            tubelet_size=tubelet_size
        )

        self.tubelet_decode = TubeletDecode(
            patch_size=self.patch_size,
            tublet_size=tubelet_size,
            conv=True,
            in_embed_dim=width,
            out_channels=channels,
        )

        if self.block_casual:
            block_mask = create_block_casual_mask(self.num_t, self.num_h, self.num_w)
        else:
            block_mask = None


        self.encoder = Transformer(
            embed_dim=width,
            depth=encoder_depth,
            num_heads=num_heads,
            checkpoint=checkpoint,
            mlp_ratio=2.67,
            norm_layer=nn.LayerNorm,
            t5_mlp=True,
            final_ln=True,
        )
        
        self.encoder_to_code = nn.Linear(
            width, code_width * 2 if variational else code_width) 
        self.code_to_decoder = nn.Linear(
            code_width, width)

        self.decoder = Transformer(
            embed_dim=width,
            depth=decoder_depth,
            num_heads=num_heads,
            checkpoint=checkpoint,
            mlp_ratio=2.67,
            norm_layer=nn.LayerNorm,
            t5_mlp=True,
            final_ln=True,
            )

        self.code_length = code_length #If code length is none, perform token compression within original tubelet token sequence
        if self.rope_theta:
            dim, mix = rope_style.split('_')
            self.dim = dim
            self.mix = mix

            if mix == 'mixed':
                self.compute_cis = partial(compute_mixed_cis, num_heads=num_heads)
                enc_freqs = []
                dec_freqs = []
                for _ in range(encoder_depth):
                    enc_freqs.append(
                        init_random_2d_freqs(dim=width // num_heads, num_heads=num_heads, theta=rope_theta)
                    )
                for _ in range(decoder_depth):
                    dec_freqs.append(
                        init_random_2d_freqs(dim=width // num_heads, num_heads=num_heads, theta=rope_theta)
                    )
                enc_freqs = torch.stack(enc_freqs, dim=1).view(2, encoder_depth, -1)
                dec_freqs = torch.stack(dec_freqs, dim=1).view(2, decoder_depth, -1)
                self.enc_freqs = nn.Parameter(enc_freqs, requires_grad=True)
                self.dec_freqs = nn.Parameter(dec_freqs, requires_grad=True)
            elif mix == 'axial':
                if dim == '1d':
                    self.compute_cis = partial(compute_freqs_cis, theta=rope_theta, dim=width // num_heads)
                else: #TODO: Add 3D RoPE for Video
                    self.compute_cis = partial(compute_axial_cis, theta=rope_theta, dim=width // num_heads)
            
            t_x, t_y = init_t_xy(end_x = img_size // patch_size, end_y = img_size // patch_size) #TODO Add 3D RoPE
            self.t_x, self.t_y = t_x, t_y

        self.posemb_name = posemb
        if 'sincos' in posemb:
            self.posemb = nn.Parameter(torch.zeros(1, self.max_num_input_tokens, width), requires_grad=False)
            self.dec_posemb = nn.Parameter(torch.zeros(1, self.max_num_input_tokens, width), requires_grad=False)
        elif posemb == 'learned':
            self.posemb = nn.Parameter(torch.randn(1, self.max_num_input_tokens, width) * 0.025, requires_grad=True)
            self.dec_posemb = nn.Parameter(torch.randn(1, self.max_num_input_tokens, width) * 0.025, requires_grad=True)
        elif posemb == 'nope':
            self.posemb = None
            self.dec_posemb = None
        elif posemb == 'dec':
            self.posemb = None
            self.dec_posemb = nn.Parameter(torch.randn(1, self.max_num_input_tokens, width) * 0.025, requires_grad=True)
        else:
            raise ValueError(f"Unknown positional embedding type {posemb}")
        
        self.lengths = np.array(lengths) if lengths is not None else None
        if self.lengths is not None:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, code_width))
        
        if not simple:
            self.code = nn.Parameter(torch.zeros(1, code_length, width), requires_grad=False) #Spawn as 1D Code
            #self.code = nn.Parameter(torch.randn(1, code_length, width), requires_grad=True) #Spawn as 1D Code
        else:
            self.code = None
            
        self.simple = simple
        self.width = width
        self.width = width
        self.variational = variational
        self.initialize_weights()
    
    def initialize_weights(self):
        #if 'sincos' in self.posemb_name:
        if self.posemb_name == 'sincos1d':
            posemb = posemb_sincos_1d(
                self.max_num_input_tokens,
                self.width,
            )
        elif self.posemb_name == 'sincos2d':
            posemb = posemb_sincos_2d(
                self.num_h,
                self.num_w,
                self.width,
            )
        else:
            posemb = posemb_sincos_3d( #TODO make these 3D for video + try RoPE
                self.num_t,
                self.num_h,
                self.num_w,
                self.width,
            )
        
        if self.posemb is not None:
            self.posemb.data.copy_(
                torch.from_numpy(posemb).float().unsqueeze(0)
            )
        
        if self.dec_posemb is not None:
            self.dec_posemb.data.copy_(
                torch.from_numpy(posemb).float().unsqueeze(0)
            )
        if not self.simple:
            #Spawn as posemb_sincos_1d
            posemb = posemb_sincos_1d(
                self.code_length,
                self.width,
            )
            self.code.data.copy_(
                torch.from_numpy(posemb).float().unsqueeze(0)
            )
        
        self.apply(self._init_weights)
        w = self.tubelet_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.0)
            nn.init.constant_(m.weight, 1.0)
    
    def finetune_decoder(self): #Set all encode(self, x) related parameters to requires_grad=False and all decode(self, code, grid_size) related parameters to requires_grad=True
        for param in self.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = True
        for param in self.code_to_decoder.parameters():
            param.requires_grad = True
        for param in self.tubelet_decode.parameters():
            param.requires_grad = True

    def encode(self, x):
        B, _, T, H, W = x.shape
        x = self.tubelet_embed(x)
        if self.posemb is not None:
            x = x + self.posemb[:, :x.shape[1], :]
        if not self.simple:
            code_length = self.code_length
            z = self.code.expand(B, -1, -1)[:, :code_length, :]
            enc_seq = torch.cat([z, x], dim=1)
        else:
            enc_seq = x
        if self.rope_theta:
            if self.mix == 'mixed':
                freqs = self.compute_cis(self.enc_freqs.to(x.device), self.t_x.to(x.device), self.t_y.to(x.device))
            else:
                if self.dim == '1d':
                    length = torch.Tensor(np.arange(enc_seq.shape[1]))
                    freqs = self.compute_cis(length.to(x.device))
                else:
                    freqs = self.compute_cis(self.t_x.to(x.device), self.t_y.to(x.device))
        else:
            freqs = None
        enc_seq = self.encoder(enc_seq, freqs=freqs)
        if not self.simple:
            z = enc_seq[:, :code_length, :]
        else:
            z = enc_seq
        code = self.encoder_to_code(z)
        if not self.variational:
            code = torch.cat((code, torch.zeros_like(code)), 2)
        code = DiagonalGaussianDistribution(code, deterministic=(not self.variational), dim=2)
        return code, (T // self.tubelet_size, H // self.patch_size, W // self.patch_size)
    
    def decode(self, code, grid_size):
        decoding_length = grid_size[0] * grid_size[1] * grid_size[2]
        z = self.code_to_decoder(code)
        if self.dec_posemb is not None:
            dec_posemb = self.dec_posemb.expand(z.shape[0], -1, -1)[:, :decoding_length, :]
            if not self.simple:
                dec_seq = torch.cat([z, dec_posemb], dim=1)
            else:
                dec_seq = z + dec_posemb
        else:
            dec_seq = z
        if self.rope_theta:
            if self.mix == 'mixed':
                freqs = self.compute_cis(self.dec_freqs.to(z.device), self.t_x.to(z.device), self.t_y.to(z.device))
            else:
                if self.dim == '1d':
                    length = torch.Tensor(np.arange(dec_seq.shape[1]))
                    freqs = self.compute_cis(length.to(z.device))
                else:
                    freqs = self.compute_cis(self.t_x.to(z.device), self.t_y.to(z.device))
        else:
            freqs = None
        dec_seq = self.decoder(dec_seq, freqs=freqs)
        if not self.simple:
            x = dec_seq[:, z.shape[1]:, :]
        else:
            x = dec_seq
        return self.tubelet_decode(x, grid_size)

    def compress_code(self, x, token_count=None):
        B, L, _ = x.shape
        mask_tokens = self.mask_token.expand_as(x).type(x.dtype)
        if token_count is not None:
            x[:, token_count:, :] = mask_tokens[:, token_count:, :]
        else:
            random_indices = np.random.randint(0, len(self.lengths), size=(B,))
            compressed_lengths = self.lengths[random_indices]
            compressed_lengths = torch.tensor(compressed_lengths, device=x.device).unsqueeze(1)
            mask = torch.arange(L).unsqueeze(0).to(x.device) >= compressed_lengths
            x[mask] = mask_tokens[mask]
        return x

    def forward(self, x, sample_posterior=True, token_count=None):
        posterior, grid_size = self.encode(x)
        if sample_posterior:
            x = posterior.sample()
        else:
            x = posterior.mode()
        if self.lengths is not None:
            x = self.compress_code(x, token_count=token_count)
        x = self.decode(x, grid_size)
        return x, posterior

def Model(**kw):  # pylint: disable=invalid-name
  """Factory function, because linen really don't like what I'm doing!"""
  return AE(**kw)
