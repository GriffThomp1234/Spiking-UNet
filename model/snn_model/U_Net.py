################################################
#@markdown ### **Network**
#@markdown
#@markdown Defines a 1D UNet architecture `ConditionalUnet1D`
#@markdown as the noies prediction network
#@markdown
#@markdown Components
#@markdown - `SinusoidalPosEmb` Positional encoding for the diffusion iteration k
#@markdown - `Downsample1d` Strided convolution to reduce temporal resolution
#@markdown - `Upsample1d` Transposed convolution to increase temporal resolution
#@markdown - `Conv1dBlock` Conv1d --> GroupNorm --> Mish
#@markdown - `ConditionalResidualBlock1D` Takes two inputs `x` and `cond`. \
#@markdown `x` is passed through 2 `Conv1dBlock` stacked together with residual connection.
#@markdown `cond` is applied to `x` with [FiLM](https://arxiv.org/abs/1709.07871) conditioning.

import math
from typing import Union
from torch import nn
import torch
from Spiking_UNet.snn.spiking_neuron import IF_Neuron

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class SpikingConv1dBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8, reset_method='reduce_by_zero'):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            IF_Neuron(v_threshold=1.0, v_reset=0.0, reset_method=reset_method)
        )

    def forward(self, x):
        return self.block(x)

class SpikingConditionalResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3, n_groups=8, reset_method='reduce_by_zero'):
        super().__init__()

        self.blocks = nn.ModuleList([
            SpikingConv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups, reset_method=reset_method),
            SpikingConv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups, reset_method=reset_method),
        ])

        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            IF_Neuron(v_threshold=1.0, v_reset=0.0, reset_method=reset_method),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1))
        )

        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)

        embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:,0,...]
        bias = embed[:,1,...]
        out = scale * out + bias

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out

class SpikingConditionalUnet1D(nn.Module):
    def __init__(self, input_dim, global_cond_dim, diffusion_step_embed_dim=256, down_dims=[256,512,1024], kernel_size=5, n_groups=8, reset_method='reduce_by_zero'):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            IF_Neuron(v_threshold=1.0, v_reset=0.0, reset_method=reset_method),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]

        self.mid_modules = nn.ModuleList([
            SpikingConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups, reset_method=reset_method),
            SpikingConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups, reset_method=reset_method),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                SpikingConditionalResidualBlock1D(dim_in, dim_out, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups, reset_method=reset_method),
                SpikingConditionalResidualBlock1D(dim_out, dim_out, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups, reset_method=reset_method),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                SpikingConditionalResidualBlock1D(dim_out*2, dim_in, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups, reset_method=reset_method),
                SpikingConditionalResidualBlock1D(dim_in, dim_in, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups, reset_method=reset_method),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

        final_conv = nn.Sequential(
            SpikingConv1dBlock(start_dim, start_dim, kernel_size=kernel_size, reset_method=reset_method),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        print("number of parameters: {:e}".format(
            sum(p.numel() for p in self.parameters()))
        )

    def forward(self, sample, timestep, global_cond=None):
        sample = sample.moveaxis(-1, -2)

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], axis=-1)

        x = sample
        h = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)
        x = x.moveaxis(-1, -2)
        return x