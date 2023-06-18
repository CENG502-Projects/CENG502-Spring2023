# Copyright (c) 2023 Devrim Cavusoglu
# Copyright (c) 2021 Phil Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
MLP Mixer. This file is taken and adapted from Phil Wang's
Pytorch implementation of MLP-Mixer. See below the original file
https://github.com/lucidrains/mlp-mixer-pytorch/blob/main/mlp_mixer_pytorch/mlp_mixer_pytorch.py
"""

from functools import partial

import torch
from einops.layers.torch import Rearrange, Reduce
from torch import nn

pair = lambda x: x if isinstance(x, tuple) else (x, x)


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x


class FFN(nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim: int,
        dropout: float = 0.0,
        mixing_type: str = "channel",
        distillation: bool = False,
    ):
        super().__init__()
        layer_dense = nn.Linear if mixing_type == "channel" else partial(nn.Conv1d, kernel_size=1)
        self.dense1 = layer_dense(dim, hidden_dim)
        self.dense2 = layer_dense(hidden_dim, dim) if not distillation else layer_dense(hidden_dim, 1)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.gelu(self.dense1(x))
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dropout2(x)
        return x


class MixerBlock(nn.Module):
    def __init__(
        self, dim, n_patches, spatial_scale: float = 0.5, channel_scale: float = 4, dropout: float = 0.0
    ):
        super().__init__()
        token_hidden = int(dim * spatial_scale)
        channel_hidden = int(dim * channel_scale)
        self.token_mixer = PreNormResidual(
            dim, FFN(n_patches, token_hidden, dropout, mixing_type="token")
        )
        self.channel_mixer = PreNormResidual(
            dim, FFN(dim, channel_hidden, dropout, mixing_type="channel")
        )

    def forward(self, z: torch.Tensor):
        u = self.token_mixer(z)
        z = self.channel_mixer(u)
        return z


class MLPMixer(nn.Module):
    def __init__(
        self,
        image_size,
        channels,
        patch_size,
        dim,
        depth,
        num_classes,
        f_spatial_expansion: float = 0.5,
        f_channel_expansion: float = 4,
        dropout=0.0,
    ):
        super().__init__()
        h, w = pair(image_size)
        assert (image_size % patch_size) == 0 and (w % h) == 0, "image must be divisible by patch size"
        n_patches = (h // patch_size) * (w // patch_size)

        self.depth = depth

        self.patchifier = Rearrange(
            "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size
        )  # Extract patches
        self.per_patch_fc = nn.Linear((patch_size ** 2) * channels, dim)
        self.mixer_blocks = nn.ModuleList(
            [
                MixerBlock(dim, n_patches, f_spatial_expansion, f_channel_expansion, dropout)
                for _ in range(depth)
            ]
        )
        self.ln = nn.LayerNorm(dim)
        self.gap = Reduce("b n c -> b c", "mean")
        self.classifier = nn.Linear(dim, num_classes)

    def forward_features(self, x):
        x = self.patchifier(x)
        z = self.per_patch_fc(x)
        for i, layer in enumerate(self.mixer_blocks):
            z = layer(z)
        z = self.ln(z)
        z = self.gap(z)
        return z

    def forward(self, x):
        z = self.forward_features(x)
        outputs = self.classifier(z)
        return outputs
