"""Fourier Neural Operator (2D).

A compact, self-contained implementation of the FNO from Li et al. (2021),
"Fourier Neural Operator for Parametric Partial Differential Equations".

The core idea: instead of a fixed convolution kernel, learn a linear
operator directly in Fourier space. Each spectral layer

    1. takes the 2D FFT of the input,
    2. keeps only the lowest `modes` frequencies,
    3. multiplies them by learned complex weights,
    4. transforms back,

and adds a pointwise (1x1 conv) residual path. Because the operator lives in
frequency space, it is resolution-invariant: a model trained at 64x64 can be
evaluated at other resolutions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """2D spectral convolution: learned multiplication on low Fourier modes."""

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # kept frequencies along dim 1
        self.modes2 = modes2  # kept frequencies along dim 2

        scale = 1.0 / (in_channels * out_channels)
        # Two corner blocks of the (Hermitian-symmetric) spectrum need weights.
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    @staticmethod
    def _compl_mul2d(inp: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # (batch, in, x, y) x (in, out, x, y) -> (batch, out, x, y)
        return torch.einsum("bixy,ioxy->boxy", inp, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, h, w = x.shape

        x_ft = torch.fft.rfft2(x)  # (batch, in, h, w//2 + 1)

        out_ft = torch.zeros(
            batch, self.out_channels, h, w // 2 + 1,
            dtype=torch.cfloat, device=x.device,
        )
        # Lowest modes from the top-left and bottom-left spectral corners.
        out_ft[:, :, : self.modes1, : self.modes2] = self._compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1:, : self.modes2] = self._compl_mul2d(
            x_ft[:, :, -self.modes1:, : self.modes2], self.weights2
        )

        return torch.fft.irfft2(out_ft, s=(h, w))


class FNO2d(nn.Module):
    """Stack of spectral layers with pointwise residual paths.

    Input/output are single-channel fields of shape (batch, 1, H, W). Two
    extra coordinate channels are appended internally so the operator can be
    spatially aware.
    """

    def __init__(
        self,
        modes: int = 12,
        width: int = 32,
        n_layers: int = 4,
        in_channels: int = 1,
    ):
        super().__init__()
        self.width = width
        self.n_layers = n_layers

        # Lift input (+ 2 coordinate channels) to `width` feature channels.
        self.lift = nn.Linear(in_channels + 2, width)

        self.spectral = nn.ModuleList(
            [SpectralConv2d(width, width, modes, modes) for _ in range(n_layers)]
        )
        self.pointwise = nn.ModuleList(
            [nn.Conv2d(width, width, 1) for _ in range(n_layers)]
        )

        self.project = nn.Sequential(
            nn.Linear(width, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    @staticmethod
    def _coord_grid(batch: int, h: int, w: int, device) -> torch.Tensor:
        ys = torch.linspace(0, 1, h, device=device)
        xs = torch.linspace(0, 1, w, device=device)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([gx, gy], dim=-1)  # (h, w, 2)
        return grid.unsqueeze(0).expand(batch, -1, -1, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, H, W) -> channels-last for the lifting MLP.
        batch, _, h, w = x.shape
        x = x.permute(0, 2, 3, 1)  # (batch, H, W, 1)

        grid = self._coord_grid(batch, h, w, x.device)
        x = torch.cat([x, grid], dim=-1)  # (batch, H, W, 3)
        x = self.lift(x)  # (batch, H, W, width)
        x = x.permute(0, 3, 1, 2)  # (batch, width, H, W)

        for spec, point in zip(self.spectral, self.pointwise):
            x = spec(x) + point(x)
            x = F.gelu(x)

        x = x.permute(0, 2, 3, 1)  # (batch, H, W, width)
        x = self.project(x)  # (batch, H, W, 1)
        return x.permute(0, 3, 1, 2)  # (batch, 1, H, W)


if __name__ == "__main__":
    model = FNO2d()
    n_params = sum(p.numel() for p in model.parameters())
    dummy = torch.randn(2, 1, 64, 64)
    out = model(dummy)
    print(f"FNO2d params: {n_params:,}")
    print(f"in {tuple(dummy.shape)} -> out {tuple(out.shape)}")
    # Resolution invariance: same weights, different grid.
    out_hi = model(torch.randn(2, 1, 96, 96))
    print(f"96x96 input  -> out {tuple(out_hi.shape)}  (resolution-invariant)")
