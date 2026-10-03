"""PyTorch port of the lightweight 3D U-Net and the composite BCE+Dice loss.

Architecture, widths and loss are identical to ``src/model.py`` (Keras); the
converted LOSO weights in ``models/pytorch/`` reproduce the Keras models'
outputs (see ``tests/test_torch_model.py`` and ``src/convert_weights.py``).

Differences from the Keras version, all behaviour-preserving:

* Tensors are channels-first: ``(N, 1, D, H, W)``.
* The network returns logits; the sigmoid lives in the loss
  (``BCEWithLogitsLoss``, numerically more stable) and in inference.
* Weights are initialised with Glorot-uniform and zero biases, as in Keras.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from . import config


class ConvBlock(nn.Module):
    """Two 3x3x3 convolutions with ReLU (Keras ``Conv3D(f, 3, padding='same')`` x2)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool3d(2)

    def forward(self, x):
        c = self.conv(x)
        return c, self.pool(c)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch * 2, out_ch)   # upsampled + skip

    def forward(self, x, skip):
        x = torch.cat([self.up(x), skip], dim=1)
        return self.conv(x)


class LightweightUNet3D(nn.Module):
    """3D U-Net with encoder widths (16, 32, 64) and a 128-filter bottleneck.

    Input ``(N, 1, D, H, W)`` with D, H, W divisible by 8; output logits of the same shape.
    """

    def __init__(self, in_channels: int = 1,
                 encoder_filters=config.ENCODER_FILTERS,
                 bottleneck_filters: int = config.BOTTLENECK_FILTERS):
        super().__init__()
        f1, f2, f3 = encoder_filters
        fb = bottleneck_filters
        self.enc1 = EncoderBlock(in_channels, f1)
        self.enc2 = EncoderBlock(f1, f2)
        self.enc3 = EncoderBlock(f2, f3)
        self.bottleneck = ConvBlock(f3, fb)
        self.dec3 = DecoderBlock(fb, f3)
        self.dec2 = DecoderBlock(f3, f2)
        self.dec1 = DecoderBlock(f2, f1)
        self.out = nn.Conv3d(f1, 1, kernel_size=1)
        self._init_like_keras()

    def _init_like_keras(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def conv_layers(self):
        """Conv layers in the same order as the Keras model's layers (used for weight conversion)."""
        return [m for m in self.modules() if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d))]

    def forward(self, x):
        c1, p1 = self.enc1(x)
        c2, p2 = self.enc2(p1)
        c3, p3 = self.enc3(p2)
        b = self.bottleneck(p3)
        d3 = self.dec3(b, c3)
        d2 = self.dec2(d3, c2)
        d1 = self.dec1(d2, c1)
        return self.out(d1)


def build_lightweight_unet() -> LightweightUNet3D:
    return LightweightUNet3D()


def load_model(path, device: str = "cpu") -> LightweightUNet3D:
    """Load a converted or PyTorch-trained ``state_dict`` (``models/pytorch/*.pt``)."""
    model = build_lightweight_unet()
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model.to(device).eval()


def dice_coef(y_true, y_pred, smooth: float = 1e-6):
    """Soft Dice coefficient on probabilities (same formula as the Keras version)."""
    y_true_f = y_true.float().reshape(-1)
    y_pred_f = y_pred.reshape(-1)
    intersection = (y_true_f * y_pred_f).sum()
    return (2.0 * intersection + smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth)


class BCEDiceLoss(nn.Module):
    """Composite loss: binary cross-entropy plus (1 - soft Dice). Takes logits."""

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, y_true):
        y_true = y_true.float()
        return self.bce(logits, y_true) + (1.0 - dice_coef(y_true, torch.sigmoid(logits)))


class TorchPredictor:
    """Wraps a PyTorch model in the Keras ``predict`` interface.

    Takes and returns channels-last NumPy arrays ``(N, D, H, W, 1)`` with sigmoid
    probabilities, so ``src.inference.sliding_window_inference`` and the
    evaluation code work unchanged with a PyTorch model.
    """

    def __init__(self, model: nn.Module, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()

    @torch.no_grad()
    def predict(self, x, batch_size: int = 8, verbose: int = 0):
        out = []
        for i in range(0, len(x), batch_size):
            batch = torch.from_numpy(np.ascontiguousarray(x[i:i + batch_size], dtype=np.float32))
            batch = batch.permute(0, 4, 1, 2, 3).to(self.device)           # NDHWC -> NCDHW
            probs = torch.sigmoid(self.model(batch)).permute(0, 2, 3, 4, 1)  # back to NDHWC
            out.append(probs.cpu().numpy())
        return np.concatenate(out)
