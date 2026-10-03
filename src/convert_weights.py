"""Convert the trained Keras LOSO models (``models/*.h5``) to PyTorch ``state_dict`` files.

Reads the HDF5 weights directly with ``h5py``, so TensorFlow is not required::

    python -m src.convert_weights                      # models/*.h5 -> models/pytorch/*.pt
    python -m src.convert_weights --h5 models/best_model_case1.h5 --out case1.pt

Keras kernel layouts are permuted to PyTorch's:

* ``Conv3D``          (kD, kH, kW, in, out)  -> ``Conv3d``          (out, in, kD, kH, kW)
* ``Conv3DTranspose`` (kD, kH, kW, out, in)  -> ``ConvTranspose3d`` (in, out, kD, kH, kW)

Both are the same axis permutation ``(4, 3, 0, 1, 2)``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch

from . import config
from .torch_model import build_lightweight_unet


def _decode(names):
    return [n.decode() if isinstance(n, bytes) else str(n) for n in names]


def read_keras_conv_weights(h5_path):
    """Return ``[(layer_name, kernel, bias), ...]`` for every conv layer, in model order."""
    with h5py.File(h5_path, "r") as f:
        root = f["model_weights"] if "model_weights" in f else f
        layers = []
        for name in _decode(root.attrs["layer_names"]):
            group = root[name]
            weight_names = _decode(group.attrs.get("weight_names", []))
            if not weight_names:
                continue   # Input, pooling, concatenate: no weights
            arrays = {w.split("/")[-1].split(":")[0]: np.array(group[w]) for w in weight_names}
            layers.append((name, arrays["kernel"], arrays["bias"]))
    return layers


def convert(h5_path, out_path=None):
    """Load a Keras ``.h5`` into the PyTorch model; optionally save its ``state_dict``."""
    model = build_lightweight_unet()
    convs = model.conv_layers()
    keras_layers = read_keras_conv_weights(h5_path)
    if len(keras_layers) != len(convs):
        raise ValueError(f"{h5_path}: {len(keras_layers)} Keras conv layers, expected {len(convs)}")

    with torch.no_grad():
        for (name, kernel, bias), layer in zip(keras_layers, convs):
            is_transpose = "transpose" in name
            if is_transpose != isinstance(layer, torch.nn.ConvTranspose3d):
                raise ValueError(f"{h5_path}: layer {name} does not match {type(layer).__name__}")
            weight = torch.from_numpy(np.ascontiguousarray(kernel.transpose(4, 3, 0, 1, 2)))
            if weight.shape != layer.weight.shape:
                raise ValueError(f"{name}: shape {tuple(weight.shape)} != {tuple(layer.weight.shape)}")
            layer.weight.copy_(weight)
            layer.bias.copy_(torch.from_numpy(bias))

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out_path)
    return model.eval()


def main():
    parser = argparse.ArgumentParser(description="Convert Keras .h5 LOSO models to PyTorch.")
    parser.add_argument("--h5", help="Single .h5 file (default: every models/*.h5)")
    parser.add_argument("--out", help="Output .pt path for --h5")
    parser.add_argument("--models-dir", default=config.MODELS_DIR)
    args = parser.parse_args()

    if args.h5:
        out = args.out or Path(args.h5).with_suffix(".pt")
        convert(args.h5, out)
        print(f"{args.h5} -> {out}")
        return

    models_dir = Path(args.models_dir)
    for h5 in sorted(models_dir.glob("*.h5")):
        out = models_dir / "pytorch" / h5.with_suffix(".pt").name
        convert(h5, out)
        print(f"{h5} -> {out}")


if __name__ == "__main__":
    main()
