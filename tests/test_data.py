"""Tests for preprocessed-data loading and the patch generator."""
import numpy as np

from src import config
from src import data


def test_generator_yields_correct_patch_shapes():
    rng = np.random.RandomState(0)
    subject = {
        "image": rng.rand(128, 128, 128).astype(np.float32),
        "label": (rng.rand(128, 128, 128) > 0.9).astype(np.uint8),
    }
    gen = data.ram_balanced_generator([subject], batch_size=2)
    x, y = next(gen)
    assert x.shape == (2, *config.PATCH_SIZE, 1)
    assert y.shape == (2, *config.PATCH_SIZE, 1)
    assert x.dtype == np.float32 or x.dtype == np.float64


def test_loaded_dataset_integrity():
    """Every released .npz must have the expected shape, dtype and binary labels."""
    dataset = data.load_preprocessed()
    if not dataset:
        import pytest
        pytest.skip("No preprocessed .npz present in data/preprocessed/.")
    for d in dataset:
        assert d["image"].shape == config.TARGET_SHAPE
        assert d["label"].shape == config.TARGET_SHAPE
        assert d["image"].dtype == np.float32
        assert d["label"].dtype == np.uint8
        assert set(np.unique(d["label"])).issubset({0, 1})
