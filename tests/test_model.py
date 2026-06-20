"""Model and loss tests. Skipped automatically when TensorFlow is unavailable."""
import numpy as np
import pytest

tf = pytest.importorskip("tensorflow", reason="TensorFlow not installed")

from src import model as m   # noqa: E402  (import after skip guard)
from src import config       # noqa: E402


def test_unet_output_shape_matches_input():
    net = m.build_lightweight_unet((96, 96, 96, 1))
    x = np.zeros((1, 96, 96, 96, 1), dtype=np.float32)
    y = net.predict(x, verbose=0)
    assert y.shape == (1, 96, 96, 96, 1)
    assert 0.0 <= float(y.min()) and float(y.max()) <= 1.0   # sigmoid output


def test_unet_is_lightweight():
    net = m.build_lightweight_unet((96, 96, 96, 1))
    # Architecture is deliberately small; guard against accidental width changes.
    assert net.count_params() < 2_000_000


def test_dice_coef_identical_is_one():
    y = tf.constant(np.ones((1, 8, 8, 8, 1), dtype=np.float32))
    assert abs(float(m.dice_coef(y, y)) - 1.0) < 1e-4


def test_bce_dice_loss_nonnegative():
    y_true = tf.constant((np.random.rand(1, 8, 8, 8, 1) > 0.5).astype(np.float32))
    y_pred = tf.constant(np.random.rand(1, 8, 8, 8, 1).astype(np.float32))
    assert float(m.bce_dice_loss(y_true, y_pred)) >= 0.0
