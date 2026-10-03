"""PyTorch port tests. Skipped automatically when PyTorch is unavailable;
the Keras-equivalence test additionally needs TensorFlow and h5py."""
import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch not installed")

from src import torch_model as tm   # noqa: E402  (import after skip guard)


def test_unet_output_shape_matches_input():
    net = tm.build_lightweight_unet().eval()
    with torch.no_grad():
        y = net(torch.zeros(1, 1, 32, 32, 32))
    assert y.shape == (1, 1, 32, 32, 32)   # logits, channels-first


def test_unet_has_same_parameter_count_as_keras():
    net = tm.build_lightweight_unet()
    # Identical architecture to src/model.py (1,400,561 parameters).
    assert sum(p.numel() for p in net.parameters()) == 1_400_561


def test_torch_predictor_matches_keras_interface():
    predictor = tm.TorchPredictor(tm.build_lightweight_unet(), device="cpu")
    y = predictor.predict(np.zeros((3, 32, 32, 32, 1), dtype=np.float32), batch_size=2)
    assert y.shape == (3, 32, 32, 32, 1)
    assert 0.0 <= float(y.min()) and float(y.max()) <= 1.0   # sigmoid probabilities


def test_dice_coef_identical_is_one():
    y = torch.ones(1, 1, 8, 8, 8)
    assert abs(float(tm.dice_coef(y, y)) - 1.0) < 1e-4


def test_bce_dice_loss_nonnegative():
    y_true = (torch.rand(1, 1, 8, 8, 8) > 0.5).float()
    logits = torch.randn(1, 1, 8, 8, 8)
    assert float(tm.BCEDiceLoss()(logits, y_true)) >= 0.0


def test_converted_weights_reproduce_keras(tmp_path):
    tf = pytest.importorskip("tensorflow", reason="TensorFlow not installed")
    pytest.importorskip("h5py", reason="h5py not installed")
    from src import model as keras_model
    from src.convert_weights import convert

    tf.random.set_seed(0)
    keras_net = keras_model.build_lightweight_unet((32, 32, 32, 1))
    h5_path = tmp_path / "random_unet.h5"
    keras_net.save(h5_path)

    x = np.random.default_rng(0).standard_normal((2, 32, 32, 32, 1)).astype(np.float32)
    expected = keras_net.predict(x, verbose=0)
    actual = tm.TorchPredictor(convert(h5_path), device="cpu").predict(x)
    assert np.abs(actual - expected).max() < 1e-4
