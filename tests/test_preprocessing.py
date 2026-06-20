"""Tests for the pure-NumPy preprocessing utilities."""
import numpy as np

from src import preprocessing as pp


def test_crop_or_pad_pads_small_volume():
    v = np.ones((100, 110, 90), dtype=np.float32)
    out = pp.center_crop_or_pad(v, (128, 128, 128))
    assert out.shape == (128, 128, 128)
    assert out.sum() == v.sum()                    # padding adds only zeros


def test_crop_or_pad_crops_large_volume():
    v = np.ones((150, 150, 150), dtype=np.float32)
    out = pp.center_crop_or_pad(v, (128, 128, 128))
    assert out.shape == (128, 128, 128)
    assert np.all(out == 1)                         # central region is all ones


def test_crop_or_pad_is_centered():
    v = np.zeros((64, 64, 64), dtype=np.float32)
    v[0, 0, 0] = 5.0
    out = pp.center_crop_or_pad(v, (128, 128, 128))
    assert out[32, 32, 32] == 5.0                   # offset by (128-64)/2 = 32


def test_zscore_zero_mean_unit_std_on_brain():
    v = np.zeros((16, 16, 16), dtype=np.float32)
    v[4:12, 4:12, 4:12] = np.random.RandomState(0).uniform(50, 150, size=(8, 8, 8))
    out = pp.normalize_zscore(v)
    brain = v > 0
    assert abs(out[brain].mean()) < 1e-4
    assert abs(out[brain].std() - 1.0) < 1e-3
    assert np.all(out[~brain] == 0)                 # background untouched


def test_zscore_all_zero_volume():
    v = np.zeros((8, 8, 8), dtype=np.float32)
    out = pp.normalize_zscore(v)
    assert np.all(out == 0)


def test_label_matching_no_substring_collision():
    labels = ["case10_seg.nii", "case1_seg.nii", "case2_seg.nii"]
    assert pp._label_for_image("case1.nii", labels) == "case1_seg.nii"
    assert pp._label_for_image("case10.nii", labels) == "case10_seg.nii"
    assert pp._label_for_image("case99.nii", labels) is None
