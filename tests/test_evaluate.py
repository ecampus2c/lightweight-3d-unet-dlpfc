"""Tests for the segmentation metrics."""
import numpy as np

from src import evaluate as ev


def _cube(shape, lo, hi):
    m = np.zeros(shape, dtype=np.uint8)
    m[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = 1
    return m


def test_dice_identical_is_one():
    m = _cube((20, 20, 20), (5, 5, 5), (15, 15, 15))
    assert abs(ev.dice_score(m, m) - 1.0) < 1e-4


def test_dice_disjoint_is_zero():
    a = _cube((20, 20, 20), (0, 0, 0), (5, 5, 5))
    b = _cube((20, 20, 20), (10, 10, 10), (15, 15, 15))
    assert ev.dice_score(a, b) < 1e-4


def test_dice_half_overlap():
    a = _cube((20, 20, 20), (0, 0, 0), (10, 10, 10))   # 1000 voxels
    b = _cube((20, 20, 20), (5, 0, 0), (15, 10, 10))   # 1000 voxels, 500 shared
    # Dice = 2*500 / (1000 + 1000) = 0.5
    assert abs(ev.dice_score(a, b) - 0.5) < 1e-3


def test_centroid_error_zero_for_identical():
    m = _cube((20, 20, 20), (5, 5, 5), (15, 15, 15))
    assert ev.centroid_error(m, m) == 0.0


def test_centroid_error_known_shift():
    a = _cube((30, 30, 30), (0, 0, 0), (10, 10, 10))
    b = _cube((30, 30, 30), (3, 0, 0), (13, 10, 10))   # shifted +3 along axis 0
    assert abs(ev.centroid_error(a, b) - 3.0) < 1e-6


def test_centroid_error_nan_when_empty():
    a = _cube((10, 10, 10), (2, 2, 2), (5, 5, 5))
    empty = np.zeros((10, 10, 10), dtype=np.uint8)
    assert np.isnan(ev.centroid_error(a, empty))


def test_center_of_mass_none_when_empty():
    assert ev.center_of_mass(np.zeros((4, 4, 4), dtype=np.uint8)) is None
