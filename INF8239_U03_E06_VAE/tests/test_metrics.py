import numpy as np
import pytest

from inf8239_u03_gen.metrics import (
    bce_per_image,
    class_entropy,
    frechet_distance,
    mse_per_image,
    nearest_distances,
    precision_recall,
)

RNG = np.random.default_rng(0)


def test_reconstruction_metrics_are_zero_for_identical_binary_images():
    x = RNG.integers(0, 2, (3, 28, 28, 1)).astype("float32")
    assert np.allclose(mse_per_image(x, x), 0)
    assert np.all(bce_per_image(x, x) < 1e-3)


def test_frechet_distance_zero_for_same_set_and_grows_with_shift():
    a = RNG.normal(size=(500, 4))
    assert frechet_distance(a, a) == pytest.approx(0.0, abs=1e-3)
    assert frechet_distance(a, a + 3) == pytest.approx(36.0, rel=0.05)


def test_precision_recall_detect_mode_collapse():
    real = RNG.normal(size=(400, 2))
    collapsed = RNG.normal(scale=0.05, size=(400, 2))  # todas las muestras en un punto
    precision, recall = precision_recall(real, collapsed, k=5)
    assert precision > 0.9 and recall < 0.3
    p_same, r_same = precision_recall(real, RNG.normal(size=(400, 2)), k=5)
    assert p_same > 0.8 and r_same > 0.8


def test_class_entropy_bounds():
    assert class_entropy(np.repeat(np.arange(10), 5)) == pytest.approx(1.0)
    assert class_entropy(np.zeros(50, dtype=int)) == pytest.approx(0.0)


def test_nearest_distances_detect_exact_copy():
    train = RNG.random((50, 28, 28, 1))
    queries = np.concatenate([train[:1], RNG.random((1, 28, 28, 1))])
    dist, idx = nearest_distances(train, queries)
    assert dist[0] == pytest.approx(0.0, abs=1e-6) and idx[0] == 0 and dist[1] > 0.1
