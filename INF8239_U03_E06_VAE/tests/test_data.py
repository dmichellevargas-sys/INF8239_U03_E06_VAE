import gzip

import numpy as np
import pytest

from inf8239_u03_gen.data import fingerprint, normalize_images, read_idx, stratified_split, validate_images


def test_normalization_contract():
    raw = np.array([[[0] * 28] * 28, [[255] * 28] * 28], dtype=np.uint8)
    result = normalize_images(raw)
    validate_images(result)
    assert result.shape == (2, 28, 28, 1)
    assert result.min() == 0 and result.max() == 1


def test_rejects_wrong_shape():
    with pytest.raises(ValueError, match="Forma inesperada"):
        validate_images(np.zeros((2, 28, 28)))


def test_rejects_values_outside_unit_interval():
    with pytest.raises(ValueError, match="entre 0 y 1"):
        validate_images(np.full((1, 28, 28, 1), 2.0))


def test_read_idx_roundtrip(tmp_path):
    images = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
    header = b"\x00\x00\x08\x03" + b"".join(int(d).to_bytes(4, "big") for d in images.shape)
    path = tmp_path / "x.gz"
    with gzip.open(path, "wb") as f:
        f.write(header + images.tobytes())
    np.testing.assert_array_equal(read_idx(path), images)


def test_stratified_split_is_disjoint_balanced_and_reproducible():
    labels = np.repeat(np.arange(10), 50)
    tr, va = stratified_split(labels, 0.1, seed=42)
    assert set(tr).isdisjoint(va) and (np.bincount(labels[va]) == 5).all()
    assert fingerprint(va) == fingerprint(stratified_split(labels, 0.1, seed=42)[1])
