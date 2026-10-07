"""Lectura IDX de Fashion-MNIST, normalización, validación y partición estratificada."""
from __future__ import annotations

import gzip
import hashlib
from pathlib import Path

import numpy as np

from .config import RAW_DIR


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as file:
        data = file.read()
    if data[0:2] != b"\x00\x00" or data[2] != 0x08:
        raise ValueError(f"Cabecera IDX no válida en {path}")
    ndim = data[3]
    shape = tuple(int.from_bytes(data[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim))
    array = np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim)
    if array.size != int(np.prod(shape)):
        raise ValueError(f"Tamaño inconsistente en {path}")
    return array.reshape(shape)


def load_fashion_mnist(raw_dir: Path = RAW_DIR):
    raw_dir = Path(raw_dir)
    if not (raw_dir / "train-images-idx3-ubyte.gz").exists():
        raise FileNotFoundError("Faltan los archivos IDX. Ejecute: uv run python scripts/download_data.py")
    return ((read_idx(raw_dir / "train-images-idx3-ubyte.gz"), read_idx(raw_dir / "train-labels-idx1-ubyte.gz")),
            (read_idx(raw_dir / "t10k-images-idx3-ubyte.gz"), read_idx(raw_dir / "t10k-labels-idx1-ubyte.gz")))


def normalize_images(images: np.ndarray) -> np.ndarray:
    return (images.astype("float32") / 255.0)[..., None]


def validate_images(images: np.ndarray) -> None:
    if images.ndim != 4 or images.shape[1:] != (28, 28, 1):
        raise ValueError(f"Forma inesperada: {images.shape}")
    if not np.isfinite(images).all() or images.min() < 0 or images.max() > 1:
        raise ValueError("Las imágenes deben contener valores finitos entre 0 y 1")


def stratified_split(labels: np.ndarray, valid_fraction: float, seed: int):
    rng = np.random.default_rng(seed)
    train_idx, valid_idx = [], []
    for cls in np.unique(labels):
        idx = np.flatnonzero(labels == cls)
        rng.shuffle(idx)
        n_valid = round(len(idx) * valid_fraction)
        valid_idx.append(idx[:n_valid])
        train_idx.append(idx[n_valid:])
    return np.sort(np.concatenate(train_idx)), np.sort(np.concatenate(valid_idx))


def fingerprint(indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()
