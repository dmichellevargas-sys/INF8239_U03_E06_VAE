"""Métricas de reconstrucción, fidelidad, diversidad y memorización."""
from __future__ import annotations

import numpy as np
from scipy import linalg
from sklearn.neighbors import NearestNeighbors


def bce_per_image(images: np.ndarray, reconstructions: np.ndarray) -> np.ndarray:
    x = images.reshape(len(images), -1)
    r = np.clip(reconstructions.reshape(len(images), -1), 1e-7, 1 - 1e-7)
    return -(x * np.log(r) + (1 - x) * np.log(1 - r)).sum(axis=1)


def mse_per_image(images: np.ndarray, reconstructions: np.ndarray) -> np.ndarray:
    return ((images - reconstructions) ** 2).reshape(len(images), -1).mean(axis=1)


def frechet_distance(features_a: np.ndarray, features_b: np.ndarray) -> float:
    """Distancia de Fréchet entre gaussianas ajustadas a dos conjuntos de características
    (la fórmula del FID de Heusel et al., 2017, aplicada con un clasificador del dominio)."""
    mu_a, mu_b = features_a.mean(axis=0), features_b.mean(axis=0)
    eps = 1e-6 * np.eye(features_a.shape[1])  # estabiliza covarianzas casi singulares
    cov_a, cov_b = np.cov(features_a, rowvar=False) + eps, np.cov(features_b, rowvar=False) + eps
    covmean = np.real(linalg.sqrtm(cov_a @ cov_b))
    return float(np.sum((mu_a - mu_b) ** 2) + np.trace(cov_a + cov_b - 2 * covmean))


def precision_recall(real: np.ndarray, fake: np.ndarray, k: int = 5) -> tuple[float, float]:
    """Precisión = fracción de muestras dentro de la variedad real (fidelidad);
    recall = fracción de reales dentro de la variedad generada (diversidad).
    Variedad = unión de bolas con radio igual a la distancia al k-ésimo vecino."""
    def radii(x):
        distances, _ = NearestNeighbors(n_neighbors=k + 1).fit(x).kneighbors(x)
        return distances[:, -1]

    def inside(points, support, support_radii):
        support_sq = (support ** 2).sum(axis=1)
        hits = []
        for begin in range(0, len(points), 500):
            block = points[begin:begin + 500]
            d2 = (block ** 2).sum(axis=1)[:, None] + support_sq[None, :] - 2 * block @ support.T
            hits.append((np.sqrt(np.maximum(d2, 0)) <= support_radii[None, :]).any(axis=1))
        return float(np.mean(np.concatenate(hits)))

    return inside(fake, real, radii(real)), inside(real, fake, radii(fake))


def class_entropy(predicted: np.ndarray, n_classes: int = 10) -> float:
    """Entropía normalizada (0-1) del histograma de clases predichas: 1 = clases uniformes."""
    p = np.bincount(predicted, minlength=n_classes) / len(predicted)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(n_classes))


def nearest_distances(reference: np.ndarray, queries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distancia RMS por píxel al vecino más cercano de `reference` (y su índice)."""
    ref = reference.reshape(len(reference), -1)
    q = queries.reshape(len(queries), -1)
    distances, idx = NearestNeighbors(n_neighbors=1, algorithm="brute").fit(ref).kneighbors(q)
    return distances.ravel() / np.sqrt(ref.shape[1]), idx.ravel()
