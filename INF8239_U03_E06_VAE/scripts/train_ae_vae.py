"""Ejercicio 06: AE vs. VAE en Fashion-MNIST con reconstrucción, muestreo, interpolación,
fidelidad, diversidad, memorización y costo.

Protocolo (declarado antes de evaluar en prueba):
  * Partición: 10 % estratificado del entrenamiento oficial como validación (semilla 42);
    la prueba oficial (10 000) solo se usa para el reporte final.
  * Modelos: AE y VAE con la MISMA arquitectura densa y latentes de 2 y 16 dimensiones.
    Adam, lote 256, hasta EPOCHS épocas, early stopping sobre la pérdida de validación.
  * Muestreo: VAE desde N(0, I); AE desde una gaussiana ajustada a sus códigos de entrenamiento.
  * Fidelidad/diversidad: distancia de Fréchet, precisión y recall k-NN y entropía de clases,
    sobre características de una CNN entrenada solo con el conjunto de entrenamiento.
  * Memoria: distancia de cada muestra a su vecino más cercano del entrenamiento, comparada con
    la de imágenes de prueba reales (que el modelo nunca vio).
  * Decisión: menor distancia de Fréchet en VALIDACIÓN entre los modelos que no memorizan
    (≤ 2 % de muestras más cercanas al entrenamiento que el percentil 1 de las imágenes reales).

Uso:  uv run python scripts/train_ae_vae.py [--epochs 15] [--limit 0]
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from time import perf_counter

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from inf8239_u03_gen.config import (
    CLASS_NAMES_ES,
    KNN_K,
    MODELS,
    N_GENERATED,
    N_MEMORY,
    N_REAL,
    REPORTS,
    SEED,
    VALID_FRACTION,
)
from inf8239_u03_gen.data import (
    fingerprint,
    load_fashion_mnist,
    normalize_images,
    stratified_split,
    validate_images,
)
from inf8239_u03_gen.metrics import (
    bce_per_image,
    class_entropy,
    frechet_distance,
    mse_per_image,
    nearest_distances,
    precision_recall,
)
from inf8239_u03_gen.models import build_ae_parts, build_classifier, build_encoder_decoder
from inf8239_u03_gen.vae import VAE, kl_per_dimension

CONFIGS = [("ae", 2), ("ae", 16), ("vae", 2), ("vae", 16)]
LABELS = {"ae_z2": "AE z=2", "ae_z16": "AE z=16", "vae_z2": "VAE z=2", "vae_z16": "VAE z=16"}
MEMORY_LIMIT = 0.02


def set_determinism() -> None:
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()


class Trained:
    def __init__(self, kind, latent, encoder, decoder, history, seconds, epochs):
        self.kind, self.latent, self.encoder, self.decoder = kind, latent, encoder, decoder
        self.name = f"{kind}_z{latent}"
        self.history, self.seconds, self.epochs = history, seconds, epochs

    def encode(self, x):
        out = self.encoder.predict(x, batch_size=1024, verbose=0)
        return out[0] if self.kind == "vae" else out

    def reconstruct(self, x):
        return self.decoder.predict(self.encode(x), batch_size=1024, verbose=0)


def prepare(limit: int):
    (x_all, y_all), (x_test, y_test) = load_fashion_mnist()
    tr_idx, va_idx = stratified_split(y_all, VALID_FRACTION, SEED)
    data = {"x_train": normalize_images(x_all[tr_idx]), "y_train": y_all[tr_idx],
            "x_valid": normalize_images(x_all[va_idx]), "y_valid": y_all[va_idx],
            "x_test": normalize_images(x_test), "y_test": y_test}
    if limit:
        for split, n in (("train", limit), ("valid", limit // 5), ("test", limit // 5)):
            data[f"x_{split}"], data[f"y_{split}"] = data[f"x_{split}"][:n], data[f"y_{split}"][:n]
    for split in ("train", "valid", "test"):
        validate_images(data[f"x_{split}"])
    split_log = {"train": len(data["y_train"]), "valid": len(data["y_valid"]), "test": len(data["y_test"]),
                 "valid_fraction": VALID_FRACTION, "seed": SEED, "limit": limit,
                 "train_index_sha256": fingerprint(tr_idx), "valid_index_sha256": fingerprint(va_idx)}
    (REPORTS / "split.json").write_text(json.dumps(split_log, indent=2), encoding="utf-8")
    return data, split_log


def train(kind: str, latent: int, data: dict, epochs: int) -> Trained:
    set_determinism()
    stop = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)]
    if kind == "ae":
        encoder, decoder = build_ae_parts(tf, latent)
        inputs = tf.keras.Input((28, 28, 1))
        model = tf.keras.Model(inputs, decoder(encoder(inputs)))
        # misma pérdida que el término de reconstrucción del VAE: BCE sumada por imagen
        model.compile(optimizer="adam", loss=lambda y, p: 784 * tf.keras.losses.binary_crossentropy(
            tf.reshape(y, (-1, 784)), tf.reshape(p, (-1, 784))))
        start = perf_counter()
        hist = model.fit(data["x_train"], data["x_train"], validation_data=(data["x_valid"], data["x_valid"]),
                         epochs=epochs, batch_size=256, callbacks=stop, verbose=2)
    else:
        encoder, decoder = build_encoder_decoder(tf, latent)
        model = VAE(encoder, decoder)
        model.compile(optimizer="adam")
        start = perf_counter()
        hist = model.fit(data["x_train"], validation_data=(data["x_valid"],), epochs=epochs,
                         batch_size=256, callbacks=stop, verbose=2)
    seconds = perf_counter() - start
    history = pd.DataFrame(hist.history)
    history.insert(0, "epoch", np.arange(1, len(history) + 1))
    history.insert(0, "model", f"{kind}_z{latent}")
    return Trained(kind, latent, encoder, decoder, history, seconds, len(history))


def train_classifier(data: dict):
    set_determinism()
    clf = build_classifier(tf)
    clf.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    clf.fit(data["x_train"], data["y_train"], validation_data=(data["x_valid"], data["y_valid"]),
            epochs=3, batch_size=128, verbose=2)
    features = tf.keras.Model(clf.inputs, clf.get_layer("features").output)
    accuracy = float((clf.predict(data["x_test"], verbose=0).argmax(1) == data["y_test"]).mean())
    return clf, features, accuracy


def sample(model: Trained, n: int, data: dict) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    if model.kind == "vae":
        z = rng.normal(size=(n, model.latent))
    else:
        codes = model.encode(data["x_train"])
        z = rng.multivariate_normal(codes.mean(0), np.cov(codes, rowvar=False), size=n)
    return model.decoder.predict(z.astype("float32"), batch_size=1024, verbose=0)


def ssim(a, b) -> np.ndarray:
    return tf.image.ssim(tf.constant(a), tf.constant(b), max_val=1.0, filter_size=7).numpy()


def run(epochs: int = 15, limit: int = 0) -> dict:
    REPORTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    data, split = prepare(limit)
    models = {}
    for kind, latent in CONFIGS:
        m = train(kind, latent, data, epochs)
        models[m.name] = m
        m.decoder.save(MODELS / f"{m.name}_decoder.keras")
        m.encoder.save(MODELS / f"{m.name}_encoder.keras")
    history = pd.concat([m.history for m in models.values()], ignore_index=True)
    history.to_csv(REPORTS / "training_history.csv", index=False)
    clf, features, clf_accuracy = train_classifier(data)

    # ---------------- reconstrucción (prueba)
    rec_rows, per_class = [], []
    xt, yt = data["x_test"], data["y_test"]
    for m in models.values():
        rec = m.reconstruct(xt)
        bce, mse, ss = bce_per_image(xt, rec), mse_per_image(xt, rec), ssim(xt, rec)
        row = {"model": m.name, "bce": bce.mean(), "mse": mse.mean(), "ssim": ss.mean(),
               "classifier_agreement": float((clf.predict(rec, verbose=0).argmax(1) == yt).mean())}
        if m.kind == "vae":
            mean, logvar = m.encoder.predict(xt, batch_size=1024, verbose=0)
            kl_dims = kl_per_dimension(mean, logvar).numpy()
            row.update({"kl": float(kl_dims.sum()), "elbo_bce_plus_kl": float(bce.mean() + kl_dims.sum()),
                        "active_dims": int((kl_dims > 0.01).sum())})
        rec_rows.append(row)
        per_class.append(pd.Series(mse).groupby(yt).mean().rename(m.name))
    rec_table = pd.DataFrame(rec_rows)
    rec_table.to_csv(REPORTS / "reconstruction_metrics.csv", index=False)
    per_class_table = pd.concat(per_class, axis=1)
    per_class_table.index = [CLASS_NAMES_ES[i] for i in per_class_table.index]
    per_class_table.to_csv(REPORTS / "reconstruction_per_class.csv")

    # ---------------- generación: fidelidad, diversidad, memoria
    rng = np.random.default_rng(SEED)
    n_real = min(N_REAL, len(data["x_valid"]), len(xt))
    real_valid = data["x_valid"][rng.choice(len(data["x_valid"]), n_real, replace=False)]
    real_test = xt[rng.choice(len(xt), n_real, replace=False)]
    f_valid = features.predict(real_valid, verbose=0)
    f_test = features.predict(real_test, verbose=0)
    test_nn, _ = nearest_distances(data["x_train"], xt[: min(N_MEMORY, len(xt))])
    copy_threshold = float(np.quantile(test_nn, 0.01))
    gen_rows, samples = [], {}
    reference = {"model": "real (validación vs. prueba)", "fd_valid": frechet_distance(f_test, f_valid),
                 "precision": precision_recall(f_valid, f_test, KNN_K)[0],
                 "recall": precision_recall(f_valid, f_test, KNN_K)[1],
                 "class_entropy": class_entropy(clf.predict(real_test, verbose=0).argmax(1)),
                 "mean_confidence": float(clf.predict(real_test, verbose=0).max(1).mean()),
                 "nn_train_median": float(np.median(test_nn)), "share_closer_than_real_p1": 0.01}
    gen_rows.append(reference)
    for m in models.values():
        start = perf_counter()
        gen = sample(m, N_GENERATED, data)
        ms = (perf_counter() - start) * 1000 / N_GENERATED
        samples[m.name] = gen
        f_gen = features.predict(gen, verbose=0)
        proba = clf.predict(gen, verbose=0)
        p_valid, r_valid = precision_recall(f_valid, f_gen, KNN_K)
        p_test, r_test = precision_recall(f_test, f_gen, KNN_K)
        nn, _ = nearest_distances(data["x_train"], gen[:N_MEMORY])
        gen_rows.append({"model": m.name, "fd_valid": frechet_distance(f_gen, f_valid),
                         "fd_test": frechet_distance(f_gen, f_test),
                         "precision_valid": p_valid, "recall_valid": r_valid,
                         "precision": p_test, "recall": r_test,
                         "class_entropy": class_entropy(proba.argmax(1)),
                         "mean_confidence": float(proba.max(1).mean()),
                         "nn_train_median": float(np.median(nn)),
                         "share_closer_than_real_p1": float((nn < copy_threshold).mean()),
                         "generation_ms_per_image": ms})
    gen_table = pd.DataFrame(gen_rows)
    candidates = gen_table[gen_table["model"].isin(models) & (gen_table["share_closer_than_real_p1"] <= MEMORY_LIMIT)]
    selected = str(candidates.sort_values("fd_valid").iloc[0]["model"])
    gen_table["selected"] = gen_table["model"] == selected
    gen_table.to_csv(REPORTS / "generation_metrics.csv", index=False)

    cost = pd.DataFrame([{
        "model": m.name, "latent_dim": m.latent, "parameters": int(m.encoder.count_params() + m.decoder.count_params()),
        "decoder_parameters": int(m.decoder.count_params()),
        "decoder_kb": (MODELS / f"{m.name}_decoder.keras").stat().st_size / 1024,
        "epochs_run": m.epochs, "train_seconds": m.seconds, "seconds_per_epoch": m.seconds / m.epochs,
    } for m in models.values()])
    cost.to_csv(REPORTS / "cost.csv", index=False)

    # ---------------- figuras
    figures(models, data, samples, clf, history, per_class_table, selected)
    memory_gallery(samples[selected], data["x_train"], selected)
    models["vae_z2"].decoder.save(MODELS / "decoder.keras")  # para la app del explorador 2D

    results = {
        "split": split, "selected_model": selected, "classifier_test_accuracy": clf_accuracy,
        "copy_threshold_rms": copy_threshold, "memory_limit": MEMORY_LIMIT,
        "selection_rule": "menor distancia de Fréchet en validación entre modelos con ≤ 2 % de muestras "
                          "más cercanas al entrenamiento que el percentil 1 de imágenes reales de prueba",
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                        "processor": platform.processor() or platform.machine(), "cpu_count": os.cpu_count(),
                        "tensorflow": tf.__version__, "gpu": [d.name for d in tf.config.list_physical_devices("GPU")],
                        "numpy": np.__version__},
    }
    (REPORTS / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return results | {"reconstruction": rec_table, "generation": gen_table, "cost": cost}


def figures(models, data, samples, clf, history, per_class_table, selected) -> None:
    xt, yt = data["x_test"], data["y_test"]
    idx = [int(np.flatnonzero(yt == c)[0]) for c in range(10)]
    rows = [("Original", xt[idx])] + [(LABELS[n], m.reconstruct(xt[idx])) for n, m in models.items()]
    fig, axes = plt.subplots(len(rows), 10, figsize=(11, 1.25 * len(rows)))
    for r, (label, imgs) in enumerate(rows):
        for c in range(10):
            axes[r, c].imshow(imgs[c].squeeze(), cmap="gray_r", vmin=0, vmax=1)
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
        axes[r, 0].set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9)
    fig.suptitle("Reconstrucción de una imagen de prueba por clase")
    fig.tight_layout()
    fig.savefig(REPORTS / "fig_reconstructions.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(len(samples), 12, figsize=(12, 1.25 * len(samples)))
    for r, (name, gen) in enumerate(samples.items()):
        for c in range(12):
            axes[r, c].imshow(gen[c].squeeze(), cmap="gray_r", vmin=0, vmax=1)
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
        axes[r, 0].set_ylabel(LABELS[name] + (" ★" if name == selected else ""), rotation=0, ha="right",
                              va="center", fontsize=9)
    fig.suptitle("Muestras nuevas (VAE: desde N(0, I); AE: desde una gaussiana ajustada a sus códigos)")
    fig.tight_layout()
    fig.savefig(REPORTS / "fig_samples.png", dpi=150)
    plt.close(fig)

    # interpolación: zapatilla -> botín y camiseta -> pantalón
    pairs = [(int(np.flatnonzero(yt == 7)[1]), int(np.flatnonzero(yt == 9)[1])),
             (int(np.flatnonzero(yt == 0)[1]), int(np.flatnonzero(yt == 1)[1]))]
    steps = np.linspace(0, 1, 10)
    rows, conf_rows = [], []
    for a, b in pairs:
        pixel = np.stack([(1 - t) * xt[a] + t * xt[b] for t in steps])
        rows.append(("Píxeles", pixel))
        for name in ("ae_z16", "vae_z16", "vae_z2"):
            m = models[name]
            za, zb = m.encode(xt[[a, b]])
            path = m.decoder.predict(np.stack([(1 - t) * za + t * zb for t in steps]), verbose=0)
            rows.append((LABELS[name], path))
        for label, imgs in rows[-4:]:
            conf_rows.append({"pair": f"{CLASS_NAMES_ES[yt[a]]} -> {CLASS_NAMES_ES[yt[b]]}", "path": label,
                              "mean_confidence": float(clf.predict(imgs, verbose=0).max(1).mean())})
    pd.DataFrame(conf_rows).to_csv(REPORTS / "interpolation_confidence.csv", index=False)
    fig, axes = plt.subplots(len(rows), 10, figsize=(10, 1.1 * len(rows)))
    for r, (label, imgs) in enumerate(rows):
        for c in range(10):
            axes[r, c].imshow(imgs[c].squeeze(), cmap="gray_r", vmin=0, vmax=1)
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
        axes[r, 0].set_ylabel(label, rotation=0, ha="right", va="center", fontsize=8)
    fig.suptitle("Interpolación: Zapatilla → Botín (arriba) y Camiseta/top → Pantalón (abajo)")
    fig.tight_layout()
    fig.savefig(REPORTS / "fig_interpolation.png", dpi=150)
    plt.close(fig)

    # espacio latente 2D del VAE
    vae2 = models["vae_z2"]
    z = vae2.encode(xt)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    sc = axes[0].scatter(z[:, 0], z[:, 1], c=yt, cmap="tab10", s=3)
    axes[0].legend(sc.legend_elements()[0], CLASS_NAMES_ES, fontsize=7, markerscale=1.5, loc="best")
    axes[0].set(title="Medias latentes del VAE z=2 (prueba)", xlabel="z1", ylabel="z2")
    from scipy.stats import norm
    grid = norm.ppf(np.linspace(0.05, 0.95, 15))
    canvas = np.zeros((28 * 15, 28 * 15))
    for i, yv in enumerate(grid[::-1]):
        decoded = vae2.decoder.predict(np.array([[xv, yv] for xv in grid], dtype="float32"), verbose=0)
        for j in range(15):
            canvas[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = decoded[j].squeeze()
    axes[1].imshow(canvas, cmap="gray_r")
    axes[1].set(title="Variedad decodificada (cuantiles 5-95 % de N(0, 1))", xticks=[], yticks=[])
    fig.tight_layout()
    fig.savefig(REPORTS / "fig_latent_space.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, h in history.groupby("model", sort=False):
        axes[0 if name.startswith("ae") else 1].plot(h["epoch"], h["val_loss"], "-o", ms=3, label=LABELS[name])
    axes[0].set(title="AE: BCE de validación", xlabel="Época", ylabel="BCE por imagen")
    axes[1].set(title="VAE: BCE + KL de validación", xlabel="Época", ylabel="Pérdida por imagen")
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS / "fig_learning_curves.png", dpi=150)
    plt.close(fig)

    ax = per_class_table.plot.barh(figsize=(8, 5))
    ax.set(title="Error de reconstrucción (MSE) por clase en prueba", xlabel="MSE por píxel")
    plt.tight_layout()
    plt.savefig(REPORTS / "fig_reconstruction_per_class.png", dpi=150)
    plt.close()


def memory_gallery(gen, x_train, name) -> None:
    dist, idx = nearest_distances(x_train, gen[:N_MEMORY])
    order = np.argsort(dist)[:8]
    fig, axes = plt.subplots(2, 8, figsize=(10, 3))
    for c, i in enumerate(order):
        axes[0, c].imshow(gen[i].squeeze(), cmap="gray_r", vmin=0, vmax=1)
        axes[0, c].set_title(f"d={dist[i]:.3f}", fontsize=8)
        axes[1, c].imshow(x_train[idx[i]].squeeze(), cmap="gray_r", vmin=0, vmax=1)
        for ax in axes[:, c]:
            ax.axis("off")
    fig.suptitle(f"Muestras de {LABELS[name]} más parecidas al entrenamiento (arriba) y su vecino (abajo)")
    fig.tight_layout()
    fig.savefig(REPORTS / "fig_memorization.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--limit", type=int, default=0, help="subconjunto para pruebas rápidas")
    args = parser.parse_args()
    r = run(args.epochs, args.limit)
    print("Seleccionado:", r["selected_model"], "| exactitud del clasificador de características:",
          round(r["classifier_test_accuracy"], 4))
    print(r["reconstruction"].round(4).to_string(index=False))
    print(r["generation"].round(4).to_string(index=False))
    print(r["cost"].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
