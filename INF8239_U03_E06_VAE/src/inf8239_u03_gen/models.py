"""Arquitecturas densas simétricas: 784 -> 256 -> 128 -> z -> 128 -> 256 -> 784.

El autoencoder (AE) y el VAE comparten exactamente el mismo codificador y decodificador;
la única diferencia es que el VAE produce media y log-varianza y se entrena con el término KL.
"""
from __future__ import annotations

HIDDEN = (256, 128)


def build_decoder(tf, latent_dim: int, name: str = "decoder"):
    inputs = tf.keras.Input((latent_dim,))
    values = inputs
    for units in reversed(HIDDEN):
        values = tf.keras.layers.Dense(units, activation="relu")(values)
    values = tf.keras.layers.Dense(784, activation="sigmoid")(values)
    outputs = tf.keras.layers.Reshape((28, 28, 1))(values)
    return tf.keras.Model(inputs, outputs, name=name)


def _encoder_trunk(tf):
    inputs = tf.keras.Input((28, 28, 1))
    values = tf.keras.layers.Flatten()(inputs)
    for units in HIDDEN:
        values = tf.keras.layers.Dense(units, activation="relu")(values)
    return inputs, values


def build_ae_parts(tf, latent_dim: int = 16):
    inputs, values = _encoder_trunk(tf)
    latent = tf.keras.layers.Dense(latent_dim, name="latent")(values)  # lineal: código continuo
    encoder = tf.keras.Model(inputs, latent, name="ae_encoder")
    return encoder, build_decoder(tf, latent_dim, "ae_decoder")


def build_autoencoder(tf, latent_dim: int = 16):
    encoder, decoder = build_ae_parts(tf, latent_dim)
    inputs = tf.keras.Input((28, 28, 1))
    return tf.keras.Model(inputs, decoder(encoder(inputs)), name=f"autoencoder_z{latent_dim}")


def build_encoder_decoder(tf, latent_dim: int = 2):
    """Codificador variacional (media y log-varianza) y decodificador."""
    inputs, values = _encoder_trunk(tf)
    mean = tf.keras.layers.Dense(latent_dim, name="z_mean")(values)
    log_variance = tf.keras.layers.Dense(latent_dim, name="z_log_var")(values)
    encoder = tf.keras.Model(inputs, [mean, log_variance], name="vae_encoder")
    return encoder, build_decoder(tf, latent_dim, "vae_decoder")


def build_classifier(tf):
    """CNN pequeña usada SOLO como extractor de características para medir fidelidad."""
    return tf.keras.Sequential([
        tf.keras.layers.Input((28, 28, 1)),
        tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu", name="features"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ], name="feature_classifier")
