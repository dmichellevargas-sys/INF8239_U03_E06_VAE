"""VAE (Kingma y Welling, 2014) con paso de entrenamiento y de validación propios."""
from __future__ import annotations

import tensorflow as tf


def reconstruction_loss(images, reconstruction):
    """Entropía cruzada binaria sumada por imagen (784 píxeles), promediada en el lote."""
    eps = 1e-7
    rec = tf.clip_by_value(reconstruction, eps, 1 - eps)
    per_pixel = -(images * tf.math.log(rec) + (1 - images) * tf.math.log(1 - rec))
    return tf.reduce_mean(tf.reduce_sum(per_pixel, axis=(1, 2, 3)))


def kl_divergence(mean, log_variance):
    """KL(q(z|x) || N(0, I)) sumada sobre dimensiones, promediada en el lote."""
    return -0.5 * tf.reduce_mean(tf.reduce_sum(1 + log_variance - tf.square(mean) - tf.exp(log_variance), axis=1))


def kl_per_dimension(mean, log_variance):
    return -0.5 * tf.reduce_mean(1 + log_variance - tf.square(mean) - tf.exp(log_variance), axis=0)


def reparameterize(mean, log_variance, seed=None):
    epsilon = tf.random.normal(tf.shape(mean), seed=seed)
    return mean + tf.exp(0.5 * log_variance) * epsilon


class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, beta: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.encoder, self.decoder, self.beta = encoder, decoder, beta
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.reconstruction_tracker = tf.keras.metrics.Mean(name="reconstruction")
        self.kl_tracker = tf.keras.metrics.Mean(name="kl")

    @property
    def metrics(self):
        return [self.loss_tracker, self.reconstruction_tracker, self.kl_tracker]

    def call(self, images, training=False):
        mean, _ = self.encoder(images, training=training)
        return self.decoder(mean, training=training)

    def _compute_losses(self, images, training):
        mean, log_variance = self.encoder(images, training=training)
        reconstruction = self.decoder(reparameterize(mean, log_variance), training=training)
        rec = reconstruction_loss(images, reconstruction)
        kl = kl_divergence(mean, log_variance)
        return rec + self.beta * kl, rec, kl

    def _update(self, total, rec, kl):
        self.loss_tracker.update_state(total)
        self.reconstruction_tracker.update_state(rec)
        self.kl_tracker.update_state(kl)
        return {m.name: m.result() for m in self.metrics}

    def train_step(self, data):
        images = data[0] if isinstance(data, tuple) else data
        with tf.GradientTape() as tape:
            total, rec, kl = self._compute_losses(images, True)
        gradients = tape.gradient(total, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        return self._update(total, rec, kl)

    def test_step(self, data):
        images = data[0] if isinstance(data, tuple) else data
        return self._update(*self._compute_losses(images, False))
