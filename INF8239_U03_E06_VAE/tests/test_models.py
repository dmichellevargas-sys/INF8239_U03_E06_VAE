import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
from inf8239_u03_gen.models import (
    build_ae_parts,
    build_autoencoder,
    build_classifier,
    build_encoder_decoder,
)
from inf8239_u03_gen.vae import (
    VAE,
    kl_divergence,
    kl_per_dimension,
    reconstruction_loss,
    reparameterize,
)


def test_autoencoder_output_contract():
    result = build_autoencoder(tf)(np.zeros((2, 28, 28, 1), dtype="float32")).numpy()
    assert result.shape == (2, 28, 28, 1)
    assert result.min() >= 0 and result.max() <= 1


def test_decoder_output_contract():
    _, decoder = build_encoder_decoder(tf, latent_dim=2)
    assert decoder(np.zeros((2, 2), dtype="float32")).numpy().shape == (2, 28, 28, 1)


def test_ae_and_vae_share_the_same_decoder_size():
    _, ae_dec = build_ae_parts(tf, 16)
    _, vae_dec = build_encoder_decoder(tf, 16)
    assert ae_dec.count_params() == vae_dec.count_params()


def test_kl_is_zero_for_standard_normal_posterior_and_positive_otherwise():
    zeros = tf.zeros((4, 3))
    assert float(kl_divergence(zeros, zeros)) == pytest.approx(0.0)
    assert float(kl_divergence(tf.ones((4, 3)), zeros)) == pytest.approx(1.5)
    np.testing.assert_allclose(kl_per_dimension(tf.ones((4, 3)), zeros).numpy(), 0.5)


def test_reconstruction_loss_is_small_for_perfect_reconstruction():
    images = tf.constant(np.random.default_rng(0).integers(0, 2, (2, 28, 28, 1)).astype("float32"))
    assert float(reconstruction_loss(images, images)) < 1e-3
    assert float(reconstruction_loss(images, 1 - images)) > 1000


def test_reparameterize_uses_mean_when_variance_is_tiny():
    mean = tf.constant([[1.0, -2.0]])
    z = reparameterize(mean, tf.fill((1, 2), -40.0), seed=1)
    np.testing.assert_allclose(z.numpy(), mean.numpy(), atol=1e-6)


def test_vae_one_training_step_returns_finite_losses():
    tf.keras.utils.set_random_seed(0)
    encoder, decoder = build_encoder_decoder(tf, 2)
    vae = VAE(encoder, decoder)
    vae.compile(optimizer="adam")
    x = np.random.default_rng(0).random((16, 28, 28, 1)).astype("float32")
    history = vae.fit(x, epochs=1, batch_size=8, verbose=0)
    assert all(np.isfinite(history.history[k][0]) for k in ("loss", "reconstruction", "kl"))
    assert vae(x[:2]).shape == (2, 28, 28, 1)


def test_classifier_exposes_feature_layer():
    clf = build_classifier(tf)
    assert clf.get_layer("features").units == 64
