from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models/decoder.keras"  # decodificador del VAE z=2
st.title("INF-8239 · Explorador del espacio latente (VAE z=2)")
st.caption("Demostración académica con Fashion-MNIST. Una muestra sintética no garantiza originalidad ni privacidad.")
if not MODEL_PATH.exists():
    st.error("Entrene los modelos antes de abrir la aplicación: uv run python scripts/run_all.py")
    st.stop()
decoder = tf.keras.models.load_model(MODEL_PATH)
x = st.slider("Latente z1", -3.0, 3.0, 0.0, 0.1)
y = st.slider("Latente z2", -3.0, 3.0, 0.0, 0.1)
image = decoder.predict(np.array([[x, y]], dtype="float32"), verbose=0)[0].squeeze()
st.image(1 - image, caption=f"Punto latente ({x:.1f}, {y:.1f})", clamp=True, width=280)
st.info("Fuera de [-3, 3] el decodificador no vio datos: las imágenes pierden realismo.")
