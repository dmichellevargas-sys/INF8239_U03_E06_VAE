"""Configuración central: fuente de datos fijada, clases, semilla y protocolo."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEED = 42

# Fashion-MNIST (Xiao et al., 2017), licencia MIT. Se descarga desde el repositorio oficial
# en un commit fijado y se verifica con SHA-256 (los mismos archivos que usa Keras).
DATASET_REPO = "https://github.com/zalandoresearch/fashion-mnist"
DATASET_COMMIT = "b2617bb6d3ffa2e429640350f613e3291e10b141"
RAW_BASE_URL = (
    "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/"
    f"{DATASET_COMMIT}/data/fashion"
)
RAW_FILES = {
    "train-images-idx3-ubyte.gz": "3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84",
    "train-labels-idx1-ubyte.gz": "a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845",
    "t10k-images-idx3-ubyte.gz": "346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073",
    "t10k-labels-idx1-ubyte.gz": "67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5",
}
RAW_DIR = ROOT / "data/raw"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]
CLASS_NAMES_ES = [
    "Camiseta/top", "Pantalón", "Suéter", "Vestido", "Abrigo",
    "Sandalia", "Camisa", "Zapatilla", "Bolso", "Botín",
]
VALID_FRACTION = 0.10

# Protocolo de evaluación generativa
N_GENERATED = 5000      # muestras por modelo para fidelidad, diversidad y memoria
N_REAL = 5000           # imágenes reales de referencia (validación o prueba)
KNN_K = 5               # vecinos para precisión y recall (Kynkäänniemi et al., 2019)
N_MEMORY = 2000         # consultas para la prueba de memorización
